"""Offline eval harness: tune the QA retrieval gate (T_v, T_b) on a labeled set.

What it does
------------
For each labeled question it runs the REAL retrieval pipeline once (query analysis
-> raw + HyDE vector probes + keyword BM25), recording:
  - best_vector : max cosine similarity across all vector probes
  - best_bm25   : max normalized ts_rank_cd over the BM25 leg
  - found       : whether the expected article URL appears in the fused top-N
Thresholds don't change those scores, so we sweep (T_v, T_b) analytically over the
recorded scores — cheap and exhaustive.

Labels (scripts/eval_set.json): a list of objects
  {"question": "...", "expect": "answer" | "refuse", "article_url": "..."?}
`article_url` is optional and only used to report retrieval hit-rate on answers.

Connects directly to Postgres + Ollama (offline op, not via the HTTP API).

Usage:
    uv run python scripts/eval_retrieval.py                      # run the sweep
    uv run python scripts/eval_retrieval.py --generate-template  # seed labels from query_logs
    uv run python scripts/eval_retrieval.py --top-k 20 --eval-file scripts/eval_set.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from sqlalchemy import text

from superbrain.app.application.qa.query_analysis import analyze_query, detect_url
from superbrain.app.application.retrieval.bm25_retriever import BM25Retriever
from superbrain.app.application.retrieval.fusion import reciprocal_rank_fusion
from superbrain.app.application.retrieval.vector_retriever import VectorRetriever
from superbrain.app.infrastructure.db.engine import (
    dispose_engine,
    get_session_factory,
    init_engine,
)
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import (
    ChunkRetrievalRepository,
)
from superbrain.app.infrastructure.embeddings.ollama_embedder import OllamaEmbedder
from superbrain.app.infrastructure.llm.ollama_llm import OllamaLLM
from superbrain.settings import get_settings

DEFAULT_EVAL_FILE = Path("scripts/eval_set.json")
T_V_GRID = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30 .. 0.80
T_B_GRID = [round(0.00 + 0.02 * i, 2) for i in range(11)]  # 0.00 .. 0.20


async def _generate_template(out: Path, limit: int) -> None:
    """Dump recent query_logs questions as an unlabeled template for hand-labeling."""
    settings = get_settings()
    init_engine(settings.database_url)
    async with get_session_factory()() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT question, aborted FROM query_logs "
                    "ORDER BY created_at DESC LIMIT :n"
                ),
                {"n": limit},
            )
        ).fetchall()
    await dispose_engine()

    seen: set[str] = set()
    entries = []
    for r in rows:
        if r.question in seen:
            continue
        seen.add(r.question)
        entries.append(
            {
                "question": r.question,
                "prior_outcome": "aborted" if r.aborted else "answered",
                "expect": "TODO: answer | refuse",
            }
        )
    out.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} unlabeled questions to {out}")
    print("Label each 'expect' as 'answer' or 'refuse', then run the sweep.")


async def _gather(eval_set: list[dict], top_k: int) -> list[dict]:
    """Run retrieval once per labeled question and record gate scores."""
    settings = get_settings()
    init_engine(settings.database_url)
    session_factory = get_session_factory()
    results: list[dict] = []

    async with httpx.AsyncClient() as http_client:
        embedder = OllamaEmbedder(settings=settings, http_client=http_client)
        llm = OllamaLLM(settings=settings, http_client=http_client)

        async with session_factory() as session:
            chunk_repo = ChunkRetrievalRepository(session)
            vector = VectorRetriever(embedder=embedder, chunk_repo=chunk_repo)
            bm25 = BM25Retriever(chunk_repo=chunk_repo)

            for item in eval_set:
                q = item["question"]
                if settings.qa_query_analysis_enabled:
                    analysis = await analyze_query(
                        llm, model=settings.ollama_query_analysis_model, question=q
                    )
                    keywords = analysis.keywords
                    passage = analysis.hypothetical_passage
                else:
                    keywords = q
                    passage = q

                probes = [q]
                if passage.strip() and passage.strip() != q.strip():
                    probes.append(passage)

                vector_lists = await vector.retrieve_multi(probes, top_k=top_k)
                bm25_chunks = await bm25.retrieve(keywords, top_k=top_k)

                best_vector = max(
                    (c.similarity_score for lst in vector_lists for c in lst),
                    default=0.0,
                )
                best_bm25 = max(
                    (c.similarity_score for c in bm25_chunks), default=0.0
                )

                found = None
                if item.get("article_url"):
                    fused = reciprocal_rank_fusion(
                        *vector_lists, bm25_chunks, top_n=settings.qa_evidence_top_n
                    )
                    found = any(c.url == item["article_url"] for c in fused)

                results.append(
                    {
                        "question": q,
                        "expect": item["expect"],
                        "best_vector": round(best_vector, 4),
                        "best_bm25": round(best_bm25, 4),
                        "found_expected": found,
                    }
                )
                print(
                    f"  [{item['expect']:>6}] v={best_vector:.3f} b={best_bm25:.3f} "
                    f"found={found}  {q[:60]}"
                )

    await dispose_engine()
    return results


def _sweep(scores: list[dict]) -> None:
    """Sweep (T_v, T_b) over recorded scores and report the best by balanced score."""
    answers = [s for s in scores if s["expect"] == "answer"]
    refuses = [s for s in scores if s["expect"] == "refuse"]
    if not answers and not refuses:
        print("no labeled answer/refuse entries — nothing to sweep.")
        return

    def passes(s: dict, t_v: float, t_b: float) -> bool:
        return s["best_vector"] >= t_v or s["best_bm25"] >= t_b

    grid = []
    for t_v in T_V_GRID:
        for t_b in T_B_GRID:
            answer_recall = (
                sum(passes(s, t_v, t_b) for s in answers) / len(answers)
                if answers else 1.0
            )
            refuse_rate = (
                sum(not passes(s, t_v, t_b) for s in refuses) / len(refuses)
                if refuses else 1.0
            )
            balanced = (answer_recall + refuse_rate) / 2
            grid.append((balanced, answer_recall, refuse_rate, t_v, t_b))

    # Best balanced; tie-break toward higher recall (lower false-refusal — the
    # original bug), then lower thresholds.
    grid.sort(key=lambda g: (g[0], g[1], -g[3], -g[4]), reverse=True)

    print(f"\nlabeled: {len(answers)} answer, {len(refuses)} refuse")
    print("\ntop (T_v, T_b) by balanced score (answer-recall + refuse-rate)/2:")
    print(f"  {'T_v':>5} {'T_b':>5} {'balanced':>9} {'ans_recall':>11} {'refuse':>8}")
    for balanced, ar, rr, t_v, t_b in grid[:8]:
        print(f"  {t_v:>5} {t_b:>5} {balanced:>9.3f} {ar:>11.3f} {rr:>8.3f}")

    best = grid[0]
    print(
        f"\nRecommended: qa_min_vector_similarity={best[3]}  "
        f"qa_min_bm25_score={best[4]}  (balanced={best[0]:.3f})"
    )
    answered_with_url = [s for s in answers if s["found_expected"] is not None]
    if answered_with_url:
        hit = sum(bool(s["found_expected"]) for s in answered_with_url)
        print(
            f"retrieval hit-rate (expected article in fused top-N): "
            f"{hit}/{len(answered_with_url)}"
        )


def _validate(eval_set: list[dict]) -> list[dict]:
    valid = []
    for item in eval_set:
        if not isinstance(item, dict) or "question" not in item:
            continue
        if item.get("expect") not in ("answer", "refuse"):
            print(f"  skipping unlabeled/invalid entry: {item.get('question', '')[:60]}")
            continue
        # Normalize: a URL question with expect=answer should carry article_url.
        if "article_url" not in item and detect_url(item["question"]):
            item["article_url"] = detect_url(item["question"])
        valid.append(item)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the QA retrieval gate on a labeled set.")
    parser.add_argument(
        "--generate-template", action="store_true",
        help="dump recent query_logs questions to label, then exit",
    )
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100, help="rows for --generate-template")
    args = parser.parse_args()

    if args.generate_template:
        template = args.eval_file.with_suffix(".template.json")
        asyncio.run(_generate_template(template, args.limit))
        return

    if not args.eval_file.exists():
        print(f"no eval set at {args.eval_file}. Run with --generate-template first.")
        return

    eval_set = _validate(json.loads(args.eval_file.read_text()))
    if not eval_set:
        print("eval set has no labeled entries.")
        return

    print(f"running retrieval for {len(eval_set)} labeled questions (top_k={args.top_k})...")
    scores = asyncio.run(_gather(eval_set, args.top_k))
    _sweep(scores)


if __name__ == "__main__":
    main()

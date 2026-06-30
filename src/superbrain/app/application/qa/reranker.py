"""SLM reranker: a pointwise relevance judge used as the precision gate.

Bi-encoder cosine (nomic) and BM25 rank are weak relevance signals — off-domain
text scores nearly as high as genuinely relevant text, so no fixed threshold on
them cleanly separates the two. This stage asks a small instruct model (phi3) to
score each candidate chunk's relevance to the CLEAN search query, in ONE batched
call, producing a calibrated signal that does separate relevant from irrelevant.

The top reranked score becomes the answer/refuse gate, and only high-scoring
chunks are kept as evidence — so the answer model never sees the scattered
grab-bag that produced the "simulated society" misrepresentation. On any failure
(LLM down, unparseable output) the caller degrades to the RRF ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.query_analysis import _strip_think
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk

log = structlog.get_logger(__name__)

_PROMPT = """You judge whether each passage helps answer a question. For EACH passage, give a relevance score from 0.0 (irrelevant) to 1.0 (directly answers the question). Be strict: a passage that is merely about a RELATED topic, not the question itself, scores low (<= 0.3).

QUESTION:
{query}

PASSAGES:
{passages}

Respond with ONLY a JSON object mapping each passage index (as a string) to its score. Include every index. Example: {{"0": 0.9, "1": 0.15, "2": 0.6}}

JSON:"""


@dataclass
class RerankResult:
    """Relevance scores aligned to the input chunk order (empty if rerank failed)."""

    scores: list[float]
    fell_back: bool


def _passage_block(chunks: list[RankedChunk], snippet_chars: int) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = " ".join((c.content or "").split())[:snippet_chars]
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)


def _parse_scores(raw: str, n: int) -> list[float] | None:
    """Strip <think>, parse the JSON map, and return n clamped scores in order."""
    cleaned = _strip_think(raw)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores: list[float] = []
    for i in range(n):
        value = data.get(str(i), data.get(i))
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        scores.append(max(0.0, min(1.0, score)))
    return scores


async def rerank(
    llm: LLMPort,
    *,
    model: str,
    query: str,
    chunks: list[RankedChunk],
    snippet_chars: int = 350,
) -> RerankResult:
    """Score each chunk's relevance to `query` in one batched LLM call.

    Returns scores aligned to `chunks`. On LLM error or unparseable output,
    returns fell_back=True with empty scores so the caller can degrade gracefully.
    """
    if not chunks:
        return RerankResult(scores=[], fell_back=False)

    prompt = _PROMPT.format(query=query, passages=_passage_block(chunks, snippet_chars))
    try:
        raw = await llm.complete(prompt, model=model, prompt_template="rerank_v1")
    except Exception as exc:  # never break QA on a rerank failure
        log.warning("rerank.llm_failed", error=str(exc))
        return RerankResult(scores=[], fell_back=True)

    scores = _parse_scores(raw, len(chunks))
    if scores is None:
        log.warning("rerank.parse_failed", raw=raw[:200])
        return RerankResult(scores=[], fell_back=True)
    return RerankResult(scores=scores, fell_back=False)

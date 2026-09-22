"""SLM reranker: a pointwise relevance judge used as the precision gate.

Bi-encoder cosine (nomic) and BM25 rank are weak relevance signals — off-domain
text scores nearly as high as genuinely relevant text, so no fixed threshold on
them cleanly separates the two. This stage asks a small instruct model (phi3) to
score each candidate chunk's relevance to the CLEAN search query, one passage per
call, producing a calibrated signal that does separate relevant from irrelevant.

One call per chunk rather than one batched call for the whole pool — a small
instruct model reliably scores a single passage but unreliably counts through a
batched list (observed losing ~50% of the scores regardless of batch size: 5/12,
then 3/6). Pointwise removes that failure mode by construction: there's nothing
to lose count of.

The top reranked score becomes the answer/refuse gate, and only high-scoring
chunks are kept as evidence — so the answer model never sees the scattered
grab-bag that produced the "simulated society" misrepresentation. An individual
chunk's call failing (LLM error, unparseable output) defaults that chunk to 0.0
— deprioritized, not dropped. Only a TOTAL failure (every call failed) makes the
caller refuse rather than answering off unranked evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import dspy
import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk
from superbrain.app.infrastructure.llm.dspy_lm import LLMPortLM

log = structlog.get_logger(__name__)


class _ScorePassage(dspy.Signature):
    """Judge whether this passage helps answer the question.

    Score from 0.0 (irrelevant) to 1.0 (directly answers the question). Be
    strict: a passage that is merely about a RELATED topic, not the question
    itself, scores low (<= 0.3).
    """

    query: str = dspy.InputField(desc="the question to judge passage relevance against")
    passage: str = dspy.InputField(desc="the passage to judge")
    score: float = dspy.OutputField(desc="relevance score from 0.0 to 1.0")


_score_passage = dspy.Predict(_ScorePassage)


@dataclass
class RerankResult:
    """Relevance scores aligned to the input chunk order.

    A chunk whose scoring call failed gets 0.0 rather than being dropped.
    fell_back is True only when EVERY call failed — a total judge outage, not
    an isolated parse blip — so the caller can refuse safely.
    """

    scores: list[float]
    fell_back: bool


async def _score_one(llm: LLMPort, *, model: str, query: str, snippet: str) -> float | None:
    """Score one passage; None on any failure (LLM error or unparseable output)."""
    stage_lm = LLMPortLM(llm, model=model, prompt_template="rerank_v1")
    try:
        with dspy.context(lm=stage_lm, adapter=dspy.JSONAdapter()):
            result = await asyncio.to_thread(_score_passage, query=query, passage=snippet)
        return max(0.0, min(1.0, float(result.score)))
    except Exception as exc:  # bad JSON, non-numeric score, LLM down — isolate to this chunk
        log.warning("rerank.item_failed", error=str(exc))
        return None


async def rerank(
    llm: LLMPort,
    *,
    model: str,
    query: str,
    chunks: list[RankedChunk],
    snippet_chars: int = 350,
) -> RerankResult:
    """Score each chunk's relevance to `query`, one passage per LLM call.

    Calls run one at a time — the shared priority queue serialises everything
    at the Ollama level regardless, so concurrency here would only complicate
    ordering without any wall-clock benefit. Returns fell_back=True only if
    every single call failed; a partial failure defaults those chunks to 0.0
    instead of discarding the whole judgment.
    """
    if not chunks:
        return RerankResult(scores=[], fell_back=False)

    results: list[float | None] = []
    for c in chunks:
        snippet = " ".join((c.content or "").split())[:snippet_chars]
        results.append(await _score_one(llm, model=model, query=query, snippet=snippet))

    if all(r is None for r in results):
        log.warning("rerank.all_failed", chunk_count=len(chunks))
        return RerankResult(scores=[], fell_back=True)

    return RerankResult(scores=[r if r is not None else 0.0 for r in results], fell_back=False)

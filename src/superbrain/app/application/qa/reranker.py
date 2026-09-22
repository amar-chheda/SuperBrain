"""SLM reranker: a pointwise relevance judge used as the precision gate.

Bi-encoder cosine (nomic) and BM25 rank are weak relevance signals — off-domain
text scores nearly as high as genuinely relevant text, so no fixed threshold on
them cleanly separates the two. This stage asks a small instruct model (phi3) to
score each candidate chunk's relevance to the CLEAN search query, in ONE batched
call, producing a calibrated signal that does separate relevant from irrelevant.

The top reranked score becomes the answer/refuse gate, and only high-scoring
chunks are kept as evidence — so the answer model never sees the scattered
grab-bag that produced the "simulated society" misrepresentation. On any failure
(LLM down, unparseable output, misaligned score count) the caller degrades to the
RRF ordering.
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


class _ScorePassages(dspy.Signature):
    """Judge whether each passage helps answer a question.

    Score each from 0.0 (irrelevant) to 1.0 (directly answers the question). Be
    strict: a passage that is merely about a RELATED topic, not the question
    itself, scores low (<= 0.3).
    """

    query: str = dspy.InputField(desc="the question to judge passage relevance against")
    passages: str = dspy.InputField(desc="numbered passages, one per line, like '[0] ...'")
    scores: list[float] = dspy.OutputField(
        desc="one relevance score 0.0-1.0 per passage, in the same order, same count as passages"
    )


_score_passages = dspy.Predict(_ScorePassages)


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


async def rerank(
    llm: LLMPort,
    *,
    model: str,
    query: str,
    chunks: list[RankedChunk],
    snippet_chars: int = 350,
) -> RerankResult:
    """Score each chunk's relevance to `query` in one batched LLM call.

    Returns scores aligned to `chunks`. On LLM error, unparseable output, or a
    score count that doesn't match the chunk count, returns fell_back=True with
    empty scores so the caller can degrade gracefully.
    """
    if not chunks:
        return RerankResult(scores=[], fell_back=False)

    stage_lm = LLMPortLM(llm, model=model, prompt_template="rerank_v1")
    try:
        with dspy.context(lm=stage_lm, adapter=dspy.JSONAdapter()):
            result = await asyncio.to_thread(
                _score_passages, query=query, passages=_passage_block(chunks, snippet_chars)
            )
    except Exception as exc:  # never break QA on a rerank failure
        log.warning("rerank.llm_failed", error=str(exc))
        return RerankResult(scores=[], fell_back=True)

    scores = result.scores
    if not isinstance(scores, list) or len(scores) != len(chunks):
        log.warning("rerank.parse_failed", scores=scores)
        return RerankResult(scores=[], fell_back=True)

    try:
        clamped = [max(0.0, min(1.0, float(s))) for s in scores]
    except (TypeError, ValueError):
        log.warning("rerank.parse_failed", scores=scores)
        return RerankResult(scores=[], fell_back=True)

    return RerankResult(scores=clamped, fell_back=False)

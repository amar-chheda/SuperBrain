"""Evidence set construction and sufficiency checks for grounded QA."""

from dataclasses import dataclass
from uuid import UUID

from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk

MIN_EVIDENCE_CHUNKS = 2
MIN_EVIDENCE_SCORE = 0.005


@dataclass
class Evidence:
    chunk_id: UUID
    article_id: UUID
    article_title: str | None
    article_url: str
    content: str
    rrf_score: float


def build_evidence_set(fused_chunks: list[RankedChunk]) -> list[Evidence]:
    return [
        Evidence(
            chunk_id=c.id,
            article_id=c.article_id,
            article_title=c.title,
            article_url=c.url,
            content=c.content,
            rrf_score=c.rrf_score,
        )
        for c in fused_chunks
    ]


def check_evidence_sufficiency(evidence: list[Evidence]) -> tuple[bool, str]:
    """Decide if there is enough evidence to attempt an answer.

    Returns (is_sufficient, reason_if_not). Refusing to answer is a deliberate
    design decision — a well-designed local system fails explicitly rather than
    hallucinating.
    """
    if len(evidence) < MIN_EVIDENCE_CHUNKS:
        return (
            False,
            f"Only {len(evidence)} chunks retrieved (minimum: {MIN_EVIDENCE_CHUNKS})",
        )
    if evidence[0].rrf_score < MIN_EVIDENCE_SCORE:
        return (
            False,
            f"Top chunk score {evidence[0].rrf_score:.4f} below threshold {MIN_EVIDENCE_SCORE}",
        )
    return True, ""

"""Models for retrieval orchestration and ranking."""

from dataclasses import dataclass

from superbrain.app.domain.models import StoredChunk


@dataclass(slots=True, frozen=True)
class RetrievalCandidate:
    """Scored candidate before final ranking assignment."""

    chunk: StoredChunk
    vector_score: float
    lexical_score: float
    fused_score: float


@dataclass(slots=True, frozen=True)
class ScoredChunk:
    """Final ranked chunk for evidence selection."""

    chunk: StoredChunk
    fused_score: float
    rank: int


@dataclass(slots=True, frozen=True)
class EvidenceSet:
    """Bundle of selected evidence chunks."""

    chunks: tuple[ScoredChunk, ...]


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Output of hybrid retrieval workflow."""

    normalized_query: str
    evidence: EvidenceSet

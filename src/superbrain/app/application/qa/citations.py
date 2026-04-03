"""Citation mapping from retrieval evidence."""

from superbrain.app.application.retrieval.models import EvidenceSet
from superbrain.app.domain.models import Citation


class CitationBuilder:
    """Build citation payloads from ranked evidence chunks."""

    def build(self, evidence: EvidenceSet) -> tuple[Citation, ...]:
        """Create citations for all provided evidence chunks."""

        return tuple(
            Citation(
                article_id=scored.chunk.article_id,
                article_title=scored.chunk.article_title,
                article_url=scored.chunk.article_url,
                chunk_id=scored.chunk.chunk_id,
                snippet=scored.chunk.chunk_text[:280],
                rank=scored.rank,
                score=scored.fused_score,
            )
            for scored in evidence.chunks
        )

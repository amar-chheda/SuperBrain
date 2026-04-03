"""Unit tests for citation building."""

from uuid import uuid4

from superbrain.app.application.qa.citations import CitationBuilder
from superbrain.app.application.retrieval.models import EvidenceSet, ScoredChunk, StoredChunk


def test_citation_builder_maps_scored_chunks() -> None:
    """Citation builder should map evidence chunks into citation objects."""

    chunk = StoredChunk(
        chunk_id=uuid4(),
        article_id=uuid4(),
        article_title="Example",
        article_url="https://example.com/x",
        chunk_text="Some supporting text.",
        embedding=[0.1, 0.2],
    )
    evidence = EvidenceSet(chunks=(ScoredChunk(chunk=chunk, fused_score=0.5, rank=1),))

    citations = CitationBuilder().build(evidence)

    assert len(citations) == 1
    assert citations[0].article_title == "Example"
    assert citations[0].article_url == "https://example.com/x"
    assert citations[0].rank == 1

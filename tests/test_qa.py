"""Smoke tests for QA parsing, RRF fusion, and evidence sufficiency — no DB or Ollama."""

from dataclasses import dataclass
from uuid import UUID

from superbrain.app.application.qa.answer_generator import parse_answer_response
from superbrain.app.application.qa.evidence_builder import (
    Evidence,
    check_evidence_sufficiency,
)
from superbrain.app.application.retrieval.fusion import reciprocal_rank_fusion
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


def _evidence(chunk_id: UUID, rrf_score: float = 0.05) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        article_id=UUID("00000000-0000-0000-0000-000000000000"),
        article_title="Test Article",
        article_url="https://example.com",
        content="Some content",
        rrf_score=rrf_score,
    )


def _chunk(chunk_id: UUID, url: str = "https://example.com") -> RankedChunk:
    return RankedChunk(
        id=chunk_id,
        article_id=UUID("00000000-0000-0000-0000-000000000000"),
        content="content",
        chunk_index=0,
        title="Title",
        url=url,
        published_at=None,
        similarity_score=0.9,
        rrf_score=0.0,
    )


def test_parse_answer_response_valid():
    # SOURCES uses 1-based integer indices into the evidence list (see GROUNDED_QA_PROMPT).
    chunk_id = UUID("bbbbbbbb-0000-0000-0000-000000000001")
    evidence = [_evidence(chunk_id)]
    raw = "The answer is X.\nSOURCES: 1"
    answer, cited = parse_answer_response(raw, evidence)
    assert answer == "The answer is X."
    assert (1, chunk_id) in cited


def test_parse_answer_response_rejects_hallucinated_source():
    evidence = [_evidence(UUID("bbbbbbbb-0000-0000-0000-000000000001"))]
    raw = "The answer is X.\nSOURCES: 00000000-0000-0000-0000-000000000000"
    answer, cited = parse_answer_response(raw, evidence)
    assert len(cited) == 0


def test_parse_answer_response_missing_sources_line():
    evidence = [_evidence(UUID("bbbbbbbb-0000-0000-0000-000000000001"))]
    raw = "The answer is X."
    answer, cited = parse_answer_response(raw, evidence)
    assert answer == "The answer is X."
    assert cited == []


def test_rrf_fusion_scores_overlap_higher():
    chunk_a = _chunk(UUID("cccccccc-0000-0000-0000-000000000001"))
    chunk_b = _chunk(UUID("dddddddd-0000-0000-0000-000000000001"))
    vector_results = [chunk_a, chunk_b]
    bm25_results = [chunk_a]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    assert fused[0].id == chunk_a.id


def test_rrf_fusion_empty_lists():
    fused = reciprocal_rank_fusion([], [])
    assert fused == []


def test_evidence_sufficiency_passes():
    evidence = [_evidence(UUID("eeeeeeee-0000-0000-0000-000000000001"), rrf_score=0.05),
                _evidence(UUID("ffffffff-0000-0000-0000-000000000001"), rrf_score=0.04)]
    ok, reason = check_evidence_sufficiency(evidence)
    assert ok
    assert reason == ""


def test_evidence_sufficiency_aborts_too_few_chunks():
    evidence = [_evidence(UUID("eeeeeeee-0000-0000-0000-000000000001"), rrf_score=0.05)]
    ok, reason = check_evidence_sufficiency(evidence)
    assert not ok
    assert "minimum" in reason


def test_evidence_sufficiency_aborts_below_score_threshold():
    evidence = [_evidence(UUID("eeeeeeee-0000-0000-0000-000000000001"), rrf_score=0.001),
                _evidence(UUID("ffffffff-0000-0000-0000-000000000001"), rrf_score=0.001)]
    ok, reason = check_evidence_sufficiency(evidence)
    assert not ok
    assert "below threshold" in reason

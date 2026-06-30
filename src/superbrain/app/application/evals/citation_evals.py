"""Citation presence and groundedness checks for QA eval."""

from superbrain.app.application.evals.types import EvalResult
from superbrain.app.application.qa.evidence_builder import Evidence
from superbrain.app.application.qa.use_case import Citation


def check_citation_presence(
    answer: str,
    citations: list[Citation],
    must_cite_urls: list[str],
) -> EvalResult:
    """Verify that all required URLs appear in the citation list."""
    cited_urls = {c.article_url for c in citations}
    missing = [url for url in must_cite_urls if url not in cited_urls]
    passed = len(missing) == 0
    score = 1.0 - (len(missing) / max(len(must_cite_urls), 1))
    return EvalResult(
        name="citation_presence",
        passed=passed,
        score=score,
        details=f"Missing citations: {missing}" if missing else "All required citations present",
        duration_ms=0,
    )


def check_groundedness(
    answer: str,
    citations: list[Citation],
    evidence: list[Evidence],
) -> EvalResult:
    """Structural groundedness: all cited chunk IDs must exist in the evidence set.

    This is a structural check only — semantic groundedness requires human eval.
    """
    evidence_ids = {str(e.chunk_id) for e in evidence}
    citation_ids = {str(c.chunk_id) for c in citations}
    hallucinated = citation_ids - evidence_ids
    passed = len(hallucinated) == 0
    score = 1.0 - (len(hallucinated) / max(len(citation_ids), 1))
    return EvalResult(
        name="groundedness",
        passed=passed,
        score=score,
        details=f"Hallucinated IDs: {hallucinated}" if hallucinated else "No hallucinated sources",
        duration_ms=0,
    )


def check_answer_keywords(
    answer: str,
    expected_keywords: list[str],
) -> EvalResult:
    """Smoke-test that the answer contains expected terms (not a semantic check)."""
    answer_lower = answer.lower()
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    passed = len(missing) == 0
    score = 1.0 - (len(missing) / max(len(expected_keywords), 1))
    return EvalResult(
        name="keyword_presence",
        passed=passed,
        score=score,
        details=f"Missing keywords: {missing}" if missing else "All keywords present",
        duration_ms=0,
    )

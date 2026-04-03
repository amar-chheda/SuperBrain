"""Smoke tests for eval harness structure."""

from superbrain.app.evals.harness import (
    RetrievalEvalCase,
    check_citation_presence,
    check_groundedness,
    run_retrieval_eval_stub,
)


def test_eval_harness_smoke() -> None:
    """Eval harness utilities should return structured pass/fail results."""

    retrieval_case = RetrievalEvalCase(question="q", expected_url_substring="example.com")
    retrieval_result = run_retrieval_eval_stub(["https://example.com/a"], retrieval_case)
    citation_result = check_citation_presence(citation_count=1)
    grounded_result = check_groundedness(supported=True, citation_count=1)

    assert retrieval_result.passed is True
    assert citation_result.passed is True
    assert grounded_result.passed is True

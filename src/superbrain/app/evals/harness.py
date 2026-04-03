"""Lightweight evaluation hooks for retrieval and grounded QA."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RetrievalEvalCase:
    """Seeded retrieval eval case."""

    question: str
    expected_url_substring: str


@dataclass(slots=True, frozen=True)
class EvalResult:
    """Generic eval result payload."""

    name: str
    passed: bool
    details: str


def check_citation_presence(citation_count: int) -> EvalResult:
    """Check citation presence requirement for generated answers."""

    return EvalResult(
        name="citation_presence",
        passed=citation_count > 0,
        details=f"citation_count={citation_count}",
    )


def check_groundedness(supported: bool, citation_count: int) -> EvalResult:
    """Check simple groundedness rule for supported answers."""

    if supported and citation_count == 0:
        return EvalResult(
            name="groundedness",
            passed=False,
            details="supported answer without citations",
        )
    return EvalResult(name="groundedness", passed=True, details="ok")


def run_retrieval_eval_stub(retrieved_urls: list[str], case: RetrievalEvalCase) -> EvalResult:
    """Run a minimal retrieval expectation check."""

    passed = any(case.expected_url_substring in url for url in retrieved_urls)
    return EvalResult(
        name="retrieval_stub",
        passed=passed,
        details=f"expected={case.expected_url_substring}, retrieved={len(retrieved_urls)}",
    )

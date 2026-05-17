"""Eval harness types and runners for retrieval and QA quality checks."""

import time
from dataclasses import dataclass, field
from uuid import UUID

from superbrain.app.application.evals.retrieval_evals import (  # noqa: F401 (re-exported)
    check_recall_at_k,
    check_url_coverage,
)
from superbrain.app.application.evals.types import EvalResult
from superbrain.app.application.retrieval.fusion import reciprocal_rank_fusion
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


@dataclass
class RetrievalEvalCase:
    case_id: str
    question: str
    expected_chunk_ids: list[UUID]
    expected_article_urls: list[str]
    top_k: int = 10


@dataclass
class QAEvalCase:
    case_id: str
    question: str
    expected_keywords: list[str]
    must_cite_urls: list[str]
    must_not_hallucinate: bool = True


async def run_retrieval_eval(
    case: RetrievalEvalCase,
    vector_retriever: object,
    bm25_retriever: object,
) -> EvalResult:
    t0 = time.monotonic()
    vector_chunks: list[RankedChunk] = await vector_retriever.retrieve(case.question, top_k=case.top_k)  # type: ignore[attr-defined]
    bm25_chunks: list[RankedChunk] = await bm25_retriever.retrieve(case.question, top_k=case.top_k)  # type: ignore[attr-defined]
    fused = reciprocal_rank_fusion(vector_chunks, bm25_chunks, top_n=case.top_k)
    duration_ms = int((time.monotonic() - t0) * 1000)

    recall = check_recall_at_k(fused, case.expected_chunk_ids, case.top_k)
    url_cov = check_url_coverage(fused, case.expected_article_urls)
    score = (recall + url_cov) / 2

    passed = score >= 0.5
    details = f"recall@{case.top_k}={recall:.2f} url_coverage={url_cov:.2f}"

    return EvalResult(
        name=f"retrieval:{case.case_id}",
        passed=passed,
        score=score,
        details=details,
        duration_ms=duration_ms,
    )


async def run_qa_eval(
    case: QAEvalCase,
    qa_use_case: object,
) -> EvalResult:
    t0 = time.monotonic()
    result = await qa_use_case.execute(case.question)  # type: ignore[attr-defined]
    duration_ms = int((time.monotonic() - t0) * 1000)

    if result.aborted:
        return EvalResult(
            name=f"qa:{case.case_id}",
            passed=False,
            score=0.0,
            details=f"Aborted: {result.abort_reason}",
            duration_ms=duration_ms,
        )

    from superbrain.app.application.evals.citation_evals import (
        check_answer_keywords,
        check_citation_presence,
    )
    sub_results = []
    if case.expected_keywords and result.answer:
        sub_results.append(check_answer_keywords(result.answer, case.expected_keywords))
    if case.must_cite_urls:
        sub_results.append(check_citation_presence(result.answer or "", result.citations, case.must_cite_urls))

    if not sub_results:
        return EvalResult(
            name=f"qa:{case.case_id}",
            passed=True,
            score=1.0,
            details="No sub-checks defined",
            duration_ms=duration_ms,
        )

    score = sum(r.score for r in sub_results) / len(sub_results)
    passed = all(r.passed for r in sub_results)
    details = " | ".join(r.details for r in sub_results)

    return EvalResult(
        name=f"qa:{case.case_id}",
        passed=passed,
        score=round(score, 3),
        details=details,
        duration_ms=duration_ms,
    )


async def run_all_evals(
    retrieval_cases: list[RetrievalEvalCase],
    qa_cases: list[QAEvalCase],
    vector_retriever: object,
    bm25_retriever: object,
    qa_use_case: object,
) -> list[EvalResult]:
    results = []
    for case in retrieval_cases:
        results.append(await run_retrieval_eval(case, vector_retriever, bm25_retriever))
    for case in qa_cases:
        results.append(await run_qa_eval(case, qa_use_case))
    return results

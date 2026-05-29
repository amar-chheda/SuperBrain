"""QA use case: hybrid retrieval + grounded answer generation."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog

from superbrain.app.application.metrics import MetricsRecorder
from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.answer_generator import generate_answer
from superbrain.app.application.qa.evidence_builder import (
    MIN_VECTOR_SIMILARITY,
    build_evidence_set,
    check_evidence_sufficiency,
)
from superbrain.app.application.retrieval.bm25_retriever import BM25Retriever
from superbrain.app.application.retrieval.fusion import reciprocal_rank_fusion
from superbrain.app.application.retrieval.vector_retriever import VectorRetriever
from superbrain.app.domain.entities import QueryLog
from superbrain.app.domain.repositories import QueryLogRepository
from superbrain.settings import Settings

log = structlog.get_logger(__name__)


@dataclass
class Citation:
    number: int
    chunk_id: UUID
    article_title: str | None
    article_url: str
    excerpt: str


@dataclass
class QAResult:
    answer: str | None
    citations: list[Citation]
    aborted: bool
    abort_reason: str | None = None
    retrieval_latency_ms: int = 0
    answer_latency_ms: int = 0


class AskQuestionUseCase:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        llm: LLMPort,
        query_log_repo: QueryLogRepository,
        metrics: MetricsRecorder,
        settings: Settings,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._llm = llm
        self._query_log_repo = query_log_repo
        self._metrics = metrics
        self._settings = settings

    async def execute(self, question: str) -> QAResult:
        t_start = time.monotonic()

        vector_chunks = await self._vector_retriever.retrieve(question, top_k=20)
        bm25_chunks = await self._bm25_retriever.retrieve(question, top_k=20)
        retrieval_ms = int((time.monotonic() - t_start) * 1000)
        self._metrics.observe("retrieval_latency_ms", retrieval_ms)

        relevant_vector_chunks = [
            c for c in vector_chunks if c.similarity_score >= MIN_VECTOR_SIMILARITY
        ]
        if not relevant_vector_chunks:
            top_score = vector_chunks[0].similarity_score if vector_chunks else 0.0
            abort_reason = (
                f"No vector matches above similarity threshold {MIN_VECTOR_SIMILARITY} "
                f"(best score: {top_score:.3f}) — topic not in knowledge base"
            )
            log.info("qa.aborted", reason=abort_reason, question=question[:100])
            self._metrics.increment("qa_aborted_total")
            self._metrics.increment("qa_aborted_low_similarity_total")
            await self._query_log_repo.save(
                QueryLog(
                    id=uuid4(),
                    question=question,
                    answer=None,
                    evidence_chunk_ids=[],
                    retrieval_latency_ms=retrieval_ms,
                    answer_latency_ms=0,
                    aborted=True,
                    abort_reason=abort_reason,
                    created_at=datetime.now(tz=timezone.utc),
                    retrieval_trace=_build_trace(vector_chunks, bm25_chunks, [], []),
                )
            )
            return QAResult(
                answer=None,
                citations=[],
                aborted=True,
                abort_reason=abort_reason,
                retrieval_latency_ms=retrieval_ms,
            )

        fused = reciprocal_rank_fusion(relevant_vector_chunks, bm25_chunks, top_n=10)
        evidence = build_evidence_set(fused)

        is_sufficient, abort_reason = check_evidence_sufficiency(evidence)
        if not is_sufficient:
            log.info("qa.aborted", reason=abort_reason, question=question[:100])
            self._metrics.increment("qa_aborted_total")

            await self._query_log_repo.save(
                QueryLog(
                    id=uuid4(),
                    question=question,
                    answer=None,
                    evidence_chunk_ids=[],
                    retrieval_latency_ms=retrieval_ms,
                    answer_latency_ms=0,
                    aborted=True,
                    abort_reason=abort_reason,
                    created_at=datetime.now(tz=timezone.utc),
                    retrieval_trace=_build_trace(vector_chunks, bm25_chunks, fused, evidence),
                )
            )
            return QAResult(
                answer=None,
                citations=[],
                aborted=True,
                abort_reason=abort_reason,
                retrieval_latency_ms=retrieval_ms,
            )

        t_answer = time.monotonic()
        answer_text, cited_pairs, prompt_sent = await generate_answer(
            self._llm,
            model=self._settings.ollama_qa_model,
            question=question,
            evidence=evidence,
        )
        answer_ms = int((time.monotonic() - t_answer) * 1000)
        self._metrics.observe("answer_latency_ms", answer_ms)
        self._metrics.increment("qa_success_total")

        evidence_by_id = {e.chunk_id: e for e in evidence}
        citations = [
            Citation(
                number=n,
                chunk_id=chunk_id,
                article_title=evidence_by_id[chunk_id].article_title,
                article_url=evidence_by_id[chunk_id].article_url,
                excerpt=evidence_by_id[chunk_id].content[:200] + "..." if len(evidence_by_id[chunk_id].content) > 200 else evidence_by_id[chunk_id].content,
            )
            for n, chunk_id in cited_pairs
            if chunk_id in evidence_by_id
        ]

        await self._query_log_repo.save(
            QueryLog(
                id=uuid4(),
                question=question,
                answer=answer_text,
                evidence_chunk_ids=[chunk_id for _, chunk_id in cited_pairs],
                retrieval_latency_ms=retrieval_ms,
                answer_latency_ms=answer_ms,
                aborted=False,
                abort_reason=None,
                created_at=datetime.now(tz=timezone.utc),
                retrieval_trace=_build_trace(vector_chunks, bm25_chunks, fused, evidence, prompt_sent),
            )
        )

        log.info(
            "qa.answered",
            question=question[:100],
            citation_count=len(citations),
            retrieval_ms=retrieval_ms,
            answer_ms=answer_ms,
        )

        return QAResult(
            answer=answer_text,
            citations=citations,
            aborted=False,
            retrieval_latency_ms=retrieval_ms,
            answer_latency_ms=answer_ms,
        )


def _build_trace(
    vector_chunks: list,
    bm25_chunks: list,
    fused: list,
    evidence: list,
    prompt_sent: str | None = None,
) -> dict[str, Any]:
    return {
        "vector_hits": [
            {"chunk_id": str(c.id), "score": round(c.similarity_score, 4), "article": c.title, "url": c.url, "content": c.content[:300]}
            for c in vector_chunks
        ],
        "bm25_hits": [
            {"chunk_id": str(c.id), "score": round(c.similarity_score, 4), "article": c.title, "url": c.url, "content": c.content[:300]}
            for c in bm25_chunks
        ],
        "fused": [
            {"chunk_id": str(c.id), "rrf_score": round(c.rrf_score, 6), "article": c.title}
            for c in fused
        ],
        "evidence": [
            {"chunk_id": str(e.chunk_id), "rrf_score": round(e.rrf_score, 6), "article": e.article_title, "url": e.article_url, "content": e.content[:500]}
            for e in evidence
        ],
        "prompt_sent": prompt_sent,
    }

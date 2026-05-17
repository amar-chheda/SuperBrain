"""QA use case: hybrid retrieval + grounded answer generation."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

import structlog

from superbrain.app.application.metrics import MetricsRecorder
from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.answer_generator import generate_answer
from superbrain.app.application.qa.evidence_builder import (
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

        fused = reciprocal_rank_fusion(vector_chunks, bm25_chunks, top_n=10)
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
        answer_text, cited_ids = await generate_answer(
            self._llm,
            model=self._settings.ollama_qa_model,
            question=question,
            evidence=evidence,
        )
        answer_ms = int((time.monotonic() - t_answer) * 1000)
        self._metrics.observe("answer_latency_ms", answer_ms)
        self._metrics.increment("qa_success_total")

        cited_set = set(cited_ids)
        citations = [
            Citation(
                chunk_id=e.chunk_id,
                article_title=e.article_title,
                article_url=e.article_url,
                excerpt=e.content[:200] + "..." if len(e.content) > 200 else e.content,
            )
            for e in evidence
            if e.chunk_id in cited_set
        ]

        await self._query_log_repo.save(
            QueryLog(
                id=uuid4(),
                question=question,
                answer=answer_text,
                evidence_chunk_ids=[e.chunk_id for e in evidence],
                retrieval_latency_ms=retrieval_ms,
                answer_latency_ms=answer_ms,
                aborted=False,
                abort_reason=None,
                created_at=datetime.now(tz=timezone.utc),
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

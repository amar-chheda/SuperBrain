"""Question answering orchestration use case."""

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from superbrain.app.application.ports import ChatModelProvider, RetrievalService
from superbrain.app.application.qa.citations import CitationBuilder
from superbrain.app.domain.models import QueryLogEntry, QueryRequest, QueryResponse
from superbrain.app.domain.repositories import QueryLogRepository
from superbrain.app.errors import ValidationError
from superbrain.app.observability.metrics import InMemoryMetricsRecorder, MetricsRecorder
from superbrain.app.observability.timing import timed
from superbrain.app.observability.tracing import TracingHook

logger = logging.getLogger(__name__)

class AskQuestionUseCase:
    """Answer a question using hybrid retrieval and grounded generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        chat_provider: ChatModelProvider,
        citation_builder: CitationBuilder,
        query_log_repository: QueryLogRepository,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        """Initialize QA workflow dependencies."""

        self._retrieval_service = retrieval_service
        self._chat_provider = chat_provider
        self._citation_builder = citation_builder
        self._query_log_repository = query_log_repository
        self._metrics = metrics or InMemoryMetricsRecorder()
        self._tracing = TracingHook("superbrain.qa")

    def ask(self, question: str, top_k: int = 6) -> QueryResponse:
        """Run grounded QA workflow and return structured response."""

        cleaned = question.strip()
        if not cleaned:
            raise ValidationError("question must not be empty")

        request = QueryRequest(id=uuid4(), question=cleaned, requested_at=datetime.now(UTC))

        with self._tracing.span("qa.retrieve"), timed() as retrieval_timing:
            retrieval_result = self._retrieval_service.retrieve(cleaned, limit=top_k)
        self._metrics.observe("qa.retrieval_latency_ms", retrieval_timing.elapsed_ms)

        with self._tracing.span("qa.generate"), timed() as generation_timing:
            generated = self._chat_provider.generate_answer(cleaned, retrieval_result.evidence)
        self._metrics.observe("qa.answer_latency_ms", generation_timing.elapsed_ms)

        evidence_citations = self._citation_builder.build(retrieval_result.evidence)
        allowed_chunk_ids = {str(citation.chunk_id) for citation in evidence_citations}
        selected_chunk_ids = [
            chunk_id for chunk_id in generated.citation_chunk_ids if chunk_id in allowed_chunk_ids
        ]
        selected_chunk_ids_set = set(selected_chunk_ids)

        if selected_chunk_ids:
            citations = tuple(
                citation
                for citation in evidence_citations
                if str(citation.chunk_id) in selected_chunk_ids_set
            )
        else:
            citations = tuple() if not generated.supported else evidence_citations[:2]

        response = QueryResponse(
            request_id=request.id,
            answer=generated.answer,
            citations=citations,
            supported=generated.supported,
            created_at=datetime.now(UTC),
        )

        self._query_log_repository.record(
            QueryLogEntry(
                query_request=request,
                query_response=response,
                retrieval_ms=retrieval_timing.elapsed_ms,
                generation_ms=generation_timing.elapsed_ms,
                evidence_chunk_ids=tuple(UUID(chunk_id) for chunk_id in selected_chunk_ids),
            )
        )
        if not generated.supported:
            self._metrics.increment("qa.low_evidence_count")

        logger.info(
            "qa_request_completed",
            extra={
                "request_id": str(request.id),
                "supported": generated.supported,
                "citation_count": len(response.citations),
            },
        )

        return response

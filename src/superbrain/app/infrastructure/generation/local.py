"""Local structured answer generation provider."""

from datetime import UTC, datetime

from superbrain.app.application.qa.models import GeneratedAnswer
from superbrain.app.application.retrieval.models import EvidenceSet
from superbrain.app.observability.model_calls import ModelCallLogger, ModelCallPayload


class LocalGroundedChatProvider:
    """Generate conservative grounded answers from provided evidence only."""

    def __init__(self, model_call_logger: ModelCallLogger | None = None) -> None:
        self._model_call_logger = model_call_logger

    def generate_answer(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        """Return structured answer constrained to retrieved evidence."""

        started_at = datetime.now(UTC)
        if not evidence.chunks:
            result = GeneratedAnswer(
                answer=(
                    "I do not have enough evidence in saved articles "
                    "to answer this question reliably."
                ),
                supported=False,
                citation_chunk_ids=[],
            )
            self._log_call(started_at=started_at, status="low_evidence")
            return result

        top_chunks = evidence.chunks[:2]
        snippets = [chunk.chunk.chunk_text[:180] for chunk in top_chunks]
        citation_ids = [str(chunk.chunk.chunk_id) for chunk in top_chunks]

        combined = " ".join(snippets)
        answer = (
            f"Based on saved articles, relevant evidence for '{question}' is: {combined}".strip()
        )

        result = GeneratedAnswer(answer=answer, supported=True, citation_chunk_ids=citation_ids)
        self._log_call(started_at=started_at, status="success")
        return result

    def _log_call(self, *, started_at: datetime, status: str) -> None:
        if self._model_call_logger is None:
            return
        self._model_call_logger.log(
            ModelCallPayload(
                provider="local_grounded",
                model_name="rule_based_qa",
                request_type="generate_answer",
                prompt_template="grounded_qa",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status=status,
            )
        )

    def health_check(self) -> bool:
        """Local rule-based provider is always healthy."""

        return True

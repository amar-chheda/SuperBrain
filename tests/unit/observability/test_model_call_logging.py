"""Unit tests for model call logging behavior."""

from datetime import UTC, datetime

from superbrain.app.infrastructure.embeddings.local import LocalHashEmbeddingProvider
from superbrain.app.observability.model_calls import ModelCallLogger, ModelCallPayload


class InMemoryModelCallLogRepository:
    """Simple repository double for model call logging tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record(self, **kwargs: object) -> None:
        self.records.append(kwargs)


def test_model_call_logger_persists_payload() -> None:
    """Logger should persist payload fields through repository."""

    repo = InMemoryModelCallLogRepository()
    logger = ModelCallLogger(repository=repo)

    payload = ModelCallPayload(
        provider="local",
        model_name="m",
        request_type="generate",
        prompt_template="template",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        status="success",
        retries=1,
        error_metadata=None,
        related_entity_id="job-1",
    )
    logger.log(payload)

    assert len(repo.records) == 1
    assert repo.records[0]["provider"] == "local"
    assert repo.records[0]["related_entity_id"] == "job-1"


def test_embedding_provider_emits_model_call_logs() -> None:
    """Embedding provider should emit model call log entries for requests."""

    repo = InMemoryModelCallLogRepository()
    logger = ModelCallLogger(repository=repo)
    provider = LocalHashEmbeddingProvider(dimensions=4, model_call_logger=logger)

    provider.embed_query("hello")
    provider.embed_documents(["a", "b"])

    assert len(repo.records) == 2
    request_types = {record["request_type"] for record in repo.records}
    assert request_types == {"embed_query", "embed_documents"}

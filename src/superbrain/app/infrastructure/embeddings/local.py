"""Local deterministic embedding provider."""

import hashlib
from datetime import UTC, datetime

from superbrain.app.application.ports import EmbeddingProvider
from superbrain.app.observability.model_calls import ModelCallLogger, ModelCallPayload


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Generate deterministic pseudo-embeddings for local development."""

    def __init__(
        self,
        dimensions: int = 16,
        model_call_logger: ModelCallLogger | None = None,
    ) -> None:
        """Initialize embedding vector size."""

        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._model_call_logger = model_call_logger

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings for a list of documents."""
        started_at = datetime.now(UTC)
        vectors = [self._embed(text) for text in texts]
        finished_at = datetime.now(UTC)
        self._log_call(
            request_type="embed_documents",
            started_at=started_at,
            finished_at=finished_at,
            status="success",
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embedding for a query."""
        started_at = datetime.now(UTC)
        vector = self._embed(text)
        finished_at = datetime.now(UTC)
        self._log_call(
            request_type="embed_query",
            started_at=started_at,
            finished_at=finished_at,
            status="success",
        )
        return vector

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = list(digest)
        vector: list[float] = []
        for index in range(self._dimensions):
            byte_value = values[index % len(values)]
            vector.append((byte_value / 255.0) * 2 - 1)
        return vector

    def _log_call(
        self,
        *,
        request_type: str,
        started_at: datetime,
        finished_at: datetime,
        status: str,
    ) -> None:
        if self._model_call_logger is None:
            return
        self._model_call_logger.log(
            ModelCallPayload(
                provider="local_hash",
                model_name=f"hash-{self._dimensions}d",
                request_type=request_type,
                prompt_template=None,
                started_at=started_at,
                finished_at=finished_at,
                status=status,
            )
        )

    def health_check(self) -> bool:
        """Local deterministic provider is always healthy."""

        return True

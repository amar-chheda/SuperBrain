"""Ollama embedding adapter implementing EmbeddingPort.

Uses Ollama's /api/embed endpoint in batch mode. Logs every call.
"""

import time
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import httpx
import structlog

from superbrain.app.application.ports import EmbeddingPort
from superbrain.app.domain.entities import ModelCallLog
from superbrain.app.infrastructure.db.engine import get_session_factory
from superbrain.app.infrastructure.db.repositories.model_call_log_repo import (
    SqlAlchemyModelCallLogRepository,
)
from superbrain.settings import Settings

log = structlog.get_logger(__name__)


class OllamaEmbedder(EmbeddingPort):
    """Generates embeddings via Ollama's /api/embed endpoint."""

    # nomic-embed-text is trained with task-instruction prefixes. Omitting them
    # (or mismatching query vs document) systematically depresses cosine scores
    # and degrades retrieval. Queries and documents MUST use the matching prefix.
    _NOMIC_TASK_PREFIXES = {
        "query": "search_query: ",
        "document": "search_document: ",
    }

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        """Initialise with settings and a shared httpx client.

        Args:
            settings: Application settings.
            http_client: Shared async httpx client.
        """
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_embedding_model
        self._client = http_client
        self._keep_alive = settings.ollama_keep_alive

    def _apply_task_prefix(
        self, texts: list[str], input_type: Literal["query", "document"]
    ) -> list[str]:
        """Prepend the nomic task-instruction prefix when using a nomic model.

        Non-nomic models are returned unchanged so the adapter stays generic.
        """
        if "nomic" not in self._model.lower():
            return texts
        prefix = self._NOMIC_TASK_PREFIXES[input_type]
        return [f"{prefix}{t}" for t in texts]

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: Literal["query", "document"] = "document",
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Non-empty list of strings to embed.
            input_type: "query" for search queries, "document" for corpus text.
                Selects the nomic task-instruction prefix so both sides share a space.

        Returns:
            List of 768-dimensional float vectors, one per input text.

        Raises:
            httpx.HTTPError: If the Ollama endpoint is unreachable.
        """
        started_at = datetime.now(UTC)
        started_mono = time.monotonic()
        status = "success"
        error_metadata: dict | None = None

        inputs = self._apply_task_prefix(texts, input_type)

        try:
            response = await self._client.post(
                f"{self._base_url}/api/embed",
                json={
                    "model": self._model,
                    "input": inputs,
                    "keep_alive": self._keep_alive,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            embeddings: list[list[float]] = response.json()["embeddings"]
            return embeddings

        except Exception as exc:
            status = "failed"
            error_metadata = {"error": str(exc)}
            log.error("embedder.failed", error=str(exc))
            raise

        finally:
            duration_ms = int((time.monotonic() - started_mono) * 1000)
            log.info("embedder.called", model=self._model,
                     batch_size=len(texts), duration_ms=duration_ms, status=status)
            await self._log_call(
                duration_ms=duration_ms,
                started_at=started_at,
                status=status,
                batch_size=len(texts),
                error_metadata=error_metadata,
            )

    async def _log_call(
        self,
        *,
        duration_ms: int,
        started_at: datetime,
        status: str,
        batch_size: int,
        error_metadata: dict | None,
    ) -> None:
        """Persist a ModelCallLog entry for this embedding call.

        Args:
            duration_ms: Total duration in milliseconds.
            started_at: When the call started.
            status: 'success' or 'failed'.
            batch_size: Number of texts embedded.
            error_metadata: Error details if the call failed.
        """
        session_factory = get_session_factory()
        if session_factory is None:
            return
        try:
            async with session_factory() as session:
                repo = SqlAlchemyModelCallLogRepository(session)
                await repo.save(ModelCallLog(
                    id=uuid4(),
                    provider="ollama",
                    model_name=self._model,
                    request_type="embedding",
                    prompt_template=f"embed_batch_{batch_size}",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    duration_ms=duration_ms,
                    status=status,  # type: ignore[arg-type]
                    retries=0,
                    error_metadata=error_metadata,
                    related_entity_id=None,
                ))
        except Exception as exc:
            log.warning("embedder.log_call_failed", error=str(exc))

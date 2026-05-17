"""Ollama LLM adapter implementing LLMPort.

Makes HTTP calls to Ollama's /api/generate endpoint with retry logic.
Logs every call to model_call_logs regardless of success or failure.
"""

import asyncio
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.domain.entities import ModelCallLog
from superbrain.app.domain.exceptions import LLMError
from superbrain.app.infrastructure.db.engine import get_session_factory
from superbrain.app.infrastructure.db.repositories.model_call_log_repo import (
    SqlAlchemyModelCallLogRepository,
)
from superbrain.settings import Settings

log = structlog.get_logger(__name__)


class OllamaLLM(LLMPort):
    """LLM completion via Ollama's /api/generate endpoint."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        """Initialise with settings and a shared httpx client.

        Args:
            settings: Application settings.
            http_client: Shared async httpx client.
        """
        self._base_url = settings.ollama_base_url
        self._client = http_client

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        prompt_template: str = "unknown",
        related_entity_id: UUID | None = None,
        max_retries: int = 3,
    ) -> str:
        """Request a text completion from Ollama.

        Retries on connection errors and 5xx responses with exponential backoff.
        Persists a ModelCallLog entry regardless of outcome.

        Args:
            prompt: The full prompt string.
            model: The Ollama model tag (e.g. 'llama3.1:8b').
            json_mode: If True, sets format='json' in the request.
            prompt_template: Template name for audit logging.
            related_entity_id: Entity UUID for audit correlation.
            max_retries: Maximum number of attempts before raising LLMError.

        Returns:
            The model's completion text.

        Raises:
            LLMError: If all retry attempts are exhausted.
        """
        started_at = datetime.now(UTC)
        started_mono = time.monotonic()
        retries = 0
        last_error: str = "unknown"

        body: dict = {"model": model, "prompt": prompt, "stream": False}
        if json_mode:
            body["format"] = "json"

        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    f"{self._base_url}/api/generate",
                    json=body,
                    timeout=120.0,
                )
                response.raise_for_status()
                text: str = response.json()["response"]

                await self._log_call(
                    model=model,
                    prompt_template=prompt_template,
                    started_at=started_at,
                    duration_ms=int((time.monotonic() - started_mono) * 1000),
                    status="success",
                    retries=retries,
                    related_entity_id=related_entity_id,
                )
                return text

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_error = str(exc)
                retries = attempt + 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

        duration_ms = int((time.monotonic() - started_mono) * 1000)
        await self._log_call(
            model=model,
            prompt_template=prompt_template,
            started_at=started_at,
            duration_ms=duration_ms,
            status="failed",
            retries=retries,
            related_entity_id=related_entity_id,
            error_metadata={"error": last_error},
        )
        raise LLMError(model, last_error)

    async def _log_call(
        self,
        *,
        model: str,
        prompt_template: str,
        started_at: datetime,
        duration_ms: int,
        status: str,
        retries: int,
        related_entity_id: UUID | None,
        error_metadata: dict | None = None,
    ) -> None:
        """Persist a ModelCallLog entry for this LLM call.

        Args:
            model: Model name used.
            prompt_template: Template identifier.
            started_at: When the call started.
            duration_ms: Total duration in milliseconds.
            status: 'success' or 'failed'.
            retries: Number of retries attempted.
            related_entity_id: Correlated entity UUID.
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
                    model_name=model,
                    request_type="extraction",
                    prompt_template=prompt_template,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    duration_ms=duration_ms,
                    status=status,  # type: ignore[arg-type]
                    retries=retries,
                    error_metadata=error_metadata,
                    related_entity_id=related_entity_id,
                ))
        except Exception as exc:
            log.warning("llm.log_call_failed", error=str(exc))

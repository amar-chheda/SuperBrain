"""Model-call logging utilities."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from superbrain.app.domain.repositories import ModelCallLogRepository


class Clock(Protocol):
    """Clock protocol for testable timestamps."""

    def now(self) -> datetime:
        """Return current timestamp."""


class UtcClock:
    """UTC clock implementation."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(slots=True, frozen=True)
class ModelCallPayload:
    """Structured model-call metadata payload."""

    provider: str
    model_name: str
    request_type: str
    prompt_template: str | None
    started_at: datetime
    finished_at: datetime
    status: str
    retries: int = 0
    error_metadata: str | None = None
    related_entity_id: str | None = None

    @property
    def duration_ms(self) -> float:
        """Return measured duration in milliseconds."""

        return (self.finished_at - self.started_at).total_seconds() * 1000


class ModelCallLogger:
    """Persist and emit model-call audit entries."""

    def __init__(self, repository: ModelCallLogRepository) -> None:
        self._repository = repository

    def log(self, payload: ModelCallPayload) -> None:
        """Persist model-call metadata payload."""

        self._repository.record(
            provider=payload.provider,
            model_name=payload.model_name,
            request_type=payload.request_type,
            prompt_template=payload.prompt_template,
            started_at=payload.started_at,
            finished_at=payload.finished_at,
            duration_ms=payload.duration_ms,
            status=payload.status,
            retries=payload.retries,
            error_metadata=payload.error_metadata,
            related_entity_id=payload.related_entity_id,
        )

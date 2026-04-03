"""Daily digest API routes and scheduler hooks."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from superbrain.app.api.dependencies import (
    get_digest_repository,
    get_generate_daily_digest_use_case,
    get_retry_failed_ingestion_use_case,
    get_scheduler,
)
from superbrain.app.api.errors import AppError
from superbrain.app.application.digest.use_case import GenerateDailyDigestUseCase
from superbrain.app.application.ingestion.retry import RetryFailedIngestionUseCase
from superbrain.app.domain.models import Digest, DigestItem
from superbrain.app.domain.repositories import DigestRepository
from superbrain.app.infrastructure.scheduling.persistent import PersistentScheduler

router = APIRouter(prefix="/digests", tags=["digests"])


class DigestItemResponse(BaseModel):
    """Serialized digest topic section."""

    topic_id: UUID | None
    topic_name: str
    summary: str
    source_urls: list[str]
    citation_article_ids: list[UUID]


class DigestResponse(BaseModel):
    """Serialized digest run payload."""

    id: UUID
    run_date: datetime
    status: str
    created_at: datetime
    items: list[DigestItemResponse]


class TriggerDigestRequest(BaseModel):
    """Request payload for on-demand digest execution."""

    notify_chat_id: str | None = None


class SchedulerTriggerResponse(BaseModel):
    """Response payload for manual scheduler trigger."""

    job_name: str
    success: bool


@router.post("/trigger", response_model=DigestResponse)
def trigger_digest_now(
    payload: TriggerDigestRequest,
    use_case: Annotated[
        GenerateDailyDigestUseCase,
        Depends(get_generate_daily_digest_use_case),
    ],
) -> DigestResponse:
    """Generate digest immediately and return persisted run."""

    digest = use_case.run(notify_chat_id=payload.notify_chat_id)
    return _to_digest_response(digest)


@router.get("/latest", response_model=DigestResponse)
def get_latest_digest(
    digest_repository: Annotated[DigestRepository, Depends(get_digest_repository)],
) -> DigestResponse:
    """Return latest persisted digest run."""

    digest = digest_repository.get_latest()
    if digest is None:
        raise AppError(code="digest_not_found", message="No digest runs found", status_code=404)
    return _to_digest_response(digest)


@router.get("/history", response_model=list[DigestResponse])
def get_digest_history(
    digest_repository: Annotated[DigestRepository, Depends(get_digest_repository)],
    limit: int = 20,
) -> list[DigestResponse]:
    """Return recent digest run history."""

    digests = digest_repository.list_recent(limit=limit)
    return [_to_digest_response(digest) for digest in digests]


@router.post("/scheduler/trigger/{job_name}", response_model=SchedulerTriggerResponse)
def trigger_scheduler_job(
    job_name: str,
    scheduler: Annotated[PersistentScheduler, Depends(get_scheduler)],
    use_case: Annotated[GenerateDailyDigestUseCase, Depends(get_generate_daily_digest_use_case)],
    retry_failed_ingestion_use_case: Annotated[
        RetryFailedIngestionUseCase,
        Depends(get_retry_failed_ingestion_use_case),
    ],
) -> SchedulerTriggerResponse:
    """Register baseline jobs and manually trigger one by name."""

    _ensure_scheduler_jobs(
        scheduler=scheduler,
        digest_use_case=use_case,
        retry_failed_ingestion_use_case=retry_failed_ingestion_use_case,
    )
    try:
        scheduler.trigger(job_name)
    except ValueError as exc:
        raise AppError(code="job_not_found", message=str(exc), status_code=404) from exc
    return SchedulerTriggerResponse(job_name=job_name, success=True)


def _ensure_scheduler_jobs(
    scheduler: PersistentScheduler,
    digest_use_case: GenerateDailyDigestUseCase,
    retry_failed_ingestion_use_case: RetryFailedIngestionUseCase,
) -> None:
    registered = set(scheduler.registered_jobs())

    if "daily_digest" not in registered:
        scheduler.register(
            name="daily_digest",
            cron_expression="0 7 * * *",
            handler=lambda: digest_use_case.run(),
        )

    if "retry_failed_ingestion" not in registered:
        scheduler.register(
            name="retry_failed_ingestion",
            cron_expression="*/30 * * * *",
            handler=lambda: retry_failed_ingestion_use_case.run(),
        )

    if "manual_digest" not in registered:
        scheduler.register(
            name="manual_digest",
            cron_expression="manual",
            handler=lambda: digest_use_case.run(),
        )


def _to_digest_response(digest: Digest) -> DigestResponse:
    return DigestResponse(
        id=digest.id,
        run_date=digest.run_date,
        status=digest.status.value,
        created_at=digest.created_at,
        items=[_to_item_response(item) for item in digest.items],
    )


def _to_item_response(item: DigestItem) -> DigestItemResponse:
    return DigestItemResponse(
        topic_id=item.topic_id,
        topic_name=item.topic_name,
        summary=item.summary,
        source_urls=list(item.source_urls),
        citation_article_ids=list(item.citation_article_ids),
    )

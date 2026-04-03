"""Ingestion API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, HttpUrl

from superbrain.app.api.dependencies import (
    get_ingest_url_use_case,
    get_ingestion_job_repository,
)
from superbrain.app.api.errors import AppError
from superbrain.app.application.ingestion.use_case import IngestUrlUseCase
from superbrain.app.domain.models import IngestionJob
from superbrain.app.domain.repositories import IngestionJobRepository

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class IngestUrlRequest(BaseModel):
    """Request payload for URL ingestion submission."""

    url: HttpUrl


class IngestionJobResponse(BaseModel):
    """Response payload describing ingestion job state."""

    job_id: UUID
    status: str
    source_url: str
    canonical_url: str
    article_id: UUID | None
    error_message: str | None


class IngestUrlResponse(BaseModel):
    """Response payload for URL ingestion submission result."""

    job_id: UUID
    status: str
    article_id: UUID | None
    duplicate: bool
    duplicate_reason: str | None
    canonical_url: str


@router.post("/jobs", response_model=IngestUrlResponse)
def submit_url_for_ingestion(
    payload: IngestUrlRequest,
    use_case: Annotated[IngestUrlUseCase, Depends(get_ingest_url_use_case)],
) -> IngestUrlResponse:
    """Submit a URL for ingestion and return resulting job metadata."""

    result = use_case.ingest(str(payload.url))
    return IngestUrlResponse(
        job_id=UUID(result.job_id),
        status=result.status.value,
        article_id=UUID(result.article_id) if result.article_id else None,
        duplicate=result.duplicate,
        duplicate_reason=result.duplicate_reason,
        canonical_url=result.canonical_url,
    )


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job_status(
    job_id: UUID,
    ingestion_jobs: Annotated[
        IngestionJobRepository,
        Depends(get_ingestion_job_repository),
    ],
) -> IngestionJobResponse:
    """Fetch ingestion job status by ID."""

    job = ingestion_jobs.get(job_id)
    if job is None:
        raise AppError(code="job_not_found", message="Ingestion job not found", status_code=404)
    return _to_job_response(job)


def _to_job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        job_id=job.id,
        status=job.status.value,
        source_url=job.source_url,
        canonical_url=job.canonical_url,
        article_id=job.article_id,
        error_message=job.error_message,
    )

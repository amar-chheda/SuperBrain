"""Ingestion job routes.

Handles job submission (POST /ingestion/jobs) and status polling
(GET /ingestion/jobs/{job_id}). Jobs are created immediately with
status='pending'; URL jobs are processed in the background via the
IngestArticleUseCase which crawls, chunks, embeds, and persists the article.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from pydantic import BaseModel, ValidationInfo, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.application.ingestion.use_case import IngestArticleUseCase
from superbrain.app.application.topics.use_cases import ClassifyArticleUseCase
from superbrain.app.domain.entities import IngestionJob
from superbrain.app.domain.exceptions import NotFoundError
from superbrain.app.infrastructure.db.engine import get_session
from superbrain.app.infrastructure.db.repositories.article_repo import (
    SqlAlchemyArticleRepository,
)
from superbrain.app.infrastructure.db.repositories.chunk_repo import (
    SqlAlchemyChunkRepository,
)
from superbrain.app.infrastructure.db.repositories.ingestion_job import (
    SqlAlchemyIngestionJobRepository,
)
from superbrain.app.infrastructure.db.repositories.topic_repo import (
    SqlAlchemyArticleTopicMatchRepository,
    SqlAlchemyTopicRepository,
)
from superbrain.settings import get_settings

router = APIRouter(prefix="/ingestion", tags=["ingestion"])
log = structlog.get_logger(__name__)


class CreateJobRequest(BaseModel):
    """Request body for submitting a new ingestion job."""

    input_type: str
    input_value: str

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v: str) -> str:
        """Ensure input_type is one of the accepted values.

        Args:
            v: The raw input_type string.

        Returns:
            The validated input_type.

        Raises:
            ValueError: If input_type is not url, pdf, or text.
        """
        if v not in ("url", "pdf", "text"):
            raise ValueError("input_type must be one of: url, pdf, text")
        return v

    @field_validator("input_value")
    @classmethod
    def validate_input_value(cls, v: str, info: ValidationInfo) -> str:
        """Validate that URL inputs are well-formed https URLs.

        Args:
            v: The raw input_value string.
            info: Pydantic validation info containing sibling field values.

        Returns:
            The validated input_value.

        Raises:
            ValueError: If input_type is 'url' but value is not an https URL.
        """
        if info.data.get("input_type") == "url" and not v.startswith("https://"):
            raise ValueError("URL inputs must start with https://")
        return v


class JobResponse(BaseModel):
    """API response shape for an ingestion job."""

    id: UUID
    input_type: str
    input_value: str
    status: str
    source: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None

    model_config = {"from_attributes": True}


def _job_to_response(job: IngestionJob) -> JobResponse:
    """Convert a domain entity to the API response model.

    Args:
        job: The domain entity.

    Returns:
        The serialisable response model.
    """
    return JobResponse(
        id=job.id,
        input_type=job.input_type,
        input_value=job.input_value,
        status=job.status,
        source=job.source,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error_message=job.error_message,
    )


async def _ingest_background(job_id: UUID, request: Request) -> None:
    """Background task: run the full ingestion pipeline for a job.

    Creates its own DB sessions since FastAPI's request session is closed
    by the time background tasks run. All dependencies are pulled from
    app.state which was populated at startup.

    Args:
        job_id: The job to process.
        request: The original request, used to access app.state.
    """
    structlog.contextvars.bind_contextvars(job_id=str(job_id))
    settings = get_settings()
    async for session in get_session():
        classify_use_case: ClassifyArticleUseCase | None = None
        if getattr(request.app.state, "classification_enabled", False):
            classify_use_case = ClassifyArticleUseCase(
                article_repo=SqlAlchemyArticleRepository(session),
                topic_repo=SqlAlchemyTopicRepository(session),
                match_repo=SqlAlchemyArticleTopicMatchRepository(session),
                llm=request.app.state.llm,
                metrics=request.app.state.metrics,
                settings=settings,
            )
        use_case = IngestArticleUseCase(
            article_repo=SqlAlchemyArticleRepository(session),
            chunk_repo=SqlAlchemyChunkRepository(session),
            ingestion_job_repo=SqlAlchemyIngestionJobRepository(session),
            crawler=request.app.state.crawler,
            embedder=request.app.state.embedder,
            llm=request.app.state.llm,
            chunker_factory=request.app.state.chunker_factory,
            metrics=request.app.state.metrics,
            settings=settings,
            classify_use_case=classify_use_case,
        )
        await use_case.execute(job_id)


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
async def create_job(
    body: CreateJobRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    """Submit a new ingestion job.

    The job is created immediately with status='pending'. Processing
    (crawl → chunk → embed → persist) runs asynchronously in the background.
    Poll GET /ingestion/jobs/{id} for status updates.

    Args:
        body: Validated request body with input_type and input_value.
        background_tasks: FastAPI background task queue.
        request: Current request (used to access app.state dependencies).
        session: Injected async database session.

    Returns:
        The created job record with status='pending'.
    """
    now = datetime.now(UTC)
    job = IngestionJob(
        id=uuid4(),
        input_type=body.input_type,  # type: ignore[arg-type]
        input_value=body.input_value,
        status="pending",
        source="api",
        created_at=now,
        updated_at=now,
    )

    repo = SqlAlchemyIngestionJobRepository(session)
    await repo.save(job)

    log.info("ingestion.job_created", job_id=str(job.id), input_type=job.input_type)
    structlog.contextvars.bind_contextvars(job_id=str(job.id))

    background_tasks.add_task(_ingest_background, job.id, request)

    return _job_to_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    """Retrieve an ingestion job by ID.

    Args:
        job_id: UUID of the job to retrieve.
        session: Injected async database session.

    Returns:
        The job record.

    Raises:
        NotFoundError: If no job with the given ID exists.
    """
    repo = SqlAlchemyIngestionJobRepository(session)
    job = await repo.find_by_id(job_id)
    if job is None:
        raise NotFoundError("IngestionJob", str(job_id))
    return _job_to_response(job)

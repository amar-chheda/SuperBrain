"""SQLAlchemy implementation of IngestionJobRepository."""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import IngestionJob
from superbrain.app.domain.repositories import IngestionJobRepository
from superbrain.app.infrastructure.db.models import IngestionJobModel

log = structlog.get_logger(__name__)


class SqlAlchemyIngestionJobRepository(IngestionJobRepository):
    """Persists and retrieves IngestionJob entities using SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an open database session.

        Args:
            session: The active async SQLAlchemy session.
        """
        self._session = session

    async def save(self, job: IngestionJob) -> None:
        """Persist a new ingestion job row.

        Args:
            job: The domain entity to persist.
        """
        model = IngestionJobModel(
            id=job.id,
            input_type=job.input_type,
            input_value=job.input_value,
            status=job.status,
            source=job.source,
            error_message=job.error_message,
            raw_text=job.raw_text,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        self._session.add(model)
        await self._session.commit()

    async def find_by_id(self, job_id: UUID) -> IngestionJob | None:
        """Find a job row by primary key and return a domain entity.

        Args:
            job_id: UUID of the job.

        Returns:
            The domain entity, or None if the row does not exist.
        """
        result = await self._session.execute(
            select(IngestionJobModel).where(IngestionJobModel.id == job_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def update_status(
        self,
        job_id: UUID,
        status: Literal["pending", "processing", "succeeded", "failed"],
        error_message: str | None = None,
    ) -> None:
        """Update a job's status and optional error message.

        Args:
            job_id: UUID of the job to update.
            status: New status value.
            error_message: Error detail when status is 'failed', else None.
        """
        result = await self._session.execute(
            select(IngestionJobModel).where(IngestionJobModel.id == job_id)
        )
        model = result.scalar_one_or_none()
        if model is not None:
            model.status = status
            model.error_message = error_message
            model.updated_at = datetime.now(UTC)
            await self._session.commit()

    async def update_crawl_result(
        self,
        job_id: UUID,
        status: Literal["pending", "processing", "succeeded", "failed"],
        raw_text: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a job's status and store crawled text or error details.

        Args:
            job_id: UUID of the job to update.
            status: New status value.
            raw_text: The crawled article text on success, else None.
            error_message: Error detail on failure, else None.
        """
        result = await self._session.execute(
            select(IngestionJobModel).where(IngestionJobModel.id == job_id)
        )
        model = result.scalar_one_or_none()
        if model is not None:
            model.status = status
            model.raw_text = raw_text
            model.error_message = error_message
            model.updated_at = datetime.now(UTC)
            await self._session.commit()

    @staticmethod
    def _to_entity(model: IngestionJobModel) -> IngestionJob:
        """Convert an ORM model row to a domain entity.

        Args:
            model: The SQLAlchemy ORM instance.

        Returns:
            The corresponding domain entity.
        """
        return IngestionJob(
            id=model.id,
            input_type=model.input_type,  # type: ignore[arg-type]
            input_value=model.input_value,
            status=model.status,  # type: ignore[arg-type]
            source=model.source,  # type: ignore[arg-type]
            error_message=model.error_message,
            raw_text=model.raw_text,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

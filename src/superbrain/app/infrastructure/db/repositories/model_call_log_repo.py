"""SQLAlchemy implementation of ModelCallLogRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import ModelCallLog
from superbrain.app.domain.repositories import ModelCallLogRepository
from superbrain.app.infrastructure.db.models import ModelCallLogModel


class SqlAlchemyModelCallLogRepository(ModelCallLogRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, log: ModelCallLog) -> None:
        self._session.add(
            ModelCallLogModel(
                id=log.id,
                provider=log.provider,
                model_name=log.model_name,
                request_type=log.request_type,
                prompt_template=log.prompt_template,
                started_at=log.started_at,
                finished_at=log.finished_at,
                duration_ms=log.duration_ms,
                status=log.status,
                retries=log.retries,
                error_metadata=log.error_metadata,
                related_entity_id=log.related_entity_id,
            )
        )
        await self._session.commit()

    async def list_recent(
        self,
        limit: int = 50,
        request_type: str | None = None,
        status: str | None = None,
    ) -> list[ModelCallLog]:
        """Return recent model call logs, optionally filtered by type and status."""
        stmt = select(ModelCallLogModel).order_by(ModelCallLogModel.started_at.desc()).limit(limit)
        if request_type:
            stmt = stmt.where(ModelCallLogModel.request_type == request_type)
        if status:
            stmt = stmt.where(ModelCallLogModel.status == status)
        result = await self._session.execute(stmt)
        return [_to_entity(m) for m in result.scalars()]

    async def list_by_entity(self, entity_id: UUID) -> list[ModelCallLog]:
        """Return all model call logs for a given related_entity_id (e.g. article ID)."""
        stmt = (
            select(ModelCallLogModel)
            .where(ModelCallLogModel.related_entity_id == entity_id)
            .order_by(ModelCallLogModel.started_at)
        )
        result = await self._session.execute(stmt)
        return [_to_entity(m) for m in result.scalars()]


def _to_entity(m: ModelCallLogModel) -> ModelCallLog:
    return ModelCallLog(
        id=m.id,
        provider=m.provider,
        model_name=m.model_name,
        request_type=m.request_type,  # type: ignore[arg-type]
        prompt_template=m.prompt_template,
        started_at=m.started_at,
        finished_at=m.finished_at,
        duration_ms=m.duration_ms,
        status=m.status,  # type: ignore[arg-type]
        retries=m.retries,
        error_metadata=m.error_metadata,
        related_entity_id=m.related_entity_id,
    )

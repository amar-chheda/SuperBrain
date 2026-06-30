"""SQLAlchemy implementation of DigestRepository."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import DigestItem, DigestRun
from superbrain.app.domain.repositories import DigestRepository
from superbrain.app.infrastructure.db.models import DigestItemModel, DigestRunModel


class SqlAlchemyDigestRepository(DigestRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_run(self, run: DigestRun) -> None:
        model = DigestRunModel(
            id=run.id,
            date_label=run.date_label,
            status=run.status,
            article_count=run.article_count,
            section_count=run.section_count,
            triggered_by=run.triggered_by,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error_message=run.error_message,
        )
        self._session.add(model)
        await self._session.commit()

    async def update_run(
        self,
        run_id: UUID,
        *,
        status: str,
        article_count: int = 0,
        section_count: int = 0,
        finished_at: object = None,
        error_message: str | None = None,
    ) -> None:
        result = await self._session.get(DigestRunModel, run_id)
        if result is None:
            return
        result.status = status
        result.article_count = article_count
        result.section_count = section_count
        result.finished_at = finished_at or datetime.now(tz=timezone.utc)
        result.error_message = error_message
        await self._session.commit()

    async def save_items(self, items: list[DigestItem]) -> None:
        for item in items:
            self._session.add(
                DigestItemModel(
                    id=item.id,
                    run_id=item.run_id,
                    topic_id=item.topic_id,
                    topic_name=item.topic_name,
                    summary=item.summary,
                    article_ids=item.article_ids,
                    article_urls=item.article_urls,
                    article_titles=item.article_titles,
                    position=item.position,
                    created_at=item.created_at,
                )
            )
        await self._session.commit()

    async def list_runs(self, limit: int = 30) -> list[DigestRun]:
        stmt = (
            select(DigestRunModel)
            .order_by(DigestRunModel.started_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [_run_to_entity(m) for m in result.scalars()]

    async def find_run_by_id(self, run_id: UUID) -> DigestRun | None:
        model = await self._session.get(DigestRunModel, run_id)
        return _run_to_entity(model) if model else None

    async def find_items_by_run(self, run_id: UUID) -> list[DigestItem]:
        stmt = (
            select(DigestItemModel)
            .where(DigestItemModel.run_id == run_id)
            .order_by(DigestItemModel.position)
        )
        result = await self._session.execute(stmt)
        return [_item_to_entity(m) for m in result.scalars()]


def _run_to_entity(m: DigestRunModel) -> DigestRun:
    return DigestRun(
        id=m.id,
        date_label=m.date_label,
        status=m.status,
        article_count=m.article_count,
        section_count=m.section_count,
        triggered_by=m.triggered_by,
        started_at=m.started_at,
        finished_at=m.finished_at,
        error_message=m.error_message,
    )


def _item_to_entity(m: DigestItemModel) -> DigestItem:
    return DigestItem(
        id=m.id,
        run_id=m.run_id,
        topic_id=m.topic_id,
        topic_name=m.topic_name,
        summary=m.summary,
        article_ids=list(m.article_ids),
        article_urls=list(m.article_urls),
        article_titles=list(m.article_titles),
        position=m.position,
        created_at=m.created_at,
    )

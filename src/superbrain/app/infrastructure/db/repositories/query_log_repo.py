"""SQLAlchemy implementation of QueryLogRepository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import QueryLog
from superbrain.app.domain.repositories import QueryLogRepository
from superbrain.app.infrastructure.db.models import QueryLogModel


class SqlAlchemyQueryLogRepository(QueryLogRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, log: QueryLog) -> None:
        self._session.add(
            QueryLogModel(
                id=log.id,
                question=log.question,
                answer=log.answer,
                evidence_chunk_ids=log.evidence_chunk_ids,
                retrieval_latency_ms=log.retrieval_latency_ms,
                answer_latency_ms=log.answer_latency_ms,
                aborted=log.aborted,
                abort_reason=log.abort_reason,
                retrieval_trace=log.retrieval_trace,
                created_at=log.created_at,
            )
        )
        await self._session.commit()

    async def list_recent(self, limit: int = 20) -> list[QueryLog]:
        stmt = (
            select(QueryLogModel)
            .order_by(QueryLogModel.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [_to_entity(m) for m in result.scalars()]


def _to_entity(m: QueryLogModel) -> QueryLog:
    return QueryLog(
        id=m.id,
        question=m.question,
        answer=m.answer,
        evidence_chunk_ids=list(m.evidence_chunk_ids or []),
        retrieval_latency_ms=m.retrieval_latency_ms or 0,
        answer_latency_ms=m.answer_latency_ms or 0,
        aborted=m.aborted,
        abort_reason=m.abort_reason,
        retrieval_trace=m.retrieval_trace,
        created_at=m.created_at,
    )

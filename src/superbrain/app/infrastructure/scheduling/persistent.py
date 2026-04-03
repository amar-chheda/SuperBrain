"""Persistent scheduler adapter with DB-backed job/run tracking."""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from superbrain.app.infrastructure.db.models import ScheduledJobRecord, ScheduledJobRunRecord


class PersistentScheduler:
    """DB-backed scheduler registry and manual trigger runner.

    This adapter is DBOS-ready: job metadata and execution history are persisted,
    and handlers are decoupled from registration records.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._handlers: dict[str, Callable[[], object]] = {}

    def register(self, name: str, cron_expression: str, handler: Callable[[], object]) -> None:
        """Register or update a scheduled job and bind runtime handler."""

        self._handlers[name] = handler
        with self._session_factory() as session:
            existing = session.scalar(
                select(ScheduledJobRecord).where(ScheduledJobRecord.name == name)
            )
            now = datetime.now(UTC)
            if existing is None:
                session.add(
                    ScheduledJobRecord(
                        id=uuid4(),
                        name=name,
                        cron_expression=cron_expression,
                        enabled=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                existing.cron_expression = cron_expression
                existing.enabled = True
                existing.updated_at = now
                session.add(existing)
            session.commit()

    def trigger(self, name: str) -> object:
        """Execute a registered job and persist execution outcome."""

        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"job not registered: {name}")

        started_at = datetime.now(UTC)
        run_id = uuid4()
        with self._session_factory() as session:
            session.add(
                ScheduledJobRunRecord(
                    id=run_id,
                    job_name=name,
                    started_at=started_at,
                    finished_at=None,
                    status="running",
                    error_message=None,
                )
            )
            session.commit()

        try:
            result = handler()
            self._finish_run(run_id=run_id, status="succeeded", error_message=None)
            return result
        except Exception as exc:
            self._finish_run(run_id=run_id, status="failed", error_message=str(exc))
            raise

    def registered_jobs(self) -> list[str]:
        """List registered runtime job names."""

        return sorted(self._handlers.keys())

    def _finish_run(
        self,
        *,
        run_id: UUID,
        status: str,
        error_message: str | None,
    ) -> None:
        with self._session_factory() as session:
            run = session.get(ScheduledJobRunRecord, run_id)
            if run is None:
                return
            run.status = status
            run.finished_at = datetime.now(UTC)
            run.error_message = error_message
            session.add(run)
            session.commit()

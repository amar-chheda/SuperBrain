"""In-process scheduler wrapping APScheduler for the daily digest job.

The run_digest callable is constructed at startup and manages its own DB
session per run, matching the same pattern used by the API background task.
"""

from collections.abc import Awaitable, Callable
from datetime import date
from typing import Literal

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from superbrain.app.domain.entities import DigestRun

log = structlog.get_logger(__name__)

RunDigestCallable = Callable[
    [date | None, Literal["scheduler", "manual", "api"]],
    Awaitable[DigestRun],
]


class SchedulerAdapter:
    def __init__(
        self,
        run_digest: RunDigestCallable,
        schedule_hour: int = 7,
    ) -> None:
        """
        run_digest: async callable(target_date, triggered_by) -> DigestRun.
                    Must manage its own DB session — called once per run.
        """
        self._run_digest = run_digest
        self._scheduler = AsyncIOScheduler()
        self._schedule_hour = schedule_hour

    def start(self) -> None:
        self._scheduler.add_job(
            self._scheduled_run,
            trigger=CronTrigger(hour=self._schedule_hour, minute=0, timezone="UTC"),
            id="daily_digest",
            replace_existing=True,
        )
        self._scheduler.start()
        log.info("scheduler.started", hour=self._schedule_hour)

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        log.info("scheduler.stopped")

    async def trigger_now(self, triggered_by: str = "manual") -> DigestRun:
        """Run the digest immediately — used by POST /digests/trigger and CLI."""
        return await self._run_digest(None, triggered_by)  # type: ignore[arg-type]

    async def _scheduled_run(self) -> None:
        try:
            await self._run_digest(None, "scheduler")
        except Exception as e:
            log.error("scheduler.digest_failed", error=str(e))

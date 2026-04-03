"""In-process scheduler abstraction for starter job orchestration."""

from collections.abc import Callable


class InMemoryScheduler:
    """Register and manually trigger named jobs in-process."""

    def __init__(self) -> None:
        self._jobs: dict[str, Callable[[], object]] = {}

    def register_daily_digest(self, handler: Callable[[], object]) -> None:
        """Register the daily digest execution handler."""

        self._jobs["daily_digest"] = handler

    def register_retry_failed_ingestion(self, handler: Callable[[], object]) -> None:
        """Register failed-ingestion retry handler."""

        self._jobs["retry_failed_ingestion"] = handler

    def register_manual(self, name: str, handler: Callable[[], object]) -> None:
        """Register an additional manually-triggerable job."""

        self._jobs[name] = handler

    def trigger(self, name: str) -> object:
        """Execute a registered job by name."""

        handler = self._jobs.get(name)
        if handler is None:
            raise ValueError(f"job not registered: {name}")
        return handler()

    def registered_jobs(self) -> list[str]:
        """List registered job names."""

        return sorted(self._jobs.keys())

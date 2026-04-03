"""Retry workflow for failed ingestion jobs."""

from superbrain.app.application.ingestion.use_case import IngestUrlUseCase
from superbrain.app.domain.repositories import IngestionJobRepository


class RetryFailedIngestionUseCase:
    """Retry failed ingestion jobs by re-submitting source URLs."""

    def __init__(
        self,
        ingestion_job_repository: IngestionJobRepository,
        ingest_url_use_case: IngestUrlUseCase,
    ) -> None:
        self._ingestion_job_repository = ingestion_job_repository
        self._ingest_url_use_case = ingest_url_use_case

    def run(self, limit: int = 10) -> int:
        """Retry failed ingestion jobs and return processed count."""

        failed_jobs = self._ingestion_job_repository.list_failed(limit=limit)
        for job in failed_jobs:
            self._ingest_url_use_case.ingest(job.source_url)
        return len(failed_jobs)

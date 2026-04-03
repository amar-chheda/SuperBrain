"""URL ingestion orchestration use case."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from superbrain.app.application.ingestion.deduplication import (
    DeduplicationReason,
    DeduplicationService,
)
from superbrain.app.application.ports import (
    ArticleExtractor,
    ChunkingStrategy,
    EmbeddingProvider,
    UrlCanonicalizer,
)
from superbrain.app.domain.models import Article, ArticleChunk, IngestionStatus, NewIngestionJob
from superbrain.app.domain.repositories import ArticleRepository, IngestionJobRepository
from superbrain.app.observability.metrics import InMemoryMetricsRecorder, MetricsRecorder
from superbrain.app.observability.timing import timed
from superbrain.app.observability.tracing import TracingHook

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class IngestUrlResult:
    """Output of URL ingestion use case."""

    job_id: str
    status: IngestionStatus
    article_id: str | None
    duplicate: bool
    duplicate_reason: str | None
    canonical_url: str


class IngestUrlUseCase:
    """Ingest an article URL into persisted article/chunk records."""

    def __init__(
        self,
        article_repository: ArticleRepository,
        ingestion_job_repository: IngestionJobRepository,
        canonicalizer: UrlCanonicalizer,
        extractor: ArticleExtractor,
        chunking_strategy: ChunkingStrategy,
        embedding_provider: EmbeddingProvider,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        """Initialize dependencies required for URL ingestion."""

        self._article_repository = article_repository
        self._ingestion_job_repository = ingestion_job_repository
        self._canonicalizer = canonicalizer
        self._extractor = extractor
        self._chunking_strategy = chunking_strategy
        self._embedding_provider = embedding_provider
        self._metrics = metrics or InMemoryMetricsRecorder()
        self._tracing = TracingHook("superbrain.ingestion")

    def ingest(self, url: str) -> IngestUrlResult:
        """Execute URL ingestion workflow and return resulting job state."""

        logger.info("ingestion_url_received", extra={"source_url": url})

        canonical_url = self._canonicalizer.canonicalize(url)
        job = self._ingestion_job_repository.create(
            NewIngestionJob(source_url=url, canonical_url=canonical_url).to_job()
        )
        self._ingestion_job_repository.update_status(job.id, IngestionStatus.RUNNING)

        deduplication_service = DeduplicationService(self._article_repository)
        with self._tracing.span("ingestion.dedup"), timed() as dedup_timing:
            dedup_result = deduplication_service.check_url(
                source_url=url,
                canonical_url=canonical_url,
            )
        logger.info(
            "ingestion_dedup_complete",
            extra={
                "job_id": str(job.id),
                "elapsed_ms": dedup_timing.elapsed_ms,
                "is_duplicate": dedup_result.is_duplicate,
                "reason": dedup_result.reason.value if dedup_result.reason else None,
            },
        )
        self._metrics.observe("ingestion.dedup_latency_ms", dedup_timing.elapsed_ms)

        if dedup_result.is_duplicate:
            self._metrics.increment("ingestion.duplicate_count")
            completed = self._ingestion_job_repository.update_status(
                job.id,
                IngestionStatus.SUCCEEDED,
                article_id=dedup_result.article_id,
            )
            return IngestUrlResult(
                job_id=str(completed.id),
                status=completed.status,
                article_id=str(completed.article_id) if completed.article_id else None,
                duplicate=True,
                duplicate_reason=dedup_result.reason.value if dedup_result.reason else None,
                canonical_url=canonical_url,
            )

        try:
            with self._tracing.span("ingestion.extract"), timed() as extraction_timing:
                extracted = self._extractor.extract(url)
            logger.info(
                "ingestion_extraction_complete",
                extra={"job_id": str(job.id), "elapsed_ms": extraction_timing.elapsed_ms},
            )
            self._metrics.observe("ingestion.extraction_latency_ms", extraction_timing.elapsed_ms)

            normalized_content = "\n".join(
                line.strip() for line in extracted.body_text.splitlines() if line.strip()
            )
            content_hash = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()

            content_dedup = deduplication_service.check_content_hash(content_hash)
            if content_dedup.is_duplicate:
                completed = self._ingestion_job_repository.update_status(
                    job.id,
                    IngestionStatus.SUCCEEDED,
                    article_id=content_dedup.article_id,
                )
                return IngestUrlResult(
                    job_id=str(completed.id),
                    status=completed.status,
                    article_id=str(completed.article_id) if completed.article_id else None,
                    duplicate=True,
                    duplicate_reason=DeduplicationReason.CONTENT_HASH.value,
                    canonical_url=canonical_url,
                )

            article = Article(
                id=uuid4(),
                source_url=url,
                canonical_url=self._canonicalizer.canonicalize(extracted.canonical_url),
                domain=extracted.domain,
                title=extracted.title,
                author=extracted.author,
                published_at=extracted.published_at,
                content=normalized_content,
                content_hash=content_hash,
                extraction_quality_score=extracted.extraction_quality_score,
                extraction_notes=extracted.extraction_notes,
                created_at=datetime.now(UTC),
            )

            persisted_article = self._article_repository.save(article)
            if extracted.raw_html is not None:
                self._article_repository.save_raw_snapshot(persisted_article.id, extracted.raw_html)

            with self._tracing.span("ingestion.chunk"), timed() as chunking_timing:
                chunk_drafts = self._chunking_strategy.chunk(normalized_content)
            logger.info(
                "ingestion_chunking_complete",
                extra={
                    "job_id": str(job.id),
                    "elapsed_ms": chunking_timing.elapsed_ms,
                    "chunk_count": len(chunk_drafts),
                },
            )
            self._metrics.observe("ingestion.chunking_latency_ms", chunking_timing.elapsed_ms)

            with self._tracing.span("ingestion.embed"), timed() as embedding_timing:
                embeddings = self._embedding_provider.embed_documents(
                    [chunk.text for chunk in chunk_drafts]
                )
            logger.info(
                "ingestion_embedding_complete",
                extra={"job_id": str(job.id), "elapsed_ms": embedding_timing.elapsed_ms},
            )
            self._metrics.observe("ingestion.embedding_latency_ms", embedding_timing.elapsed_ms)

            chunks = [
                ArticleChunk(
                    id=uuid4(),
                    article_id=persisted_article.id,
                    index=chunk.index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=embeddings[index],
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )
                for index, chunk in enumerate(chunk_drafts)
            ]
            self._article_repository.save_chunks(chunks)
            logger.info(
                "ingestion_persistence_complete",
                extra={"job_id": str(job.id), "article_id": str(persisted_article.id)},
            )

            completed = self._ingestion_job_repository.update_status(
                job.id,
                IngestionStatus.SUCCEEDED,
                article_id=persisted_article.id,
            )
            self._metrics.increment("ingestion.success_count")
            return IngestUrlResult(
                job_id=str(completed.id),
                status=completed.status,
                article_id=str(completed.article_id) if completed.article_id else None,
                duplicate=False,
                duplicate_reason=None,
                canonical_url=canonical_url,
            )
        except Exception as exc:
            failed = self._ingestion_job_repository.update_status(
                job.id,
                IngestionStatus.FAILED,
                error_message=str(exc),
            )
            self._metrics.increment("ingestion.failure_count")
            logger.exception("ingestion_failed", extra={"job_id": str(job.id)})
            return IngestUrlResult(
                job_id=str(failed.id),
                status=failed.status,
                article_id=None,
                duplicate=False,
                duplicate_reason=None,
                canonical_url=canonical_url,
            )

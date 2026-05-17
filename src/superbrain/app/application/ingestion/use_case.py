"""Ingestion pipeline use case.

Orchestrates the full article ingestion flow: crawl → dedup → chunk → embed → persist.
This is called by the background task registered in the ingestion API routes.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

from superbrain.app.application.ingestion.chunking_agent import decide_chunking_strategy
from superbrain.app.application.ingestion.dedup import compute_content_hash
from superbrain.app.application.metrics import MetricsRecorder
from superbrain.app.application.ports import CrawlerPort, EmbeddingPort, LLMPort
from superbrain.app.domain.entities import Article, Chunk
from superbrain.app.domain.exceptions import CrawlerError
from superbrain.app.domain.repositories import (
    ArticleRepository,
    ChunkRepository,
    IngestionJobRepository,
)
from superbrain.app.infrastructure.chunkers.factory import ChunkerFactory
from superbrain.app.infrastructure.chunkers.fixed import count_tokens
from superbrain.app.infrastructure.crawlers.url_utils import canonicalise_url
from superbrain.settings import Settings

if TYPE_CHECKING:
    from superbrain.app.application.topics.use_cases import ClassifyArticleUseCase

log = structlog.get_logger(__name__)


class IngestArticleUseCase:
    """Orchestrates the full article ingestion pipeline.

    Responsibilities (in order):
        1. Load the ingestion job
        2. Mark job as 'processing'
        3. Crawl the URL (or use stored raw_text for text/pdf jobs)
        4. Canonicalise URL and compute content hash
        5. Dedup check — skip if already ingested
        6. Persist Article with status='processing'
        7. LLM decides chunking strategy
        8. Chunk the text
        9. Embed all chunks (single batch call)
        10. Persist Chunk objects
        11. Mark Article as 'succeeded'
        12. Mark Job as 'succeeded'
    """

    def __init__(
        self,
        article_repo: ArticleRepository,
        chunk_repo: ChunkRepository,
        ingestion_job_repo: IngestionJobRepository,
        crawler: CrawlerPort,
        embedder: EmbeddingPort,
        llm: LLMPort,
        chunker_factory: ChunkerFactory,
        metrics: MetricsRecorder,
        settings: Settings,
        classify_use_case: ClassifyArticleUseCase | None = None,
    ) -> None:
        """Initialise with all required dependencies.

        Args:
            article_repo: Repository for Article persistence.
            chunk_repo: Repository for Chunk persistence.
            ingestion_job_repo: Repository for IngestionJob persistence.
            crawler: Web crawler backend.
            embedder: Embedding model backend.
            llm: LLM backend for chunking strategy decisions.
            chunker_factory: Factory that returns the right chunker by strategy.
            metrics: Shared in-memory metrics recorder.
            settings: Application settings (model names, etc.).
            classify_use_case: Optional topic classifier — when provided, every
                newly ingested article is classified immediately after embedding.
        """
        self._article_repo = article_repo
        self._chunk_repo = chunk_repo
        self._job_repo = ingestion_job_repo
        self._crawler = crawler
        self._embedder = embedder
        self._llm = llm
        self._chunker_factory = chunker_factory
        self._metrics = metrics
        self._settings = settings
        self._classify_use_case = classify_use_case

    async def execute(self, job_id: UUID) -> None:
        """Run the full ingestion pipeline for the given job.

        Loads the job, crawls the content, deduplicates, chunks, embeds,
        and persists everything. Updates job status throughout.

        Args:
            job_id: UUID of the IngestionJob to process.

        Raises:
            Exception: Re-raises any pipeline failure after marking the job failed.
        """
        structlog.contextvars.bind_contextvars(job_id=str(job_id))

        job = await self._job_repo.find_by_id(job_id)
        if job is None:
            log.error("ingestion.job_not_found", job_id=str(job_id))
            return

        await self._job_repo.update_status(job_id, "processing")

        try:
            # Step 3: Crawl
            crawl_start = time.monotonic()
            if job.input_type == "url":
                crawl_result = await self._crawler.fetch(job.input_value)
            else:
                # pdf/text jobs carry their content directly in input_value
                from superbrain.app.application.ports import CrawlResult
                crawl_result = CrawlResult(
                    url=job.input_value,
                    canonical_url=job.input_value,
                    raw_text=job.input_value if job.input_type == "text" else "",
                    title=None,
                    author=None,
                    published_at=None,
                    status_code=200,
                )
            crawl_ms = int((time.monotonic() - crawl_start) * 1000)
            self._metrics.observe("crawl_latency_ms", crawl_ms)

            if not crawl_result.raw_text:
                raise CrawlerError("Crawl returned empty text")

            # Step 4: Canonicalise + hash
            canonical_url = canonicalise_url(crawl_result.url)
            content_hash = compute_content_hash(crawl_result.raw_text)

            # Step 5: Dedup
            existing = await self._article_repo.find_by_hash(content_hash)
            if existing is not None:
                await self._job_repo.update_status(job_id, "succeeded")
                self._metrics.increment("ingestion_dedup_total")
                log.info("ingestion.dedup.skipped", article_id=str(existing.id))
                return

            # Step 6: Persist article (status=processing)
            article = Article(
                id=uuid4(),
                url=crawl_result.url,
                canonical_url=canonical_url,
                content_hash=content_hash,
                raw_text=crawl_result.raw_text,
                title=crawl_result.title,
                author=crawl_result.author,
                published_at=crawl_result.published_at,
                ingested_at=datetime.now(UTC),
                status="processing",
            )
            await self._article_repo.save(article)
            structlog.contextvars.bind_contextvars(article_id=str(article.id))

            # Step 7: LLM decides chunking strategy
            decision_start = time.monotonic()
            strategy = await decide_chunking_strategy(
                self._llm,
                model=self._settings.ollama_classification_model,
                article_text=crawl_result.raw_text,
                title=crawl_result.title,
                related_entity_id=article.id,
            )
            decision_ms = int((time.monotonic() - decision_start) * 1000)
            self._metrics.observe("chunk_decision_latency_ms", decision_ms)

            # Step 8: Chunk
            chunker = self._chunker_factory.get(strategy)
            chunk_texts = chunker.chunk(crawl_result.raw_text, strategy)

            if not chunk_texts:
                raise ValueError("Chunker returned no chunks")

            # Step 9: Embed (single batch call)
            embed_start = time.monotonic()
            embeddings = await self._embedder.embed(chunk_texts)
            embed_ms = int((time.monotonic() - embed_start) * 1000)
            self._metrics.observe("embedding_latency_ms", embed_ms)

            # Step 10: Build Chunk objects
            chunks = [
                Chunk(
                    id=uuid4(),
                    article_id=article.id,
                    content=text,
                    chunk_index=i,
                    strategy=strategy,
                    token_count=count_tokens(text),
                    embedding=embedding,
                )
                for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings))
            ]

            # Step 11: Persist chunks
            await self._chunk_repo.save_many(chunks)

            # Step 12: Mark article succeeded
            await self._article_repo.update_status(article.id, "succeeded")

            # Step 13: Mark job succeeded
            await self._job_repo.update_status(job_id, "succeeded")

            # Step 14: Classify article against active topics (optional)
            if self._classify_use_case is not None:
                try:
                    await self._classify_use_case.execute(article.id)
                except Exception as exc:
                    log.warning("ingestion.classification_failed",
                                article_id=str(article.id), error=str(exc))

            self._metrics.increment("ingestion_success_total")
            log.info(
                "ingestion.succeeded",
                article_id=str(article.id),
                chunk_count=len(chunks),
                strategy=strategy,
                crawl_ms=crawl_ms,
                embed_ms=embed_ms,
            )

        except Exception as exc:
            await self._job_repo.update_status(job_id, "failed", error_message=str(exc))
            self._metrics.increment("ingestion_failure_total")
            log.error("ingestion.failed", job_id=str(job_id), error=str(exc))
            raise

"""Topic classification use cases.

ClassifyArticleUseCase — classify a single article against all active topics.
ReclassifyTopicUseCase — re-run classification for all articles when a topic changes.
TopicCRUDUseCase — create, update (versioned), and archive topics.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import structlog

from superbrain.app.application.metrics import MetricsRecorder
from superbrain.app.application.ports import LLMPort
from superbrain.app.application.topics.classifier import TopicMatch, classify_article
from superbrain.app.domain.entities import Article, ArticleTopicMatch
from superbrain.app.domain.exceptions import NotFoundError
from superbrain.app.domain.repositories import (
    ArticleRepository,
    ArticleTopicMatchRepository,
    TopicRepository,
)
from superbrain.settings import Settings

log = structlog.get_logger(__name__)


class ClassifyArticleUseCase:
    """Classify a single article against all active topics and persist matches.

    Replaces all existing matches for the article on each run so that
    reclassification is idempotent.
    """

    def __init__(
        self,
        article_repo: ArticleRepository,
        topic_repo: TopicRepository,
        match_repo: ArticleTopicMatchRepository,
        llm: LLMPort,
        metrics: MetricsRecorder,
        settings: Settings,
    ) -> None:
        """Initialise with all required dependencies.

        Args:
            article_repo: Repository for Article retrieval.
            topic_repo: Repository for Topic retrieval.
            match_repo: Repository for ArticleTopicMatch persistence.
            llm: LLM backend.
            metrics: Shared in-memory metrics recorder.
            settings: Application settings (model names).
        """
        self._article_repo = article_repo
        self._topic_repo = topic_repo
        self._match_repo = match_repo
        self._llm = llm
        self._metrics = metrics
        self._settings = settings

    async def execute(self, article_id: UUID) -> list[ArticleTopicMatch]:
        """Classify an article and persist the results.

        Args:
            article_id: UUID of the article to classify.

        Returns:
            List of persisted ArticleTopicMatch records.

        Raises:
            NotFoundError: If no article with the given ID exists.
        """
        article = await self._article_repo.find_by_id(article_id)
        if article is None:
            raise NotFoundError("Article", str(article_id))

        topics = await self._topic_repo.list_active()
        if not topics:
            log.info("classification.skipped", reason="no_active_topics",
                     article_id=str(article_id))
            return []

        matches: list[TopicMatch] = await classify_article(
            self._llm,
            model=self._settings.ollama_classification_model,
            article=article,
            topics=topics,
        )

        topic_version_map = {t.id: t.version for t in topics}

        records = [
            ArticleTopicMatch(
                id=uuid4(),
                article_id=article_id,
                topic_id=m.topic_id,
                topic_version=topic_version_map.get(m.topic_id, 1),
                confidence=m.confidence,
                reason=m.reason,
                classified_at=datetime.now(UTC),
            )
            for m in matches
        ]

        await self._match_repo.delete_by_article(article_id)
        await self._match_repo.save_many(records)

        self._metrics.increment("classification_success_total")
        self._metrics.observe("topic_match_count", len(records))
        log.info(
            "classification.completed",
            article_id=str(article_id),
            match_count=len(records),
        )
        return records


class ReclassifyTopicUseCase:
    """Re-run classification for all articles against a single changed topic.

    Called when a topic definition changes (PUT /topics/{id}). Runs article
    by article — slow but safe. Never stops on a single article failure.
    """

    def __init__(
        self,
        article_repo: ArticleRepository,
        topic_repo: TopicRepository,
        match_repo: ArticleTopicMatchRepository,
        llm: LLMPort,
        settings: Settings,
    ) -> None:
        """Initialise with all required dependencies.

        Args:
            article_repo: Repository for Article retrieval.
            topic_repo: Repository for Topic retrieval.
            match_repo: Repository for ArticleTopicMatch persistence.
            llm: LLM backend.
            settings: Application settings (model names).
        """
        self._article_repo = article_repo
        self._topic_repo = topic_repo
        self._match_repo = match_repo
        self._llm = llm
        self._settings = settings

    async def execute(self, topic_id: UUID) -> None:
        """Reclassify all active articles against the updated topic.

        Args:
            topic_id: UUID of the topic to reclassify against.

        Raises:
            NotFoundError: If the topic does not exist.
        """
        topic = await self._topic_repo.find_by_id(topic_id)
        if topic is None:
            raise NotFoundError("Topic", str(topic_id))

        articles: list[Article] = await self._article_repo.list_all_active()

        log.info("reclassification.started", topic_id=str(topic_id),
                 article_count=len(articles))

        for article in articles:
            try:
                matches: list[TopicMatch] = await classify_article(
                    self._llm,
                    model=self._settings.ollama_classification_model,
                    article=article,
                    topics=[topic],
                )

                records = [
                    ArticleTopicMatch(
                        id=uuid4(),
                        article_id=article.id,
                        topic_id=m.topic_id,
                        topic_version=topic.version,
                        confidence=m.confidence,
                        reason=m.reason,
                        classified_at=datetime.now(UTC),
                    )
                    for m in matches
                ]

                await self._match_repo.upsert_for_topic(article.id, topic_id, records)

            except Exception as exc:
                log.error("reclassification.article_failed",
                          article_id=str(article.id), error=str(exc))
                continue

        log.info("reclassification.completed", topic_id=str(topic_id))

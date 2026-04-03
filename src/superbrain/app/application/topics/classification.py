"""Article classification and reclassification workflows."""

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from superbrain.app.application.topics.models import TopicClassificationDecision, TopicWithVersion
from superbrain.app.domain.models import TopicMatch
from superbrain.app.domain.repositories import (
    ArticleRepository,
    ArticleTopicMatchRepository,
    TopicRepository,
)
from superbrain.app.errors import NotFoundError
from superbrain.app.observability.metrics import InMemoryMetricsRecorder, MetricsRecorder
from superbrain.app.observability.tracing import TracingHook


class TopicClassifier(Protocol):
    """Abstraction for assigning topics to an article."""

    def classify(
        self,
        article_text: str,
        topics: list[TopicWithVersion],
    ) -> list[TopicClassificationDecision]:
        """Return per-topic classification decisions for an article."""
        ...


class ClassifyArticleUseCase:
    """Classify one article against active topic definitions."""

    def __init__(
        self,
        article_repository: ArticleRepository,
        topic_repository: TopicRepository,
        match_repository: ArticleTopicMatchRepository,
        classifier: TopicClassifier,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        """Initialize dependencies for article classification."""

        self._article_repository = article_repository
        self._topic_repository = topic_repository
        self._match_repository = match_repository
        self._classifier = classifier
        self._metrics = metrics or InMemoryMetricsRecorder()
        self._tracing = TracingHook("superbrain.classification")

    def classify(self, article_id: UUID) -> list[TopicMatch]:
        """Classify an article and persist topic matches."""

        article = self._article_repository.get(article_id)
        if article is None:
            raise NotFoundError("article not found")

        topics = [
            TopicWithVersion(topic=topic, version=version)
            for topic, version in self._topic_repository.list_active_with_latest_versions()
        ]
        with self._tracing.span("classification.run"):
            decisions = self._classifier.classify(article.content, topics)

        matched = [decision for decision in decisions if decision.matched]
        now = datetime.now(UTC)
        matches = [
            TopicMatch(
                article_id=article.id,
                topic_id=decision.topic_id,
                topic_version_id=decision.topic_version_id,
                score=decision.score,
                rationale=decision.rationale,
                disqualifiers=decision.disqualifiers,
                classified_at=now,
            )
            for decision in matched
        ]
        saved = self._match_repository.replace_for_article(article.id, matches)
        self._metrics.observe("classification.match_count", float(len(saved)))
        return saved


class ReclassifyArticlesUseCase:
    """Reclassify a scoped set of articles when topic definitions change."""

    def __init__(
        self,
        article_repository: ArticleRepository,
        classify_article_use_case: ClassifyArticleUseCase,
    ) -> None:
        """Initialize dependencies for bulk reclassification."""

        self._article_repository = article_repository
        self._classify_article_use_case = classify_article_use_case

    def reclassify(
        self,
        article_ids: list[UUID] | None = None,
        limit: int = 100,
    ) -> int:
        """Reclassify scoped articles and return processed count."""

        articles = self._article_repository.list_articles(limit=limit, article_ids=article_ids)
        for article in articles:
            self._classify_article_use_case.classify(article.id)
        return len(articles)

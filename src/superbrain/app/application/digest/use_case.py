"""Daily digest generation orchestration."""

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from uuid import UUID

from superbrain.app.application.digest.deduplication import DigestDeduper
from superbrain.app.application.digest.generator import DigestGenerator
from superbrain.app.application.digest.models import DigestSourceArticle
from superbrain.app.application.ports import TelegramClient
from superbrain.app.domain.models import Digest, DigestItem, DigestStatus, TopicMatch
from superbrain.app.domain.repositories import (
    ArticleRepository,
    ArticleTopicMatchRepository,
    DigestRepository,
    TopicRepository,
)
from superbrain.app.observability.metrics import InMemoryMetricsRecorder, MetricsRecorder
from superbrain.app.observability.tracing import TracingHook

logger = logging.getLogger(__name__)


class GenerateDailyDigestUseCase:
    """Generate and optionally dispatch a daily digest."""

    def __init__(
        self,
        article_repository: ArticleRepository,
        topic_repository: TopicRepository,
        match_repository: ArticleTopicMatchRepository,
        digest_repository: DigestRepository,
        deduper: DigestDeduper,
        generator: DigestGenerator,
        notifier: TelegramClient | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        """Initialize digest workflow dependencies."""

        self._article_repository = article_repository
        self._topic_repository = topic_repository
        self._match_repository = match_repository
        self._digest_repository = digest_repository
        self._deduper = deduper
        self._generator = generator
        self._notifier = notifier
        self._metrics = metrics or InMemoryMetricsRecorder()
        self._tracing = TracingHook("superbrain.digest")

    def run(
        self,
        *,
        run_date: datetime | None = None,
        notify_chat_id: str | None = None,
    ) -> Digest:
        """Execute digest generation for the previous UTC day by default."""

        base = run_date or datetime.now(UTC)
        target_day = (base - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        start = target_day
        end = target_day + timedelta(days=1)

        digest = self._digest_repository.create_run(run_date=start)

        try:
            with self._tracing.span("digest.select_sources"):
                articles = self._article_repository.list_between(start=start, end=end)
            if not articles:
                empty_digest = Digest(
                    id=digest.id,
                    run_date=digest.run_date,
                    status=DigestStatus.SUCCEEDED,
                    created_at=digest.created_at,
                    items=tuple(),
                )
                self._metrics.increment("digest.empty_count")
                return self._digest_repository.complete_run(empty_digest)

            article_ids = [article.id for article in articles]
            matches = self._match_repository.list_for_articles(article_ids)
            topics = {
                topic.id: topic
                for topic in self._topic_repository.list_all(active_only=False)
            }

            grouped_matches: dict[UUID, list[TopicMatch]] = defaultdict(list)
            for match in matches:
                grouped_matches[match.article_id].append(match)

            sources: list[DigestSourceArticle] = []
            for article in articles:
                article_matches = grouped_matches.get(article.id)
                if not article_matches:
                    sources.append(
                        DigestSourceArticle(
                            article=article,
                            topic_id=None,
                            topic_name="Uncategorized",
                        )
                    )
                    continue

                for match in article_matches:
                    topic = topics.get(match.topic_id)
                    topic_name = topic.name if topic is not None else "Unknown Topic"
                    sources.append(
                        DigestSourceArticle(
                            article=article,
                            topic_id=match.topic_id,
                            topic_name=topic_name,
                        )
                    )

            with self._tracing.span("digest.generate_sections"):
                deduped_sources = self._deduper.dedupe(sources)
                section_drafts = self._generator.generate_sections(deduped_sources)

            completed = Digest(
                id=digest.id,
                run_date=digest.run_date,
                status=DigestStatus.SUCCEEDED,
                created_at=digest.created_at,
                items=tuple(
                    DigestItem(
                        topic_id=section.topic_id,
                        topic_name=section.topic_name,
                        summary=section.summary,
                        source_urls=section.source_urls,
                        citation_article_ids=section.citation_article_ids,
                    )
                    for section in section_drafts
                ),
            )
            completed_digest = self._digest_repository.complete_run(completed)
            self._metrics.increment("digest.success_count")
            self._metrics.observe("digest.section_count", float(len(completed_digest.items)))

            if notify_chat_id and self._notifier is not None:
                message = self._render_notification(completed_digest)
                self._notifier.send_message(chat_id=notify_chat_id, text=message)

            return completed_digest
        except Exception as exc:
            logger.exception("digest_generation_failed")
            self._metrics.increment("digest.failure_count")
            return self._digest_repository.fail_run(digest.id, str(exc))

    def _render_notification(self, digest: Digest) -> str:
        lines = [f"Daily Digest ({digest.run_date.date().isoformat()})"]
        if not digest.items:
            lines.append("No relevant articles yesterday.")
            return "\n".join(lines)

        for item in digest.items:
            lines.append(f"- {item.topic_name}: {item.summary}")
        return "\n".join(lines)

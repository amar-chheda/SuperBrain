"""Daily digest generation use case — map-reduce over ingested articles."""

import dataclasses
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

import structlog

from superbrain.app.application.digest.deduplicator import deduplicate_sources_within_group
from superbrain.app.application.digest.grouper import group_by_topic, join_matches
from superbrain.app.application.digest.selector import select_articles_for_digest
from superbrain.app.application.digest.summariser import summarise_topic_group
from superbrain.app.application.metrics import MetricsRecorder
from superbrain.app.application.ports import LLMPort
from superbrain.app.domain.entities import DigestItem, DigestRun
from superbrain.app.domain.repositories import (
    ArticleRepository,
    ArticleTopicMatchRepository,
    DigestRepository,
    TopicRepository,
)
from superbrain.settings import Settings

log = structlog.get_logger(__name__)


class GenerateDailyDigestUseCase:
    def __init__(
        self,
        article_repo: ArticleRepository,
        match_repo: ArticleTopicMatchRepository,
        topic_repo: TopicRepository,
        digest_repo: DigestRepository,
        llm: LLMPort,
        metrics: MetricsRecorder,
        settings: Settings,
    ) -> None:
        self._article_repo = article_repo
        self._match_repo = match_repo
        self._topic_repo = topic_repo
        self._digest_repo = digest_repo
        self._llm = llm
        self._metrics = metrics
        self._settings = settings

    async def execute(
        self,
        target_date: date | None = None,
        triggered_by: Literal["scheduler", "manual", "api"] = "scheduler",
    ) -> DigestRun:
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        run = DigestRun(
            id=uuid4(),
            date_label=target_date,
            status="running",
            triggered_by=triggered_by,
            started_at=datetime.now(tz=timezone.utc),
        )
        await self._digest_repo.save_run(run)

        try:
            articles = await self._article_repo.list_by_date(target_date)
            matches = await self._match_repo.list_by_article_ids(
                [a.id for a in articles]
            )
            articles_with_topics = join_matches(articles, matches)
            selected = select_articles_for_digest(articles_with_topics, target_date)

            if not selected:
                log.info("digest.empty", date=str(target_date))
                await self._digest_repo.update_run(
                    run.id,
                    status="succeeded",
                    article_count=0,
                    section_count=0,
                    finished_at=datetime.now(tz=timezone.utc),
                )
                self._metrics.increment("digest_empty_total")
                return dataclasses.replace(run, status="succeeded")

            topics = await self._topic_repo.list_active()
            groups = group_by_topic(selected, topics)
            groups = [deduplicate_sources_within_group(g) for g in groups]

            items = []
            for position, group in enumerate(groups):
                summary = await summarise_topic_group(
                    self._llm,
                    model=self._settings.ollama_digest_model,
                    group=group,
                )
                if not summary:
                    continue
                items.append(
                    DigestItem(
                        id=uuid4(),
                        run_id=run.id,
                        topic_id=group.topic.id,
                        topic_name=group.topic.name,
                        summary=summary,
                        article_ids=[a.id for a in group.articles],
                        article_urls=[a.url for a in group.articles],
                        article_titles=[a.title or "" for a in group.articles],
                        position=position,
                        created_at=datetime.now(tz=timezone.utc),
                    )
                )

            await self._digest_repo.save_items(items)
            await self._digest_repo.update_run(
                run.id,
                status="succeeded",
                article_count=len(selected),
                section_count=len(items),
                finished_at=datetime.now(tz=timezone.utc),
            )

            self._metrics.increment("digest_success_total")
            self._metrics.observe("digest_section_count", len(items))
            log.info(
                "digest.succeeded",
                date=str(target_date),
                sections=len(items),
                articles=len(selected),
            )
            return dataclasses.replace(
                run, status="succeeded", section_count=len(items), article_count=len(selected)
            )

        except Exception as e:
            await self._digest_repo.update_run(
                run.id,
                status="failed",
                finished_at=datetime.now(tz=timezone.utc),
                error_message=str(e),
            )
            self._metrics.increment("digest_failure_total")
            log.error("digest.failed", date=str(target_date), error=str(e))
            raise

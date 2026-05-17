"""SQLAlchemy implementation of TopicRepository and ArticleTopicMatchRepository."""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import ArticleTopicMatch, Topic
from superbrain.app.domain.repositories import ArticleTopicMatchRepository, TopicRepository
from superbrain.app.infrastructure.db.models import ArticleTopicMatchModel, TopicModel


class SqlAlchemyTopicRepository(TopicRepository):
    """Persists and retrieves Topic entities using SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an open database session.

        Args:
            session: The active async SQLAlchemy session.
        """
        self._session = session

    async def save(self, topic: Topic) -> None:
        """Persist a new topic row.

        Args:
            topic: The domain entity to persist.
        """
        model = TopicModel(
            id=topic.id,
            name=topic.name,
            version=topic.version,
            description=topic.description,
            examples=topic.examples,
            priority=topic.priority,
            status=topic.status,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self._session.add(model)
        await self._session.commit()

    async def list_active(self) -> list[Topic]:
        """Return all topics with status='active', ordered by priority descending.

        Returns:
            Active topics ordered by priority descending.
        """
        result = await self._session.execute(
            select(TopicModel)
            .where(TopicModel.status == "active")
            .order_by(TopicModel.priority.desc())
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def list_all(self, include_archived: bool = False) -> list[Topic]:
        """Return all topics, optionally including archived ones.

        Args:
            include_archived: If True, include archived topics.

        Returns:
            Topics ordered by priority descending.
        """
        stmt = select(TopicModel).order_by(TopicModel.priority.desc())
        if not include_archived:
            stmt = stmt.where(TopicModel.status == "active")
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_by_id(self, topic_id: UUID) -> Topic | None:
        """Find a topic by primary key.

        Args:
            topic_id: UUID of the topic.

        Returns:
            The domain entity, or None if not found.
        """
        result = await self._session.execute(
            select(TopicModel).where(TopicModel.id == topic_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def set_status(
        self, topic_id: UUID, status: Literal["active", "archived"]
    ) -> None:
        """Update a topic's status.

        Args:
            topic_id: UUID of the topic to update.
            status: New status value.
        """
        result = await self._session.execute(
            select(TopicModel).where(TopicModel.id == topic_id)
        )
        model = result.scalar_one_or_none()
        if model is not None:
            model.status = status
            model.updated_at = datetime.now(UTC)
            await self._session.commit()

    @staticmethod
    def _to_entity(model: TopicModel) -> Topic:
        """Convert an ORM model to a domain entity.

        Args:
            model: The SQLAlchemy ORM instance.

        Returns:
            The corresponding domain entity.
        """
        return Topic(
            id=model.id,
            name=model.name,
            version=model.version,
            description=model.description,
            examples=model.examples or [],
            priority=model.priority,
            status=model.status,  # type: ignore[arg-type]
        )


class SqlAlchemyArticleTopicMatchRepository(ArticleTopicMatchRepository):
    """Persists and retrieves ArticleTopicMatch entities using SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an open database session.

        Args:
            session: The active async SQLAlchemy session.
        """
        self._session = session

    async def save_many(self, matches: list[ArticleTopicMatch]) -> None:
        """Persist a batch of topic matches.

        Args:
            matches: The matches to save.
        """
        for match in matches:
            model = ArticleTopicMatchModel(
                id=match.id,
                article_id=match.article_id,
                topic_id=match.topic_id,
                topic_version=match.topic_version,
                confidence=match.confidence,
                reason=match.reason,
                classified_at=match.classified_at,
            )
            self._session.add(model)
        if matches:
            await self._session.commit()

    async def delete_by_article(self, article_id: UUID) -> None:
        """Delete all matches for a given article.

        Args:
            article_id: UUID of the article.
        """
        await self._session.execute(
            delete(ArticleTopicMatchModel).where(
                ArticleTopicMatchModel.article_id == article_id
            )
        )
        await self._session.commit()

    async def upsert_for_topic(
        self,
        article_id: UUID,
        topic_id: UUID,
        matches: list[ArticleTopicMatch],
    ) -> None:
        """Replace the match for a specific article-topic pair.

        Args:
            article_id: UUID of the article.
            topic_id: UUID of the topic being reclassified.
            matches: New matches (may be empty if article no longer matches).
        """
        await self._session.execute(
            delete(ArticleTopicMatchModel).where(
                ArticleTopicMatchModel.article_id == article_id,
                ArticleTopicMatchModel.topic_id == topic_id,
            )
        )
        for match in matches:
            self._session.add(ArticleTopicMatchModel(
                id=match.id,
                article_id=match.article_id,
                topic_id=match.topic_id,
                topic_version=match.topic_version,
                confidence=match.confidence,
                reason=match.reason,
                classified_at=match.classified_at,
            ))
        await self._session.commit()

    async def find_by_article(self, article_id: UUID) -> list[ArticleTopicMatch]:
        """Return all matches for a given article.

        Args:
            article_id: UUID of the article.

        Returns:
            All topic matches for the article.
        """
        result = await self._session.execute(
            select(ArticleTopicMatchModel).where(
                ArticleTopicMatchModel.article_id == article_id
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def list_by_article_ids(
        self, article_ids: list[UUID]
    ) -> list[ArticleTopicMatch]:
        if not article_ids:
            return []
        result = await self._session.execute(
            select(ArticleTopicMatchModel).where(
                ArticleTopicMatchModel.article_id.in_(article_ids)
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    @staticmethod
    def _to_entity(model: ArticleTopicMatchModel) -> ArticleTopicMatch:
        """Convert an ORM model to a domain entity.

        Args:
            model: The SQLAlchemy ORM instance.

        Returns:
            The corresponding domain entity.
        """
        return ArticleTopicMatch(
            id=model.id,
            article_id=model.article_id,
            topic_id=model.topic_id,
            topic_version=model.topic_version,
            confidence=model.confidence,  # type: ignore[arg-type]
            reason=model.reason,
            classified_at=model.classified_at,
        )

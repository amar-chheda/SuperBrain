"""SQLAlchemy implementation of ArticleRepository."""

from datetime import date
from typing import Literal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import Article
from superbrain.app.domain.repositories import ArticleRepository
from superbrain.app.infrastructure.db.models import ArticleModel


class SqlAlchemyArticleRepository(ArticleRepository):
    """Persists and retrieves Article entities using SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an open database session.

        Args:
            session: The active async SQLAlchemy session.
        """
        self._session = session

    async def save(self, article: Article) -> None:
        """Persist a new article row.

        Args:
            article: The domain entity to persist.
        """
        model = ArticleModel(
            id=article.id,
            url=article.url,
            canonical_url=article.canonical_url,
            content_hash=article.content_hash,
            raw_text=article.raw_text,
            title=article.title,
            author=article.author,
            published_at=article.published_at,
            ingested_at=article.ingested_at,
            status=article.status,
        )
        self._session.add(model)
        await self._session.commit()

    async def find_by_hash(self, content_hash: str) -> Article | None:
        """Find an article by content hash for deduplication.

        Args:
            content_hash: SHA-256 hash of the article's raw text.

        Returns:
            The domain entity, or None if not found.
        """
        result = await self._session.execute(
            select(ArticleModel).where(ArticleModel.content_hash == content_hash)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_id(self, article_id: UUID) -> Article | None:
        """Find an article by primary key.

        Args:
            article_id: UUID of the article.

        Returns:
            The domain entity, or None if not found.
        """
        result = await self._session.execute(
            select(ArticleModel).where(ArticleModel.id == article_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_canonical_url(self, canonical_url: str) -> Article | None:
        """Find an article by its canonical URL (most recent if duplicated).

        Args:
            canonical_url: The canonicalised URL to look up.

        Returns:
            The most recently ingested matching article, or None if not found.
        """
        result = await self._session.execute(
            select(ArticleModel)
            .where(ArticleModel.canonical_url == canonical_url)
            .order_by(ArticleModel.ingested_at.desc())
            .limit(1)
        )
        model = result.scalars().first()
        return self._to_entity(model) if model else None

    async def update_status(
        self,
        article_id: UUID,
        status: Literal["pending", "processing", "succeeded", "failed"],
    ) -> None:
        """Update an article's processing status.

        Args:
            article_id: UUID of the article to update.
            status: The new status value.
        """
        result = await self._session.execute(
            select(ArticleModel).where(ArticleModel.id == article_id)
        )
        model = result.scalar_one_or_none()
        if model is not None:
            model.status = status
            await self._session.commit()

    async def list_all_active(self) -> list[Article]:
        """Return all articles with status='succeeded'.

        Returns:
            All successfully ingested articles.
        """
        result = await self._session.execute(
            select(ArticleModel).where(ArticleModel.status == "succeeded")
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def list_by_date(self, target_date: date) -> list[Article]:
        """List articles ingested on a given date.

        Args:
            target_date: The date to filter by.

        Returns:
            List of articles ingested on that date.
        """
        result = await self._session.execute(
            select(ArticleModel).where(
                func.date(ArticleModel.ingested_at) == target_date
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    @staticmethod
    def _to_entity(model: ArticleModel) -> Article:
        """Convert an ORM row to a domain entity.

        Args:
            model: The SQLAlchemy ORM instance.

        Returns:
            The corresponding domain entity.
        """
        return Article(
            id=model.id,
            url=model.url,
            canonical_url=model.canonical_url,
            content_hash=model.content_hash,
            raw_text=model.raw_text,
            title=model.title,
            author=model.author,
            published_at=model.published_at,
            ingested_at=model.ingested_at,
            status=model.status,  # type: ignore[arg-type]
        )

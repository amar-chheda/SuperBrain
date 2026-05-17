"""SQLAlchemy implementation of ChunkRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.domain.entities import Chunk
from superbrain.app.domain.repositories import ChunkRepository
from superbrain.app.infrastructure.db.models import ChunkModel


class SqlAlchemyChunkRepository(ChunkRepository):
    """Persists and retrieves Chunk entities using SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an open database session.

        Args:
            session: The active async SQLAlchemy session.
        """
        self._session = session

    async def save_many(self, chunks: list[Chunk]) -> None:
        """Persist a batch of chunks in a single transaction.

        Args:
            chunks: The chunks to save.
        """
        models = [
            ChunkModel(
                id=chunk.id,
                article_id=chunk.article_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                strategy=chunk.strategy,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
            )
            for chunk in chunks
        ]
        self._session.add_all(models)
        await self._session.commit()

    async def find_by_article(self, article_id: UUID) -> list[Chunk]:
        """Find all chunks for an article, ordered by chunk_index.

        Args:
            article_id: UUID of the parent article.

        Returns:
            Ordered list of chunks for the article.
        """
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.article_id == article_id)
            .order_by(ChunkModel.chunk_index)
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    @staticmethod
    def _to_entity(model: ChunkModel) -> Chunk:
        """Convert an ORM row to a domain entity.

        Args:
            model: The SQLAlchemy ORM instance.

        Returns:
            The corresponding domain entity.
        """
        embedding = model.embedding
        return Chunk(
            id=model.id,
            article_id=model.article_id,
            content=model.content,
            chunk_index=model.chunk_index,
            strategy=model.strategy,  # type: ignore[arg-type]
            token_count=model.token_count,
            embedding=list(embedding) if embedding is not None else None,
        )

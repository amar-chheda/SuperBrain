"""Repository for hybrid chunk retrieval (vector similarity + full-text search)."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger(__name__)


@dataclass
class RankedChunk:
    """A chunk returned from retrieval with its ranking score."""

    id: UUID
    article_id: UUID
    content: str
    chunk_index: int
    title: str | None
    url: str
    published_at: datetime | None
    similarity_score: float
    rrf_score: float = field(default=0.0)


class ChunkRetrievalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_vector(
        self, embedding: list[float], top_k: int = 20
    ) -> list[RankedChunk]:
        """Return top_k chunks ranked by cosine similarity to the query embedding."""
        vector_literal = "[" + ",".join(str(v) for v in embedding) + "]"
        sql = text(
            """
            SELECT
                c.id,
                c.article_id,
                c.content,
                c.chunk_index,
                a.title,
                a.url,
                a.published_at,
                1 - (c.embedding <=> :embedding::vector) AS similarity_score
            FROM chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE a.status = 'succeeded'
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> :embedding::vector
            LIMIT :top_k
            """
        )
        result = await self._session.execute(
            sql, {"embedding": vector_literal, "top_k": top_k}
        )
        rows = result.fetchall()
        return [_row_to_ranked_chunk(row) for row in rows]

    async def find_by_text(self, query: str, top_k: int = 20) -> list[RankedChunk]:
        """Return top_k chunks ranked by PostgreSQL full-text search (BM25-like)."""
        sql = text(
            """
            SELECT
                c.id,
                c.article_id,
                c.content,
                c.chunk_index,
                a.title,
                a.url,
                a.published_at,
                ts_rank_cd(c.content_tsv, plainto_tsquery('english', :query)) AS similarity_score
            FROM chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE c.content_tsv @@ plainto_tsquery('english', :query)
              AND a.status = 'succeeded'
            ORDER BY similarity_score DESC
            LIMIT :top_k
            """
        )
        result = await self._session.execute(sql, {"query": query, "top_k": top_k})
        rows = result.fetchall()
        return [_row_to_ranked_chunk(row) for row in rows]


def _row_to_ranked_chunk(row: object) -> RankedChunk:
    return RankedChunk(
        id=row.id,
        article_id=row.article_id,
        content=row.content,
        chunk_index=row.chunk_index,
        title=row.title,
        url=row.url,
        published_at=row.published_at,
        similarity_score=float(row.similarity_score),
    )

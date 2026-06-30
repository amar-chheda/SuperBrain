"""Repository for hybrid chunk retrieval (vector similarity + full-text search)."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _to_or_tsquery(text_value: str) -> str:
    """Build a sanitized OR `to_tsquery` expression ('a | b | c') from free text.

    OR-semantics (vs plainto_tsquery's AND) keeps BM25 recall-oriented: a few core
    keywords no longer have to ALL appear in one chunk. Precision is restored later
    by the reranker. Only alphanumeric tokens are kept, so to_tsquery never errors.
    """
    tokens = [t for t in _WORD_RE.findall(text_value.lower()) if len(t) >= 2]
    return " | ".join(dict.fromkeys(tokens))


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
            f"""
            SELECT
                c.id,
                c.article_id,
                c.content,
                c.chunk_index,
                a.title,
                a.url,
                a.published_at,
                1 - (c.embedding <=> '{vector_literal}'::vector) AS similarity_score
            FROM chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE a.status = 'succeeded'
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> '{vector_literal}'::vector
            LIMIT :top_k
            """
        )
        result = await self._session.execute(sql, {"top_k": top_k})
        rows = result.fetchall()
        return [_row_to_ranked_chunk(row) for row in rows]

    async def find_by_text(self, query: str, top_k: int = 20) -> list[RankedChunk]:
        """Return top_k chunks ranked by PostgreSQL full-text search (BM25-like).

        Uses OR-semantics (`to_tsquery('a | b | c')`) for recall — a chunk need not
        contain every term. The `32` normalization flag scales ts_rank_cd to
        rank/(rank+1), bounding it to [0, 1) so it is comparable against a fixed
        lexical floor (qa_min_bm25_score). ts_rank_cd still rewards chunks that
        match more terms with higher density.
        """
        tsq = _to_or_tsquery(query)
        if not tsq:
            return []
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
                ts_rank_cd(
                    c.content_tsv, to_tsquery('english', :tsq), 32
                ) AS similarity_score
            FROM chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE c.content_tsv @@ to_tsquery('english', :tsq)
              AND a.status = 'succeeded'
            ORDER BY similarity_score DESC
            LIMIT :top_k
            """
        )
        result = await self._session.execute(sql, {"tsq": tsq, "top_k": top_k})
        rows = result.fetchall()
        return [_row_to_ranked_chunk(row) for row in rows]

    async def find_by_article(
        self, article_id: UUID, limit: int = 50
    ) -> list[RankedChunk]:
        """Return one article's chunks (ordered by position) as RankedChunk rows.

        Used for direct article summarization (URL intent). similarity_score is a
        constant 1.0 — these are an exact article match, not a ranked search.
        """
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
                1.0 AS similarity_score
            FROM chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE c.article_id = :article_id
            ORDER BY c.chunk_index
            LIMIT :limit
            """
        )
        result = await self._session.execute(
            sql, {"article_id": article_id, "limit": limit}
        )
        return [_row_to_ranked_chunk(row) for row in result.fetchall()]


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

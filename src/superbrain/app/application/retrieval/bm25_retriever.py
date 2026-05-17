"""BM25-style lexical retriever using PostgreSQL full-text search."""

from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import (
    ChunkRetrievalRepository,
    RankedChunk,
)


class BM25Retriever:
    def __init__(self, chunk_repo: ChunkRetrievalRepository) -> None:
        self._chunk_repo = chunk_repo

    async def retrieve(self, query: str, top_k: int = 20) -> list[RankedChunk]:
        """Return top_k chunks ranked by PostgreSQL tsvector full-text search."""
        return await self._chunk_repo.find_by_text(query=query, top_k=top_k)

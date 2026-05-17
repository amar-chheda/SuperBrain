"""Vector similarity retriever using pgvector cosine distance."""

from superbrain.app.application.ports import EmbeddingPort
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import (
    ChunkRetrievalRepository,
    RankedChunk,
)


class VectorRetriever:
    def __init__(
        self, embedder: EmbeddingPort, chunk_repo: ChunkRetrievalRepository
    ) -> None:
        self._embedder = embedder
        self._chunk_repo = chunk_repo

    async def retrieve(self, query: str, top_k: int = 20) -> list[RankedChunk]:
        """Embed the query and return the top_k most similar chunks by cosine similarity."""
        [query_embedding] = await self._embedder.embed([query])
        return await self._chunk_repo.find_by_vector(
            embedding=query_embedding, top_k=top_k
        )

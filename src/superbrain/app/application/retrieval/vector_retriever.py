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
        [query_embedding] = await self._embedder.embed([query], input_type="query")
        return await self._chunk_repo.find_by_vector(
            embedding=query_embedding, top_k=top_k
        )

    async def retrieve_multi(
        self, queries: list[str], top_k: int = 20
    ) -> list[list[RankedChunk]]:
        """Embed several query probes in ONE batch and return a ranked list per probe.

        Used for multi-probe retrieval (raw query + HyDE passage). Batching the
        embeddings into a single call avoids extra Ollama round-trips. All probes
        are embedded with the "query" task prefix so they share nomic's space with
        the stored documents.
        """
        if not queries:
            return []
        embeddings = await self._embedder.embed(queries, input_type="query")
        return [
            await self._chunk_repo.find_by_vector(embedding=embedding, top_k=top_k)
            for embedding in embeddings
        ]

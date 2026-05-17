"""Chunker factory — maps strategy names to chunker instances.

Creates a single instance of each chunker and returns the appropriate one
for the given strategy. All chunkers implement ChunkerPort.
"""

from typing import Literal

from superbrain.app.application.ports import ChunkerPort
from superbrain.app.infrastructure.chunkers.fixed import FixedChunker
from superbrain.app.infrastructure.chunkers.recursive import RecursiveChunker
from superbrain.app.infrastructure.chunkers.semantic import SemanticChunker


class ChunkerFactory:
    """Provides chunker instances keyed by strategy name.

    Instantiates each chunker once at construction time and reuses them
    for all calls. Chunkers are stateless so sharing is safe.
    """

    def __init__(self) -> None:
        """Initialise all chunker instances with default parameters."""
        self._chunkers: dict[str, ChunkerPort] = {
            "semantic": SemanticChunker(max_tokens=400),
            "recursive": RecursiveChunker(max_tokens=400, overlap_chars=100),
            "fixed": FixedChunker(chunk_size=512, overlap=64),
        }

    def get(
        self, strategy: Literal["semantic", "recursive", "fixed"]
    ) -> ChunkerPort:
        """Return the chunker for the given strategy.

        Args:
            strategy: One of 'semantic', 'recursive', or 'fixed'.

        Returns:
            The corresponding ChunkerPort implementation.
        """
        return self._chunkers[strategy]

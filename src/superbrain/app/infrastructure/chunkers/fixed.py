"""Fixed-size token chunker using tiktoken.

Splits text into chunks of exactly chunk_size tokens with overlap.
Fastest and simplest — use as a baseline or for long/unstructured text.
"""

from typing import Literal

import tiktoken

from superbrain.app.application.ports import ChunkerPort

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MIN_CHARS = 50


class FixedChunker(ChunkerPort):
    """Splits text into fixed-size token windows with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        """Initialise with chunk size and overlap in tokens.

        Args:
            chunk_size: Target chunk size in tokens.
            overlap: Number of tokens to repeat at the start of each chunk.
        """
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(
        self,
        text: str,
        strategy: Literal["semantic", "recursive", "fixed"],
    ) -> list[str]:
        """Split text into fixed-size token chunks with overlap.

        Args:
            text: The full text to split.
            strategy: Ignored — this chunker always applies fixed splitting.

        Returns:
            List of text chunks, each at least 50 characters.
        """
        tokens = _ENCODING.encode(text)
        chunks: list[str] = []
        step = self._chunk_size - self._overlap
        start = 0

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk_text = _ENCODING.decode(tokens[start:end])
            if len(chunk_text) >= _MIN_CHARS:
                chunks.append(chunk_text)
            start += step

        return chunks


def count_tokens(text: str) -> int:
    """Count tokens in a text string using cl100k_base encoding.

    Args:
        text: The text to count tokens for.

    Returns:
        Number of tokens.
    """
    return len(_ENCODING.encode(text))

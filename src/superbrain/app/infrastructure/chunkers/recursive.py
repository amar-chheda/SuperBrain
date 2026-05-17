"""Recursive character splitter.

Tries separators in order (paragraphs → sentences → words → chars).
Best for mixed content — articles with headings and prose.
"""

from typing import Literal

from superbrain.app.application.ports import ChunkerPort
from superbrain.app.infrastructure.chunkers.fixed import count_tokens

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_MIN_CHARS = 50


class RecursiveChunker(ChunkerPort):
    """Splits text using a hierarchy of separators with character overlap."""

    def __init__(self, max_tokens: int = 400, overlap_chars: int = 100) -> None:
        """Initialise with a token budget and character overlap.

        Args:
            max_tokens: Maximum tokens per chunk.
            overlap_chars: Characters from the previous chunk to prepend as context.
        """
        self._max_tokens = max_tokens
        self._overlap_chars = overlap_chars

    def chunk(
        self,
        text: str,
        strategy: Literal["semantic", "recursive", "fixed"],
    ) -> list[str]:
        """Split text recursively using a separator hierarchy.

        Args:
            text: The full text to split.
            strategy: Ignored — this chunker always applies recursive splitting.

        Returns:
            List of text chunks, each at least 50 characters.
        """
        raw_chunks = self._split(text, _SEPARATORS)
        merged = self._merge_with_overlap(raw_chunks)
        return [c for c in merged if len(c) >= _MIN_CHARS]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that reduces chunk size.

        Args:
            text: Text to split.
            separators: Ordered list of separators to try.

        Returns:
            List of text fragments.
        """
        if not separators or count_tokens(text) <= self._max_tokens:
            return [text]

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            # Character-level split as last resort
            size = self._max_tokens * 4  # rough char estimate
            return [text[i:i + size] for i in range(0, len(text), size)]

        parts = text.split(sep)
        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if count_tokens(part) > self._max_tokens:
                result.extend(self._split(part, rest))
            else:
                result.append(part)
        return result

    def _merge_with_overlap(self, fragments: list[str]) -> list[str]:
        """Merge small fragments and add character overlap between chunks.

        Args:
            fragments: List of text fragments from _split().

        Returns:
            List of merged chunks with overlap.
        """
        chunks: list[str] = []
        current = ""

        for fragment in fragments:
            if not current:
                current = fragment
            elif count_tokens(current + " " + fragment) <= self._max_tokens:
                current = current + " " + fragment
            else:
                chunks.append(current)
                overlap = current[-self._overlap_chars:] if self._overlap_chars else ""
                current = (overlap + " " + fragment).strip() if overlap else fragment

        if current:
            chunks.append(current)

        return chunks

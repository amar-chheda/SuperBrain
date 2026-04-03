"""Chunking strategies for ingestion indexing."""

import re

from superbrain.app.application.ports import ChunkDraft


class ParagraphChunkingStrategy:
    """Chunk text while preferring paragraph and heading boundaries."""

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 150) -> None:
        """Initialize chunk sizing and overlap configuration."""

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")

        self._max_chars = max_chars
        self._overlap_chars = overlap_chars

    def chunk(self, text: str) -> list[ChunkDraft]:
        """Split text into ordered chunk drafts with traceability ranges."""

        normalized = text.strip()
        if not normalized:
            return []

        blocks = [
            segment.strip()
            for segment in re.split(r"\n\s*\n", normalized)
            if segment.strip()
        ]

        chunks: list[ChunkDraft] = []
        cursor = 0
        index = 0

        while cursor < len(normalized):
            window_end = min(len(normalized), cursor + self._max_chars)
            preferred_cut = normalized.rfind("\n\n", cursor, window_end)
            if preferred_cut == -1 or preferred_cut <= cursor:
                preferred_cut = normalized.rfind("\n", cursor, window_end)
            if preferred_cut == -1 or preferred_cut <= cursor:
                preferred_cut = window_end

            chunk_text = normalized[cursor:preferred_cut].strip()
            if not chunk_text:
                cursor = preferred_cut + 1
                continue

            token_count = len(chunk_text.split())
            chunks.append(
                ChunkDraft(
                    index=index,
                    text=chunk_text,
                    token_count=token_count,
                    char_start=cursor,
                    char_end=preferred_cut,
                )
            )
            index += 1

            if preferred_cut >= len(normalized):
                break
            next_cursor = max(0, preferred_cut - self._overlap_chars)
            cursor = preferred_cut if next_cursor <= cursor else next_cursor

        if not chunks and blocks:
            block_text = "\n\n".join(blocks)
            chunks.append(
                ChunkDraft(
                    index=0,
                    text=block_text,
                    token_count=len(block_text.split()),
                    char_start=0,
                    char_end=len(block_text),
                )
            )

        return chunks

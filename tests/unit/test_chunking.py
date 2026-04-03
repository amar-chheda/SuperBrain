"""Unit tests for paragraph chunking strategy."""

from superbrain.app.application.ingestion.chunking import ParagraphChunkingStrategy


def test_chunking_splits_large_text_with_overlap() -> None:
    """Chunker should split oversized text into ordered chunks."""

    chunker = ParagraphChunkingStrategy(max_chars=40, overlap_chars=8)
    text = "\n\n".join(
        [
            "# Heading",
            "Paragraph one has several words and some detail.",
            "Paragraph two continues with additional context.",
        ]
    )

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert chunks[0].char_start < chunks[0].char_end


def test_chunking_returns_empty_for_blank_text() -> None:
    """Chunker should return empty list when input is blank."""

    chunker = ParagraphChunkingStrategy()
    assert chunker.chunk("   \n\n") == []

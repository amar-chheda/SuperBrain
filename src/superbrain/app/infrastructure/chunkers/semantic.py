"""Sentence-boundary semantic chunker using NLTK.

Groups sentences into chunks until the token budget is reached.
Best for flowing prose — news articles, essays, blog posts.
"""

from typing import Literal

from nltk.tokenize import sent_tokenize

from superbrain.app.application.ports import ChunkerPort
from superbrain.app.infrastructure.chunkers.fixed import count_tokens

_MIN_CHARS = 50


class SemanticChunker(ChunkerPort):
    """Splits text on sentence boundaries, grouping by token budget."""

    def __init__(self, max_tokens: int = 400) -> None:
        """Initialise with a maximum token budget per chunk.

        Args:
            max_tokens: Maximum tokens allowed per chunk before starting a new one.
        """
        self._max_tokens = max_tokens

    def chunk(
        self,
        text: str,
        strategy: Literal["semantic", "recursive", "fixed"],
    ) -> list[str]:
        """Split text into sentence-grouped chunks.

        Includes the last sentence of the previous chunk as overlap context
        at the start of the next chunk. Never splits mid-sentence.

        Args:
            text: The full text to split.
            strategy: Ignored — this chunker always applies semantic splitting.

        Returns:
            List of text chunks, each at least 50 characters.
        """
        sentences = sent_tokenize(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        overlap_sentence: str | None = None

        for sentence in sentences:
            s_tokens = count_tokens(sentence)

            if current_tokens + s_tokens > self._max_tokens and current:
                chunk_text = " ".join(current)
                if len(chunk_text) >= _MIN_CHARS:
                    chunks.append(chunk_text)
                overlap_sentence = current[-1]
                current = [overlap_sentence, sentence] if overlap_sentence else [sentence]
                current_tokens = count_tokens(" ".join(current))
            else:
                current.append(sentence)
                current_tokens += s_tokens

        if current:
            chunk_text = " ".join(current)
            if len(chunk_text) >= _MIN_CHARS:
                chunks.append(chunk_text)

        return chunks

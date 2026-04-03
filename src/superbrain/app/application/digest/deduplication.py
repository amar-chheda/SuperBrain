"""Digest deduplication policy implementations."""

from typing import Protocol

from superbrain.app.application.digest.models import DigestSourceArticle


class DigestDeduper(Protocol):
    """Abstraction for deduplicating digest source articles."""

    def dedupe(self, sources: list[DigestSourceArticle]) -> list[DigestSourceArticle]:
        """Return deduplicated list of source articles."""
        ...


class CanonicalUrlDigestDeduper(DigestDeduper):
    """Deduplicate source articles by canonical URL with content hash fallback hook."""

    def dedupe(self, sources: list[DigestSourceArticle]) -> list[DigestSourceArticle]:
        """Drop duplicates while preserving stable original order."""

        seen_keys: set[str] = set()
        output: list[DigestSourceArticle] = []

        for source in sources:
            key = source.article.canonical_url or source.article.content_hash
            if key in seen_keys:
                continue
            seen_keys.add(key)
            output.append(source)
        return output

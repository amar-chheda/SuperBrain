"""Local article extractor implementations."""

import re
from html import unescape
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from superbrain.app.application.ports import ArticleExtractor, ExtractedArticle


class HttpArticleExtractor:
    """Fetch and parse article content from raw HTML pages."""

    def extract(self, url: str) -> ExtractedArticle:
        """Extract article-like content from a URL using basic HTML parsing."""

        request = Request(url, headers={"User-Agent": "Superbrain/0.1"})
        with urlopen(request, timeout=15) as response:
            html = response.read().decode("utf-8", errors="ignore")

        title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = unescape(title_match.group(1).strip()) if title_match is not None else "Untitled"

        body_text = re.sub(r"<script.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        body_text = re.sub(r"<style.*?</style>", " ", body_text, flags=re.IGNORECASE | re.DOTALL)
        body_text = re.sub(r"<[^>]+>", " ", body_text)
        body_text = unescape(re.sub(r"\s+", " ", body_text)).strip()

        parsed = urlparse(url)
        canonical = (
            f"{(parsed.scheme or 'https').lower()}://"
            f"{parsed.netloc.lower()}{parsed.path or '/'}"
        )

        return ExtractedArticle(
            title=title,
            canonical_url=canonical,
            source_url=url,
            domain=parsed.hostname or "",
            author=None,
            published_at=None,
            body_text=body_text,
            raw_html=html,
            extraction_quality_score=0.6,
            extraction_notes="basic_html_parser",
        )


class FallbackArticleExtractor:
    """Fallback extractor for cases where primary extraction fails."""

    def extract(self, url: str) -> ExtractedArticle:
        """Return minimal extraction output when content extraction fails."""

        parsed = urlparse(url)
        fallback_title = (
            parsed.path.strip("/").replace("-", " ").replace("_", " ").strip() or "Untitled"
        )
        canonical = (
            f"{(parsed.scheme or 'https').lower()}://"
            f"{parsed.netloc.lower()}{parsed.path or '/'}"
        )

        return ExtractedArticle(
            title=fallback_title.title(),
            canonical_url=canonical,
            source_url=url,
            domain=parsed.hostname or "",
            author=None,
            published_at=None,
            body_text=fallback_title,
            raw_html=None,
            extraction_quality_score=0.1,
            extraction_notes="fallback_extractor",
        )


class ChainedArticleExtractor:
    """Try primary extractor and fallback to secondary extractor on failure."""

    def __init__(self, primary: ArticleExtractor, fallback: ArticleExtractor) -> None:
        """Initialize chain with primary and fallback extractors."""

        self._primary = primary
        self._fallback = fallback

    def extract(self, url: str) -> ExtractedArticle:
        """Extract using primary, then fallback if primary raises."""

        try:
            return self._primary.extract(url)
        except Exception:
            return self._fallback.extract(url)

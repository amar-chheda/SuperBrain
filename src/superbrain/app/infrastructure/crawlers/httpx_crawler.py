"""Lightweight static page crawler using httpx and BeautifulSoup.

Fast (~200-500ms per page) but will fail on SPAs and JS-rendered content.
Use when pages load content with plain HTML — no JavaScript required.
"""

import time
from datetime import datetime

import chardet
import httpx
import structlog
from bs4 import BeautifulSoup

from superbrain.app.application.ports import CrawlerPort, CrawlResult
from superbrain.app.domain.exceptions import CrawlerError
from superbrain.app.infrastructure.crawlers.text_extractor import (
    extract_clean_text,
    normalise_text,
)
from superbrain.app.infrastructure.crawlers.url_utils import canonicalise_url
from superbrain.settings import Settings

log = structlog.get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class HttpxCrawler(CrawlerPort):
    """Lightweight static page crawler via httpx.

    Use for simple HTML pages, RSS feeds, and any site that doesn't require JS.
    Faster than SpiderCrawler but fails on SPAs and JS-rendered content.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """Initialise with shared settings and an httpx client.

        Args:
            settings: Application settings.
            client: A shared async httpx client (managed by the app lifespan).
        """
        self.settings = settings
        self.client = client

    async def fetch(self, url: str) -> CrawlResult:
        """Fetch a static page and extract its text content.

        Args:
            url: The URL to crawl.

        Returns:
            Parsed crawl result with extracted text and metadata.

        Raises:
            CrawlerError: If the page cannot be fetched or returns a non-2xx status.
        """
        canonical = canonicalise_url(url)
        started = time.monotonic()

        log.info("crawler.fetch.started", url=canonical, backend="httpx")

        try:
            response = await self.client.get(
                canonical,
                headers=_HEADERS,
                follow_redirects=True,
                timeout=15.0,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            log.error("crawler.fetch.failed", url=canonical, backend="httpx",
                      reason=str(exc), duration_ms=duration_ms)
            raise CrawlerError(canonical, str(exc), cause=exc) from exc

        if not response.is_success:
            duration_ms = int((time.monotonic() - started) * 1000)
            reason = f"HTTP {response.status_code}"
            log.error("crawler.fetch.failed", url=canonical, backend="httpx",
                      reason=reason, duration_ms=duration_ms)
            raise CrawlerError(canonical, reason)

        # Detect encoding
        content_type = response.headers.get("content-type", "")
        encoding: str | None = None
        if "charset=" in content_type:
            encoding = content_type.split("charset=")[-1].split(";")[0].strip()
        if not encoding:
            detected = chardet.detect(response.content)
            encoding = detected.get("encoding") or "utf-8"

        html = response.content.decode(encoding, errors="replace")
        soup = BeautifulSoup(html, "lxml")

        raw_text = normalise_text(extract_clean_text(soup))

        title: str | None = None
        tag = soup.find("title")
        if tag:
            title = tag.get_text(strip=True) or None
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True) or None

        author = _meta(soup, ["og:author", "article:author", "author"])
        published_at = _parse_date(_meta(soup, ["article:published_time", "og:published_time"]))

        duration_ms = int((time.monotonic() - started) * 1000)
        log.info("crawler.fetch.succeeded", url=canonical, backend="httpx",
                 text_length=len(raw_text), duration_ms=duration_ms)

        return CrawlResult(
            url=url,
            canonical_url=canonical,
            raw_text=raw_text,
            title=title,
            author=author,
            published_at=published_at,
            status_code=response.status_code,
        )


def _meta(soup: BeautifulSoup, names: list[str]) -> str | None:
    """Extract the first matching meta tag content.

    Args:
        soup: Parsed document.
        names: List of property/name values to try in order.

    Returns:
        The content attribute value, or None if not found.
    """
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or \
              soup.find("meta", attrs={"name": name})
        if tag:
            content = tag.get("content")
            if content:
                return str(content).strip() or None
    return None


def _parse_date(value: str | None) -> datetime | None:
    """Parse an ISO 8601 date string into a datetime, or return None.

    Args:
        value: ISO 8601 string or None.

    Returns:
        Parsed datetime with timezone info, or None on failure.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

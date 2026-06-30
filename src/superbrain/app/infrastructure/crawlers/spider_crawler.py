"""JS-rendering crawler using the Spider REST API directly.

POSTs to https://api.spider.cloud/scrape with Authorization: Bearer <key>.
Slower than httpx (~2-5s) but renders JavaScript and handles dynamic content.
Falls back to HttpxCrawler via FallbackCrawler when this raises CrawlerError.
"""

import time
from datetime import datetime

import httpx
import structlog

from superbrain.app.application.ports import CrawlerPort, CrawlResult
from superbrain.app.domain.exceptions import CrawlerError
from superbrain.app.infrastructure.crawlers.text_extractor import clean_spider_markdown
from superbrain.app.infrastructure.crawlers.url_utils import canonicalise_url
from superbrain.settings import Settings

log = structlog.get_logger(__name__)

_SPIDER_API_URL = "https://api.spider.cloud/scrape"


class SpiderCrawler(CrawlerPort):
    """Scrapes pages via the Spider cloud API (JS-rendering capable).

    Requires a valid API key with credits in SUPERBRAIN_SPIDER_API_KEY.
    """

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        """Initialise with settings and a shared httpx client.

        Args:
            settings: Application settings (reads spider_api_key).
            http_client: Shared async httpx client for the API call.
        """
        self._api_key = settings.spider_api_key
        self._client = http_client

    async def fetch(self, url: str) -> CrawlResult:
        """Fetch a page via the Spider API and return extracted text.

        Args:
            url: The URL to scrape.

        Returns:
            Parsed crawl result with markdown content and metadata.

        Raises:
            CrawlerError: If the API call fails or returns empty content.
        """
        canonical = canonicalise_url(url)
        started = time.monotonic()
        log.info("crawler.fetch.started", url=canonical, backend="spider")

        try:
            response = await self._client.post(
                _SPIDER_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": canonical,
                    "request": "smart",
                    "return_format": "markdown",
                    "metadata": True,
                },
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            reason = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            log.error("crawler.fetch.failed", url=canonical, backend="spider",
                      reason=reason, duration_ms=duration_ms)
            raise CrawlerError(canonical, reason, cause=exc) from exc
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            log.error("crawler.fetch.failed", url=canonical, backend="spider",
                      reason=str(exc), duration_ms=duration_ms)
            raise CrawlerError(canonical, str(exc), cause=exc) from exc

        data = response.json()

        # Response is a list of page objects
        if not data or not isinstance(data, list) or not data[0].get("content"):
            duration_ms = int((time.monotonic() - started) * 1000)
            reason = f"Spider returned empty content: {str(data)[:200]}"
            log.error("crawler.fetch.failed", url=canonical, backend="spider",
                      reason=reason, duration_ms=duration_ms)
            raise CrawlerError(canonical, reason)

        page = data[0]
        raw_text: str = clean_spider_markdown(page.get("content", ""))
        metadata: dict = page.get("metadata", {}) or {}

        title: str | None = metadata.get("title") or metadata.get("og:title") or None
        author: str | None = metadata.get("author") or metadata.get("og:author") or None
        published_at = _parse_date(
            metadata.get("article:published_time") or metadata.get("og:published_time")
        )
        status_code: int = page.get("status", 200)

        duration_ms = int((time.monotonic() - started) * 1000)
        log.info("crawler.fetch.succeeded", url=canonical, backend="spider",
                 text_length=len(raw_text), duration_ms=duration_ms)

        return CrawlResult(
            url=url,
            canonical_url=canonical,
            raw_text=raw_text,
            title=title,
            author=author,
            published_at=published_at,
            status_code=status_code,
        )


def _parse_date(value: str | None) -> datetime | None:
    """Parse an ISO 8601 date string into a datetime, or return None.

    Args:
        value: ISO 8601 string or None.

    Returns:
        Parsed datetime, or None on failure.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

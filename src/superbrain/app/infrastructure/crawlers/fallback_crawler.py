"""Fallback crawler: Spider with HttpxCrawler as a safety net.

Tries Spider first (JS rendering, better content extraction). If Spider raises
CrawlerError for any reason — rate limit, empty content, network error — it
falls back to HttpxCrawler transparently. The caller never sees the failure.
"""

import structlog

from superbrain.app.application.ports import CrawlerPort, CrawlResult
from superbrain.app.domain.exceptions import CrawlerError
from superbrain.app.infrastructure.crawlers.httpx_crawler import HttpxCrawler
from superbrain.app.infrastructure.crawlers.spider_crawler import SpiderCrawler

log = structlog.get_logger(__name__)


class FallbackCrawler(CrawlerPort):
    """Tries Spider; falls back to HttpxCrawler on any CrawlerError.

    Attributes:
        _primary: SpiderCrawler — used first.
        _fallback: HttpxCrawler — used if primary fails.
    """

    def __init__(self, primary: SpiderCrawler, fallback: HttpxCrawler) -> None:
        """Initialise with both crawler backends.

        Args:
            primary: Spider-based JS-rendering crawler.
            fallback: Httpx-based static HTML crawler.
        """
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, url: str) -> CrawlResult:
        """Fetch a URL, falling back to httpx if Spider fails.

        Args:
            url: The URL to crawl.

        Returns:
            CrawlResult from whichever backend succeeded.

        Raises:
            CrawlerError: If both backends fail.
        """
        try:
            return await self._primary.fetch(url)
        except CrawlerError as exc:
            log.warning(
                "crawler.fallback.triggered",
                url=url,
                spider_error=str(exc),
            )
            return await self._fallback.fetch(url)

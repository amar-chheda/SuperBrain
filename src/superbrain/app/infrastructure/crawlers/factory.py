"""Crawler backend factory.

The only place in the codebase that knows both crawler implementations exist.
Switching backends is a one-line config change — no code change required.
"""

import httpx

from superbrain.app.application.ports import CrawlerPort
from superbrain.app.infrastructure.crawlers.fallback_crawler import FallbackCrawler
from superbrain.app.infrastructure.crawlers.httpx_crawler import HttpxCrawler
from superbrain.app.infrastructure.crawlers.spider_crawler import SpiderCrawler
from superbrain.settings import Settings


def get_crawler(settings: Settings, http_client: httpx.AsyncClient) -> CrawlerPort:
    """Return the configured crawler backend.

    'spider' returns a FallbackCrawler: Spider first, HttpxCrawler on failure.
    'httpx' returns HttpxCrawler directly (no JS rendering, no fallback).

    Args:
        settings: Application settings. Reads settings.crawler_backend.
        http_client: Shared async httpx client (used by HttpxCrawler).

    Returns:
        The active CrawlerPort implementation.

    Raises:
        ValueError: If settings.crawler_backend is not a known value.
    """
    if settings.crawler_backend == "spider":
        return FallbackCrawler(
            primary=SpiderCrawler(settings, http_client),
            fallback=HttpxCrawler(settings, http_client),
        )
    if settings.crawler_backend == "httpx":
        return HttpxCrawler(settings, http_client)
    raise ValueError(f"Unknown crawler backend: {settings.crawler_backend!r}")

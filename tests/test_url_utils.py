"""Smoke tests for URL canonicalisation — no DB or Ollama required."""

from superbrain.app.infrastructure.crawlers.url_utils import canonicalise_url


def test_canonicalise_removes_utm():
    url = "https://example.com/article?utm_source=twitter&utm_medium=social"
    assert canonicalise_url(url) == "https://example.com/article"


def test_canonicalise_lowercases_host():
    url = "https://Example.COM/path"
    assert canonicalise_url(url) == "https://example.com/path"


def test_canonicalise_removes_fragment():
    url = "https://example.com/article#section-2"
    assert canonicalise_url(url) == "https://example.com/article"


def test_canonicalise_preserves_meaningful_query_params():
    url = "https://example.com/search?q=python"
    result = canonicalise_url(url)
    assert "q=python" in result


def test_canonicalise_removes_trailing_slash():
    url = "https://example.com/article/"
    result = canonicalise_url(url)
    assert not result.endswith("/")

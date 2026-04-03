"""Unit tests for URL canonicalization behavior."""

from superbrain.app.application.ingestion.url import DefaultUrlCanonicalizer


def test_canonicalize_normalizes_case_and_tracking_params() -> None:
    """Canonicalizer should lowercase and strip common tracking parameters."""

    canonicalizer = DefaultUrlCanonicalizer()
    result = canonicalizer.canonicalize(
        "HTTPS://Example.COM/path/?utm_source=newsletter&fbclid=123&id=42"
    )
    assert result == "https://example.com/path?id=42"


def test_canonicalize_sorts_query_params_and_trims_trailing_slash() -> None:
    """Canonicalizer should normalize query ordering and trailing slashes."""

    canonicalizer = DefaultUrlCanonicalizer()
    result = canonicalizer.canonicalize("https://example.com/a/b/?z=9&a=1")
    assert result == "https://example.com/a/b?a=1&z=9"

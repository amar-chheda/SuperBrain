"""URL canonicalisation utilities.

Produces stable, normalised URLs for deduplication. Two URLs pointing to the
same article must produce the same canonical URL regardless of tracking params,
trailing slashes, or fragment identifiers.
"""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid",
})


def canonicalise_url(url: str) -> str:
    """Produce a stable, normalised URL for deduplication.

    Strips tracking parameters, normalises scheme/host casing, removes default
    ports, sorts remaining query params, and drops fragments.

    Args:
        url: The raw URL string to normalise.

    Returns:
        The canonicalised URL string.
    """
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()

    # Remove default ports
    if host.endswith(":80") and scheme == "http":
        host = host[:-3]
    elif host.endswith(":443") and scheme == "https":
        host = host[:-4]

    # Strip tracking params, sort the rest
    filtered = sorted(
        (k, v) for k, v in parse_qsl(parsed.query) if k not in _TRACKING_PARAMS
    )
    query = urlencode(filtered)

    # Remove trailing slash unless path is just "/"
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, host, path, parsed.params, query, ""))

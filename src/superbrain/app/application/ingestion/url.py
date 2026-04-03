"""URL canonicalization helpers."""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAM_EXACT = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


class DefaultUrlCanonicalizer:
    """Canonicalize URLs for robust deduplication."""

    def canonicalize(self, url: str) -> str:
        """Normalize URL components and strip common tracking params."""

        parsed = urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        host = parsed.hostname.lower() if parsed.hostname else ""
        port = f":{parsed.port}" if parsed.port else ""

        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
            if path == "":
                path = "/"

        cleaned_params: list[tuple[str, str]] = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            lowered = key.lower()
            if lowered.startswith("utm_") or lowered in TRACKING_PARAM_EXACT:
                continue
            cleaned_params.append((key, value))
        cleaned_params.sort(key=lambda item: (item[0], item[1]))

        query = urlencode(cleaned_params, doseq=True)
        netloc = f"{host}{port}"

        canonical = urlunparse((scheme, netloc, path, "", query, ""))
        return canonical

"""Content deduplication helpers for the ingestion pipeline.

Computes a stable content hash used to detect articles that have already
been ingested, preventing duplicate embeddings for the same content.
"""

import hashlib


def compute_content_hash(text: str) -> str:
    """Compute a SHA-256 hash of the article's raw text.

    The hash is used as a stable fingerprint for deduplication. Two articles
    with identical raw text will produce the same hash regardless of URL.

    Args:
        text: The normalised raw text of the article.

    Returns:
        Lowercase hex-encoded SHA-256 digest (64 characters).
    """
    return hashlib.sha256(text.encode()).hexdigest()

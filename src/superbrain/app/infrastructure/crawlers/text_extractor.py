"""Shared HTML-to-clean-text extraction logic.

Both crawler backends pass their parsed HTML through these functions to produce
consistent, boilerplate-free article text. Order of operations in
extract_clean_text() matters — do not reorder the steps.
"""

import html
import re
import unicodedata

from bs4 import BeautifulSoup, Tag

_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form",
               "iframe", "noscript"]

_NOISE_CLASSES = ["nav", "menu", "sidebar", "footer", "header", "cookie",
                  "banner", "advertisement", "popup", "modal"]

# Whole-word pattern: matches "nav" and "nav-bar" but not "navigate" or "main-navigation"
_NOISE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _NOISE_CLASSES) + r")\b",
    re.IGNORECASE,
)


def _has_noise_class(tag: Tag) -> bool:
    """Return True if a tag has a class or id that suggests boilerplate content.

    Uses whole-word matching so 'main-navigation' matches 'nav' but
    'main-content' does not match 'nav'.

    Args:
        tag: The BeautifulSoup tag to inspect.

    Returns:
        True if the tag looks like navigation, ads, or other non-content.
    """
    if not tag.attrs:
        return False
    for attr in ("class", "id"):
        values = tag.get(attr, [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            if _NOISE_PATTERN.search(value):
                return True
    return False


def extract_clean_text(soup: BeautifulSoup) -> str:
    """Remove boilerplate and return article body text.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Cleaned article text with boilerplate removed.
    """
    # 1. Remove noise tags entirely
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # 2. Remove elements whose class/id suggests boilerplate
    for tag in soup.find_all(True):
        if isinstance(tag, Tag) and _has_noise_class(tag):
            tag.decompose()

    # 3. Find the main content block
    content: Tag | BeautifulSoup | None = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", class_="content")
    )

    if content is None:
        # Fallback: largest <div> by text length
        divs = soup.find_all("div")
        if divs:
            content = max(divs, key=lambda d: len(d.get_text()))
        else:
            content = soup

    # 4. Extract text
    raw = content.get_text(separator="\n", strip=True)

    # 5. Collapse runs of blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()


_FOOTER_MARKERS = re.compile(
    r"^(more from (this )?author|related articles?|next up|you might (also )?like|"
    r"recommended for you|## more\b|## related\b|## next\b)",
    re.IGNORECASE,
)
# Lines that are only a markdown link — pure nav items
_PURE_LINK_LINE = re.compile(r"^\s*\[.*?\]\(https?://[^\)]+\)\s*\**\s*$")
# Lines that are only markdown image syntax
_IMAGE_LINE = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")
# Lines that are only social-sharing placeholders: "* **" or "* *" or just "* "
_SOCIAL_STUB = re.compile(r"^\s*\*\s*\*{0,2}\s*$")


def clean_spider_markdown(text: str) -> str:
    """Remove nav, footer, and boilerplate from Spider's markdown output.

    Spider returns the full rendered page as markdown, including site navigation,
    social sharing buttons, and related-article footers. This function strips all
    of that using content patterns that hold across any page structure:

    1. Everything before the first ``# `` h1 heading is site chrome (nav/header).
    2. Lines that are only markdown links, images, or social stubs are discarded.
    3. Content is truncated at recognised footer markers.
    4. HTML entities left in by Spider are decoded.
    5. Excessive blank lines are collapsed.

    Args:
        text: Raw markdown string returned by the Spider API.

    Returns:
        Cleaned article text suitable for chunking and embedding.
    """
    lines = text.splitlines()

    # 1. Drop everything before the first h1 (# ...) — that's all site nav
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^#\s+\S", line):
            start = i
            break

    # 2. Truncate at footer markers
    end = len(lines)
    for i, line in enumerate(lines[start:], start):
        if _FOOTER_MARKERS.search(line.strip()):
            end = i
            break

    # 3. Filter noisy lines within the article body
    kept: list[str] = []
    for line in lines[start:end]:
        if _PURE_LINK_LINE.match(line):
            continue
        if _IMAGE_LINE.match(line):
            continue
        if _SOCIAL_STUB.match(line):
            continue
        kept.append(line)

    # 4. Decode HTML entities Spider leaves in (&rsquo; &mdash; &amp; etc.)
    cleaned = html.unescape("\n".join(kept))

    # 5. Collapse runs of blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def normalise_text(raw: str) -> str:
    """Normalise unicode and whitespace in extracted text.

    Apply after extract_clean_text(). Fixes encoding artifacts and collapses
    internal whitespace without losing paragraph structure.

    Args:
        raw: The raw extracted text string.

    Returns:
        Normalised text string.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

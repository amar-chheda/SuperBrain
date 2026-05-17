# Superbrain — Block 3: Web Scraping Layer

## Context

Superbrain is a local-only agentic AI system. This is Block 3. It depends on Blocks 1 and 2 being complete — specifically the `CrawlerPort` interface defined in `application/ports.py` and the `IngestionJob` entity from Block 2.

This block implements the two concrete crawler backends behind the `CrawlerPort` abstraction, plus the config-driven switch between them. The ingestion pipeline (Block 4) will call `CrawlerPort.fetch()` without knowing which backend is active — that's the entire point of the port.

This is also a key teaching moment for the conference: showing the audience that swapping a heavyweight JS-rendering crawler for a lightweight static one is a one-line config change, not a code change, because the boundary was defined correctly.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    └── infrastructure/
        └── crawlers/
            ├── __init__.py
            ├── factory.py        # returns correct backend from settings
            ├── spider_crawler.py # Spider (Rust-backed, JS rendering)
            ├── httpx_crawler.py  # httpx (fast, static pages)
            └── text_extractor.py # shared HTML → clean text logic
```

### 2. Spider crawler (`spider_crawler.py`)

Spider is a Rust-based crawler with a Python client. Install: `pip install spider-client`.

```python
class SpiderCrawler(CrawlerPort):
    """
    Full JS-rendering crawler via spider-rs.
    Use when pages load content dynamically (SPAs, paywalled articles, JS-heavy sites).
    Slower (~2-5s per page) but handles the full DOM.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    async def fetch(self, url: str) -> CrawlResult:
        # Spider Python client usage:
        # from spider import Spider
        # app = Spider(api_key=None)  # local mode — no API key needed
        # result = app.scrape_url(url, params={"return_format": "markdown"})

        # Steps:
        # 1. Canonicalise the URL before fetching (see section 5)
        # 2. Call Spider with return_format="markdown" to get clean text
        # 3. Extract title, author, published_at from the result metadata if available
        # 4. Build and return a CrawlResult
        # 5. On any exception, raise CrawlerError with the original exception as cause
        ...
```

Key Spider configuration to apply:
- `return_format`: use `"markdown"` — Spider will strip nav, ads, boilerplate and return clean article text
- `request`: use `"smart"` mode — Spider decides whether to use a headless browser or not
- Set a user agent that looks like a real browser
- Timeout: 30 seconds

### 3. httpx crawler (`httpx_crawler.py`)

```python
class HttpxCrawler(CrawlerPort):
    """
    Lightweight static page crawler via httpx.
    Use for simple HTML pages, RSS feeds, and any site that doesn't require JS.
    Fast (~200-500ms per page) but will fail on SPAs and JS-rendered content.

    Teaching note: this is the tradeoff the audience needs to understand.
    Spider is slower and heavier but reliable. httpx is fast but brittle.
    The config switch lets you demonstrate both live.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def fetch(self, url: str) -> CrawlResult:
        # Steps:
        # 1. Canonicalise the URL
        # 2. GET the URL with httpx, follow redirects, timeout=15s
        # 3. Check status code — raise CrawlerError if not 2xx
        # 4. Detect encoding from Content-Type header or chardet fallback
        # 5. Parse HTML with BeautifulSoup
        # 6. Pass the parsed HTML to text_extractor.extract()
        # 7. Extract title from <title> or <h1>
        # 8. Extract author from meta tags (og:author, article:author, name="author")
        # 9. Extract published_at from meta tags (article:published_time, og:published_time)
        # 10. Build and return a CrawlResult
        ...
```

### 4. Shared text extractor (`text_extractor.py`)

Both crawlers must produce clean, consistent text. The httpx crawler needs this explicitly; the Spider crawler gets markdown directly but should still pass through normalisation.

```python
def extract_clean_text(soup: BeautifulSoup) -> str:
    """
    Remove everything that isn't article content.
    Order of operations matters — do not change the sequence.
    """
    # 1. Remove: <script>, <style>, <nav>, <header>, <footer>,
    #            <aside>, <form>, <iframe>, <noscript>, <ads>
    # 2. Remove elements with class/id containing: 'nav', 'menu', 'sidebar',
    #    'footer', 'header', 'cookie', 'banner', 'ad', 'popup', 'modal'
    # 3. Find the main content block:
    #    Try in order: <article>, <main>, [role="main"], <div class="content">,
    #    largest <div> by text length as fallback
    # 4. Extract text with soup.get_text(separator="\n", strip=True)
    # 5. Collapse runs of blank lines to a single blank line
    # 6. Strip leading/trailing whitespace
    # Return the cleaned string
    ...


def normalise_text(raw: str) -> str:
    """
    Apply after extract_clean_text. Normalises unicode, fixes common encoding
    artifacts, collapses internal whitespace.
    """
    # 1. unicodedata.normalize("NFKC", raw)
    # 2. Replace \r\n and \r with \n
    # 3. Collapse 3+ consecutive newlines to 2
    # 4. Strip
    ...
```

### 5. URL canonicalisation

Create `infrastructure/crawlers/url_utils.py`:

```python
def canonicalise_url(url: str) -> str:
    """
    Produce a stable, normalised URL for deduplication.
    Two URLs that point to the same article must produce the same canonical URL.
    """
    # 1. Parse with urllib.parse.urlparse
    # 2. Lowercase the scheme and host
    # 3. Remove default ports (80 for http, 443 for https)
    # 4. Remove tracking parameters:
    #    utm_source, utm_medium, utm_campaign, utm_term, utm_content,
    #    fbclid, gclid, ref, source, mc_cid, mc_eid
    # 5. Sort remaining query parameters alphabetically
    # 6. Remove trailing slash from path (unless path is just "/")
    # 7. Remove fragment (#section)
    # Return the reconstructed URL string
```

### 6. Crawler factory (`factory.py`)

```python
def get_crawler(settings: Settings, http_client: httpx.AsyncClient) -> CrawlerPort:
    """
    Returns the correct crawler backend based on settings.crawler_backend.
    This is the only place in the codebase that knows both implementations exist.
    """
    if settings.crawler_backend == "spider":
        return SpiderCrawler(settings)
    elif settings.crawler_backend == "httpx":
        return HttpxCrawler(settings, http_client)
    else:
        raise ValueError(f"Unknown crawler backend: {settings.crawler_backend}")
```

Register the crawler as a FastAPI dependency, created once on startup and shared across requests.

### 7. CrawlerError

Add to `domain/exceptions.py`:

```python
class CrawlerError(Exception):
    def __init__(self, url: str, reason: str, cause: Exception | None = None):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to crawl {url}: {reason}")
```

Map `CrawlerError` → HTTP 422 in the global error handler.

### 8. Logging requirements

Every `fetch()` call must emit structured log lines at these points:
```
INFO  crawler.fetch.started   {url, backend, request_id, job_id}
INFO  crawler.fetch.succeeded {url, backend, text_length, duration_ms}
ERROR crawler.fetch.failed    {url, backend, reason, duration_ms}
```

### 9. Wire crawler into job status updates

Update `POST /ingestion/jobs` handler:

When a job is created with `input_type="url"`, immediately attempt to crawl in the background:
1. Update job status to `"processing"`
2. Call `crawler.fetch(url)` — **do not await in the request handler**, use `BackgroundTasks`
3. On success: store the `CrawlResult` on the job (add a `raw_text` field to `IngestionJob` or persist separately — your choice, document it)
4. On `CrawlerError`: update job status to `"failed"`, store `error_message`

For now, after crawling succeeds, update status to `"succeeded"` — Block 4 will replace this with the full ingestion pipeline.

---

## Dependencies to add

```toml
"spider-client>=0.0.27",
"httpx>=0.27",             # already present from Block 1
"beautifulsoup4>=4.12",
"lxml>=5.0",               # faster BS4 parser
"chardet>=5.0",            # encoding detection fallback
```

---

## Config to add to `.env.example`

```
CRAWLER_BACKEND=httpx   # or "spider" for JS rendering
```

---

## Definition of done

- [ ] `GET /ingestion/jobs/{id}` after submitting a real URL shows `status="succeeded"` and non-empty `raw_text`
- [ ] Switching `CRAWLER_BACKEND=spider` in `.env` and restarting uses Spider with no code changes
- [ ] Switching `CRAWLER_BACKEND=httpx` uses httpx with no code changes
- [ ] `canonicalise_url("https://example.com/article?utm_source=twitter")` returns `"https://example.com/article"`
- [ ] Crawling a URL that doesn't exist returns `status="failed"` with a meaningful `error_message`
- [ ] Log lines include `backend`, `url`, and `duration_ms` on every fetch attempt
- [ ] Extracted text from a real news article contains no nav/footer/ad content
- [ ] `extract_clean_text` tested against at least two real article HTML fixtures

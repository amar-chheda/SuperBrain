# Superbrain — Block 7: Daily Digest Pipeline

## Context

Superbrain is a local-only agentic AI system. This is Block 7. It depends on Blocks 4 and 5 being complete — articles must be ingested and classified by topic before a digest can be generated.

This block implements the daily digest: a scheduled pipeline that selects yesterday's articles, groups them by topic, deduplicates sources, summarises each group using LiquidAI LFM-7B, and persists the result. It replaces the stub `POST /digests/trigger` route from Block 2.

**Conference teaching note:** The digest pipeline is a map-reduce pattern implemented as an orchestrated flow. This is a good moment to show your audience how "agentic" doesn't always mean a ReAct loop — sometimes the most reliable and testable design is a deterministic sequence of steps with LLM calls only at the points where language generation is actually needed.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    ├── application/
    │   ├── digest/
    │   │   ├── __init__.py
    │   │   ├── use_case.py          # orchestrates the full pipeline
    │   │   ├── selector.py          # selects and filters articles
    │   │   ├── grouper.py           # groups articles by topic
    │   │   ├── deduplicator.py      # removes duplicate sources within a group
    │   │   └── summariser.py        # LLM summarisation prompt + parser
    │   └── scheduler/
    │       ├── __init__.py
    │       └── adapter.py           # in-process scheduler with manual trigger
    └── infrastructure/
        └── db/
            └── repositories/
                └── digest_repo.py
```

### 2. Alembic migration — digest tables

```sql
CREATE TABLE digest_runs (
    id              UUID PRIMARY KEY,
    date_label      DATE NOT NULL,        -- the date the digest covers (yesterday)
    status          VARCHAR(20) NOT NULL DEFAULT 'running',
    article_count   INTEGER NOT NULL DEFAULT 0,
    section_count   INTEGER NOT NULL DEFAULT 0,
    triggered_by    VARCHAR(20) NOT NULL, -- "scheduler" | "manual" | "api"
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    error_message   TEXT
);

CREATE TABLE digest_items (
    id              UUID PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES digest_runs(id) ON DELETE CASCADE,
    topic_id        UUID NOT NULL REFERENCES topics(id),
    topic_name      VARCHAR(100) NOT NULL,
    summary         TEXT NOT NULL,
    article_ids     UUID[] NOT NULL,
    article_urls    TEXT[] NOT NULL,
    article_titles  TEXT[] NOT NULL,
    position        INTEGER NOT NULL,     -- display order
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_digest_runs_date ON digest_runs(date_label);
CREATE INDEX idx_digest_items_run ON digest_items(run_id);
```

### 3. Article selector (`application/digest/selector.py`)

```python
def select_articles_for_digest(
    articles: list[Article],
    matches: list[ArticleTopicMatch],
    date: date,
) -> list[ArticleWithTopics]:
    """
    Select articles ingested on a specific date and join their topic matches.

    Filtering rules:
    - Only articles with status="succeeded"
    - Only articles ingested on the target date (ingested_at::date = date)
    - Only articles that have at least one topic match with confidence "high" or "medium"
      (low-confidence matches are excluded from digests)
    - Deduplicate by canonical_url before returning
    """
    ...


def deduplicate_by_url(articles: list[ArticleWithTopics]) -> list[ArticleWithTopics]:
    """
    If two articles have the same canonical_url (can happen with crawler retries),
    keep only the most recently ingested one.
    """
    seen_urls: dict[str, ArticleWithTopics] = {}
    for article in sorted(articles, key=lambda a: a.ingested_at):
        seen_urls[article.canonical_url] = article
    return list(seen_urls.values())
```

### 4. Grouper (`application/digest/grouper.py`)

```python
def group_by_topic(
    articles: list[ArticleWithTopics],
    topics: list[Topic],
) -> list[TopicGroup]:
    """
    Group articles by their matched topics.

    Rules:
    - An article can appear in multiple topic groups if it matched multiple topics
    - Groups are ordered by topic priority (highest first)
    - Groups with fewer than MIN_ARTICLES_PER_SECTION (default: 1) are excluded
    - Topics with no matching articles produce no group
    """
    topic_map = {t.id: t for t in topics}
    groups: dict[UUID, list[ArticleWithTopics]] = defaultdict(list)

    for article in articles:
        for match in article.topic_matches:
            if match.confidence in ("high", "medium"):
                groups[match.topic_id].append(article)

    result = []
    for topic_id, group_articles in groups.items():
        topic = topic_map.get(topic_id)
        if not topic:
            continue
        result.append(TopicGroup(
            topic=topic,
            articles=group_articles,
        ))

    # Sort by topic priority descending, then by article count descending
    result.sort(key=lambda g: (-g.topic.priority, -len(g.articles)))
    return result
```

### 5. Deduplicator (`application/digest/deduplicator.py`)

```python
def deduplicate_sources_within_group(group: TopicGroup) -> TopicGroup:
    """
    Within a topic group, remove articles from the same domain that cover
    the same story. Keep the highest-confidence match.

    Strategy:
    - Extract the domain from each article's URL (e.g. "techcrunch.com")
    - If the same domain appears multiple times, keep only the article with
      the highest-confidence topic match
    - This prevents the digest from being dominated by one source

    Note: This is a heuristic. It will occasionally discard genuinely different
    articles from the same domain. That's an acceptable tradeoff for digest quality.
    """
    seen_domains: dict[str, ArticleWithTopics] = {}
    CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}

    for article in group.articles:
        domain = extract_domain(article.url)
        existing = seen_domains.get(domain)

        if existing is None:
            seen_domains[domain] = article
        else:
            # Keep the one with higher confidence match for this topic
            existing_conf = max(
                (CONFIDENCE_ORDER.get(m.confidence, 0)
                 for m in existing.topic_matches if m.topic_id == group.topic.id),
                default=0,
            )
            new_conf = max(
                (CONFIDENCE_ORDER.get(m.confidence, 0)
                 for m in article.topic_matches if m.topic_id == group.topic.id),
                default=0,
            )
            if new_conf > existing_conf:
                seen_domains[domain] = article

    return dataclasses.replace(group, articles=list(seen_domains.values()))


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    # Remove "www." prefix for dedup purposes
    return re.sub(r"^www\.", "", parsed.netloc.lower())
```

### 6. Summariser (`application/digest/summariser.py`)

Uses **LiquidAI LFM-7B** (`settings.ollama_digest_model = "lfm2-7b"` or whatever the Ollama tag is). This model is chosen for digest summarisation because it handles sequential, multi-document synthesis well.

**Fallback:** If LiquidAI is not available via Ollama, fall back to `settings.ollama_qa_model` (Llama 3.1 8B). Make this configurable.

```python
DIGEST_SUMMARY_PROMPT = """You are a news digest writer. Your job is to write a brief, informative summary of the following articles about {topic_name}.

TOPIC: {topic_name}
TOPIC DESCRIPTION: {topic_description}

ARTICLES:
{articles_block}

RULES:
- Write a summary of 3 to 6 sentences covering the key developments across these articles
- Synthesise the articles — do not summarise each one individually
- Focus on what is new, significant, or actionable
- Write in a neutral, factual tone
- Do not include phrases like "According to the articles" or "The articles discuss"
- Do not use bullet points — write in prose
- Do not hallucinate any information not present in the articles above

Write the summary now:"""


def format_articles_block(articles: list[ArticleWithTopics]) -> str:
    """
    Format articles for the digest prompt. Include title and a text excerpt.
    Keep total length under ~2000 characters to stay within local model context.
    """
    MAX_EXCERPT_CHARS = 400
    MAX_ARTICLES = 5  # cap even if more are available — context limit

    lines = []
    for i, article in enumerate(articles[:MAX_ARTICLES], start=1):
        lines.append(f"Article {i}: {article.title or article.url}")
        excerpt = article.raw_text[:MAX_EXCERPT_CHARS].replace("\n", " ").strip()
        lines.append(excerpt)
        lines.append("")
    return "\n".join(lines)


async def summarise_topic_group(
    llm: LLMPort,
    model: str,
    group: TopicGroup,
) -> str:
    """
    Generate a summary for one topic group.
    Returns the summary text.
    """
    if not group.articles:
        return ""

    prompt = DIGEST_SUMMARY_PROMPT.format(
        topic_name=group.topic.name,
        topic_description=group.topic.description,
        articles_block=format_articles_block(group.articles),
    )

    summary = await llm.complete(
        prompt,
        model=model,
        prompt_template="digest_summary_v1",
    )
    return summary.strip()
```

### 7. Digest use case (`application/digest/use_case.py`)

```python
class GenerateDailyDigestUseCase:
    async def execute(
        self,
        target_date: date | None = None,
        triggered_by: str = "scheduler",
    ) -> DigestRun:
        # Default to yesterday
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        # Create the run record
        run = DigestRun(
            id=uuid4(),
            date_label=target_date,
            status="running",
            triggered_by=triggered_by,
            started_at=datetime.utcnow(),
        )
        await self.digest_repo.save_run(run)

        try:
            # 1. Load articles and their topic matches for the target date
            articles = await self.article_repo.list_by_date(target_date)
            matches = await self.match_repo.list_by_article_ids(
                [a.id for a in articles]
            )
            articles_with_topics = join_matches(articles, matches)

            # 2. Select eligible articles
            selected = select_articles_for_digest(articles_with_topics, target_date)

            if not selected:
                log.info("digest.empty", date=str(target_date))
                await self.digest_repo.update_run(
                    run.id, status="succeeded", article_count=0, section_count=0
                )
                self.metrics.increment("digest_empty_total")
                return dataclasses.replace(run, status="succeeded")

            # 3. Load active topics
            topics = await self.topic_repo.list_active()

            # 4. Group by topic
            groups = group_by_topic(selected, topics)

            # 5. Deduplicate sources within each group
            groups = [deduplicate_sources_within_group(g) for g in groups]

            # 6. Summarise each group (sequential — local LLM, one at a time)
            items = []
            for position, group in enumerate(groups):
                summary = await summarise_topic_group(
                    self.llm,
                    model=self.settings.ollama_digest_model,
                    group=group,
                )
                if not summary:
                    continue

                item = DigestItem(
                    id=uuid4(),
                    run_id=run.id,
                    topic_id=group.topic.id,
                    topic_name=group.topic.name,
                    summary=summary,
                    article_ids=[a.id for a in group.articles],
                    article_urls=[a.url for a in group.articles],
                    article_titles=[a.title or "" for a in group.articles],
                    position=position,
                    created_at=datetime.utcnow(),
                )
                items.append(item)

            # 7. Persist all items
            await self.digest_repo.save_items(items)

            # 8. Mark run succeeded
            await self.digest_repo.update_run(
                run.id,
                status="succeeded",
                article_count=len(selected),
                section_count=len(items),
                finished_at=datetime.utcnow(),
            )

            self.metrics.increment("digest_success_total")
            self.metrics.observe("digest_section_count", len(items))
            log.info("digest.succeeded", date=str(target_date), sections=len(items))

            return dataclasses.replace(run, status="succeeded", section_count=len(items))

        except Exception as e:
            await self.digest_repo.update_run(
                run.id, status="failed", error_message=str(e)
            )
            self.metrics.increment("digest_failure_total")
            log.error("digest.failed", date=str(target_date), error=str(e))
            raise
```

### 8. Scheduler adapter (`application/scheduler/adapter.py`)

Simple in-process scheduler using `APScheduler`. Abstracted so it can be swapped later.

```python
class SchedulerAdapter:
    def __init__(self, digest_use_case: GenerateDailyDigestUseCase):
        self.scheduler = AsyncIOScheduler()
        self.digest_use_case = digest_use_case

    def start(self):
        # Run daily at 07:00 UTC
        self.scheduler.add_job(
            self._run_digest,
            trigger=CronTrigger(hour=7, minute=0, timezone="UTC"),
            id="daily_digest",
            replace_existing=True,
        )
        self.scheduler.start()
        log.info("scheduler.started")

    def stop(self):
        self.scheduler.shutdown()

    async def trigger_now(self, triggered_by: str = "manual") -> DigestRun:
        """Manual trigger — used by POST /digests/trigger and CLI."""
        return await self.digest_use_case.execute(triggered_by=triggered_by)

    async def _run_digest(self):
        try:
            await self.digest_use_case.execute(triggered_by="scheduler")
        except Exception as e:
            log.error("scheduler.digest_failed", error=str(e))
```

Start and stop the scheduler in the FastAPI lifespan:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.stop()
```

### 9. Digest API routes (`api/digests.py` — replaces Block 2 stubs)

```
POST /digests/trigger
```
Body (optional): `{ "date": "2024-01-15" }` — defaults to yesterday if omitted
Returns the `DigestRun` record immediately (generation continues in background)

```
GET /digests
```
Returns list of recent `DigestRun` records (last 30 days), ordered by date descending

```
GET /digests/{run_id}
```
Returns a `DigestRun` with its `DigestItem` list

Response shape:
```json
{
  "id": "uuid",
  "date_label": "2024-01-15",
  "status": "succeeded",
  "article_count": 12,
  "section_count": 4,
  "triggered_by": "manual",
  "started_at": "2024-01-16T07:00:00Z",
  "finished_at": "2024-01-16T07:02:30Z",
  "items": [
    {
      "topic_name": "Machine Learning",
      "summary": "Several papers published this week...",
      "article_titles": ["Title 1", "Title 2"],
      "article_urls": ["https://...", "https://..."],
      "position": 0
    }
  ]
}
```

Also add `superbrain digest trigger` to the CLI.

---

## Dependencies to add

```toml
"apscheduler>=3.10",
```

---

## Config to add to `.env.example`

```
OLLAMA_DIGEST_MODEL=llama3.1:8b   # replace with lfm2-7b when available via Ollama
DIGEST_SCHEDULE_HOUR=7            # UTC hour to run daily digest
```

---

## Definition of done

- [ ] `POST /digests/trigger` creates a `DigestRun` and returns immediately
- [ ] After the run completes, `GET /digests/{run_id}` shows `status="succeeded"` with `section_count > 0` (given ingested articles from yesterday)
- [ ] Each `DigestItem` has a non-empty `summary` and at least one `article_url`
- [ ] Running `POST /digests/trigger` with no ingested articles for yesterday results in `status="succeeded"` with `section_count=0` (not an error)
- [ ] Duplicate sources from the same domain within a topic group are deduplicated
- [ ] The scheduler starts on app startup and logs `scheduler.started`
- [ ] `superbrain digest trigger` CLI command works
- [ ] `ModelCallLog` records exist for every summarisation LLM call
- [ ] Digest metrics recorded: success/failure/empty counters, section count
- [ ] A digest run that fails partway through (e.g. LLM error on one section) marks the run as `failed` and logs the error — it does not silently succeed with partial output

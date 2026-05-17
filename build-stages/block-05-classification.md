# Superbrain — Block 5: Topic Classification

## Context

Superbrain is a local-only agentic AI system. This is Block 5. It depends on Block 4 being complete — specifically ingested articles and chunks in the database, and the `LLMPort` / `OllamaLLM` adapter from Block 4.

This block adds the topic system: CRUD + versioning for topic definitions, and a classifier that assigns ingested articles to topics using Phi-3 Mini. Classification runs after ingestion and is also re-triggered when a topic definition changes.

**Conference teaching note:** This block demonstrates two important things for your audience:
1. Why structured JSON output prompts require a completely different prompt style than free-form text prompts — the model must return machine-readable output that the system depends on.
2. Why topic versioning exists: when you update a topic definition, previously classified articles may now be wrong. The reclassification workflow forces you to think explicitly about this — something cloud AI wrappers hide from you.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    ├── application/
    │   └── topics/
    │       ├── __init__.py
    │       ├── use_cases.py         # classify, reclassify, CRUD
    │       └── classifier.py        # Phi-3 Mini classification prompt + parser
    └── infrastructure/
        └── db/
            └── repositories/
                └── topic_repo.py
```

### 2. Alembic migration — topics and matches tables

```sql
CREATE TABLE topics (
    id          UUID PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    description TEXT NOT NULL,
    examples    JSONB NOT NULL DEFAULT '[]',   -- list of example article titles
    priority    INTEGER NOT NULL DEFAULT 0,
    status      VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(name, version)
);

CREATE TABLE article_topic_matches (
    id              UUID PRIMARY KEY,
    article_id      UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    topic_id        UUID NOT NULL REFERENCES topics(id),
    topic_version   INTEGER NOT NULL,
    confidence      VARCHAR(10) NOT NULL,   -- "high" | "medium" | "low"
    reason          TEXT NOT NULL,
    classified_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(article_id, topic_id)            -- one match per article-topic pair
);

CREATE INDEX idx_matches_article ON article_topic_matches(article_id);
CREATE INDEX idx_matches_topic ON article_topic_matches(topic_id);
```

### 3. Topic CRUD routes (`api/topics.py` — replace stubs from Block 2)

```
GET  /topics                    → list all topics (active only by default, ?include_archived=true for all)
POST /topics                    → create a new topic (version=1)
GET  /topics/{id}               → get topic by id
PUT  /topics/{id}               → update topic (creates a new version, does NOT modify existing)
DELETE /topics/{id}             → set status="archived" (never hard delete)
POST /topics/{id}/reclassify    → trigger reclassification of all articles against this topic
```

**PUT semantics — this is important:** Updating a topic must create a new version record, not overwrite the existing one. Existing `article_topic_matches` reference `topic_version` — you must be able to audit which definition was used when.

```python
async def update_topic(topic_id: UUID, body: TopicUpdateRequest) -> TopicResponse:
    existing = await topic_repo.find_by_id(topic_id)
    if not existing:
        raise NotFoundError(f"Topic {topic_id} not found")

    new_version = Topic(
        id=uuid4(),          # new row, new ID
        name=existing.name,  # name stays the same — it's the topic's identity
        version=existing.version + 1,
        description=body.description,
        examples=body.examples,
        priority=body.priority,
        status="active",
    )
    # Archive the old version
    await topic_repo.set_status(existing.id, "archived")
    await topic_repo.save(new_version)
    return TopicResponse.from_domain(new_version)
```

Request/response shapes:

```python
class TopicCreateRequest(BaseModel):
    name: str
    description: str
    examples: list[str] = []   # example article titles that belong to this topic
    priority: int = 0          # higher = shown first in digest

class TopicUpdateRequest(BaseModel):
    description: str
    examples: list[str] = []
    priority: int = 0

class TopicResponse(BaseModel):
    id: UUID
    name: str
    version: int
    description: str
    examples: list[str]
    priority: int
    status: str
    created_at: datetime
```

### 4. Classifier (`application/topics/classifier.py`)

Uses **Phi-3 Mini** (configured as `settings.ollama_classification_model`). Phi-3 Mini is chosen over Llama 3.1 8B here because classification is a structured, constrained task — it needs to pick from a defined list and return JSON. Phi-3 Mini handles this reliably at lower resource cost.

**The prompt is the hardest part of this block.** Local models struggle with multi-label classification when topics are numerous. The solution: send topics in small batches and be absolutely explicit about the output format.

```python
CLASSIFICATION_PROMPT = """You are a content classifier. Your job is to determine which of the following topics an article belongs to.

ARTICLE TITLE: {title}

ARTICLE EXCERPT (first 800 characters):
{excerpt}

TOPICS TO EVALUATE:
{topics_block}

TASK:
For each topic listed above, decide if this article matches it.
An article matches a topic if its main subject is directly relevant to that topic.

RULES:
- An article can match multiple topics, or zero topics
- Only match a topic if you are confident the article is primarily about that subject
- Do not match a topic just because the article mentions it in passing
- You must respond with ONLY a valid JSON array. No explanation. No preamble. No markdown fences.

The JSON array must contain one object per matching topic, with exactly these fields:
- "topic_id": the exact topic ID string from the list above
- "confidence": one of exactly "high", "medium", or "low"
- "reason": one sentence explaining why this article matches

If no topics match, respond with an empty array: []

Example valid response for two matching topics:
[{{"topic_id": "abc-123", "confidence": "high", "reason": "Article is primarily about Python programming"}}, {{"topic_id": "def-456", "confidence": "low", "reason": "Article briefly covers machine learning applications"}}]"""


def format_topics_block(topics: list[Topic]) -> str:
    """
    Format topics for the prompt. Keep it compact — local models have limited context.
    Only include the fields the model needs to make a decision.
    """
    lines = []
    for t in topics:
        lines.append(f"ID: {t.id}")
        lines.append(f"Name: {t.name}")
        lines.append(f"Description: {t.description}")
        if t.examples:
            lines.append(f"Examples: {', '.join(t.examples[:3])}")  # max 3 examples
        lines.append("")  # blank line between topics
    return "\n".join(lines)
```

```python
async def classify_article(
    llm: LLMPort,
    model: str,
    article: Article,
    topics: list[Topic],
    batch_size: int = 5,
) -> list[TopicMatch]:
    """
    Classify an article against all active topics.
    Batches topics to avoid overflowing the model's context window.

    Returns a list of TopicMatch objects for topics that matched.
    """
    if not topics:
        return []

    all_matches = []

    # Process topics in batches to stay within context window
    for batch_start in range(0, len(topics), batch_size):
        batch = topics[batch_start : batch_start + batch_size]

        prompt = CLASSIFICATION_PROMPT.format(
            title=article.title or "Unknown",
            excerpt=article.raw_text[:800],
            topics_block=format_topics_block(batch),
        )

        raw = await llm.complete(
            prompt,
            model=model,
            json_mode=True,
            prompt_template="article_classification_v1",
        )

        batch_matches = parse_classification_response(raw, batch)
        all_matches.extend(batch_matches)

    return all_matches


def parse_classification_response(
    raw: str, topics: list[Topic]
) -> list[TopicMatch]:
    """
    Parse and validate the classifier's JSON output.
    Must be extremely defensive — local models misbehave.
    """
    valid_ids = {str(t.id) for t in topics}

    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        parsed = json.loads(cleaned.strip())

        if not isinstance(parsed, list):
            log.warning("classification.invalid_response", reason="not a list", raw=raw[:200])
            return []

        matches = []
        for item in parsed:
            topic_id = item.get("topic_id", "")
            confidence = item.get("confidence", "")
            reason = item.get("reason", "")

            # Validate topic_id is one we sent
            if topic_id not in valid_ids:
                log.warning("classification.unknown_topic_id", topic_id=topic_id)
                continue

            # Validate confidence is one of the three valid values
            if confidence not in ("high", "medium", "low"):
                log.warning("classification.invalid_confidence", confidence=confidence)
                confidence = "low"  # conservative fallback

            matches.append(TopicMatch(
                topic_id=UUID(topic_id),
                confidence=confidence,
                reason=reason or "No reason provided",
            ))

        return matches

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("classification.parse_failed", error=str(e), raw=raw[:200])
        return []  # never crash ingestion because classification failed
```

### 5. Classification use cases (`application/topics/use_cases.py`)

```python
class ClassifyArticleUseCase:
    async def execute(self, article_id: UUID) -> list[ArticleTopicMatch]:
        article = await self.article_repo.find_by_id(article_id)
        topics = await self.topic_repo.list_active()

        matches = await classify_article(
            self.llm,
            model=self.settings.ollama_classification_model,
            article=article,
            topics=topics,
        )

        # Persist: replace all existing matches for this article
        # (reclassification replaces, not appends)
        await self.match_repo.delete_by_article(article_id)

        records = [
            ArticleTopicMatch(
                id=uuid4(),
                article_id=article_id,
                topic_id=m.topic_id,
                topic_version=self._get_topic_version(m.topic_id, topics),
                confidence=m.confidence,
                reason=m.reason,
                classified_at=datetime.utcnow(),
            )
            for m in matches
        ]

        await self.match_repo.save_many(records)
        return records


class ReclassifyTopicUseCase:
    """
    Re-runs classification for ALL articles against a specific topic.
    Triggered when a topic definition changes (PUT /topics/{id}).
    This can be slow — run as a background job.
    """
    async def execute(self, topic_id: UUID) -> None:
        topic = await self.topic_repo.find_by_id(topic_id)
        articles = await self.article_repo.list_all_active()

        log.info("reclassification.started", topic_id=str(topic_id),
                 article_count=len(articles))

        for article in articles:
            try:
                matches = await classify_article(
                    self.llm,
                    model=self.settings.ollama_classification_model,
                    article=article,
                    topics=[topic],  # only reclassify against the changed topic
                )
                # Update only this topic's match for this article
                await self.match_repo.upsert_for_topic(article.id, topic_id, matches)
            except Exception as e:
                log.error("reclassification.article_failed",
                          article_id=str(article.id), error=str(e))
                continue  # don't stop reclassification because one article failed

        log.info("reclassification.completed", topic_id=str(topic_id))
```

### 6. Wire classification into the ingestion pipeline

After Block 4's ingestion use case marks a job as succeeded, it should trigger classification:

In `IngestArticleUseCase.execute()`, after step 13 (update job status to succeeded), add:
```python
# Trigger classification as a follow-up (non-blocking)
# This is a separate use case — ingestion doesn't depend on it succeeding
await self.classify_use_case.execute(article.id)
```

Or alternatively, trigger via `BackgroundTasks` in the API layer — either approach is acceptable, document your choice.

### 7. Classification API routes

```
POST /topics/classify/articles/{article_id}
```
- Classify a single article against all active topics
- Returns the list of matches
- Used for manual re-classification and testing

```
GET /articles/{article_id}/topics
```
- Return all topic matches for an article
- Include `confidence`, `reason`, `topic_version`

---

## Definition of done

- [ ] `POST /topics` creates a topic and it appears in `GET /topics`
- [ ] `PUT /topics/{id}` creates a new version and archives the old one — old version still queryable
- [ ] `POST /topics/classify/articles/{article_id}` returns matches for a real ingested article
- [ ] Classification results are persisted in `article_topic_matches`
- [ ] `ModelCallLog` records exist for every classification LLM call, with `prompt_template="article_classification_v1"`
- [ ] `parse_classification_response` handles malformed JSON without raising — returns `[]` and logs a warning
- [ ] `parse_classification_response` rejects topic IDs that weren't in the batch (hallucination guard)
- [ ] `POST /topics/{id}/reclassify` re-classifies all articles against the updated topic definition
- [ ] Reclassification stores the new `topic_version` in each match record
- [ ] Classification metrics: match count per run recorded in `MetricsRecorder`

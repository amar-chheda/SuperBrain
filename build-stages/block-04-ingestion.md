# Superbrain — Block 4: Ingestion Pipeline

## Context

Superbrain is a local-only agentic AI system. This is Block 4. It depends on Blocks 1, 2, and 3 being complete — specifically:
- `CrawlResult` from Block 3 (the raw text input)
- `Article`, `Chunk`, `ModelCallLog` entities from Block 1
- `IngestionJob` from Block 2 (job status tracking)
- `EmbeddingPort`, `LLMPort`, `ChunkerPort` interfaces from Block 1

This block is the pipeline's core. It takes a `CrawlResult`, decides how to chunk it, embeds those chunks, and persists everything. It also makes the first real LLM calls in the system.

**This is the most important teaching block for the conference.** The chunking strategy decision — where a local LLM reasons about what chunking approach to use — is the centrepiece demo moment. The prompt you write here must be precise, explicit, and structured because an 8B model has no room for ambiguity. Write the prompt like you're leaving instructions for a junior engineer who will follow them literally.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    ├── application/
    │   └── ingestion/
    │       ├── __init__.py
    │       ├── use_case.py          # orchestrates the full pipeline
    │       ├── chunking_agent.py    # LLM decides chunking strategy
    │       └── dedup.py             # hash-based deduplication
    └── infrastructure/
        ├── chunkers/
        │   ├── __init__.py
        │   ├── semantic.py          # semantic chunker (sentence boundaries)
        │   ├── recursive.py         # recursive character splitter
        │   └── fixed.py             # fixed-size with overlap
        ├── embeddings/
        │   ├── __init__.py
        │   └── ollama_embedder.py   # nomic-embed-text via Ollama
        ├── llm/
        │   ├── __init__.py
        │   └── ollama_llm.py        # LLM completion via Ollama
        └── db/
            └── repositories/
                ├── article_repo.py
                └── chunk_repo.py
```

### 2. Alembic migration — articles and chunks tables

```sql
CREATE TABLE articles (
    id              UUID PRIMARY KEY,
    url             TEXT NOT NULL,
    canonical_url   TEXT NOT NULL UNIQUE,
    content_hash    TEXT NOT NULL UNIQUE,
    raw_text        TEXT NOT NULL,
    title           TEXT,
    author          TEXT,
    published_at    TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
);

CREATE TABLE chunks (
    id              UUID PRIMARY KEY,
    article_id      UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    strategy        VARCHAR(20) NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       VECTOR(768),       -- nomic-embed-text output dimension
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chunks_article_id ON chunks(article_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

Also add to `model_call_logs` table:
```sql
CREATE TABLE model_call_logs (
    id                  UUID PRIMARY KEY,
    provider            VARCHAR(50) NOT NULL,
    model_name          VARCHAR(100) NOT NULL,
    request_type        VARCHAR(50) NOT NULL,
    prompt_template     TEXT NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ NOT NULL,
    duration_ms         INTEGER NOT NULL,
    status              VARCHAR(20) NOT NULL,
    retries             INTEGER NOT NULL DEFAULT 0,
    error_metadata      JSONB,
    related_entity_id   UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 3. Ollama LLM adapter (`infrastructure/llm/ollama_llm.py`)

Implements `LLMPort`. Makes HTTP calls to Ollama's `/api/generate` endpoint.

```python
class OllamaLLM(LLMPort):
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient):
        self.base_url = settings.ollama_base_url
        self.client = http_client

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> str:
        # POST to {base_url}/api/generate
        # Body: {"model": model, "prompt": prompt, "stream": false,
        #        "format": "json" if json_mode else None}
        # Retry on connection errors and 5xx with exponential backoff
        # Raise LLMError on exhausted retries
        # Return response["response"] string
        ...
```

**Critical:** Every call to `complete()` must:
1. Record `started_at` before the call
2. Record `finished_at` and `duration_ms` after
3. Persist a `ModelCallLog` regardless of success or failure
4. Include the `prompt_template` name (not the full prompt — pass it as a parameter)
5. Include `related_entity_id` (the article ID being processed)

Add `LLMError` to `domain/exceptions.py`.

### 4. Ollama embedder (`infrastructure/embeddings/ollama_embedder.py`)

Implements `EmbeddingPort`.

```python
class OllamaEmbedder(EmbeddingPort):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        # POST to {base_url}/api/embed
        # Body: {"model": settings.ollama_embedding_model, "input": texts}
        # Ollama's /api/embed accepts a list — use batch mode
        # Return list of float vectors
        # Record ModelCallLog for every call (request_type="embedding")
        ...
```

### 5. Chunkers (`infrastructure/chunkers/`)

Implement `ChunkerPort`. Each chunker takes raw text and returns a list of string chunks.

#### `semantic.py` — sentence-boundary chunker
- Split on sentence boundaries using `nltk.sent_tokenize` or `spacy`
- Group sentences into chunks until token count approaches `max_tokens` (default: 400)
- Overlap: include the last sentence of the previous chunk as the first sentence of the next
- Never split mid-sentence

#### `recursive.py` — recursive character splitter
- Split hierarchy: `\n\n` → `\n` → `. ` → ` ` → character
- Try each separator in order; if a split produces a chunk over `max_tokens`, split again with the next separator
- Overlap: last `overlap_chars` (default: 100) characters of previous chunk prepended to next

#### `fixed.py` — fixed-size chunker
- Split into chunks of exactly `chunk_size` tokens (default: 512) with `overlap` tokens (default: 64)
- Use `tiktoken` (cl100k_base encoding) for token counting
- Simplest and fastest — use as baseline for comparison

All chunkers must:
- Return chunks of at least 50 characters (discard shorter fragments)
- Record `token_count` for each chunk

### 6. Chunking strategy agent (`application/ingestion/chunking_agent.py`)

This is the first agentic decision in the system. A local LLM (Llama 3.1 8B) inspects the article and decides which chunking strategy to use.

**This prompt must be extremely precise.** Write it as if for a junior engineer who will follow it literally. No ambiguity. Every decision path must be explicit.

```python
CHUNKING_STRATEGY_PROMPT = """You are a text chunking strategy selector. Your job is to read the article metadata below and return a JSON object specifying which chunking strategy to use.

ARTICLE METADATA:
- Title: {title}
- Word count: {word_count}
- First 500 characters: {preview}
- Detected structure: {structure_hints}

CHUNKING STRATEGIES AVAILABLE:
1. "semantic" — splits on sentence boundaries and groups by meaning
   USE WHEN: the text is flowing prose, news articles, essays, blog posts
   DO NOT USE WHEN: the text has lists, tables, code, or clear section headers

2. "recursive" — splits on paragraph breaks, then sentences, then words
   USE WHEN: the text has clear paragraph structure but mixed content types
   USE WHEN: articles with some lists or subheadings mixed with prose

3. "fixed" — splits into fixed token windows with overlap
   USE WHEN: the text is very long (over 5000 words) and structure is unclear
   USE WHEN: the text contains code, technical documentation, or tabular data
   USE WHEN: you are uncertain which other strategy to use

DECISION RULES (apply in order, stop at first match):
- If word_count > 5000 → use "fixed"
- If structure_hints contains "code_block" or "table" → use "fixed"
- If structure_hints contains "heading" and word_count < 5000 → use "recursive"
- Otherwise → use "semantic"

YOU MUST respond with ONLY a valid JSON object. No explanation. No preamble. No markdown.
The JSON must have exactly this structure:
{{"strategy": "semantic" | "recursive" | "fixed", "reason": "one sentence"}}

Example valid response:
{{"strategy": "semantic", "reason": "flowing prose news article under 5000 words"}}"""
```

```python
async def decide_chunking_strategy(
    llm: LLMPort,
    model: str,
    article_text: str,
    title: str | None,
) -> Literal["semantic", "recursive", "fixed"]:
    word_count = len(article_text.split())
    preview = article_text[:500]
    structure_hints = detect_structure(article_text)  # see below

    prompt = CHUNKING_STRATEGY_PROMPT.format(
        title=title or "Unknown",
        word_count=word_count,
        preview=preview,
        structure_hints=", ".join(structure_hints) if structure_hints else "none detected",
    )

    raw = await llm.complete(prompt, model=model, json_mode=True,
                              prompt_template="chunking_strategy_v1")
    parsed = parse_strategy_response(raw)  # see below
    return parsed


def detect_structure(text: str) -> list[str]:
    """
    Cheap heuristic scan — no LLM needed.
    Returns a list of detected structure types.
    """
    hints = []
    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        hints.append("heading")
    if "```" in text or "    " in text:
        hints.append("code_block")
    if re.search(r"^\s*[\|\+]", text, re.MULTILINE):
        hints.append("table")
    if re.search(r"^\s*[-*]\s", text, re.MULTILINE):
        hints.append("list")
    return hints


def parse_strategy_response(raw: str) -> Literal["semantic", "recursive", "fixed"]:
    """
    Parse the LLM's JSON response. Must be defensive — local models
    sometimes emit extra whitespace, trailing commas, or markdown fences
    even when told not to.
    """
    # 1. Strip markdown code fences if present (```json ... ```)
    # 2. Strip leading/trailing whitespace
    # 3. json.loads()
    # 4. Validate "strategy" key exists and is one of the three valid values
    # 5. If parsing fails or strategy is invalid → default to "fixed" and log a warning
    #    (never crash the ingestion pipeline because the LLM misbehaved)
    ...
```

**Why the fallback to "fixed" matters for your talk:** This is a real failure mode with local models. They sometimes emit markdown fences, extra text, or invalid JSON despite being told not to. Your system must handle this gracefully. The fallback to "fixed" is conservative — it always produces *something* — and it's logged so you can observe how often it happens.

### 7. Ingestion use case (`application/ingestion/use_case.py`)

This orchestrates the full pipeline. It is called by the background task created in Block 2/3.

```python
class IngestArticleUseCase:
    def __init__(
        self,
        article_repo: ArticleRepository,
        chunk_repo: ChunkRepository,
        model_call_log_repo: ModelCallLogRepository,
        ingestion_job_repo: IngestionJobRepository,
        crawler: CrawlerPort,
        embedder: EmbeddingPort,
        llm: LLMPort,
        chunker_factory: ChunkerFactory,
        settings: Settings,
    ): ...

    async def execute(self, job_id: UUID) -> None:
        # 1. Load the job
        job = await self.ingestion_job_repo.find_by_id(job_id)

        # 2. Update status to "processing"
        await self.ingestion_job_repo.update_status(job_id, "processing")

        try:
            # 3. Crawl (if url) — CrawlResult already on job from Block 3,
            #    OR call crawler here for pdf/text types
            crawl_result = await self._get_content(job)

            # 4. Canonicalise + hash
            canonical_url = canonicalise_url(crawl_result.url)
            content_hash = hashlib.sha256(
                crawl_result.raw_text.encode()
            ).hexdigest()

            # 5. Dedup check
            existing = await self.article_repo.find_by_hash(content_hash)
            if existing:
                await self.ingestion_job_repo.update_status(job_id, "succeeded")
                log.info("ingestion.dedup.skipped", article_id=str(existing.id))
                return

            # 6. Persist article (status="processing")
            article = Article(
                id=uuid4(),
                url=crawl_result.url,
                canonical_url=canonical_url,
                content_hash=content_hash,
                raw_text=crawl_result.raw_text,
                title=crawl_result.title,
                author=crawl_result.author,
                published_at=crawl_result.published_at,
                ingested_at=datetime.utcnow(),
                status="processing",
            )
            await self.article_repo.save(article)

            # 7. Decide chunking strategy (LLM call)
            strategy = await decide_chunking_strategy(
                self.llm,
                model=self.settings.ollama_qa_model,
                article_text=crawl_result.raw_text,
                title=crawl_result.title,
            )

            # 8. Chunk
            chunker = self.chunker_factory.get(strategy)
            chunk_texts = chunker.chunk(crawl_result.raw_text, strategy)

            # 9. Embed (batch all chunks in one call)
            embeddings = await self.embedder.embed(chunk_texts)

            # 10. Build Chunk objects
            chunks = [
                Chunk(
                    id=uuid4(),
                    article_id=article.id,
                    content=text,
                    chunk_index=i,
                    strategy=strategy,
                    token_count=count_tokens(text),
                    embedding=embedding,
                )
                for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings))
            ]

            # 11. Persist chunks
            await self.chunk_repo.save_many(chunks)

            # 12. Update article status to "succeeded"
            await self.article_repo.update_status(article.id, "succeeded")

            # 13. Update job status to "succeeded"
            await self.ingestion_job_repo.update_status(job_id, "succeeded")

            log.info(
                "ingestion.succeeded",
                article_id=str(article.id),
                chunk_count=len(chunks),
                strategy=strategy,
            )

        except Exception as e:
            await self.ingestion_job_repo.update_status(
                job_id, "failed", error_message=str(e)
            )
            log.error("ingestion.failed", job_id=str(job_id), error=str(e))
            raise
```

### 8. Ingestion metrics

Add these to `MetricsRecorder` (create `application/metrics.py` if it doesn't exist):

```python
# Counters
ingestion_success_total: int
ingestion_failure_total: int
ingestion_dedup_total: int

# Latency histograms (track as list of ms values for now)
crawl_latency_ms: list[int]
chunking_decision_latency_ms: list[int]
embedding_latency_ms: list[int]
```

Record each metric at the appropriate step in the use case.

---

## Dependencies to add

```toml
"nltk>=3.8",
"tiktoken>=0.7",
```

Download NLTK punkt tokenizer on startup:
```python
import nltk
nltk.download("punkt", quiet=True)
```

---

## Definition of done

- [ ] Submitting a URL via `POST /ingestion/jobs` results in a fully ingested article with chunks and embeddings in the database
- [ ] `GET /ingestion/jobs/{id}` shows `status="succeeded"` after pipeline completes
- [ ] A `ModelCallLog` row exists for the chunking strategy LLM call
- [ ] A `ModelCallLog` row exists for the embedding call
- [ ] Submitting the same URL twice results in the second job completing with `status="succeeded"` but no duplicate article in the database (dedup worked)
- [ ] Chunks table contains embeddings (non-null `VECTOR(768)` values)
- [ ] `parse_strategy_response` correctly handles malformed JSON by returning `"fixed"` with a logged warning
- [ ] `detect_structure` correctly identifies headings, code blocks, and lists in test fixtures
- [ ] Ingestion metrics are recorded for each pipeline run
- [ ] All LLM calls log `duration_ms`, `status`, and `prompt_template`

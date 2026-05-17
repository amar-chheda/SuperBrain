# Superbrain — Data Flow

This document traces how data moves through the system for each major operation. Every diagram shows the full path from entry point to storage (or response), including which services are called, what is persisted, and where asynchronous hand-offs occur.

---

## 1. URL Ingestion

A URL submitted to the API is processed asynchronously in four sequential phases: crawl → chunk → embed → classify.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI<br/>/ingestion/jobs
    participant DB as PostgreSQL
    participant Crawler as Crawler<br/>(httpx / Spider)
    participant Chunker as Chunker<br/>(semantic/recursive/fixed)
    participant Embedder as OllamaEmbedder<br/>(nomic-embed-text)
    participant LLM_QA as OllamaLLM<br/>(llama3.1:8b)
    participant LLM_CLS as OllamaLLM<br/>(phi3:mini)

    Client->>API: POST /ingestion/jobs {url}
    API->>DB: INSERT ingestion_jobs (status=pending)
    API-->>Client: 202 Accepted {job_id}

    Note over API,LLM_CLS: Background task begins (FastAPI BackgroundTasks)

    API->>Crawler: fetch(url)
    Crawler-->>API: raw_html
    API->>API: extract_text(html) → raw_text
    API->>DB: UPDATE ingestion_jobs SET raw_text, status=crawled

    API->>DB: SELECT articles WHERE canonical_url = url
    alt Already ingested (same content_hash)
        API->>DB: UPDATE ingestion_jobs SET status=deduped
    else New content
        API->>DB: INSERT articles (status=pending)
        API->>LLM_QA: decide_chunking_strategy(raw_text)
        LLM_QA-->>API: {strategy: "semantic"|"recursive"|"fixed"}
        API->>Chunker: chunk(raw_text, strategy)
        Chunker-->>API: list[Chunk]

        loop for each chunk
            API->>Embedder: embed(chunk.content)
            Embedder-->>API: vector[768]
            API->>DB: INSERT chunks (content, embedding, content_tsv)
        end

        API->>DB: UPDATE articles SET status=ready
        API->>DB: UPDATE ingestion_jobs SET status=complete

        API->>LLM_CLS: classify(article, topics)
        LLM_CLS-->>API: [{topic_id, confidence, reason}]
        API->>DB: INSERT article_topic_matches
    end
```

### What is written to the database

| Table | When | Contents |
|---|---|---|
| `ingestion_jobs` | Immediately on POST | id, input_type, input_value, status=pending |
| `ingestion_jobs` | After crawl | raw_text, status=crawled |
| `ingestion_jobs` | After dedup check | status=deduped (if duplicate) or status=complete |
| `articles` | After dedup check (new only) | url, canonical_url, content_hash, raw_text, title, status=ready |
| `chunks` | After each embedding | content, chunk_index, strategy, token_count, embedding, content_tsv |
| `article_topic_matches` | After classification | article_id, topic_id, topic_version, confidence, reason |
| `model_call_logs` | After each LLM call | request_type, model_name, duration_ms, status |

---

## 2. QA (Question Answering)

A question flows synchronously through retrieval, fusion, evidence gating, and generation — then the full interaction is logged.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI<br/>/qa/ask
    participant Embedder as OllamaEmbedder<br/>(nomic-embed-text)
    participant DB as PostgreSQL
    participant Fuser as RRF Fusion
    participant Evidence as Evidence Builder
    participant LLM as OllamaLLM<br/>(llama3.1:8b)

    Client->>API: POST /qa/ask {question}

    API->>Embedder: embed(question)
    Embedder-->>API: query_vector[768]

    par Vector retrieval
        API->>DB: SELECT chunks ORDER BY embedding <=> query_vector LIMIT 20
        DB-->>API: vector_chunks (ranked by cosine distance)
    and BM25 retrieval
        API->>DB: SELECT chunks WHERE content_tsv @@ plainto_tsquery(question) LIMIT 20
        DB-->>API: bm25_chunks (ranked by ts_rank)
    end

    API->>Fuser: reciprocal_rank_fusion(vector_chunks, bm25_chunks, top_n=10)
    Fuser-->>API: fused_chunks (RRF score = Σ 1/(60+rank))

    API->>Evidence: build_evidence_set(fused_chunks)
    Evidence-->>API: evidence_set + sufficiency_check

    alt Evidence insufficient (< 2 chunks or max_score < 0.005)
        API->>DB: INSERT query_logs (aborted=true, abort_reason)
        API-->>Client: 200 {aborted: true, abort_reason}
    else Evidence sufficient
        API->>LLM: generate_answer(question, evidence)
        LLM-->>API: answer text with SOURCES: [chunk_ids]

        API->>API: parse_answer_response()<br/>reject hallucinated chunk IDs
        API->>DB: INSERT query_logs (question, answer, latencies)
        API-->>Client: 200 {answer, citations[]}
    end
```

### RRF Score Formula

For each chunk appearing in the ranked lists:

```
score = Σ  1 / (60 + rank_i)
```

A chunk that ranks 1st in vector search and 3rd in BM25 gets score `1/61 + 1/63 ≈ 0.032`. A chunk appearing in only one list gets a single term. Chunks are then sorted descending by score and truncated to `top_n=10`.

---

## 3. Digest Generation

The daily digest is a map-reduce pipeline: articles are selected, grouped by topic, deduplicated within each group, then summarised one group at a time.

```mermaid
sequenceDiagram
    autonumber
    participant Scheduler as APScheduler<br/>(daily @ 07:00 UTC)
    participant Digest as Digest Pipeline
    participant DB as PostgreSQL
    participant LLM as OllamaLLM<br/>(llama3.1:8b)

    Scheduler->>Digest: trigger(target_date=yesterday)

    Digest->>DB: INSERT digest_runs (status=running)

    Digest->>DB: SELECT articles WHERE status=ready<br/>AND ingested_at >= target_date<br/>AND topic_match.confidence IN (high, medium)
    DB-->>Digest: articles[]

    Digest->>Digest: deduplicate_by_url()<br/>(keep most-recent per canonical_url)

    Digest->>DB: SELECT article_topic_matches for article_ids
    DB-->>Digest: matches[]

    Digest->>Digest: join_matches(articles, matches)
    Digest->>Digest: group_by_topic()<br/>(sort by topic.priority DESC, count DESC)

    loop for each topic group
        Digest->>Digest: deduplicate_sources_within_group()<br/>(one article per domain)
        Digest->>LLM: summarise(topic_name, articles[])
        LLM-->>Digest: 3-6 sentence prose summary
        Digest->>DB: INSERT digest_items (run_id, topic_id, summary, article_ids, position)
    end

    Digest->>DB: UPDATE digest_runs SET status=complete,<br/>article_count, section_count, finished_at
```

### Digest trigger paths

```mermaid
graph LR
    SCHED[APScheduler<br/>CronTrigger UTC] -->|daily| PIPELINE[Digest Pipeline]
    API[POST /digests/trigger] -->|BackgroundTask| PIPELINE
    CLI[superbrain digest trigger] -->|HTTP POST| API
    PIPELINE --> DB[(PostgreSQL<br/>digest_runs<br/>digest_items)]
```

---

## 4. Telegram Bot

The Telegram webhook converts a chat message into an ingestion or QA API call and relays the result back to the user.

```mermaid
sequenceDiagram
    autonumber
    actor User as Telegram User
    participant TG as Telegram Servers
    participant Bot as Bot Webhook Handler<br/>/bot/webhook
    participant API as Superbrain API

    User->>TG: sends message (URL or question)
    TG->>Bot: POST /bot/webhook {Update}

    Bot->>Bot: parse Update → extract text

    alt Message looks like a URL
        Bot->>API: POST /ingestion/jobs {url}
        API-->>Bot: {job_id, status}
        Bot->>TG: sendMessage "Ingesting job_id …"
        TG-->>User: reply
    else Message is a question
        Bot->>API: POST /qa/ask {question}
        API-->>Bot: {answer, citations[]}
        Bot->>TG: sendMessage answer + citations
        TG-->>User: reply
    end
```

---

## 5. Topic Lifecycle and Reclassification

When a topic definition changes, all existing articles are re-classified against the updated definition.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI<br/>/topics/{id}
    participant DB as PostgreSQL
    participant Classifier as Classifier<br/>(phi3:mini)

    Client->>API: PUT /topics/{id} {name, description, examples}
    API->>DB: INSERT topics (version=old+1, status=active)
    API->>DB: UPDATE topics SET status=archived WHERE id=old_id
    API-->>Client: 200 {new_topic}

    Client->>API: POST /topics/{id}/reclassify
    API-->>Client: 202 Accepted

    Note over API,Classifier: Background task

    API->>DB: SELECT articles WHERE status=ready
    loop for each article (batched)
        API->>Classifier: classify(article, [updated_topic])
        Classifier-->>API: [{topic_id, confidence, reason}]
        API->>DB: INSERT article_topic_matches (topic_version=new)
    end
```

---

## 6. Observability Data Flow

Every model call and every QA query is logged to PostgreSQL for later inspection.

```mermaid
graph TD
    subgraph Application Layer
        ING[Ingestion Pipeline]
        QA[QA Pipeline]
        DIGEST[Digest Pipeline]
        CLASS[Classifier]
        METRICS[InMemoryMetricsRecorder]
    end

    subgraph Infrastructure Layer
        LLM_ADAPTER[OllamaLLM Adapter]
        MODEL_LOG_REPO[ModelCallLogRepo]
        QUERY_LOG_REPO[QueryLogRepo]
    end

    subgraph Storage
        DB[(PostgreSQL)]
        MEMORY[(In-process memory)]
    end

    subgraph Observability API
        OBS[GET /observe/metrics<br/>GET /observe/model-calls<br/>GET /observe/query-logs<br/>GET /observe/jobs/trace<br/>GET /observe/evals/run]
    end

    ING -->|increment / observe| METRICS
    QA -->|increment / observe| METRICS
    DIGEST -->|increment / observe| METRICS
    CLASS -->|increment / observe| METRICS

    METRICS --> MEMORY

    LLM_ADAPTER -->|save ModelCallLog| MODEL_LOG_REPO
    QA -->|save QueryLog| QUERY_LOG_REPO
    MODEL_LOG_REPO --> DB
    QUERY_LOG_REPO --> DB

    OBS -->|snapshot()| METRICS
    OBS -->|list_recent()| MODEL_LOG_REPO
    OBS -->|list_recent()| QUERY_LOG_REPO
```

### Metrics collected

| Metric name | Type | Emitted by |
|---|---|---|
| `ingestion_success_total` | counter | Ingestion pipeline |
| `ingestion_failure_total` | counter | Ingestion pipeline |
| `ingestion_dedup_total` | counter | Ingestion pipeline |
| `crawl_latency_ms` | histogram (p50/p95/p99) | Ingestion pipeline |
| `chunk_decision_latency_ms` | histogram | Ingestion pipeline |
| `embedding_latency_ms` | histogram | Ingestion pipeline |
| `classification_success_total` | counter | Classifier |
| `topic_match_count` | histogram | Classifier |
| `retrieval_latency_ms` | histogram | QA pipeline |
| `answer_latency_ms` | histogram | QA pipeline |
| `qa_success_total` | counter | QA pipeline |
| `qa_aborted_total` | counter | QA pipeline |
| `digest_success_total` | counter | Digest pipeline |
| `digest_failure_total` | counter | Digest pipeline |
| `digest_sections_total` | counter | Digest pipeline |

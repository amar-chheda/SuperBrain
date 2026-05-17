# Superbrain — Architecture

## Overview

Superbrain is a fully local agentic AI system. Every model call — embeddings, classification, chunking decisions, QA answers, digest summaries — runs on hardware you control via [Ollama](https://ollama.ai). There is no cloud AI dependency at runtime.

The system is built around a standard **hexagonal architecture** (also called ports-and-adapters): the domain and application layers are pure Python with no framework dependencies; infrastructure adapters (SQLAlchemy, httpx, Ollama, APScheduler) are plugged in at the boundary.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **API framework** | FastAPI | ≥ 0.115 |
| **ASGI server** | Uvicorn (with standard extras) | ≥ 0.30 |
| **Data validation** | Pydantic v2 + pydantic-settings | ≥ 2.7 |
| **Database** | PostgreSQL 16 with pgvector extension | — |
| **ORM** | SQLAlchemy (async) | ≥ 2.0 |
| **DB driver** | asyncpg | ≥ 0.29 |
| **Migrations** | Alembic | ≥ 1.13 |
| **Vector search** | pgvector (cosine similarity, ivfflat index) | ≥ 0.3 |
| **Full-text search** | PostgreSQL tsvector / GIN index (BM25-style) | — |
| **Local LLM runtime** | Ollama (HTTP API) | — |
| **Embedding model** | nomic-embed-text (768-dim) | via Ollama |
| **QA / chunking model** | llama3.1:8b | via Ollama |
| **Classification model** | phi3:mini | via Ollama |
| **Digest model** | llama3.1:8b (swap to lfm2-7b when available) | via Ollama |
| **HTTP client** | httpx (async) | ≥ 0.27 |
| **Web scraping** | httpx + BeautifulSoup4 (static) / Spider.cloud (JS) | — |
| **Tokenisation** | tiktoken (cl100k_base) | ≥ 0.7 |
| **Sentence splitting** | NLTK punkt tokenizer | ≥ 3.8 |
| **Telegram bot** | python-telegram-bot (webhooks) | ≥ 21.0 |
| **CLI** | Typer | ≥ 0.12 |
| **Scheduler** | APScheduler (AsyncIOScheduler) | ≥ 3.10 |
| **Structured logging** | structlog (JSON / console dual-mode) | ≥ 24.0 |
| **Runtime** | Python 3.12 |  |
| **Package manager** | uv | — |

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        Entry Points                         │
│   FastAPI (HTTP)    Telegram bot    CLI (Typer)             │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     Application Layer                       │
│   Use cases · Ports (interfaces) · MetricsRecorder          │
│   Pipelines: Ingestion · QA · Classification · Digest       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                       Domain Layer                          │
│   Entities (pure dataclasses) · Repository contracts        │
│   Exceptions · No framework dependencies                    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   Infrastructure Layer                      │
│   SQLAlchemy repos · Ollama adapters · Crawlers             │
│   Chunkers · APScheduler · asyncpg                          │
└─────────────────────────────────────────────────────────────┘
```

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph Inputs
        API[REST API<br/>POST /ingestion/jobs]
        TG[Telegram Bot<br/>webhook]
        CLI[CLI<br/>superbrain ingest url]
    end

    subgraph Core["Superbrain Core (FastAPI)"]
        direction TB
        ING[Ingestion Pipeline]
        CLASS[Classifier<br/>Phi-3 Mini]
        QA[QA Pipeline<br/>Llama 3.1 8B]
        DIGEST[Digest Pipeline<br/>Llama 3.1 8B]
        SCHED[APScheduler<br/>daily @ 07:00 UTC]
    end

    subgraph Storage
        PG[(PostgreSQL 16<br/>+ pgvector)]
    end

    subgraph Models["Ollama (local)"]
        EMB[nomic-embed-text<br/>768-dim embeddings]
        LLM_QA[llama3.1:8b<br/>QA + chunking]
        LLM_CLS[phi3:mini<br/>classification]
    end

    subgraph Crawlers
        HTTPX[httpx crawler<br/>static pages]
        SPIDER[Spider.cloud<br/>JS-rendered pages]
    end

    API --> ING
    TG --> ING
    CLI --> API

    ING --> Crawlers
    ING --> EMB
    ING --> LLM_QA
    ING --> CLASS
    ING --> PG

    CLASS --> LLM_CLS
    CLASS --> PG

    QA --> EMB
    QA --> PG
    QA --> LLM_QA

    SCHED --> DIGEST
    DIGEST --> PG
    DIGEST --> LLM_QA

    PG --> QA
    PG --> DIGEST
```

---

## Database Schema

Six domain tables plus three observability tables:

```mermaid
erDiagram
    ingestion_jobs {
        uuid id PK
        varchar input_type
        text input_value
        varchar status
        text raw_text
        timestamptz created_at
    }

    articles {
        uuid id PK
        text url
        text canonical_url
        text content_hash
        text raw_text
        text title
        varchar status
        timestamptz ingested_at
    }

    chunks {
        uuid id PK
        uuid article_id FK
        text content
        int chunk_index
        varchar strategy
        int token_count
        vector embedding
        tsvector content_tsv
    }

    topics {
        uuid id PK
        varchar name
        int version
        text description
        jsonb examples
        int priority
        varchar status
    }

    article_topic_matches {
        uuid id PK
        uuid article_id FK
        uuid topic_id FK
        int topic_version
        varchar confidence
        text reason
    }

    digest_runs {
        uuid id PK
        date date_label
        varchar status
        int article_count
        int section_count
        varchar triggered_by
    }

    digest_items {
        uuid id PK
        uuid run_id FK
        uuid topic_id FK
        text summary
        uuid[] article_ids
        int position
    }

    model_call_logs {
        uuid id PK
        varchar request_type
        varchar model_name
        int duration_ms
        varchar status
        uuid related_entity_id
    }

    query_logs {
        uuid id PK
        text question
        text answer
        bool aborted
        int retrieval_latency_ms
        int answer_latency_ms
    }

    articles ||--o{ chunks : "has"
    articles ||--o{ article_topic_matches : "matched to"
    topics ||--o{ article_topic_matches : "matched by"
    digest_runs ||--o{ digest_items : "contains"
    topics ||--o{ digest_items : "appears in"
```

---

## Key Design Decisions

**Local-only by design.** No calls to OpenAI, Anthropic, or any cloud model provider. All inference runs through Ollama's local HTTP API. This makes the system fully air-gappable.

**Hexagonal architecture.** Use cases depend on abstract repository and port interfaces, not concrete implementations. Swapping the database or the LLM runtime requires only a new adapter — the application logic is unchanged.

**Async throughout.** FastAPI + asyncpg + httpx + APScheduler all run on the same asyncio event loop. No threads except for the metrics lock (fine-grained, sub-microsecond).

**Postgres as the only store.** Vector search (pgvector), full-text search (tsvector), relational joins, and JSONB metadata all live in one database. No separate vector DB, no Redis, no Elasticsearch.

**One session per request.** SQLAlchemy sessions are created at the start of each request (or background task) and closed at the end. No global session, no leaked connections.

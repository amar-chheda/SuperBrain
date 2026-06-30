# Superbrain

A fully local agentic AI system that ingests web content, classifies it by topic, answers questions using retrieved evidence, and generates daily digest summaries — with every model call running on your own hardware via Ollama.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.12 | |
| uv | latest | `pip install uv` |
| PostgreSQL | 16 | with pgvector extension |
| Ollama | latest | `ollama.ai` |
| Docker | optional | for PostgreSQL via compose |

**Pull the required models into Ollama before starting:**

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull phi3:mini
```

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd Superbrain

# 2. Install dependencies
uv sync

# 3. Start PostgreSQL (or use an existing instance)
docker compose up -d

# 4. Copy and configure environment
cp .env.example .env
# Edit .env — set SUPERBRAIN_DATABASE_URL at minimum

# 5. Run database migrations
alembic upgrade head

# 6. Start the API server
uvicorn superbrain.main:app --reload

# 7. Verify health
curl http://localhost:8000/health
# {"status":"ok","db":"connected","ollama":"connected"}
```

---

## Environment Variables

All variables use the `SUPERBRAIN_` prefix (set in `.env` or the shell).

| Variable | Default | Description |
|---|---|---|
| `SUPERBRAIN_DATABASE_URL` | — | **Required.** PostgreSQL connection string |
| `SUPERBRAIN_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `SUPERBRAIN_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model tag |
| `SUPERBRAIN_OLLAMA_QA_MODEL` | `llama3.1:8b` | QA + chunking decision model |
| `SUPERBRAIN_OLLAMA_CLASSIFICATION_MODEL` | `phi3:mini` | Article classification model |
| `SUPERBRAIN_OLLAMA_DIGEST_MODEL` | `llama3.1:8b` | Digest summarisation model |
| `SUPERBRAIN_DIGEST_SCHEDULE_HOUR` | `7` | UTC hour to run the daily digest |
| `SUPERBRAIN_CRAWLER_BACKEND` | `httpx` | `httpx` (static) or `spider` (JS-rendered) |
| `SUPERBRAIN_SPIDER_API_KEY` | — | Required only when `CRAWLER_BACKEND=spider` |
| `SUPERBRAIN_TELEGRAM_BOT_TOKEN` | — | Optional — enables Telegram bot |
| `SUPERBRAIN_TELEGRAM_WEBHOOK_URL` | — | Optional — Telegram webhook endpoint |
| `SUPERBRAIN_API_BASE_URL` | `http://localhost:8000` | Used by the CLI |
| `SUPERBRAIN_LOG_LEVEL` | `INFO` | Log level |
| `SUPERBRAIN_LOG_FORMAT` | `json` | `json` or `text` |

---

## API Reference

### Ingestion

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingestion/jobs` | Submit a URL, PDF, or text for ingestion |
| `GET` | `/ingestion/jobs/{job_id}` | Poll job status |

### QA

| Method | Path | Description |
|---|---|---|
| `POST` | `/qa/ask` | Ask a question; returns grounded answer + citations |

### Topics

| Method | Path | Description |
|---|---|---|
| `GET` | `/topics` | List all active topics |
| `POST` | `/topics` | Create a topic |
| `GET` | `/topics/{id}` | Get topic by ID |
| `PUT` | `/topics/{id}` | Update topic (creates new version, archives old) |
| `DELETE` | `/topics/{id}` | Archive a topic |
| `POST` | `/topics/{id}/reclassify` | Re-classify all articles against updated topic |
| `POST` | `/topics/classify/articles/{id}` | Classify a single article |
| `GET` | `/articles/{id}/topics` | Get topic matches for an article |

### Digests

| Method | Path | Description |
|---|---|---|
| `POST` | `/digests/trigger` | Trigger a digest run (async, returns immediately) |
| `GET` | `/digests` | List recent digest runs |
| `GET` | `/digests/{run_id}` | Get a digest run with its full section list |

### Observability

| Method | Path | Description |
|---|---|---|
| `GET` | `/observe/metrics` | Live metrics snapshot (counters + percentiles) |
| `GET` | `/observe/model-calls` | Recent model call logs (filterable) |
| `GET` | `/observe/jobs/{job_id}/trace` | Full ingestion trace for a job |
| `GET` | `/observe/query-logs` | Recent QA question/answer history |
| `GET` | `/observe/evals/run` | Run the eval suite and return results |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | DB + Ollama liveness check |

---

## CLI

```bash
# Ingest a URL
superbrain ingest url https://example.com/article

# Check job status
superbrain ingest status <job-id>

# Trigger a digest run
superbrain digest trigger
superbrain digest trigger --date 2024-01-15

# Check API health
superbrain health
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests run without a database or Ollama — all network calls are avoided via pure unit tests of parsing and fusion logic.

---

## Repository Structure

```
Superbrain/
├── src/superbrain/
│   ├── main.py                        # FastAPI app factory + lifespan
│   ├── settings.py                    # Pydantic settings (SUPERBRAIN_ prefix)
│   ├── logging_config.py              # structlog JSON/console setup
│   ├── middleware.py                  # RequestID middleware
│   └── app/
│       ├── api/                       # HTTP route handlers
│       │   ├── router.py              # Top-level router
│       │   ├── health.py
│       │   ├── ingestion.py
│       │   ├── qa.py
│       │   ├── topics.py
│       │   ├── digests.py
│       │   └── observability.py
│       ├── domain/                    # Pure domain layer (no framework deps)
│       │   ├── entities.py            # Dataclass entities
│       │   ├── repositories.py        # Abstract repository contracts
│       │   └── exceptions.py
│       ├── application/               # Use cases and ports
│       │   ├── ports.py               # LLMPort, EmbeddingPort, CrawlerPort, ChunkerPort
│       │   ├── metrics.py             # InMemoryMetricsRecorder
│       │   ├── ingestion/             # Ingestion use case + chunking agent
│       │   ├── qa/                    # QA use case + answer generator
│       │   ├── retrieval/             # VectorRetriever, BM25Retriever, RRF fusion
│       │   ├── topics/                # Classifier + topic use cases
│       │   ├── digest/                # Digest pipeline (select/group/dedup/summarise)
│       │   ├── scheduler/             # APScheduler adapter
│       │   └── evals/                 # Eval harness + retrieval/citation checks
│       ├── infrastructure/            # Concrete adapters
│       │   ├── db/
│       │   │   ├── engine.py          # Async engine + session factory
│       │   │   ├── models.py          # SQLAlchemy ORM models
│       │   │   └── repositories/      # One file per aggregate
│       │   ├── crawlers/              # httpx + spider backends
│       │   ├── chunkers/              # semantic / recursive / fixed + factory
│       │   ├── embeddings/            # OllamaEmbedder
│       │   └── llm/                   # OllamaLLM with retry + ModelCallLog
│       ├── bot/                       # Telegram webhook handler
│       └── cli/                       # Typer CLI commands
├── migrations/                        # Alembic migration scripts
│   └── versions/
│       ├── 001_enable_pgvector.py
│       ├── 002_create_ingestion_jobs.py
│       ├── 003_add_raw_text_to_ingestion_jobs.py
│       ├── 004_add_articles_chunks_model_call_logs.py
│       ├── 005_add_query_logs_and_fts.py
│       └── 006_add_digest_tables.py
├── tests/                             # Unit tests (no DB/Ollama required)
│   ├── test_chunking_agent.py
│   ├── test_classification.py
│   ├── test_qa.py
│   └── test_url_utils.py
├── docs/                              # This documentation
├── build-stages/                      # Block-by-block implementation specs
├── docker-compose.yml                 # PostgreSQL + pgvector
├── pyproject.toml
└── .env.example
```

---

## Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Check current revision
alembic current

# Generate a new migration (after changing models.py)
alembic revision --autogenerate -m "description"
```

Migration history:

| Revision | Description |
|---|---|
| 001 | Enable pgvector extension |
| 002 | Create `ingestion_jobs` table |
| 003 | Add `raw_text` column to `ingestion_jobs` |
| 004 | Create `articles`, `chunks`, `model_call_logs` tables |
| 005 | Create `query_logs` table; add `content_tsv` FTS column to `chunks` |
| 006 | Create `digest_runs` and `digest_items` tables |

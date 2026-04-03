# Superbrain

Superbrain is a local-first personal article intelligence assistant.

Phase 5 adds daily digest generation and scheduling abstractions on top of ingestion, retrieval, and topic classification.

## Stack

- Python 3.12+
- `uv` for dependency and command management
- FastAPI
- SQLAlchemy + Alembic
- PostgreSQL + `pgvector`

## Project Layout

```text
superbrain/
  pyproject.toml
  uv.lock
  README.md
  .env.example
  docker-compose.yml
  migrations/
  docs/
  src/superbrain/
    app/
      main.py
      config/
      api/
      bot/
      domain/
      application/
      infrastructure/
      observability/
      prompts/
      workflows/
      tasks/
      cli/
  tests/
    unit/
    integration/
    e2e/
```

## Setup

1. Install dependencies:
```bash
uv sync
```
2. Create environment file:
```bash
cp .env.example .env
```
3. Start PostgreSQL:
```bash
docker compose up -d db
```
Optional observability stack (Prometheus + Grafana):
```bash
docker compose up -d prometheus grafana
```
4. Apply migrations:
```bash
uv run alembic upgrade head
```

## Run

```bash
uv run uvicorn superbrain.app.main:app --reload
```

Health check:

```bash
curl -i http://localhost:8000/health
```

Ingest a URL:

```bash
curl -X POST http://localhost:8000/ingestion/jobs \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/article?utm_source=newsletter"}'
```

Check ingestion job:

```bash
curl http://localhost:8000/ingestion/jobs/<job_id>
```

Ask a grounded question:

```bash
curl -X POST http://localhost:8000/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What improves retrieval reliability?","top_k":6}'
```

Create a topic:

```bash
curl -X POST http://localhost:8000/topics \
  -H "Content-Type: application/json" \
  -d '{
    "name":"work",
    "description":"Engineering and architecture content",
    "positive_examples":["system design","backend platform"],
    "negative_examples":["travel","recipes"],
    "priority":10
  }'
```

Classify an article:

```bash
curl -X POST http://localhost:8000/topics/classify/articles/<article_id>
```

Trigger daily digest now:

```bash
curl -X POST http://localhost:8000/digests/trigger -H "Content-Type: application/json" -d '{}'
```

Get latest digest:

```bash
curl http://localhost:8000/digests/latest
```

Check model runtime health:

```bash
curl http://localhost:8000/health/models
```

Prometheus metrics:

```bash
curl http://localhost:8000/metrics
```

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## Observability Notes

- Operational observability and eval hook guidance:
  `docs/operations-observability.md`
- End-to-end architecture and workflow diagrams:
  `docs/flow-diagrams.md`
- Prometheus scrape config and Grafana provisioning:
  `observability/`

## What is intentionally not implemented yet

- Telegram bot command handling
- DBOS-backed production scheduler/runtime

These are added in later prompt-pack phases.

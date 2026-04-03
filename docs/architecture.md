# Architecture Overview

Superbrain uses a layered architecture designed for incremental delivery and clean boundaries.

## Layers

- `domain`: Framework-free entities, value objects, and repository contracts.
- `application`: Service interfaces and use-case orchestration.
- `infrastructure`: SQLAlchemy persistence, model/provider adapters, and external integrations.
- `api` / `bot` / `cli`: Delivery interfaces that invoke application services.

## Baseline decisions in Phase 1-6

- FastAPI app factory with startup lifespan and shared wiring.
- Typed environment-driven settings via `pydantic-settings`.
- JSON structured logging with request/job correlation IDs.
- PostgreSQL + SQLAlchemy + Alembic migration foundation.
- Repository contracts in domain layer before concrete implementations.
- Provider/service interfaces to support swappable local model runtimes.
- Ingestion use case orchestrates canonicalization, dedup, extraction, chunking, embedding, and persistence.
- Hybrid retrieval fuses vector and lexical ranks with reciprocal-rank fusion.
- Grounded QA returns structured answers and citations only from retrieved evidence.
- Topics are versioned definitions (description/examples/priority/status) with explicit reclassification workflows.
- Daily digest pipeline selects yesterday's articles, groups by topic, dedupes, and persists digest runs/items.
- Scheduling is abstracted through an in-process scheduler with manual trigger hooks, ready for DBOS migration.
- Model and embedding calls are audited in persistent `model_call_logs`.
- Metrics are captured through a recorder abstraction and emitted from core workflows.
- Eval hooks provide retrieval/citation/groundedness checks without a full eval platform yet.

## Current extension points

- `superbrain.app.application.ports` defines provider and service protocols.
- `superbrain.app.application.ingestion` provides canonicalization, deduplication, chunking, and use-case orchestration.
- `superbrain.app.application.retrieval` provides candidate scoring and fusion.
- `superbrain.app.application.qa` handles citation construction and grounded QA orchestration.
- `superbrain.app.application.topics` provides topic CRUD/versioning and classification/reclassification use cases.
- `superbrain.app.application.digest` provides selection/dedup/grouping and daily digest orchestration.
- `superbrain.app.domain.repositories` defines persistence contracts.
- `superbrain.app.infrastructure.db` provides DB engine/session and ORM models for ingestion entities.
- `query_logs` persist question/answer timings and evidence references.
- `article_topic_matches` persist explainable article-topic assignment metadata.
- `digest_runs` and `digest_items` persist generated digest history.
- `migrations/` establishes schema versioning and `pgvector` extension setup.

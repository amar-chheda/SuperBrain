# Superbrain — Build Guide for Claude Code

## What is Superbrain?

A local-only agentic AI system built on FastAPI, PostgreSQL, and Ollama. It ingests web pages, chunks and embeds them locally using small models, classifies them by topic, answers questions with grounded citations, and generates daily digests — with zero cloud AI calls. All inference runs on your machine.

**Architecture in one sentence:** Delivery interfaces → Orchestration (agentic routing) → Application use cases → Infrastructure adapters → PostgreSQL + Ollama.

---

## How to use these specs

Each spec file is a self-contained feature brief for one build block. Give Claude Code one file at a time. Do not skip blocks — each one depends on the previous.

**Start every session with this context:**
> "I'm building Superbrain, a local-only agentic AI system. Here is the spec for the next block to build: [paste or attach the spec file]. The previous blocks are already implemented. Build everything in this spec."

---

## Build order (strict — each block depends on the one above)

| Block | File | What it delivers |
|-------|------|-----------------|
| 1 | `block-01-foundation.md` | FastAPI app, settings, logging, domain entities, repository contracts, port interfaces, PostgreSQL + Alembic, `/health` |
| 2 | `block-02-delivery.md` | Telegram bot, FastAPI REST routes, CLI, request ID middleware, job submission and tracking |
| 3 | `block-03-scraping.md` | Spider crawler adapter, httpx crawler adapter, config switch, URL canonicalisation, text extraction |
| 4 | `block-04-ingestion.md` | Chunking strategy agent (LLM), embedder, ingestion pipeline, deduplication, model call logging |
| 5 | `block-05-classification.md` | Topic CRUD + versioning, Phi-3 Mini classifier, structured JSON output, article-topic matching |
| 6 | `block-06-qa.md` | Hybrid retrieval (vector + BM25 + RRF), grounded QA, low-evidence abort, citation builder |
| 7 | `block-07-digest.md` | Article selection, topic grouping, source deduplication, LLM summarisation, scheduler |
| 8 | `block-08-observability.md` | MetricsRecorder, eval harness, groundedness checks, observability API, smoke tests |

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| API framework | FastAPI + uvicorn | async, typed, fast |
| Package manager | uv | fast, reproducible |
| Settings | pydantic-settings | typed env config |
| Logging | structlog | JSON structured logs |
| Database | PostgreSQL 16 + pgvector | vector + relational in one store |
| ORM + migrations | SQLAlchemy (async) + Alembic | typed, async |
| Local model runtime | Ollama | one API for all models |
| QA model | Llama 3.1 8B | strong instruction following at 8B |
| Classification model | Phi-3 Mini 3.8B | reliable structured output, small |
| Digest model | LiquidAI LFM-7B | sequential synthesis |
| Embeddings | nomic-embed-text | purpose-built, 768d, long context |
| Crawler (JS) | Spider (Rust-backed) | handles SPAs and dynamic pages |
| Crawler (static) | httpx + BeautifulSoup | fast, lightweight |
| Scheduler | APScheduler | in-process cron |
| Bot | python-telegram-bot v20+ | async webhook |
| CLI | Typer | clean, typed |
| Tests | pytest + pytest-asyncio | standard async test stack |

---

## Local model setup (run before starting)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1:8b
ollama pull phi3:mini
ollama pull nomic-embed-text
# For LiquidAI — check Ollama model library for current tag
# ollama pull lfm2-7b  (or equivalent)
```

---

## Key design decisions to preserve across all blocks

**1. Port/adapter pattern throughout.** Every external dependency (crawler, LLM, embedder, database) is accessed through an abstract interface defined in `application/ports.py` or `domain/repositories.py`. Infrastructure implementations live in `infrastructure/`. This is what makes the Spider ↔ httpx switch a one-line config change and what makes tests possible without Ollama running.

**2. Every LLM call is logged.** `ModelCallLog` records are persisted for every call to Ollama, including failures and retries. This is non-negotiable — it's how you debug a local AI system.

**3. Prompts are explicit and defensive.** Local 7-8B models need unambiguous instructions. Every prompt in this system includes explicit decision rules, exact output format specification, and fallback handling for malformed output. Never trust a local model to "figure it out."

**4. The ingestion pipeline never crashes on classification failure.** Classification is best-effort. If the classifier returns invalid JSON, logs a warning, records the failure, and moves on. Ingestion status is `succeeded` regardless of whether classification succeeded.

**5. QA aborts rather than hallucinating.** If the evidence set is below threshold, the system returns `aborted=true` with a reason. This is a design decision, not a failure mode.

**6. Request IDs and job IDs flow through every log line.** Every log line emitted during request handling must include `request_id`. Every log line during a background job must include `job_id`. Use structlog's context vars for this.

---

## Dependency graph (visual)

```
Block 1: Foundation
    └── Block 2: Delivery
            └── Block 3: Scraping
                    └── Block 4: Ingestion ──────────────────┐
                            └── Block 5: Classification       │
                            └── Block 6: Retrieval + QA ◄────┘
                                    └── Block 7: Digest
                                            └── Block 8: Observability (closes the loop across all)
```

---

## Conference demo path (end-to-end)

Once all 8 blocks are built, the live demo follows this path:

1. **Send a URL to the Telegram bot** → bot creates an `IngestionJob` and replies with job ID
2. **Watch the trace** at `GET /observe/jobs/{id}/trace` → show the audience each model call: chunking strategy decision (Llama 3.1 8B), embedding (nomic-embed-text), classification (Phi-3 Mini)
3. **Ask a question** via `POST /qa/ask` → receive a grounded answer with citations
4. **Show what happens without evidence** → ask about something not in the knowledge base → system returns `aborted=true`
5. **Trigger a digest** via `POST /digests/trigger` → show the grouped, summarised output
6. **Show the metrics** at `GET /observe/metrics` → counters, latencies, eval scores
7. **Run the eval suite** at `GET /observe/evals/run` → show pass/fail for retrieval recall and groundedness

---

## Common issues when building with Claude Code

- **Always specify which block is being built** — Claude Code needs context about what already exists
- **Reference the domain entity names exactly** — `Article`, `Chunk`, `Topic`, `IngestionJob` etc. are defined in Block 1 and referenced in every subsequent block
- **If a block generates a migration, run it** before testing the next block
- **Check `definition of done`** at the bottom of each spec before moving to the next block — it's a checklist, not a suggestion

# Superbrain — Block 2: Delivery Interfaces

## Context

Superbrain is a local-only agentic AI system. This is Block 2. It depends entirely on Block 1 being complete — the FastAPI app factory, domain entities, port interfaces, and database foundation must all exist before starting this block.

This block wires up every surface through which a user sends data into Superbrain. The goal is to have working endpoints for job submission before any of the downstream pipeline exists. Each route creates an `IngestionJob` record and returns immediately — the actual work happens in later blocks.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    ├── api/
    │   ├── __init__.py
    │   ├── router.py            # mounts all sub-routers
    │   ├── health.py            # GET /health (already in Block 1, move here)
    │   ├── ingestion.py         # POST /ingestion/jobs
    │   ├── qa.py                # POST /qa/ask (stub — wired in Block 6)
    │   ├── topics.py            # CRUD stubs (wired in Block 5)
    │   └── digests.py           # POST /digests/trigger stub (wired in Block 7)
    ├── bot/
    │   ├── __init__.py
    │   └── telegram.py          # Telegram webhook handler
    └── cli/
        ├── __init__.py
        └── commands.py          # CLI entry points via typer
```

### 2. Ingestion job entity and repository

Add to `domain/entities.py`:

```python
@dataclass
class IngestionJob:
    id: UUID
    input_type: Literal["url", "pdf", "text"]
    input_value: str              # the URL, file path, or raw text
    status: Literal["pending", "processing", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    source: Literal["api", "telegram", "cli", "scheduler"] = "api"
```

Add to `domain/repositories.py`:

```python
class IngestionJobRepository(ABC):
    @abstractmethod
    async def save(self, job: IngestionJob) -> None: ...
    @abstractmethod
    async def find_by_id(self, job_id: UUID) -> IngestionJob | None: ...
    @abstractmethod
    async def update_status(
        self,
        job_id: UUID,
        status: Literal["pending", "processing", "succeeded", "failed"],
        error_message: str | None = None,
    ) -> None: ...
```

Add an Alembic migration to create the `ingestion_jobs` table:

```sql
CREATE TABLE ingestion_jobs (
    id          UUID PRIMARY KEY,
    input_type  VARCHAR(10) NOT NULL,
    input_value TEXT NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    source      VARCHAR(20) NOT NULL DEFAULT 'api',
    error_message TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 3. FastAPI ingestion routes (`api/ingestion.py`)

```
POST /ingestion/jobs
```

Request body:
```json
{
  "input_type": "url",
  "input_value": "https://example.com/article"
}
```

- Validate `input_type` is one of `url`, `pdf`, `text`
- For `url`: validate it is a well-formed https URL
- Create an `IngestionJob` with `status="pending"`, `source="api"`
- Persist via `IngestionJobRepository`
- Return `202 Accepted` with the job record
- Log the job creation with `job_id` in the structured log

```
GET /ingestion/jobs/{job_id}
```

- Return the job record by ID
- `404` if not found
- This is how callers poll for job completion

Response shape for both:
```json
{
  "id": "uuid",
  "input_type": "url",
  "input_value": "https://...",
  "status": "pending",
  "source": "api",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "error_message": null
}
```

### 4. Stub routes (return `501 Not Implemented` until later blocks)

**`api/qa.py`**
```
POST /qa/ask
```
Body: `{ "question": "..." }`
Returns: `{"detail": "QA pipeline not yet implemented"}` with HTTP 501

**`api/topics.py`**
```
GET  /topics
POST /topics
GET  /topics/{id}
PUT  /topics/{id}
```
All return 501 for now.

**`api/digests.py`**
```
POST /digests/trigger
GET  /digests
GET  /digests/{id}
```
All return 501 for now.

### 5. Request ID middleware

Create `middleware/request_id.py`:

```python
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.unbind_contextvars("request_id")
        return response
```

Every log line during a request must include the `request_id`. Every response must include `X-Request-ID` header.

### 6. Telegram bot webhook (`bot/telegram.py`)

Use `python-telegram-bot` (async version, v20+).

The bot listens for messages and creates `IngestionJob` records exactly like the API does. Supported message formats:

- A bare URL → `input_type="url"`, `source="telegram"`
- A message starting with `/ask ` → stub, reply "QA not yet available"
- Anything else → reply "Send me a URL to ingest"

The Telegram bot webhook endpoint must be registered as a FastAPI route:
```
POST /bot/telegram/webhook
```

The bot token is loaded from settings:
```python
telegram_bot_token: str | None = None
telegram_webhook_url: str | None = None
```

If `telegram_bot_token` is None, the bot is disabled silently (no error on startup).

On receiving a URL:
1. Create an `IngestionJob` with `source="telegram"`
2. Persist it
3. Reply to the user: "Got it — ingesting <url>. Job ID: <id>"

### 7. CLI (`cli/commands.py`)

Use `typer`. Entry point: `superbrain`

```bash
superbrain ingest url <url>
# Creates an IngestionJob with source="cli", prints job ID

superbrain ingest status <job_id>
# Polls and prints job status

superbrain health
# Hits GET /health and pretty-prints the result
```

Register as a script in `pyproject.toml`:
```toml
[project.scripts]
superbrain = "superbrain.cli.commands:app"
```

### 8. Error handling

Create `api/errors.py` with a global exception handler registered on the FastAPI app:

| Exception class     | HTTP status |
|---------------------|-------------|
| `ValidationError`   | 400         |
| `NotFoundError`     | 404         |
| `ConflictError`     | 409         |
| Unhandled exception | 500         |

All error responses use this shape:
```json
{
  "error": "not_found",
  "message": "IngestionJob abc123 not found",
  "request_id": "uuid"
}
```

Define `NotFoundError`, `ConflictError`, and `ValidationError` as custom exceptions in `domain/exceptions.py`.

---

## Dependencies to add

```toml
"python-telegram-bot[webhooks]>=21.0",
"typer>=0.12",
```

---

## Definition of done

- [ ] `POST /ingestion/jobs` with a valid URL returns `202` and a job record with `status="pending"`
- [ ] `GET /ingestion/jobs/{id}` returns the job or `404`
- [ ] `X-Request-ID` header is present on every response
- [ ] Every log line for a request contains `request_id` and `job_id` fields
- [ ] Stub routes return `501` with a clear message
- [ ] Sending a URL to the Telegram bot creates an `IngestionJob` and replies with the job ID
- [ ] `superbrain ingest url <url>` creates a job and prints its ID
- [ ] `superbrain health` prints the health check result
- [ ] Invalid input (bad URL, unknown input_type) returns `400` with a structured error body

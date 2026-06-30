# Superbrain — Block 1: Project Foundation

## Context

Superbrain is a local-only agentic AI system built on FastAPI, PostgreSQL, and Ollama. It ingests web pages, chunks and embeds them locally, classifies them by topic, answers questions with grounded citations, and generates daily digests — all without any cloud AI calls.

This is Block 1. It has no dependencies on any other block. Everything else in the system builds on top of what you create here. Get the boundaries right — bad abstractions here ripple through every block that follows.

---

## What to build

### 1. Project scaffold

Use **uv** as the package manager. Python 3.12+.

```
superbrain/
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── alembic.ini
├── migrations/
│   └── versions/
└── src/
    └── superbrain/
        ├── __init__.py
        ├── main.py                  # FastAPI app factory
        ├── settings.py              # pydantic-settings config
        ├── logging_config.py        # structured JSON logging
        └── app/
            ├── domain/
            │   ├── __init__.py
            │   ├── entities.py      # core domain models
            │   └── repositories.py  # abstract repository contracts
            ├── application/
            │   └── ports.py         # provider/service interfaces
            └── infrastructure/
                └── db/
                    ├── engine.py    # SQLAlchemy engine + session
                    └── models.py    # ORM models (empty stubs for now)
```

### 2. Settings (`settings.py`)

Use `pydantic-settings` with a `.env` file. Required fields:

```python
class Settings(BaseSettings):
    # Database
    database_url: str  # postgresql+asyncpg://...

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_qa_model: str = "llama3.1:8b"
    ollama_classification_model: str = "phi3:mini"
    ollama_digest_model: str = "llama3.1:8b"

    # Crawler
    crawler_backend: Literal["spider", "httpx"] = "httpx"

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

### 3. Structured logging (`logging_config.py`)

- JSON log output using `structlog` or `python-json-logger`
- Every log line must include: `timestamp`, `level`, `logger`, `message`
- Middleware must inject `request_id` (UUID) into every log line for a request
- Background jobs must inject `job_id` into every log line for that job's lifecycle
- Log format switchable via `Settings.log_format` (json in prod, text in dev)

### 4. Domain entities (`domain/entities.py`)

Pure Python dataclasses or Pydantic models. No SQLAlchemy, no FastAPI here. These are the core objects the system reasons about:

```python
@dataclass
class Article:
    id: UUID
    url: str
    canonical_url: str
    content_hash: str
    raw_text: str
    title: str | None
    author: str | None
    published_at: datetime | None
    ingested_at: datetime
    status: Literal["pending", "processing", "succeeded", "failed"]

@dataclass
class Chunk:
    id: UUID
    article_id: UUID
    content: str
    chunk_index: int
    strategy: Literal["semantic", "recursive", "fixed"]
    token_count: int

@dataclass
class Topic:
    id: UUID
    name: str
    version: int
    description: str
    examples: list[str]
    priority: int
    status: Literal["active", "archived"]

@dataclass
class QueryLog:
    id: UUID
    question: str
    answer: str
    evidence_chunk_ids: list[UUID]
    retrieval_latency_ms: int
    answer_latency_ms: int
    created_at: datetime

@dataclass
class ModelCallLog:
    id: UUID
    provider: str
    model_name: str
    request_type: str           # "embedding" | "extraction" | "classification" | "qa" | "digest"
    prompt_template: str
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    status: Literal["success", "failed"]
    retries: int
    error_metadata: dict | None
    related_entity_id: UUID | None
```

### 5. Repository contracts (`domain/repositories.py`)

Abstract base classes only — no implementations here. These define what the infrastructure layer must provide:

```python
class ArticleRepository(ABC):
    @abstractmethod
    async def save(self, article: Article) -> None: ...
    @abstractmethod
    async def find_by_hash(self, content_hash: str) -> Article | None: ...
    @abstractmethod
    async def find_by_id(self, article_id: UUID) -> Article | None: ...
    @abstractmethod
    async def list_by_date(self, date: date) -> list[Article]: ...

class ChunkRepository(ABC):
    @abstractmethod
    async def save_many(self, chunks: list[Chunk]) -> None: ...
    @abstractmethod
    async def find_by_article(self, article_id: UUID) -> list[Chunk]: ...

class TopicRepository(ABC):
    @abstractmethod
    async def save(self, topic: Topic) -> None: ...
    @abstractmethod
    async def list_active(self) -> list[Topic]: ...
    @abstractmethod
    async def find_by_id(self, topic_id: UUID) -> Topic | None: ...

class ModelCallLogRepository(ABC):
    @abstractmethod
    async def save(self, log: ModelCallLog) -> None: ...

class QueryLogRepository(ABC):
    @abstractmethod
    async def save(self, log: QueryLog) -> None: ...
```

### 6. Port interfaces (`application/ports.py`)

Abstract interfaces for external providers. Implementations live in infrastructure. This is what makes the system swappable:

```python
class CrawlerPort(ABC):
    @abstractmethod
    async def fetch(self, url: str) -> CrawlResult: ...

@dataclass
class CrawlResult:
    url: str
    canonical_url: str
    raw_text: str
    title: str | None
    author: str | None
    published_at: datetime | None
    status_code: int

class EmbeddingPort(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class LLMPort(ABC):
    @abstractmethod
    async def complete(self, prompt: str, *, model: str, json_mode: bool = False) -> str: ...

class ChunkerPort(ABC):
    @abstractmethod
    def chunk(self, text: str, strategy: Literal["semantic", "recursive", "fixed"]) -> list[str]: ...
```

### 7. FastAPI app factory (`main.py`)

```python
def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(title="Superbrain", lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)  # injects request_id into logs
    app.include_router(health_router)        # GET /health → {"status": "ok"}

    return app
```

- Use `@asynccontextmanager` lifespan for startup/shutdown
- DB engine created on startup, disposed on shutdown
- Settings loaded once via `lru_cache`

### 8. Database setup (`infrastructure/db/`)

- Async SQLAlchemy engine using `asyncpg`
- `get_session()` async context manager for dependency injection
- Alembic configured for async migrations
- Initial migration: enable `pgvector` extension, create placeholder tables (flesh out in Block 4)
- `docker-compose.yml` must include a PostgreSQL 16 service with `pgvector` image (`pgvector/pgvector:pg16`)

### 9. `GET /health` route

Returns:
```json
{
  "status": "ok",
  "db": "connected" | "error",
  "ollama": "connected" | "error"
}
```

Checks DB connectivity and Ollama `/api/tags` reachability. This is your smoke test.

---

## Dependencies to install

```toml
[project]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "alembic>=1.13",
    "structlog>=24.0",
    "httpx>=0.27",
    "python-ulid>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-httpx>=0.30",
]
```

---

## Definition of done

- [ ] `docker compose up` starts PostgreSQL with pgvector
- [ ] `alembic upgrade head` runs without error
- [ ] `uvicorn superbrain.main:app` starts without error
- [ ] `GET /health` returns `{"status": "ok", "db": "connected", "ollama": "connected"}`
- [ ] A log line is emitted for every request containing `request_id`
- [ ] All abstract base classes are importable from their modules
- [ ] Settings load correctly from `.env`
- [ ] `pytest` runs (even with 0 tests) without import errors

# System Diagram

```mermaid
flowchart TD
    User[User] --> Telegram[Telegram Bot]
    User --> API[FastAPI API]

    Telegram --> App[Application Services]
    API --> App

    App --> Domain[Domain Models + Contracts]
    App --> Repos[Repository Interfaces]
    App --> Ports[Provider Interfaces]

    Repos --> Infra[(Infrastructure Adapters)]
    Ports --> Infra

    Infra --> Postgres[(PostgreSQL + pgvector)]
    Infra --> LocalLLM[Local Model Runtime\n(Ollama/LM Studio)]

    Scheduler[Scheduler] --> App
    App --> QA[Grounded QA Use Case]
    QA --> Retrieval[Hybrid Retrieval]
    Retrieval --> Postgres
    App --> Topics[Topic Service + Classifier]
    Topics --> Postgres
    App --> Digest[Daily Digest Use Case]
    Digest --> Postgres
    Scheduler --> Digest
```

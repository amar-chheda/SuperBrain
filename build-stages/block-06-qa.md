# Superbrain — Block 6: Retrieval and Grounded QA

## Context

Superbrain is a local-only agentic AI system. This is Block 6. It depends on Block 4 being complete — chunks and embeddings must exist in the database. Block 5 (classification) is not strictly required for QA, but topic metadata enriches the evidence set and should be included if available.

This block implements hybrid retrieval (vector + BM25 fused via Reciprocal Rank Fusion) and the grounded QA pipeline that answers questions using only retrieved evidence. It replaces the stub `POST /qa/ask` route created in Block 2.

**This is the most important demo block for your conference.** Two teaching moments:
1. The hybrid retrieval fusion — showing why combining vector similarity and keyword search outperforms either alone with local models.
2. The low-evidence abort — the system *refuses* to answer when it doesn't have enough grounded evidence, rather than hallucinating. This is a deliberate design decision that local LLMs make necessary and explicit.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    ├── application/
    │   ├── retrieval/
    │   │   ├── __init__.py
    │   │   ├── vector_retriever.py    # pgvector similarity search
    │   │   ├── bm25_retriever.py      # keyword/lexical search
    │   │   └── fusion.py             # Reciprocal Rank Fusion
    │   └── qa/
    │       ├── __init__.py
    │       ├── use_case.py            # orchestrates retrieval + answer generation
    │       ├── evidence_builder.py    # assembles evidence set from chunks
    │       └── answer_generator.py    # grounded answer prompt + parser
    └── infrastructure/
        └── db/
            └── repositories/
                └── chunk_retrieval_repo.py
```

### 2. Alembic migration — query logs

```sql
CREATE TABLE query_logs (
    id                  UUID PRIMARY KEY,
    question            TEXT NOT NULL,
    answer              TEXT,
    evidence_chunk_ids  UUID[] NOT NULL DEFAULT '{}',
    retrieval_latency_ms INTEGER,
    answer_latency_ms   INTEGER,
    aborted             BOOLEAN NOT NULL DEFAULT false,
    abort_reason        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 3. Vector retriever (`application/retrieval/vector_retriever.py`)

Queries pgvector using cosine similarity against chunk embeddings.

```python
class VectorRetriever:
    def __init__(self, embedder: EmbeddingPort, chunk_repo: ChunkRetrievalRepository):
        self.embedder = embedder
        self.chunk_repo = chunk_repo

    async def retrieve(self, query: str, top_k: int = 20) -> list[RankedChunk]:
        """
        Embed the query, then find the top_k most similar chunks by cosine similarity.
        Returns chunks with their similarity score.
        """
        [query_embedding] = await self.embedder.embed([query])

        results = await self.chunk_repo.find_by_vector(
            embedding=query_embedding,
            top_k=top_k,
        )
        return results
```

In `ChunkRetrievalRepository`, the vector search query is:

```sql
SELECT
    c.id,
    c.article_id,
    c.content,
    c.chunk_index,
    a.title,
    a.url,
    a.published_at,
    1 - (c.embedding <=> $1::vector) AS similarity_score
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE a.status = 'succeeded'
ORDER BY c.embedding <=> $1::vector
LIMIT $2;
```

### 4. BM25 lexical retriever (`application/retrieval/bm25_retriever.py`)

BM25 is keyword-based retrieval. It finds chunks that literally contain the query terms — complementary to vector search, which finds semantic matches.

Use `rank_bm25` library for in-memory BM25, OR use PostgreSQL full-text search. **Prefer PostgreSQL FTS** to keep everything in one database:

```sql
-- Add a tsvector column to chunks for FTS (add to migration)
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX idx_chunks_fts ON chunks USING gin(content_tsv);
```

```python
class BM25Retriever:
    async def retrieve(self, query: str, top_k: int = 20) -> list[RankedChunk]:
        """
        Full-text keyword search using PostgreSQL tsvector.
        """
        results = await self.chunk_repo.find_by_text(
            query=query,
            top_k=top_k,
        )
        return results
```

```sql
-- BM25/FTS query
SELECT
    c.id,
    c.article_id,
    c.content,
    c.chunk_index,
    a.title,
    a.url,
    a.published_at,
    ts_rank_cd(c.content_tsv, plainto_tsquery('english', $1)) AS similarity_score
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE c.content_tsv @@ plainto_tsquery('english', $1)
  AND a.status = 'succeeded'
ORDER BY similarity_score DESC
LIMIT $2;
```

### 5. Reciprocal Rank Fusion (`application/retrieval/fusion.py`)

RRF merges two ranked lists into one. The insight: a chunk that ranks 5th in vector search AND 8th in BM25 is more trustworthy than one that ranks 1st in only one list.

```python
def reciprocal_rank_fusion(
    vector_results: list[RankedChunk],
    bm25_results: list[RankedChunk],
    k: int = 60,
    top_n: int = 10,
) -> list[RankedChunk]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score for a chunk = sum over each list of: 1 / (k + rank)
    where rank is 1-indexed position in that list.
    k=60 is the standard constant — higher k reduces the impact of top ranks.

    Chunks that appear in both lists get scores from both, so they
    naturally rise to the top of the fused list.
    """
    scores: dict[UUID, float] = defaultdict(float)
    chunk_map: dict[UUID, RankedChunk] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        scores[chunk.id] += 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk.id] += 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [
        dataclasses.replace(chunk_map[cid], rrf_score=scores[cid])
        for cid in sorted_ids[:top_n]
    ]
```

### 6. Evidence builder (`application/qa/evidence_builder.py`)

```python
@dataclass
class Evidence:
    chunk_id: UUID
    article_id: UUID
    article_title: str | None
    article_url: str
    content: str
    rrf_score: float

MIN_EVIDENCE_CHUNKS = 2      # abort if fewer than this many chunks retrieved
MIN_EVIDENCE_SCORE = 0.005   # abort if top chunk RRF score is below this threshold

def build_evidence_set(fused_chunks: list[RankedChunk]) -> list[Evidence]:
    return [
        Evidence(
            chunk_id=c.id,
            article_id=c.article_id,
            article_title=c.title,
            article_url=c.url,
            content=c.content,
            rrf_score=c.rrf_score,
        )
        for c in fused_chunks
    ]

def check_evidence_sufficiency(evidence: list[Evidence]) -> tuple[bool, str]:
    """
    Decide if there is enough evidence to answer.
    Returns (is_sufficient, reason_if_not).

    This is a design decision, not a technical limitation.
    A well-designed local system refuses to answer rather than hallucinate.
    """
    if len(evidence) < MIN_EVIDENCE_CHUNKS:
        return False, f"Only {len(evidence)} chunks retrieved (minimum: {MIN_EVIDENCE_CHUNKS})"
    if evidence[0].rrf_score < MIN_EVIDENCE_SCORE:
        return False, f"Top chunk score {evidence[0].rrf_score:.4f} below threshold {MIN_EVIDENCE_SCORE}"
    return True, ""
```

### 7. Answer generator (`application/qa/answer_generator.py`)

Uses **Llama 3.1 8B** (`settings.ollama_qa_model`). The prompt enforces groundedness — the model is explicitly forbidden from using any knowledge outside the provided evidence.

```python
GROUNDED_QA_PROMPT = """You are a question-answering assistant. Your job is to answer the question below using ONLY the evidence provided. You are not allowed to use any knowledge outside of the provided evidence.

QUESTION:
{question}

EVIDENCE:
{evidence_block}

RULES:
- Answer using ONLY information present in the evidence above
- If the evidence does not contain enough information to answer the question, say exactly: "I cannot answer this question based on the available evidence."
- Do not speculate, infer, or use outside knowledge
- Keep your answer concise — 2 to 5 sentences unless the question requires more detail
- After your answer, list the source IDs you used, in this exact format:
  SOURCES: chunk_id_1, chunk_id_2

You must always end your response with a SOURCES line, even if you could not answer.

Begin your answer now:"""


def format_evidence_block(evidence: list[Evidence]) -> str:
    """
    Format evidence chunks for the prompt.
    Each chunk gets a clear ID and source label so the model can reference them.
    """
    lines = []
    for i, e in enumerate(evidence, start=1):
        lines.append(f"[CHUNK {e.chunk_id}]")
        lines.append(f"Source: {e.article_title or e.article_url}")
        lines.append(e.content)
        lines.append("")  # blank line between chunks
    return "\n".join(lines)


async def generate_answer(
    llm: LLMPort,
    model: str,
    question: str,
    evidence: list[Evidence],
) -> tuple[str, list[UUID]]:
    """
    Generate a grounded answer and extract cited chunk IDs.
    Returns (answer_text, list_of_cited_chunk_ids).
    """
    prompt = GROUNDED_QA_PROMPT.format(
        question=question,
        evidence_block=format_evidence_block(evidence),
    )

    raw = await llm.complete(prompt, model=model, prompt_template="grounded_qa_v1")
    answer, cited_ids = parse_answer_response(raw, evidence)
    return answer, cited_ids


def parse_answer_response(
    raw: str, evidence: list[Evidence]
) -> tuple[str, list[UUID]]:
    """
    Split the model's response into answer text and cited source IDs.
    Must be defensive about the SOURCES line format.
    """
    valid_ids = {str(e.chunk_id) for e in evidence}

    # Split on "SOURCES:" line
    parts = re.split(r"\nSOURCES:\s*", raw, maxsplit=1)
    answer_text = parts[0].strip()

    if len(parts) < 2:
        log.warning("qa.missing_sources_line", raw=raw[:200])
        return answer_text, []

    sources_raw = parts[1].strip()
    cited_ids = []
    for id_str in re.split(r"[,\s]+", sources_raw):
        id_str = id_str.strip()
        if id_str in valid_ids:
            cited_ids.append(UUID(id_str))
        elif id_str:
            log.warning("qa.invalid_source_id", id=id_str)  # hallucinated ID

    return answer_text, cited_ids
```

### 8. QA use case (`application/qa/use_case.py`)

```python
class AskQuestionUseCase:
    async def execute(self, question: str) -> QAResult:
        t_start = time.monotonic()

        # 1. Retrieve from both sources
        vector_chunks = await self.vector_retriever.retrieve(question, top_k=20)
        bm25_chunks = await self.bm25_retriever.retrieve(question, top_k=20)
        retrieval_ms = int((time.monotonic() - t_start) * 1000)

        # 2. Fuse
        fused = reciprocal_rank_fusion(vector_chunks, bm25_chunks, top_n=10)

        # 3. Build evidence set
        evidence = build_evidence_set(fused)

        # 4. Check sufficiency — abort if insufficient
        is_sufficient, abort_reason = check_evidence_sufficiency(evidence)
        if not is_sufficient:
            log.info("qa.aborted", reason=abort_reason, question=question[:100])
            self.metrics.increment("qa_low_evidence_total")

            await self.query_log_repo.save(QueryLog(
                id=uuid4(),
                question=question,
                answer=None,
                evidence_chunk_ids=[],
                retrieval_latency_ms=retrieval_ms,
                answer_latency_ms=0,
                aborted=True,
                abort_reason=abort_reason,
                created_at=datetime.utcnow(),
            ))

            return QAResult(
                answer=None,
                citations=[],
                aborted=True,
                abort_reason=abort_reason,
            )

        # 5. Generate answer
        t_answer = time.monotonic()
        answer_text, cited_ids = await generate_answer(
            self.llm,
            model=self.settings.ollama_qa_model,
            question=question,
            evidence=evidence,
        )
        answer_ms = int((time.monotonic() - t_answer) * 1000)

        # 6. Build citations (only for cited chunks)
        cited_evidence = [e for e in evidence if e.chunk_id in set(cited_ids)]
        citations = [
            Citation(
                chunk_id=e.chunk_id,
                article_title=e.article_title,
                article_url=e.article_url,
                excerpt=e.content[:200] + "..." if len(e.content) > 200 else e.content,
            )
            for e in cited_evidence
        ]

        # 7. Persist query log
        await self.query_log_repo.save(QueryLog(
            id=uuid4(),
            question=question,
            answer=answer_text,
            evidence_chunk_ids=[e.chunk_id for e in evidence],
            retrieval_latency_ms=retrieval_ms,
            answer_latency_ms=answer_ms,
            aborted=False,
            created_at=datetime.utcnow(),
        ))

        return QAResult(
            answer=answer_text,
            citations=citations,
            aborted=False,
            retrieval_latency_ms=retrieval_ms,
            answer_latency_ms=answer_ms,
        )
```

### 9. Wire up the QA route (`api/qa.py` — replaces Block 2 stub)

```
POST /qa/ask
```

Request:
```json
{ "question": "What is Reciprocal Rank Fusion?" }
```

Response (when answer found):
```json
{
  "answer": "Reciprocal Rank Fusion is...",
  "aborted": false,
  "citations": [
    {
      "chunk_id": "uuid",
      "article_title": "How RAG Works",
      "article_url": "https://...",
      "excerpt": "Reciprocal Rank Fusion merges two ranked..."
    }
  ],
  "retrieval_latency_ms": 45,
  "answer_latency_ms": 2300
}
```

Response (when aborted):
```json
{
  "answer": null,
  "aborted": true,
  "abort_reason": "Only 1 chunk retrieved (minimum: 2)",
  "citations": []
}
```

---

## Dependencies to add

```toml
"rank-bm25>=0.2",   # only needed if not using PostgreSQL FTS
```

---

## Definition of done

- [ ] `POST /qa/ask` with a question about an ingested article returns a grounded answer with citations
- [ ] `POST /qa/ask` with a question about something not in the knowledge base returns `aborted=true` with a clear reason
- [ ] The `SOURCES:` line in the LLM response is correctly parsed and matched to real chunk IDs
- [ ] Hallucinated source IDs (chunk IDs not in the evidence set) are logged as warnings and excluded from citations
- [ ] A `QueryLog` row is persisted for every question — both answered and aborted
- [ ] `ModelCallLog` records exist for the embedding call (query embedding) and the LLM call (answer generation)
- [ ] `retrieval_latency_ms` and `answer_latency_ms` are populated on every response
- [ ] RRF fusion correctly scores chunks that appear in both vector and BM25 results higher than chunks appearing in only one
- [ ] Asking the same question twice returns consistent answers (determinism check — set Ollama temperature=0)
- [ ] QA metrics recorded: retrieval latency, answer latency, low-evidence counter

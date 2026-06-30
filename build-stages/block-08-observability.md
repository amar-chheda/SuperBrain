# Superbrain — Block 8: Observability and Eval Harness

## Context

Superbrain is a local-only agentic AI system. This is Block 8 — the final block. It depends on all previous blocks being complete. Every pipeline (ingestion, classification, QA, digest) must exist before this block can close the observability loop across them.

This block is not just "adding logging" — it's about making the system provably trustworthy. You should be able to trace any failure to its exact source, measure whether retrieval quality is degrading, and demonstrate groundedness of QA answers. For your conference demo, this block gives you the "here's what the system is actually doing" moment that makes local AI systems credible.

**Conference teaching note:** This block is the argument for why local systems are better for learning than cloud APIs. Every call is logged. Every decision is observable. Nothing is hidden behind a vendor's dashboard. You can show the audience a single ingestion job traced end-to-end through every layer — crawler, chunker, embedder, classifier — with timing and status for each step.

---

## What to build

### 1. File structure additions

```
src/superbrain/
└── app/
    ├── application/
    │   ├── metrics.py               # MetricsRecorder abstraction + InMemoryRecorder
    │   └── evals/
    │       ├── __init__.py
    │       ├── harness.py           # eval runner + result types
    │       ├── retrieval_evals.py   # retrieval quality checks
    │       ├── citation_evals.py    # citation presence and groundedness
    │       └── fixtures/
    │           ├── retrieval_cases.py
    │           └── qa_cases.py
    └── api/
        └── observability.py         # GET /observe/* routes
```

### 2. MetricsRecorder (`application/metrics.py`)

A simple abstraction that can be replaced with a real metrics backend later. For now, in-memory only.

```python
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol


class MetricsRecorder(Protocol):
    def increment(self, name: str, value: int = 1) -> None: ...
    def observe(self, name: str, value: float) -> None: ...
    def snapshot(self) -> dict: ...


class InMemoryMetricsRecorder:
    """
    Thread-safe in-memory metrics store.
    Counters accumulate. Observations are stored as lists (for percentile calc later).
    Reset on process restart — acceptable for now.
    """

    def __init__(self):
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        self._observations: dict[str, list[float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._observations:
                self._observations[name] = []
            self._observations[name].append(value)

    def snapshot(self) -> dict:
        with self._lock:
            obs_summary = {}
            for name, values in self._observations.items():
                if values:
                    sorted_v = sorted(values)
                    obs_summary[name] = {
                        "count": len(values),
                        "mean": round(sum(values) / len(values), 2),
                        "p50": round(sorted_v[len(sorted_v) // 2], 2),
                        "p95": round(sorted_v[int(len(sorted_v) * 0.95)], 2),
                        "p99": round(sorted_v[int(len(sorted_v) * 0.99)], 2),
                    }
            return {
                "counters": dict(self._counters),
                "observations": obs_summary,
            }
```

**Ensure every pipeline uses `MetricsRecorder`:**

| Pipeline | Counter | Observation |
|---|---|---|
| Ingestion | `ingestion_success_total`, `ingestion_failure_total`, `ingestion_dedup_total` | `crawl_latency_ms`, `chunk_decision_latency_ms`, `embedding_latency_ms` |
| Classification | `classification_success_total`, `classification_failure_total` | `classification_latency_ms`, `topic_match_count` |
| QA | `qa_success_total`, `qa_aborted_total`, `qa_low_evidence_total` | `retrieval_latency_ms`, `answer_latency_ms` |
| Digest | `digest_success_total`, `digest_failure_total`, `digest_empty_total` | `digest_section_count`, `digest_summary_latency_ms` |

Go back through Blocks 4–7 and verify every metric point is actually wired up. Add any that are missing.

### 3. Eval harness types (`application/evals/harness.py`)

```python
@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float           # 0.0 to 1.0
    details: str
    duration_ms: int


@dataclass
class RetrievalEvalCase:
    case_id: str
    question: str
    expected_chunk_ids: list[UUID]    # chunk IDs that MUST appear in top-k results
    expected_article_urls: list[str]  # article URLs that should be retrieved
    top_k: int = 10


@dataclass
class QAEvalCase:
    case_id: str
    question: str
    expected_keywords: list[str]      # words that must appear in the answer
    must_cite_urls: list[str]         # article URLs that must be cited
    must_not_hallucinate: bool = True # if True, check no invented source IDs


async def run_retrieval_eval(
    case: RetrievalEvalCase,
    vector_retriever,
    bm25_retriever,
) -> EvalResult:
    """Run a single retrieval eval case. Returns pass/fail with score."""
    ...


async def run_qa_eval(
    case: QAEvalCase,
    qa_use_case,
) -> EvalResult:
    """Run a single QA eval case. Returns pass/fail with score."""
    ...


async def run_all_evals(
    retrieval_cases: list[RetrievalEvalCase],
    qa_cases: list[QAEvalCase],
    vector_retriever,
    bm25_retriever,
    qa_use_case,
) -> list[EvalResult]:
    results = []
    for case in retrieval_cases:
        result = await run_retrieval_eval(case, vector_retriever, bm25_retriever)
        results.append(result)
    for case in qa_cases:
        result = await run_qa_eval(case, qa_use_case)
        results.append(result)
    return results
```

### 4. Retrieval evals (`application/evals/retrieval_evals.py`)

```python
def check_recall_at_k(
    retrieved_chunks: list[RankedChunk],
    expected_chunk_ids: list[UUID],
    k: int,
) -> float:
    """
    Recall@k: what fraction of expected chunks appear in the top-k results?
    Score = (number of expected chunks found in top-k) / (total expected chunks)
    """
    top_k_ids = {c.id for c in retrieved_chunks[:k]}
    found = sum(1 for eid in expected_chunk_ids if eid in top_k_ids)
    return found / len(expected_chunk_ids) if expected_chunk_ids else 0.0


def check_url_coverage(
    retrieved_chunks: list[RankedChunk],
    expected_urls: list[str],
) -> float:
    """
    What fraction of expected article URLs appear in retrieved results?
    """
    retrieved_urls = {c.url for c in retrieved_chunks}
    found = sum(1 for url in expected_urls if url in retrieved_urls)
    return found / len(expected_urls) if expected_urls else 0.0
```

### 5. Citation and groundedness evals (`application/evals/citation_evals.py`)

```python
def check_citation_presence(
    answer: str,
    citations: list[Citation],
    must_cite_urls: list[str],
) -> EvalResult:
    """
    Verify that required URLs are actually cited in the answer.
    Passes if all must_cite_urls appear in the citation list.
    """
    cited_urls = {c.article_url for c in citations}
    missing = [url for url in must_cite_urls if url not in cited_urls]

    passed = len(missing) == 0
    score = 1.0 - (len(missing) / max(len(must_cite_urls), 1))

    return EvalResult(
        name="citation_presence",
        passed=passed,
        score=score,
        details=f"Missing citations: {missing}" if missing else "All required citations present",
        duration_ms=0,
    )


def check_groundedness(
    answer: str,
    citations: list[Citation],
    evidence: list[Evidence],
) -> EvalResult:
    """
    Groundedness check: verify that all cited chunk IDs exist in the evidence set.
    An uncited chunk ID in the answer = hallucinated source.

    This is a structural check, not a semantic one. Semantic groundedness
    (did the model actually use the content?) requires human evaluation.
    """
    evidence_ids = {str(e.chunk_id) for e in evidence}
    citation_ids = {str(c.chunk_id) for c in citations}

    hallucinated = citation_ids - evidence_ids
    passed = len(hallucinated) == 0
    score = 1.0 - (len(hallucinated) / max(len(citation_ids), 1))

    return EvalResult(
        name="groundedness",
        passed=passed,
        score=score,
        details=f"Hallucinated IDs: {hallucinated}" if hallucinated else "No hallucinated sources",
        duration_ms=0,
    )


def check_answer_keywords(
    answer: str,
    expected_keywords: list[str],
) -> EvalResult:
    """
    Simple keyword presence check. Not a substitute for semantic evaluation,
    but useful for smoke testing that the answer is on-topic.
    """
    answer_lower = answer.lower()
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    passed = len(missing) == 0
    score = 1.0 - (len(missing) / max(len(expected_keywords), 1))

    return EvalResult(
        name="keyword_presence",
        passed=passed,
        score=score,
        details=f"Missing keywords: {missing}" if missing else "All keywords present",
        duration_ms=0,
    )
```

### 6. Observability API routes (`api/observability.py`)

These routes are for the conference demo and for operational debugging. Not production-hardened, but fully functional.

```
GET /observe/metrics
```
Returns the current `MetricsRecorder.snapshot()` — all counters and percentile summaries.

```
GET /observe/model-calls?limit=50&request_type=qa
```
Returns recent `ModelCallLog` records, filterable by `request_type` and `status`. Useful for seeing how long LLM calls are taking.

```
GET /observe/jobs/{job_id}/trace
```
The most important demo endpoint. Returns a full trace of a single ingestion job:
```json
{
  "job": { "id": "...", "status": "succeeded", "input_value": "https://..." },
  "article": { "id": "...", "title": "...", "chunk_count": 8, "strategy": "semantic" },
  "model_calls": [
    {
      "request_type": "chunking_strategy",
      "model_name": "llama3.1:8b",
      "duration_ms": 1240,
      "status": "success",
      "prompt_template": "chunking_strategy_v1"
    },
    {
      "request_type": "embedding",
      "model_name": "nomic-embed-text",
      "duration_ms": 340,
      "status": "success"
    },
    {
      "request_type": "classification",
      "model_name": "phi3:mini",
      "duration_ms": 890,
      "status": "success"
    }
  ],
  "topic_matches": [
    { "topic_name": "Machine Learning", "confidence": "high" }
  ],
  "total_duration_ms": 2470
}
```

```
GET /observe/evals/run
```
Runs the full eval suite and returns results. Expensive — runs LLM calls. For demo and CI use only, not for production traffic.

```
GET /observe/query-logs?limit=20
```
Returns recent `QueryLog` records — question, answer, retrieval/answer latency, whether aborted.

### 7. Structured log completeness audit

Go back through every pipeline and verify these log lines exist with the correct fields:

**Ingestion:**
```
INFO  ingestion.started           {job_id, url, source}
INFO  ingestion.dedup.skipped     {job_id, existing_article_id}
INFO  ingestion.chunk.decided     {job_id, strategy, reason}
INFO  ingestion.embedding.done    {job_id, chunk_count, duration_ms}
INFO  ingestion.succeeded         {job_id, article_id, chunk_count, strategy}
ERROR ingestion.failed            {job_id, error, stage}
```

**QA:**
```
INFO  qa.retrieval.done           {question_preview, vector_count, bm25_count, fused_count, duration_ms}
INFO  qa.aborted                  {reason, question_preview}
INFO  qa.answered                 {cited_count, answer_length, retrieval_ms, answer_ms}
WARN  qa.invalid_source_id        {id}
WARN  qa.missing_sources_line     {raw_preview}
```

**Classification:**
```
INFO  classification.started      {article_id, topic_count}
INFO  classification.done         {article_id, match_count, duration_ms}
WARN  classification.parse_failed {error, raw_preview}
WARN  classification.unknown_topic_id {topic_id}
```

**Digest:**
```
INFO  digest.started              {date, triggered_by}
INFO  digest.empty                {date}
INFO  digest.section.done         {topic_name, article_count, duration_ms}
INFO  digest.succeeded            {date, section_count, article_count}
ERROR digest.failed               {date, error}
```

### 8. Pytest smoke tests

Create `tests/` with at minimum these smoke tests. They should pass without a running database or Ollama (use mocks/stubs):

```python
# tests/test_chunking_agent.py
def test_parse_strategy_response_valid():
    raw = '{"strategy": "semantic", "reason": "flowing prose"}'
    result = parse_strategy_response(raw)
    assert result == "semantic"

def test_parse_strategy_response_with_markdown_fence():
    raw = '```json\n{"strategy": "recursive", "reason": "has headings"}\n```'
    result = parse_strategy_response(raw)
    assert result == "recursive"

def test_parse_strategy_response_invalid_falls_back_to_fixed():
    raw = "I cannot determine the strategy."
    result = parse_strategy_response(raw)
    assert result == "fixed"

def test_parse_strategy_response_unknown_strategy_falls_back():
    raw = '{"strategy": "unknown_value", "reason": "test"}'
    result = parse_strategy_response(raw)
    assert result == "fixed"


# tests/test_classification.py
def test_parse_classification_valid():
    topic = build_test_topic(id=UUID("aaaaaaaa-0000-0000-0000-000000000001"))
    raw = f'[{{"topic_id": "aaaaaaaa-0000-0000-0000-000000000001", "confidence": "high", "reason": "directly relevant"}}]'
    result = parse_classification_response(raw, [topic])
    assert len(result) == 1
    assert result[0].confidence == "high"

def test_parse_classification_rejects_hallucinated_id():
    topic = build_test_topic(id=UUID("aaaaaaaa-0000-0000-0000-000000000001"))
    raw = '[{"topic_id": "00000000-0000-0000-0000-000000000000", "confidence": "high", "reason": "test"}]'
    result = parse_classification_response(raw, [topic])
    assert len(result) == 0  # hallucinated ID rejected

def test_parse_classification_empty_array():
    raw = "[]"
    result = parse_classification_response(raw, [])
    assert result == []


# tests/test_qa.py
def test_parse_answer_response_valid():
    evidence = [build_test_evidence(chunk_id=UUID("bbbbbbbb-0000-0000-0000-000000000001"))]
    raw = "The answer is X.\nSOURCES: bbbbbbbb-0000-0000-0000-000000000001"
    answer, cited = parse_answer_response(raw, evidence)
    assert answer == "The answer is X."
    assert UUID("bbbbbbbb-0000-0000-0000-000000000001") in cited

def test_parse_answer_response_rejects_hallucinated_source():
    evidence = [build_test_evidence(chunk_id=UUID("bbbbbbbb-0000-0000-0000-000000000001"))]
    raw = "The answer is X.\nSOURCES: 00000000-0000-0000-0000-000000000000"
    answer, cited = parse_answer_response(raw, evidence)
    assert len(cited) == 0  # hallucinated ID not included

def test_rrf_fusion_scores_overlap_higher():
    chunk_a = build_ranked_chunk(id=UUID("cccccccc-0000-0000-0000-000000000001"))
    chunk_b = build_ranked_chunk(id=UUID("dddddddd-0000-0000-0000-000000000001"))
    # chunk_a appears in both lists, chunk_b in only one
    vector_results = [chunk_a, chunk_b]
    bm25_results = [chunk_a]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    assert fused[0].id == chunk_a.id  # chunk_a should rank first

def test_evidence_sufficiency_aborts_below_threshold():
    evidence = [build_test_evidence(rrf_score=0.001)]  # below MIN_EVIDENCE_SCORE
    is_sufficient, reason = check_evidence_sufficiency(evidence)
    assert not is_sufficient
    assert "below threshold" in reason


# tests/test_url_utils.py
def test_canonicalise_removes_utm():
    url = "https://example.com/article?utm_source=twitter&utm_medium=social"
    assert canonicalise_url(url) == "https://example.com/article"

def test_canonicalise_lowercases_host():
    url = "https://Example.COM/path"
    assert canonicalise_url(url) == "https://example.com/path"

def test_canonicalise_removes_fragment():
    url = "https://example.com/article#section-2"
    assert canonicalise_url(url) == "https://example.com/article"
```

---

## Definition of done

- [ ] `GET /observe/metrics` returns a JSON snapshot with counters and percentile summaries populated from real pipeline runs
- [ ] `GET /observe/jobs/{job_id}/trace` returns a full trace of a real ingestion job including all model call steps
- [ ] `GET /observe/model-calls?request_type=qa` returns only QA model call logs
- [ ] `GET /observe/evals/run` runs the eval suite and returns pass/fail for all cases
- [ ] All smoke tests in `tests/` pass with `pytest`
- [ ] `check_groundedness` correctly identifies hallucinated chunk IDs and marks the eval as failed
- [ ] `check_citation_presence` fails when a required URL is missing from citations
- [ ] All structured log lines listed in section 7 are present and include the required fields
- [ ] `InMemoryMetricsRecorder.snapshot()` returns correct p50/p95/p99 values
- [ ] The full end-to-end demo path works: ingest a URL via Telegram → wait for ingestion → ask a question → receive a cited answer → view the trace at `/observe/jobs/{id}/trace`

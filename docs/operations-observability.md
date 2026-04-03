# Operations: Observability and Eval Hooks

## What gets logged

### Structured application logs
- JSON log lines include timestamp, level, logger, message, request ID, and job ID.
- Ingestion/QA/digest workflows emit stage-level completion/failure logs.

### Model call logs (persisted)
- Stored in `model_call_logs`.
- Fields include:
  - `provider`
  - `model_name`
  - `request_type`
  - `prompt_template`
  - `started_at` / `finished_at`
  - `duration_ms`
  - `status`
  - `retries`
  - `error_metadata`
  - `related_entity_id`

### Scheduler execution logs (persisted)
- `scheduled_jobs` tracks registered cron/manual jobs.
- `scheduled_job_runs` stores execution status, timing, and error metadata.

## What gets measured

Using `MetricsRecorder` abstraction (`InMemoryMetricsRecorder` currently):
- ingestion:
  - success/failure counters
  - duplicate counter
  - dedup/extraction/chunking/embedding latencies
- QA:
  - retrieval latency
  - answer latency
  - low-evidence counter
- classification:
  - match count observations
- digest:
  - success/failure/empty counters
  - section count observations

## Where to look during failures

1. API-level error payloads for mapped exception code/status.
2. Structured app logs for stage breakdown and correlation IDs.
3. `ingestion_jobs`, `query_logs`, and `digest_runs` tables for workflow status.
4. `model_call_logs` for provider-level call traces and failures.

## Error taxonomy

- `ValidationError` -> HTTP 400
- `NotFoundError` -> HTTP 404
- `ConflictError` -> HTTP 409
- fallback uncaught errors -> HTTP 500

## Evaluation hooks

Lightweight harness is in `src/superbrain/app/evals/harness.py`:
- `RetrievalEvalCase`
- `run_retrieval_eval_stub(...)`
- `check_citation_presence(...)`
- `check_groundedness(...)`

### How to add future evals

1. Add a new eval case dataclass for the task type.
2. Add a pure function returning `EvalResult`.
3. Keep model/provider calls behind interfaces so test doubles are easy.
4. Add smoke tests first, then task-specific benchmark fixtures.

## Future extension

- Replace local-only metrics deployment with centralized remote telemetry backend.
- Add tracing spans around repository and provider boundaries.
- Integrate scheduled eval batches and persistent eval result history.

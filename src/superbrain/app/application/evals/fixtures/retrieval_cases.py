"""Retrieval eval fixtures — populate after ingesting real articles."""

from superbrain.app.application.evals.harness import RetrievalEvalCase

# Populated at demo time after articles are ingested.
# Each case must reference chunk IDs and URLs that actually exist in the DB.
RETRIEVAL_CASES: list[RetrievalEvalCase] = []

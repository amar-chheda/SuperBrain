"""QA eval fixtures — populate after ingesting real articles."""

from superbrain.app.application.evals.harness import QAEvalCase

# Populated at demo time after articles are ingested.
# Each case must reference keywords and URLs that actually exist in the KB.
QA_CASES: list[QAEvalCase] = []

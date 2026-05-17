"""Generic metrics abstraction for the Superbrain application layer.

InMemoryMetricsRecorder is the only implementation for now — process-local,
resets on restart. Sufficient for conference demo and development.
"""

from threading import Lock
from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsRecorder(Protocol):
    def increment(self, name: str, value: int = 1) -> None: ...
    def observe(self, name: str, value: float) -> None: ...
    def snapshot(self) -> dict: ...


class InMemoryMetricsRecorder:
    """Thread-safe in-memory metrics store.

    Counters accumulate. Observations are stored as sorted lists for
    percentile calculation. Both reset on process restart.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        self._observations: dict[str, list[float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._observations.setdefault(name, []).append(value)

    def snapshot(self) -> dict:
        with self._lock:
            obs_summary: dict[str, object] = {}
            for name, values in self._observations.items():
                if values:
                    sv = sorted(values)
                    n = len(sv)
                    obs_summary[name] = {
                        "count": n,
                        "mean": round(sum(sv) / n, 2),
                        "p50": round(sv[n // 2], 2),
                        "p95": round(sv[int(n * 0.95)], 2),
                        "p99": round(sv[int(n * 0.99)], 2),
                    }
            return {
                "counters": dict(self._counters),
                "observations": obs_summary,
            }

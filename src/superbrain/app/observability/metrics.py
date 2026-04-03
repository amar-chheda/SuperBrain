"""Metrics abstraction and in-memory recorder."""

from collections import defaultdict

from prometheus_client import Counter, Histogram, generate_latest


class MetricsRecorder:
    """Abstraction for counters and latency observations."""

    def increment(self, name: str, value: int = 1) -> None:
        """Increment named counter metric."""

    def observe(self, name: str, value: float) -> None:
        """Record observed metric value."""


class InMemoryMetricsRecorder(MetricsRecorder):
    """Simple in-memory metrics sink for local development and tests."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.observations: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.observations[name].append(value)


class PrometheusMetricsRecorder(MetricsRecorder):
    """Prometheus-backed metrics recorder."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def increment(self, name: str, value: int = 1) -> None:
        counter = self._counters.get(name)
        if counter is None:
            counter = Counter(_sanitize(name), f"Counter for {name}")
            self._counters[name] = counter
        counter.inc(value)

    def observe(self, name: str, value: float) -> None:
        histogram = self._histograms.get(name)
        if histogram is None:
            histogram = Histogram(_sanitize(name), f"Histogram for {name}")
            self._histograms[name] = histogram
        histogram.observe(value)

    def render(self) -> bytes:
        """Return Prometheus exposition payload."""

        return generate_latest()


def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")

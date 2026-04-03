"""Timing helper utilities."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class TimerResult:
    """Mutable holder for measured elapsed milliseconds."""

    elapsed_ms: float = 0.0


@contextmanager
def timed() -> Iterator[TimerResult]:
    """Measure wall-clock execution time for a block in milliseconds."""

    result = TimerResult()
    started_at = perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (perf_counter() - started_at) * 1000

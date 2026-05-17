"""Shared eval result type — kept separate to avoid circular imports."""

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    details: str
    duration_ms: int

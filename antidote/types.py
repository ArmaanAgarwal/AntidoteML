"""Shared types. Frozen by the contract in ANTIDOTEML_PLAN.md. Owner: Person 4."""

from dataclasses import dataclass


@dataclass
class Score:
    worker_id: int
    norm_ratio: float | None  # norm / median norm this round
    cosine: float | None  # cosine similarity to the coordinate-wise median update
    anomaly: float  # >= 0, higher = more suspicious
    reputation: float  # 0..1, starts at 1
    status: str  # "trusted" | "suspect" | "ejected"
    reason: str  # "" when trusted

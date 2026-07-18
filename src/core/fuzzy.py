"""Triangular fuzzy number and membership function.

Ported from legacy/benchmark-fsspb/algorithm/fuzzy.py (task 0.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TriangularFuzzyNumber:
    """A triangular fuzzy number T_tilde = (T_L, T_M, T_R)."""

    T_L: float
    T_M: float
    T_R: float

    def __post_init__(self) -> None:
        if not (self.T_L <= self.T_M <= self.T_R):
            raise ValueError(
                f"TFN must satisfy T_L <= T_M <= T_R, got "
                f"({self.T_L}, {self.T_M}, {self.T_R})"
            )

    @property
    def delta_L(self) -> float:
        return self.T_M - self.T_L

    @property
    def delta_R(self) -> float:
        return self.T_R - self.T_M

    @property
    def spread(self) -> float:
        return self.T_R - self.T_L

    def mu(self, x: float) -> float:
        """Return mu_T(x) in [0, 1]."""
        if x < self.T_L or x > self.T_R:
            return 0.0
        if x == self.T_M:
            return 1.0
        if x < self.T_M:
            dL = self.delta_L
            return 1.0 if dL == 0 else (x - self.T_L) / dL
        dR = self.delta_R
        return 1.0 if dR == 0 else (self.T_R - x) / dR

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        """Return the closed interval {x : mu(x) >= alpha} for alpha in (0, 1]."""
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
        return (self.T_L + alpha * self.delta_L,
                self.T_R - alpha * self.delta_R)

    def to_dict(self) -> dict:
        return {"T_L": self.T_L, "T_M": self.T_M, "T_R": self.T_R}

    @classmethod
    def from_dict(cls, d: dict) -> "TriangularFuzzyNumber":
        return cls(T_L=d["T_L"], T_M=d["T_M"], T_R=d["T_R"])

    def __repr__(self) -> str:
        return f"TFN(L={self.T_L}, M={self.T_M}, R={self.T_R})"

"""Uniform solver interface (task 0.4).

Every solver subclasses `Solver` and implements `search`, scoring candidates
only through the protocol-owned `BudgetedObjective`. The public entry point
`solve` times the run, absorbs the protocol's stop signal, and returns a
`SolverResult` carrying the subset, evals used, and wall time — and NOTHING
about quality. Reported quality is always recomputed downstream by
`src/core/evaluate.py`. (R2.5, R3.1, R4.3)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective, ProtocolConfig, StopSearch


@dataclass(frozen=True)
class SolverResult:
    """What a solver run yields. Deliberately contains no quality field."""

    subset_indices: Tuple[int, ...]
    evals_used: int
    wall_time: float
    stop_reason: str  # "budget_exhausted" | "plateau" | "solver_done"


class Solver(ABC):
    """Base class for all solvers."""

    name: str = "unnamed"
    # False for wall-clock-limited anytime solvers (e.g. CP-SAT), whose
    # incumbent at the time limit is not a pure function of the seed.
    deterministic: bool = True

    @abstractmethod
    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        """Run the search, scoring candidates via objective.evaluate().

        May return early (exact solvers that finish); the protocol's stop
        signal (StopSearch) is handled by solve(), so implementations may
        simply let it propagate. The result subset is the protocol-tracked
        best-evaluated subset, so there is nothing to return.
        """

    def solve(self, instance: Instance, config: ProtocolConfig,
              rng: np.random.Generator) -> SolverResult:
        objective = BudgetedObjective(instance, config)
        t0 = time.perf_counter()
        try:
            self.search(instance, objective, rng)
            stop_reason = objective.stop_reason or "solver_done"
        except StopSearch as sig:
            stop_reason = sig.reason
        wall = time.perf_counter() - t0
        if objective.best_subset is None:
            # Solver evaluated nothing; the empty subset is the honest answer.
            best = ()
        else:
            best = objective.best_subset
        return SolverResult(
            subset_indices=best,
            evals_used=objective.evals_used,
            wall_time=wall,
            stop_reason=stop_reason,
        )

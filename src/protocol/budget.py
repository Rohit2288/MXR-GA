"""Protocol-owned evaluation budget and termination (task 0.5).

Every solver scores candidates ONLY through `BudgetedObjective.evaluate`.
The objective counts every evaluation, tracks the best subset seen, and
raises a stop signal when the shared budget is exhausted or the shared
plateau criterion fires. Identical tolerance and patience for every solver —
no solver has private termination privileges. (R4.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from src.core.evaluate import evaluate
from src.core.instance import Instance


class StopSearch(Exception):
    """Base signal: the protocol has ended this run."""

    reason: str = "stopped"


class BudgetExhausted(StopSearch):
    reason = "budget_exhausted"


class PlateauReached(StopSearch):
    reason = "plateau"


@dataclass(frozen=True)
class ProtocolConfig:
    """Shared run protocol. One config object is used for ALL solvers in an
    experiment; per-solver overrides are forbidden by construction."""

    max_evals: int
    plateau_tol: float = 1e-9      # improvement in mu below this doesn't reset patience
    plateau_patience: Optional[int] = None  # evals since last improvement; None disables


class BudgetedObjective:
    """Counting, best-tracking, stop-signalling objective for one run."""

    def __init__(self, instance: Instance, config: ProtocolConfig):
        self.instance = instance
        self.config = config
        self.evals_used = 0
        self.best_mu: float = float("-inf")
        self.best_subset: Optional[Tuple[int, ...]] = None
        self._evals_since_improvement = 0
        self._stopped: Optional[str] = None

    @property
    def stop_reason(self) -> Optional[str]:
        return self._stopped

    def evaluate(self, subset_indices: Iterable[int]) -> float:
        """Score a candidate subset; counts one evaluation.

        Raises BudgetExhausted / PlateauReached when the protocol says stop.
        The evaluation that triggers the stop is still counted and tracked.
        """
        if self._stopped is not None:
            # A solver ignored the stop signal; refuse further evaluations.
            raise BudgetExhausted() if self._stopped == BudgetExhausted.reason \
                else PlateauReached()

        ev = evaluate(self.instance, subset_indices)
        self.evals_used += 1

        if ev.mu > self.best_mu + self.config.plateau_tol:
            self._evals_since_improvement = 0
        else:
            self._evals_since_improvement += 1
        if ev.mu > self.best_mu:
            self.best_mu = ev.mu
            self.best_subset = ev.subset_indices

        if self.evals_used >= self.config.max_evals:
            self._stopped = BudgetExhausted.reason
            raise BudgetExhausted()
        if (self.config.plateau_patience is not None
                and self._evals_since_improvement >= self.config.plateau_patience):
            self._stopped = PlateauReached.reason
            raise PlateauReached()
        return ev.mu

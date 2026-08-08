"""Fuzzy Knapsack (FKP): the second domain for the clipped-objective study.

Formulation (task 2.16)
-----------------------
Items carry a profit p_i and a weight w_i. A subset S is feasible when
sum_{i in S} w_i <= C. The objective is the membership degree of the total
profit in a triangular target (P^L, P^M, P^R):

    mu(S) = 0                                   if S is infeasible,
            max(0, 1 - d(profit(S)))            otherwise,

with d the asymmetric weighted distance of Proposition 1. Reported quality
is mu and nothing else, exactly as in FSSP.

Why this formulation
--------------------
It keeps the clamped objective identical to FSSP while changing the
combinatorial core: a genuine capacity constraint now exists. Any plateau
effect that reproduces here is therefore attributable to the clipping, not
to subset-sum structure -- which is the claim the paper needs to support.

Note that FKP has TWO sources of flatness, where FSSP had one: mu vanishes
both outside the profit support and on every infeasible subset. The
unclamped guide must therefore supply gradient in both directions, which is
what `crisp_guide` below does.

This module deliberately mirrors src/core/fuzzy.py + evaluate.py rather than
generalising them: the FSSP scorer is cited in the reviewer response as a
single, auditable object, and widening it to carry a second problem class
would weaken that claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

from src.core.fuzzy import TriangularFuzzyNumber


class InvalidSubsetError(ValueError):
    """Raised when a solver returns an index set that is not a valid subset."""


@dataclass(frozen=True)
class KnapsackInstance:
    instance_id: str
    profits: Tuple[int, ...]
    weights: Tuple[int, ...]
    capacity: int
    target: TriangularFuzzyNumber
    n: int
    correlation: str          # uncorrelated | weakly | strongly (Pisinger-style)
    seed: int
    metadata: dict = field(repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.profits) != self.n or len(self.weights) != self.n:
            raise ValueError("profits/weights length must equal n")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


@dataclass(frozen=True)
class KnapsackEvaluation:
    subset_indices: Tuple[int, ...]
    profit: int
    weight: int
    feasible: bool
    mu: float


def evaluate_knapsack(instance: KnapsackInstance,
                      subset_indices: Iterable[int]) -> KnapsackEvaluation:
    """THE scorer for FKP. Validates indices, recomputes profit and weight
    from the actual items, and returns the membership degree. No solver ever
    reports its own quality."""
    idx = tuple(subset_indices)
    n = instance.n
    if idx:
        if set(map(type, idx)) != {int}:
            for i in idx:
                if type(i) is not int:
                    raise InvalidSubsetError(f"non-integer index {i!r}")
        if len(set(idx)) != len(idx):
            raise InvalidSubsetError("duplicate index")
        if min(idx) < 0 or max(idx) >= n:
            raise InvalidSubsetError(f"index out of range for n={n}")
    profit = sum(map(instance.profits.__getitem__, idx))
    weight = sum(map(instance.weights.__getitem__, idx))
    feasible = weight <= instance.capacity
    mu = instance.target.mu(profit) if feasible else 0.0
    return KnapsackEvaluation(tuple(sorted(idx)), profit, weight, feasible, mu)


def crisp_guide(instance: KnapsackInstance):
    """Unclamped guidance signal for FKP.

    g(S) = 1 - d(profit)  -  lambda * relative capacity overrun

    Inside the profit support and within capacity this equals mu. Outside
    either, where mu is flat at zero, g keeps decreasing, so the search still
    has a direction to follow. The overrun penalty is scaled so that a
    full-capacity violation is comparable in magnitude to a full-tolerance
    profit miss, which keeps the two terms on one scale without tuning.
    """
    t = instance.target
    d_l = t.delta_L if t.delta_L > 0 else 1.0
    d_r = t.delta_R if t.delta_R > 0 else 1.0
    p_m = t.T_M
    cap = instance.capacity

    def guide(profit: float, weight: float, mu: float) -> float:
        d = (p_m - profit) / d_l if profit <= p_m else (profit - p_m) / d_r
        overrun = max(0.0, (weight - cap) / cap)
        return 1.0 - d - overrun

    return guide


def dp_exact_knapsack(instance: KnapsackInstance,
                      max_cells: int = 400_000_000
                      ) -> Optional[KnapsackEvaluation]:
    """Exact maximiser of mu, by DP over profit tracking minimum weight.

    The natural knapsack DP -- over capacity, tracking maximum profit -- is
    invalid here: mu is not monotone in profit, it peaks at the modal target
    P^M, so a higher-profit subset can score worse. Enumerating attainable
    (weight, profit) pairs is valid but costs O(C x P).

    Instead, for every attainable profit p we record the MINIMUM weight
    needed to attain exactly p. A profit is then achievable within capacity
    iff minw[p] <= C, and the optimum is the achievable p maximising mu(p).
    This is O(n x sum(profits)) time and O(sum(profits)) memory, which is
    tractable at the sizes used here.

    Returns None when the table would exceed `max_cells`.
    """
    import numpy as np

    n, cap = instance.n, instance.capacity
    total_profit = int(sum(instance.profits))
    if n * (total_profit + 1) > max_cells:
        return None

    INF = np.iinfo(np.int64).max // 4
    minw = np.full(total_profit + 1, INF, dtype=np.int64)
    minw[0] = 0
    # choice[i, p] is True when item i is taken on the best path to profit p
    # at the time item i was processed; used for reconstruction.
    choice = np.zeros((n, total_profit + 1), dtype=bool)

    reach = 0
    for i in range(n):
        p_i = int(instance.profits[i])
        w_i = int(instance.weights[i])
        if p_i <= 0:
            continue
        hi = min(reach + p_i, total_profit)
        # Descending so each item is used at most once.
        src = minw[hi - p_i::-1] if hi - p_i >= 0 else np.empty(0, np.int64)
        seg = slice(hi, p_i - 1, -1)
        cand = minw[hi - p_i::-1] + w_i
        cur = minw[seg]
        better = cand < cur
        if better.any():
            minw[seg] = np.where(better, cand, cur)
            choice[i, seg] = better
        reach = hi

    feasible = np.nonzero(minw <= cap)[0]
    if feasible.size == 0:
        return evaluate_knapsack(instance, [])
    mus = np.array([instance.target.mu(int(p)) for p in feasible])
    best_p = int(feasible[int(np.argmax(mus))])

    idx, p = [], best_p
    for i in range(n - 1, -1, -1):
        if p >= 0 and choice[i, p]:
            idx.append(i)
            p -= int(instance.profits[i])
    return evaluate_knapsack(instance, sorted(idx))

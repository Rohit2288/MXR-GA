"""Alpha-cut FPTAS, rewritten with subset certification (task 1.7; R2.5, R3.1).

Legacy bug (do not reproduce): the old implementation returned the scaled
reconstructed sum s*K, which corresponds to NO real subset; recomputing mu on
that phantom sum produced the 43% Suite-U overshoot the reviewers flagged.

This rewrite keeps the algorithmic framework — binary search on alpha, an
approximate scaled subset-sum feasibility test per alpha-cut interval — but
every feasibility test now returns an EXPLICIT subset, reconstructed from
per-item prefix reachability bitsets over the scaled sums. Each candidate
subset is scored through the protocol objective on its true unscaled item
sum, so the reported membership can never exceed the true optimum: it is the
membership of a real subset. Overshoot is impossible by construction.

Procedure (formal spec in paper_fragments/alpha_cut_spec.tex):
  1. alpha-grid: binary search on alpha in (0, 1], ceil(log2(1/eps))
     iterations, preceded by a test at alpha = 1 and followed by a fallback
     test at alpha = eps.
  2. Interval: [L_a, R_a] = alpha-cut of the target at level alpha.
  3. Rounding: K = max(1, eps * R_a / n); item i scales to
     max(1, floor(a_i / K)); the window scales to [ceil(L_a/K), floor(R_a/K)].
  4. Scaled DP: prefix reachability bitsets over scaled sums 0..floor(R_a/K);
     the largest reachable scaled sum in the window is selected.
  5. Reconstruction: walk the prefix bitsets backward to an explicit index
     set; evaluate it through the shared scorer (one protocol evaluation per
     alpha probe, ~log2(1/eps)+2 total).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


def _scaled_dp_subset(items, L: float, R: float, eps: float) -> Optional[List[int]]:
    """Explicit subset whose SCALED sum is the largest reachable value in the
    scaled window, or None if the scaled window contains no reachable sum."""
    n = len(items)
    if n == 0 or R <= 0:
        return None
    K = max(1.0, eps * R / n)
    scaled = [max(1, int(math.floor(a / K))) for a in items]
    s_hi = int(math.floor(R / K))
    s_lo = max(0, int(math.ceil(L / K)))
    if s_hi <= 0 or s_lo > s_hi:
        return None

    mask = (1 << (s_hi + 1)) - 1
    prefixes = [1]
    reachable = 1
    for a in scaled:
        reachable |= (reachable << a)
        reachable &= mask
        prefixes.append(reachable)

    window = (reachable >> s_lo) & ((1 << (s_hi - s_lo + 1)) - 1)
    if window == 0:
        return None
    s = s_lo + window.bit_length() - 1  # highest reachable scaled sum in window

    subset: List[int] = []
    for k in range(n, 0, -1):
        if (prefixes[k - 1] >> s) & 1:
            continue
        subset.append(k - 1)
        s -= scaled[k - 1]
    assert s == 0, "scaled reconstruction failed"
    return sorted(subset)


class AlphaCut(Solver):
    name = "alpha_cut"

    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        target = instance.target
        items = instance.items
        eps = self.eps

        def probe(alpha: float) -> Optional[float]:
            """Feasibility test at level alpha; scores the certified subset
            (if any) through the protocol and returns its true mu."""
            L_a, R_a = target.alpha_cut(alpha)
            subset = _scaled_dp_subset(items, L_a, R_a, eps)
            if subset is None:
                return None
            return objective.evaluate(subset)

        if probe(1.0) is not None:
            pass  # best-seen tracking is protocol-owned; nothing to record
        lo, hi = 0.0, 1.0
        n_iters = max(1, int(math.ceil(math.log2(1.0 / eps))))
        for _ in range(n_iters):
            mid = 0.5 * (lo + hi)
            if probe(mid) is not None:
                lo = mid
            else:
                hi = mid
        if lo == 0.0:
            probe(eps)  # legacy fallback probe at the lowest grid level

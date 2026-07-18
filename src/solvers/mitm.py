"""Meet-in-the-middle exact solver (task 1.3; R2.7, R3.3).

Splits the items into two halves, enumerates all subset sums of each half
(numpy doubling; the array index IS the inclusion mask), and for each
left-half sum s_A finds the right-half partners bracketing T_M - s_A by
binary search. Because mu is unimodal in the total sum (increasing below
T_M, decreasing above), those two bracketing partners are the only
candidates for the maximum given s_A, so the sweep is exact.

n ceiling (documented for the paper): each half enumerates 2^(n/2) sums, so
memory and time scale as O(2^(n/2)). The default limit is n <= 44
(2^22 = 4.2M int64 sums per half, ~34 MB); beyond that `feasible()` is
False and the solver returns the evaluated empty subset. In this project's
corpora MITM therefore covers Suite U (n <= 22) and the n <= 22 slices
of the other suites — the exact-for-small/medium-n baseline.

Evaluation accounting: like the DP, one protocol evaluation — the final
reconstructed subset (cost is wall-clock; both are logged, R4.3).
"""

from __future__ import annotations

import numpy as np

from src.core.brute_force import _mu_vectorized, all_subset_sums
from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


class MITM(Solver):
    name = "mitm"

    def __init__(self, max_n: int = 44):
        self.max_n = max_n

    def feasible(self, instance: Instance) -> tuple:
        if instance.n > self.max_n:
            return False, f"n={instance.n} > max_n={self.max_n} (2^(n/2) blowup)"
        return True, ""

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        ok, _reason = self.feasible(instance)
        if not ok:
            objective.evaluate([])
            return

        n = instance.n
        target = instance.target
        h = n // 2
        left_items = instance.items[:h]
        right_items = instance.items[h:]

        sums_l = all_subset_sums(left_items)            # index = mask over left
        sums_r = all_subset_sums(right_items)           # index = mask over right
        order_r = np.argsort(sums_r, kind="stable")
        sorted_r = sums_r[order_r]

        # For each left sum, the two right sums bracketing T_M - s_A.
        want = target.T_M - sums_l.astype(np.float64)
        pos = np.searchsorted(sorted_r, want)           # first >= want
        lo = np.clip(pos - 1, 0, len(sorted_r) - 1)     # last <= want
        hi = np.clip(pos, 0, len(sorted_r) - 1)

        best_mu = -1.0
        best_pair = (0, 0)                              # (left mask, right mask)
        for cand in (lo, hi):
            totals = sums_l + sorted_r[cand]
            mu = _mu_vectorized(totals, target.T_L, target.T_M, target.T_R)
            k = int(np.argmax(mu))
            if float(mu[k]) > best_mu:
                best_mu = float(mu[k])
                best_pair = (k, int(order_r[cand[k]]))

        mask_l, mask_r = best_pair
        subset = [i for i in range(h) if (mask_l >> i) & 1] + \
                 [h + i for i in range(n - h) if (mask_r >> i) & 1]
        objective.evaluate(subset)

"""Exact bitset DP with subset reconstruction (task 1.2; R2.7, R3.3).

Rewrite of legacy generator/reference_solvers/dp_exact.py, which returned only
the best reachable sum and no subset. Here the DP stores per-item prefix
reachability bitsets and reconstructs an EXPLICIT subset achieving the optimal
sum; the harness then scores that subset's true unscaled sum through the
single scorer (architecture rule 1).

Exactness domain: integer items; sums are tracked exactly up to floor(T_R)
(sums above T_R have mu = 0, so truncation loses nothing). The optimum over
the window [ceil(T_L), floor(T_R)] is the global optimum of mu.

Feasibility limits (documented for the paper): memory is n * (T_R + 1) bits
for the prefix bitsets; runs are refused above `max_state_bits` (default 4e9
bits = 500 MB) or when T_R exceeds `max_tr`. Use `feasible(instance)` to
pre-filter in experiments.

Evaluation accounting: an exact solver does not sample candidate subsets, so
it consumes exactly ONE protocol evaluation — the final reconstructed subset.
Its cost is reported as wall-clock time (R4.3: both are logged for every
solver). If the instance is outside the feasibility domain, the solver
evaluates the empty subset and returns (never a fabricated result).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


class DPBitset(Solver):
    name = "dp_exact"

    def __init__(self, max_tr: int = 10_000_000, max_state_bits: int = 4_000_000_000):
        self.max_tr = max_tr
        self.max_state_bits = max_state_bits

    def feasible(self, instance: Instance) -> Tuple[bool, str]:
        t_r = math.floor(instance.target.T_R)
        if t_r > self.max_tr:
            return False, f"T_R={t_r} > max_tr={self.max_tr}"
        state_bits = instance.n * (t_r + 1)
        if state_bits > self.max_state_bits:
            return False, (f"n*(T_R+1)={state_bits} bits "
                           f"> max_state_bits={self.max_state_bits}")
        return True, ""

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        ok, _reason = self.feasible(instance)
        if not ok:
            objective.evaluate([])
            return

        items = instance.items
        n = instance.n
        target = instance.target
        t_lo = max(0, math.ceil(target.T_L))
        t_hi = math.floor(target.T_R)
        if t_hi < 0:
            objective.evaluate([])
            return

        mask = (1 << (t_hi + 1)) - 1
        # prefixes[k] = bitset of sums reachable using items[0..k-1], <= t_hi.
        prefixes: List[int] = [1]
        reachable = 1
        for a in items:
            reachable |= (reachable << a)
            reachable &= mask
            prefixes.append(reachable)

        best_mu = -1.0
        best_s = -1
        for s in range(t_lo, t_hi + 1):
            if (reachable >> s) & 1:
                m = target.mu(s)
                if m > best_mu:
                    best_mu = m
                    best_s = s
        if best_s < 0:
            # No reachable sum with mu > 0 exists; the empty subset (mu = 0)
            # is a true optimum.
            objective.evaluate([])
            return

        # Reconstruct an explicit subset for best_s by walking prefixes back.
        subset: List[int] = []
        s = best_s
        for k in range(n, 0, -1):
            if (prefixes[k - 1] >> s) & 1:
                continue  # sum s already reachable without item k-1
            subset.append(k - 1)
            s -= items[k - 1]
        assert s == 0, "reconstruction failed"
        objective.evaluate(sorted(subset))

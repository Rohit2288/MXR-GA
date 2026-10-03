"""Lattice-reduction baseline for FSSP (Phase 11, strong-baseline check).

The classical attack on subset sum (Lagarias-Odlyzko; Coster, Joux,
LaMacchia, Odlyzko, Schnorr, Stern = "CJLOSS"): the lattice spanned by

    b_i     = (2 e_i, N a_i)          i = 1..n
    b_{n+1} = (1, ..., 1, N T_M)

contains (2x - 1, N(sum x a - T_M)), whose first n entries are +-1 and whose
last entry is 0 when x hits T_M exactly. Reduction (LLL, then BKZ with
growing block size) finds such short vectors when the density n/log2(max a)
is low enough (< ~0.94 provably-on-average; it degrades above). Every row
of each reduced basis whose first n entries are all +-1 is read as a subset
(both signs) and scored through the protocol, so near misses inside the
fuzzy window count too. Randomised restarts (row permutations) continue
until the protocol's wall-clock budget ends.

Dimension n + 1, so the solver declines n > max_n (default 150); larger
lattices do not reduce usefully within seconds. One protocol evaluation per
candidate subset read off a reduced basis.
"""

from __future__ import annotations

import math

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


class Lattice(Solver):
    name = "lattice"
    deterministic = False  # BKZ is wall-clock limited

    def __init__(self, max_n: int = 150):
        self.max_n = max_n
        self.last_stats: dict = {}

    def feasible(self, instance: Instance):
        if instance.n > self.max_n:
            return False, f"n={instance.n} > max_n={self.max_n} (lattice dim)"
        return True, ""

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        from fpylll import BKZ, LLL, IntegerMatrix
        from fpylll.algorithms.bkz2 import BKZReduction

        ok, _ = self.feasible(instance)
        if not ok:
            objective.evaluate([])
            return
        n = instance.n
        a = [int(x) for x in instance.items]
        t = int(round(instance.target.T_M))
        big_n = max(n, int(math.isqrt(n)) + 1) * 4
        stats = self.last_stats = {"restarts": 0, "candidates": 0}

        def read_off(B):
            for r in range(B.nrows):
                row = [B[r, j] for j in range(n)]
                if all(v in (1, -1) for v in row):
                    for sign in (1, -1):
                        subset = [j for j in range(n) if row[j] * sign == 1]
                        stats["candidates"] += 1
                        objective.evaluate(subset)

        while True:
            perm = rng.permutation(n) if stats["restarts"] else np.arange(n)
            B = IntegerMatrix(n + 1, n + 1)
            for row_i, i in enumerate(perm):
                B[row_i, int(i)] = 2
                B[row_i, n] = big_n * a[int(i)]
            for j in range(n):
                B[n, j] = 1
            B[n, n] = big_n * t
            LLL.reduction(B)
            read_off(B)
            for block in (10, 20, 30):
                left = objective.time_left()
                if left < 0.05:
                    objective.evaluate([])   # lets the protocol stop the run
                block = min(block, n + 1)
                params = BKZ.Param(block_size=block,
                                   max_time=max(0.05, min(left, 60.0)),
                                   flags=BKZ.AUTO_ABORT | BKZ.MAX_TIME)
                BKZReduction(B)(params)
                read_off(B)
            stats["restarts"] += 1
            objective.evaluate([])           # budget check between restarts

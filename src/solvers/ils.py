"""Iterated local search baseline (task 1.6; R2.7).

Local search: best-improvement single-bit flips to a local optimum, then a
capped 1-1 swap pass (smallest-K included x smallest-K excluded items, K=20,
the same cap as FH-GA's swap phase); any swap improvement returns to the
flip phase. Perturbation: `perturb_strength` uniform random bit flips from
the incumbent; the incumbent is updated only on improvement (better-accept
ILS). Termination is protocol-owned.
"""

from __future__ import annotations

from itertools import compress
from typing import List

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.protocol.guide import make_guide
from src.solvers.base import Solver


class ILS(Solver):
    name = "ils"

    def __init__(self, perturb_strength: int = 3, swap_k: int = 20,
                 guide_mode: str = "mu", name: str = None):
        self.perturb_strength = perturb_strength
        self.swap_k = swap_k
        self.guide_mode = guide_mode
        if name is not None:
            self.name = name
        elif guide_mode != "mu":
            self.name = f"ils_{guide_mode}"

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        items = instance.items
        guide = make_guide(self.guide_mode, instance)

        positions = range(n)

        def _eval(chrom) -> float:
            # itertools.compress is the C-level equivalent of
            # [i for i, b in enumerate(chrom) if b] and yields the same list.
            idx = list(compress(positions, chrom))
            mu = objective.evaluate(idx)
            if guide is None:
                return mu
            return guide(sum(items[i] for i in idx), mu)

        self._eval_fn = _eval
        density = min(1.0, instance.target.T_M / max(sum(items), 1))
        chrom: List[int] = [1 if rng.random() < density else 0
                            for _ in range(n)]
        f = _eval(chrom)
        f = self._local_search(chrom, f, objective, items, n)
        inc_chrom, inc_f = list(chrom), f
        while True:
            chrom = list(inc_chrom)
            for i in rng.choice(n, size=min(self.perturb_strength, n),
                                replace=False):
                chrom[int(i)] = 1 - chrom[int(i)]
            f = _eval(chrom)
            f = self._local_search(chrom, f, objective, items, n)
            if f > inc_f:
                inc_chrom, inc_f = list(chrom), f

    def _eval(self, chrom, objective) -> float:
        return self._eval_fn(chrom)

    def _local_search(self, chrom, f, objective, items, n) -> float:
        while True:
            # Best-improvement single flips.
            improved = True
            while improved:
                improved = False
                best_i, best_f = -1, f
                for i in range(n):
                    chrom[i] = 1 - chrom[i]
                    cand_f = self._eval(chrom, objective)
                    chrom[i] = 1 - chrom[i]
                    if cand_f > best_f + 1e-12:
                        best_i, best_f = i, cand_f
                if best_i >= 0:
                    chrom[best_i] = 1 - chrom[best_i]
                    f = best_f
                    improved = True
            # Capped best-improvement 1-1 swap pass.
            ones = sorted((i for i in range(n) if chrom[i]),
                          key=lambda i: items[i])[: self.swap_k]
            zeros = sorted((i for i in range(n) if not chrom[i]),
                           key=lambda i: items[i])[: self.swap_k]
            best_pair, best_f = None, f
            for i in ones:
                for j in zeros:
                    chrom[i], chrom[j] = 0, 1
                    cand_f = self._eval(chrom, objective)
                    chrom[i], chrom[j] = 1, 0
                    if cand_f > best_f + 1e-12:
                        best_pair, best_f = (i, j), cand_f
            if best_pair is None:
                return f
            i, j = best_pair
            chrom[i], chrom[j] = 0, 1
            f = best_f

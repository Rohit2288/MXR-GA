"""Baseline solvers ported to the protocol interface (task 1.8; R4.3).

Ports of legacy generator/reference_solvers/{vanilla_ga, random_search,
greedy_center}.py. The legacy asymmetries are fixed by construction:
vanilla_ga's private early exit at mu >= 1-1e-9 is removed and its default
budget (legacy 50k, vs random_search's 10k) is replaced by the shared
ProtocolConfig — all solvers now get identical budgets and the identical
protocol plateau early-exit.

GreedyCenter is deterministic and finite: it evaluates its constructed
subset once and returns (stop_reason "solver_done").
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.protocol.guide import make_guide
from src.solvers.base import Solver


def _density(instance: Instance) -> float:
    return min(1.0, instance.target.T_M / max(sum(instance.items), 1))


class RandomSearch(Solver):
    name = "random_search"

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        density = _density(instance)
        while True:
            chrom = rng.random(n) < density
            objective.evaluate([int(i) for i in np.flatnonzero(chrom)])


class GreedyCenter(Solver):
    name = "greedy_center"

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        items = instance.items
        center = instance.target.T_M / n
        order = sorted(range(n), key=lambda i: abs(items[i] - center))
        subset: List[int] = []
        s = 0
        for i in order:
            if s + items[i] <= instance.target.T_R:
                subset.append(i)
                s += items[i]
        objective.evaluate(sorted(subset))


class VanillaGA(Solver):
    name = "vanilla_ga"

    def __init__(self, pop_size=None, p_c: float = 0.9, p_m=None,
                 guide_mode: str = "mu"):
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        # "crisp": a conventional GA steered by the shared crisp guide -- the
        # comparator R4 (round 2) asked for. Default "mu" is unchanged.
        self.guide_mode = guide_mode
        if guide_mode != "mu":
            self.name = f"vanilla_ga_{guide_mode}"

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        N = self.pop_size if self.pop_size is not None else max(50, 2 * n)
        p_c = self.p_c
        p_m = self.p_m if self.p_m is not None else 1.0 / n
        density = _density(instance)
        guide = make_guide(self.guide_mode, instance)
        items = instance.items

        def evaluate(chrom: List[int]) -> float:
            idx = [i for i, b in enumerate(chrom) if b]
            mu = objective.evaluate(idx)
            return mu if guide is None else guide(sum(items[i] for i in idx), mu)

        pop = [[1 if rng.random() < density else 0 for _ in range(n)]
               for _ in range(N)]
        fits = [evaluate(c) for c in pop]

        def tournament() -> int:
            cands = rng.choice(N, size=3, replace=False)
            return max((int(i) for i in cands), key=lambda i: fits[i])

        while True:
            elite_idx = max(range(N), key=lambda i: fits[i])
            new_pop = [list(pop[elite_idx])]
            new_fit = [fits[elite_idx]]
            while len(new_pop) < N:
                p1, p2 = pop[tournament()], pop[tournament()]
                if rng.random() < p_c:
                    c1 = [a if rng.random() < 0.5 else b
                          for a, b in zip(p1, p2)]
                    c2 = [a if rng.random() < 0.5 else b
                          for a, b in zip(p2, p1)]
                else:
                    c1, c2 = list(p1), list(p2)
                for c in (c1, c2):
                    for k in range(n):
                        if rng.random() < p_m:
                            c[k] = 1 - c[k]
                for c in (c1, c2):
                    if len(new_pop) >= N:
                        break
                    new_pop.append(c)
                    new_fit.append(evaluate(c))
            pop, fits = new_pop, new_fit

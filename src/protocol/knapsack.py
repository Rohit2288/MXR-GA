"""Protocol and solvers for the Fuzzy Knapsack replication (task 2.16).

Mirrors src/protocol/budget.py: the objective owns the evaluation budget and
the plateau termination rule, counts every evaluation, tracks the best subset
by the membership degree, and optionally applies the shared unclamped guide.
No solver has private termination privileges and none reports its own
quality (R4.3).

Solvers are deliberately the same family used for FSSP -- random search,
greedy, a vanilla GA, iterated local search and simulated annealing -- so
that the comparison across the two domains is like-for-like.
"""

from __future__ import annotations

import math
from itertools import compress
from typing import Iterable, List, Optional, Tuple

import numpy as np

from src.core.knapsack import (KnapsackInstance, crisp_guide,
                               evaluate_knapsack)
from src.protocol.budget import (BudgetExhausted, PlateauReached,
                                 ProtocolConfig, StopSearch)


class BudgetedKnapsackObjective:
    def __init__(self, instance: KnapsackInstance, config: ProtocolConfig,
                 guide_mode: str = "mu"):
        self.instance = instance
        self.config = config
        self.evals_used = 0
        self.best_mu = float("-inf")
        self.best_subset: Optional[Tuple[int, ...]] = None
        self._since = 0
        self._stopped: Optional[str] = None
        self._guide = crisp_guide(instance) if guide_mode == "crisp" else None
        if guide_mode not in ("mu", "crisp"):
            raise ValueError(f"unknown guide mode {guide_mode!r}")

    @property
    def stop_reason(self) -> Optional[str]:
        return self._stopped

    def evaluate(self, subset_indices: Iterable[int]) -> float:
        if self._stopped is not None:
            raise (BudgetExhausted() if self._stopped == BudgetExhausted.reason
                   else PlateauReached())
        ev = evaluate_knapsack(self.instance, subset_indices)
        self.evals_used += 1
        if ev.mu > self.best_mu + self.config.plateau_tol:
            self._since = 0
        else:
            self._since += 1
        if ev.mu > self.best_mu:
            self.best_mu = ev.mu
            self.best_subset = ev.subset_indices
        if self.evals_used >= self.config.max_evals:
            self._stopped = BudgetExhausted.reason
            raise BudgetExhausted()
        if (self.config.plateau_patience is not None
                and self._since >= self.config.plateau_patience):
            self._stopped = PlateauReached.reason
            raise PlateauReached()
        if self._guide is None:
            return ev.mu
        return self._guide(ev.profit, ev.weight, ev.mu)


class KnapsackSolver:
    name = "solver"

    def solve(self, instance: KnapsackInstance, config: ProtocolConfig,
              rng: np.random.Generator):
        import time
        obj = BudgetedKnapsackObjective(instance, config,
                                        getattr(self, "guide_mode", "mu"))
        t0 = time.perf_counter()
        try:
            self.search(instance, obj, rng)
        except StopSearch:
            pass
        wall = time.perf_counter() - t0
        subset = obj.best_subset if obj.best_subset is not None else ()
        return subset, obj.evals_used, wall, obj.stop_reason

    def search(self, instance, objective, rng) -> None:
        raise NotImplementedError


def _density(instance: KnapsackInstance) -> float:
    """Inclusion probability matching the capacity, shared by all solvers."""
    total_w = sum(instance.weights)
    return min(1.0, instance.capacity / max(total_w, 1))


class KnapRandomSearch(KnapsackSolver):
    name = "random_search"

    def search(self, instance, objective, rng):
        n = instance.n
        p = _density(instance)
        pos = range(n)
        while True:
            chrom = rng.random(n) < p
            objective.evaluate(list(compress(pos, chrom)))


class KnapGreedy(KnapsackSolver):
    name = "greedy"

    def search(self, instance, objective, rng):
        """Profit-density greedy, then a repair pass toward the modal target."""
        n = instance.n
        order = sorted(range(n),
                       key=lambda i: instance.profits[i] / instance.weights[i],
                       reverse=True)
        chosen, w, p = [], 0, 0
        for i in order:
            if w + instance.weights[i] <= instance.capacity:
                chosen.append(i)
                w += instance.weights[i]
                p += instance.profits[i]
                objective.evaluate(sorted(chosen))
                if p >= instance.target.T_M:
                    break
        while True:
            objective.evaluate(sorted(chosen))


class KnapILS(KnapsackSolver):
    name = "ils"

    def __init__(self, guide_mode: str = "mu", perturb: int = 3,
                 name: str = None):
        self.guide_mode = guide_mode
        self.perturb = perturb
        if name:
            self.name = name
        elif guide_mode != "mu":
            self.name = f"ils_{guide_mode}"

    def search(self, instance, objective, rng):
        n = instance.n
        pos = range(n)
        chrom = list((rng.random(n) < _density(instance)).astype(int))
        f = objective.evaluate(list(compress(pos, chrom)))
        f = self._ls(chrom, f, objective, pos, n)
        inc, inc_f = list(chrom), f
        while True:
            chrom = list(inc)
            for i in rng.choice(n, size=min(self.perturb, n), replace=False):
                chrom[int(i)] = 1 - chrom[int(i)]
            f = objective.evaluate(list(compress(pos, chrom)))
            f = self._ls(chrom, f, objective, pos, n)
            if f > inc_f:
                inc, inc_f = list(chrom), f

    @staticmethod
    def _ls(chrom, f, objective, pos, n):
        improved = True
        while improved:
            improved = False
            best_i, best_f = -1, f
            for i in range(n):
                chrom[i] = 1 - chrom[i]
                cand = objective.evaluate(list(compress(pos, chrom)))
                chrom[i] = 1 - chrom[i]
                if cand > best_f + 1e-12:
                    best_i, best_f = i, cand
            if best_i >= 0:
                chrom[best_i] = 1 - chrom[best_i]
                f = best_f
                improved = True
        return f


class KnapSimAnneal(KnapsackSolver):
    name = "sim_anneal"
    DEFAULTS = {"mu": (0.1, 1e-4), "crisp": (100.0, 1e-2)}

    def __init__(self, guide_mode: str = "mu", t0: float = None,
                 t_end: float = None, name: str = None):
        self.guide_mode = guide_mode
        d0, d1 = self.DEFAULTS[guide_mode]
        self.t0 = d0 if t0 is None else t0
        self.t_end = d1 if t_end is None else t_end
        if name:
            self.name = name
        elif guide_mode != "mu":
            self.name = f"sim_anneal_{guide_mode}"

    def search(self, instance, objective, rng):
        n = instance.n
        pos = range(n)
        chrom = list((rng.random(n) < _density(instance)).astype(int))
        cur = objective.evaluate(list(compress(pos, chrom)))
        ratio = math.log(self.t_end / self.t0)
        max_evals = objective.config.max_evals
        while True:
            i = int(rng.integers(0, n))
            chrom[i] = 1 - chrom[i]
            cand = objective.evaluate(list(compress(pos, chrom)))
            temp = self.t0 * math.exp(ratio * objective.evals_used / max_evals)
            delta = cand - cur
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                cur = cand
            else:
                chrom[i] = 1 - chrom[i]


class KnapVanillaGA(KnapsackSolver):
    name = "vanilla_ga"

    def __init__(self, pop: int = None, p_c: float = 0.8):
        self.pop = pop
        self.p_c = p_c

    def search(self, instance, objective, rng):
        n = instance.n
        pos = range(n)
        pop_size = self.pop or max(50, 4 * n)
        p_m = 1.0 / n
        pop = [list((rng.random(n) < _density(instance)).astype(int))
               for _ in range(pop_size)]
        fit = [objective.evaluate(list(compress(pos, c))) for c in pop]
        while True:
            new_pop, new_fit = [], []
            order = sorted(range(pop_size), key=lambda i: fit[i], reverse=True)
            for e in order[:2]:
                new_pop.append(list(pop[e]))
                new_fit.append(fit[e])
            while len(new_pop) < pop_size:
                a = max(rng.choice(pop_size, 3), key=lambda i: fit[i])
                b = max(rng.choice(pop_size, 3), key=lambda i: fit[i])
                c1 = list(pop[a])
                if rng.random() < self.p_c:
                    mask = rng.random(n) < 0.5
                    c1 = [pop[a][i] if mask[i] else pop[b][i] for i in range(n)]
                for i in range(n):
                    if rng.random() < p_m:
                        c1[i] = 1 - c1[i]
                new_pop.append(c1)
                new_fit.append(objective.evaluate(list(compress(pos, c1))))
            pop, fit = new_pop, new_fit

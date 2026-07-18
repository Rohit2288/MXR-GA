"""Simulated annealing baseline (task 1.5; R2.7).

Single-bit-flip neighborhood; Metropolis acceptance exp(delta_mu / T);
exponential cooling from T0 to T_end scheduled over the consumed fraction of
the shared evaluation budget (so the schedule is budget-aware and identical
in shape for every budget). Initial state: Bernoulli(T_M / sum(items))
inclusion, the same density rule as FH-GA's random initialization stratum.

Default parameters were selected by grid search on a held-out tuning slice
(instances with generator seed s00, RNG seeds disjoint from evaluation
seeds) — see experiments/02_sa_tuning.py; the chosen values and the full
grid results are recorded in results/sa_tuning/.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


class SimAnneal(Solver):
    name = "sim_anneal"

    def __init__(self, t0: float = 0.1, t_end: float = 1e-4):
        if not (t0 > t_end > 0):
            raise ValueError("need t0 > t_end > 0")
        self.t0 = t0
        self.t_end = t_end

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        density = min(1.0, instance.target.T_M / max(sum(instance.items), 1))
        chrom: List[int] = [1 if rng.random() < density else 0
                            for i in range(n)]
        cur_f = objective.evaluate([i for i, b in enumerate(chrom) if b])
        log_ratio = math.log(self.t_end / self.t0)
        max_evals = objective.config.max_evals
        while True:
            i = int(rng.integers(0, n))
            chrom[i] = 1 - chrom[i]
            cand_f = objective.evaluate([j for j, b in enumerate(chrom) if b])
            frac = objective.evals_used / max_evals
            temp = self.t0 * math.exp(log_ratio * frac)
            delta = cand_f - cur_f
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                cur_f = cand_f
            else:
                chrom[i] = 1 - chrom[i]  # reject: undo

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
from itertools import compress
from typing import List

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.protocol.guide import make_guide
from src.solvers.base import Solver


class SimAnneal(Solver):
    name = "sim_anneal"

    # Tuned per guide on the same held-out slice, seeds and budget
    # (experiments/02_sa_tuning.py). The crisp guide is unbounded below, so
    # its acceptance deltas are on a different scale and it needs -- and gets
    # -- its own temperature schedule; reusing the mu schedule freezes the
    # chain. Tuning both arms is what makes the comparison fair (R2.7, R4.3).
    DEFAULTS = {"mu": (0.1, 1e-4), "crisp": (100.0, 1e-2)}

    def __init__(self, t0: float = None, t_end: float = None,
                 guide_mode: str = "mu", name: str = None):
        d_t0, d_tend = self.DEFAULTS[guide_mode]
        t0 = d_t0 if t0 is None else t0
        t_end = d_tend if t_end is None else t_end
        if not (t0 > t_end > 0):
            raise ValueError("need t0 > t_end > 0")
        self.t0 = t0
        self.t_end = t_end
        self.guide_mode = guide_mode
        if name is not None:
            self.name = name
        elif guide_mode != "mu":
            self.name = f"sim_anneal_{guide_mode}"

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        items = instance.items
        guide = make_guide(self.guide_mode, instance)
        positions = range(n)

        def score(idx) -> float:
            mu = objective.evaluate(idx)
            if guide is None:
                return mu
            return guide(sum(map(items.__getitem__, idx)), mu)

        density = min(1.0, instance.target.T_M / max(sum(instance.items), 1))
        chrom: List[int] = [1 if rng.random() < density else 0
                            for i in range(n)]
        # compress is the C-level equivalent of the enumerate comprehension.
        cur_f = score(list(compress(positions, chrom)))
        log_ratio = math.log(self.t_end / self.t0)
        max_evals = objective.config.max_evals
        while True:
            i = int(rng.integers(0, n))
            chrom[i] = 1 - chrom[i]
            cand_f = score(list(compress(positions, chrom)))
            frac = objective.evals_used / max_evals
            temp = self.t0 * math.exp(log_ratio * frac)
            delta = cand_f - cur_f
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                cur_f = cand_f
            else:
                chrom[i] = 1 - chrom[i]  # reject: undo

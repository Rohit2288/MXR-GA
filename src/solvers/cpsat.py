"""OR-Tools CP-SAT solver for FSSP (task 1.4; R2.7, R3.3).

Piecewise-linear mu maximization in exact integer arithmetic. With
S = sum a_i x_i, dL = T_M - T_L, dR = T_R - T_M (all integers in these
corpora), mu(S) = min((S-T_L)/dL, (T_R-S)/dR) inside the support and 0
outside, so maximizing the integer variable

    z <= (S - T_L) * dR,   z <= (T_R - S) * dL,   z in [-B, dL*dR]

maximizes mu = z / (dL * dR) whenever some subset has mu > 0; if the true
optimum is mu = 0, z* < 0 and whatever incumbent subset the solver holds is
returned (its recomputed mu is 0 — honest either way). Degenerate targets
(dL = 0 or dR = 0) do not occur in the corpora and fall back to an evaluated
empty subset.

Time-limited: `max_time_s` wall-clock (an anytime exact method — status
OPTIMAL proves the optimum, FEASIBLE yields the incumbent). Seeded from the
protocol rng for determinism (single worker). One protocol evaluation: the
returned subset.
"""

from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


class CPSAT(Solver):
    name = "cpsat"
    deterministic = False  # incumbent at the wall-clock limit is time-dependent

    def __init__(self, max_time_s: float = 10.0):
        self.max_time_s = max_time_s
        self.last_status: str = ""

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        target = instance.target
        t_l, t_m, t_r = (int(target.T_L), int(target.T_M), int(target.T_R))
        d_l, d_r = t_m - t_l, t_r - t_m
        if d_l <= 0 or d_r <= 0:
            objective.evaluate([])
            return

        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(instance.n)]
        total = sum(int(a) * xi for a, xi in zip(instance.items, x))
        s_max = sum(instance.items)
        lb = min((0 - t_l) * d_r, (t_r - s_max) * d_l, 0)
        z = model.new_int_var(lb, d_l * d_r, "z")
        model.add(z <= (total - t_l) * d_r)
        model.add(z <= (t_r - total) * d_l)
        model.maximize(z)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time_s
        solver.parameters.num_workers = 1  # determinism
        solver.parameters.random_seed = int(rng.integers(1, 2**31 - 1))
        status = solver.solve(model)
        self.last_status = solver.status_name(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            subset = [i for i in range(instance.n)
                      if solver.value(x[i])]
        else:
            subset = []
        objective.evaluate(subset)

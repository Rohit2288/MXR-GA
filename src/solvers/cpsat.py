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

    def __init__(self, max_time_s: float = 10.0, num_workers: int = 1,
                 formulation: str = "mu"):
        self.max_time_s = max_time_s
        # "mu": maximise the linearised membership (default, as published).
        # "closest": minimise wl*under + wr*over with sum + under - over = T_M,
        # an alternative closest-sum model (Phase 12.5 strong-competitor check).
        self.formulation = formulation
        # 1 (default) = deterministic single worker, one core like every
        # other solver; >1 = CP-SAT's parallel portfolio (strong-baseline check).
        self.num_workers = num_workers
        self.last_status: str = ""

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        target = instance.target
        t_l, t_m, t_r = (int(target.T_L), int(target.T_M), int(target.T_R))
        d_l, d_r = t_m - t_l, t_r - t_m
        if d_l <= 0 or d_r <= 0:
            objective.evaluate([])
            return

        # Pre-scale when the mu-linearization products would overflow CP-SAT's
        # int64 domain (huge-value instances, e.g. Suite A Avis-Lai). The
        # model becomes approximate at resolution `scale`, but the returned
        # subset is real and is rescored on true unscaled sums — honest by
        # construction, exactly like the certified alpha-cut FPTAS.
        items = [int(a) for a in instance.items]
        bound = (sum(items) + abs(t_l) + t_r + 1) * max(d_l, d_r)
        limit = 4 * 10**18
        scale = 1
        while bound // (scale * scale) > limit:
            scale *= 2
        if scale > 1:
            items = [max(1, a // scale) for a in items]
            t_l, t_m, t_r = t_l // scale, t_m // scale, t_r // scale
            d_l, d_r = t_m - t_l, t_r - t_m
            if d_l <= 0 or d_r <= 0:
                objective.evaluate([])
                return
        self.last_scale = scale

        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(instance.n)]
        total = sum(a * xi for a, xi in zip(items, x))
        s_max = sum(items)
        if self.formulation == "closest":
            hi = s_max + abs(t_m)
            under = model.new_int_var(0, hi, "under")
            over = model.new_int_var(0, hi, "over")
            model.add(total + under - over == t_m)
            model.minimize(d_r * under + d_l * over)
        else:
            lb = min((0 - t_l) * d_r, (t_r - s_max) * d_l, 0)
            z = model.new_int_var(lb, d_l * d_r, "z")
            model.add(z <= (total - t_l) * d_r)
            model.add(z <= (t_r - total) * d_l)
            model.maximize(z)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time_s
        solver.parameters.num_workers = self.num_workers
        solver.parameters.random_seed = int(rng.integers(1, 2**31 - 1))
        status = solver.solve(model)
        self.last_status = solver.status_name(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            subset = [i for i in range(instance.n)
                      if solver.value(x[i])]
        else:
            subset = []
        objective.evaluate(subset)

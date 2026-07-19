"""Shared work-unit runner for the sweep drivers (04, 06).

Lives in the importable src package so multiprocessing spawn workers can
re-import it by name — loading a driver script by file path breaks worker
pickling (the 06 stage failure of 2026-07-19).
"""

from __future__ import annotations

import json

import numpy as np

from src.core.evaluate import evaluate
from src.core.instance import load_instance
from src.protocol.budget import ProtocolConfig

PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)
STOCHASTIC = ["fhga_fast", "fhga_thorough", "vanilla_ga", "random_search",
              "sim_anneal", "ils"]
DETERMINISTIC = ["dp_exact", "mitm", "greedy_center", "alpha_cut", "cpsat"]
CPSAT_TIME_S = 10.0
FIELDS = ["solver", "instance_id", "suite", "n", "distribution", "seed",
          "status", "mu", "subset_sum", "subset_size", "evals_used",
          "wall_time", "stop_reason", "budget", "extra"]


def make_solver(name: str):
    from src.solvers.alpha_cut import AlphaCut
    from src.solvers.baselines import GreedyCenter, RandomSearch, VanillaGA
    from src.solvers.cpsat import CPSAT
    from src.solvers.dp_bitset import DPBitset
    from src.solvers.fhga import FHGA
    from src.solvers.ils import ILS
    from src.solvers.mitm import MITM
    from src.solvers.sim_anneal import SimAnneal
    return {
        "fhga_fast": lambda: FHGA("fast"),
        "fhga_thorough": lambda: FHGA("thorough"),
        "vanilla_ga": VanillaGA,
        "random_search": RandomSearch,
        "sim_anneal": SimAnneal,
        "ils": ILS,
        "dp_exact": DPBitset,
        "mitm": MITM,
        "greedy_center": GreedyCenter,
        "alpha_cut": AlphaCut,
        "cpsat": lambda: CPSAT(max_time_s=CPSAT_TIME_S),
    }[name]()


def run_unit(args):
    solver_name, inst_path, seed = args
    inst = load_instance(inst_path)
    solver = make_solver(solver_name)
    base = {
        "solver": solver_name, "instance_id": inst.instance_id,
        "suite": inst.suite or "", "n": inst.n,
        "distribution": inst.distribution, "seed": seed,
        "budget": PROTOCOL.max_evals,
    }
    if hasattr(solver, "feasible"):
        ok, reason = solver.feasible(inst)
        if not ok:
            return {**base, "status": "infeasible", "mu": "", "subset_sum": "",
                    "subset_size": "", "evals_used": 0, "wall_time": "",
                    "stop_reason": "", "extra": json.dumps({"reason": reason})}
    res = solver.solve(inst, PROTOCOL, np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    extra = {}
    if getattr(solver, "last_stats", None):
        extra = solver.last_stats
    if getattr(solver, "last_status", ""):
        extra["cpsat_status"] = solver.last_status
    return {**base, "status": "ok", "mu": repr(ev.mu),
            "subset_sum": ev.subset_sum, "subset_size": len(ev.subset_indices),
            "evals_used": res.evals_used, "wall_time": repr(res.wall_time),
            "stop_reason": res.stop_reason, "extra": json.dumps(extra)}

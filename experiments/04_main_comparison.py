"""Task 2.1 — main comparison: ALL solvers x ALL FSSP-2 instances. (R2.6, R3.4, R4.2, E)

Shared protocol for every solver: max_evals=50,000, plateau_tol=1e-9,
plateau_patience=10,000 (identical budget and early-exit — R4.3). Stochastic
solvers: seeds 0..19 (20 seeds). Deterministic/exact solvers: one run.
CP-SAT: one run, 10 s wall-clock limit. DP/MITM instances outside their
documented feasibility domains produce status="infeasible" rows (excluded
from quality stats, reported as coverage).

Parallel (worker pool) with checkpoint/resume: existing rows in results.csv
are skipped on restart, so the script can be interrupted freely.

Usage:
  .venv/bin/python experiments/04_main_comparison.py [--workers 6] [--limit N]
Output:
  results/main_comparison/results.csv + manifest.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.protocol.runner import git_commit

PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)
STOCHASTIC = ["fhga_fast", "fhga_thorough", "vanilla_ga", "random_search",
              "sim_anneal", "ils"]
DETERMINISTIC = ["dp_exact", "mitm", "greedy_center", "alpha_cut", "cpsat"]
SEEDS = list(range(20))
CPSAT_TIME_S = 10.0

OUT_DIR = REPO / "results" / "main_comparison"
CSV_PATH = OUT_DIR / "results.csv"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="limit number of instances (pilot mode)")
    args = ap.parse_args()

    paths = list(iter_instance_paths(FSSP2_DIR))
    if args.limit:
        paths = paths[:: max(1, len(paths) // args.limit)][: args.limit]

    units = []
    for p in paths:
        for name in STOCHASTIC:
            units.extend((name, p, s) for s in SEEDS)
        units.extend((name, p, 0) for name in DETERMINISTIC)

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add((row["solver"], row["instance_id"], int(row["seed"])))
    todo = []
    for (name, p, s) in units:
        iid = p.stem
        if (name, iid, s) not in done:
            todo.append((name, p, s))
    print(f"{len(units)} units total, {len(done)} done, {len(todo)} to run",
          flush=True)

    # Mix solvers/sizes so workers stay balanced.
    rng = np.random.default_rng(0)
    rng.shuffle(todo)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    t0 = time.perf_counter()
    n_done = 0
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with mp.Pool(args.workers) as pool:
            for row in pool.imap_unordered(run_unit, todo, chunksize=4):
                writer.writerow(row)
                n_done += 1
                if n_done % 200 == 0:
                    fh.flush()
                    rate = n_done / (time.perf_counter() - t0)
                    eta_h = (len(todo) - n_done) / rate / 3600
                    print(f"  {n_done}/{len(todo)} "
                          f"({rate:.1f}/s, ETA {eta_h:.1f}h)", flush=True)

    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/04_main_comparison.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": PROTOCOL.max_evals,
                         "plateau_tol": PROTOCOL.plateau_tol,
                         "plateau_patience": PROTOCOL.plateau_patience},
            "seeds": SEEDS, "stochastic": STOCHASTIC,
            "deterministic": DETERMINISTIC, "cpsat_time_s": CPSAT_TIME_S,
            "n_instances": len(paths),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

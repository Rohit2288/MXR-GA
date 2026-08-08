"""Task 2.16 — the Fuzzy Knapsack replication run (R2.7, R4.3 generalisation).

Runs the same solver family used for FSSP on the FKP corpus, under the same
protocol, with the metaheuristics in both mu-steered and guided arms. The
question is whether the clipped-objective effect established on FSSP
reproduces on a problem with a genuine capacity constraint:

  H1  mu-steered ILS and SA stall relative to their guided counterparts;
  H2  the per-instance plateau mass predicts the size of that gain.

Both are pre-registered here in the sense that the analysis in
experiments/31 computes exactly these two quantities and nothing else.

Usage: python3 experiments/30_fkp_run.py [--seeds 10] [--max-seconds 33]
Output: results/fkp/results.csv + manifest.json
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

from src.core.fuzzy import TriangularFuzzyNumber  # noqa: E402
from src.core.knapsack import (KnapsackInstance, dp_exact_knapsack,  # noqa: E402
                               evaluate_knapsack)
from src.protocol.budget import ProtocolConfig  # noqa: E402
from src.protocol.knapsack import (KnapGreedy, KnapILS,  # noqa: E402
                                   KnapRandomSearch, KnapSimAnneal,
                                   KnapVanillaGA)
from src.protocol.runner import git_commit  # noqa: E402

FKP_DIR = REPO / "benchmarks" / "fkp"
OUT_DIR = REPO / "results" / "fkp"
CSV_PATH = OUT_DIR / "results.csv"

PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)

SOLVERS = ["ils", "ils_crisp", "sim_anneal", "sim_anneal_crisp",
           "random_search", "vanilla_ga", "greedy", "dp_exact"]
DETERMINISTIC = {"greedy", "dp_exact"}
FIELDS = ["solver", "instance_id", "n", "correlation", "tol_frac", "seed",
          "status", "mu", "profit", "weight", "feasible", "subset_size",
          "evals_used", "wall_time", "stop_reason", "budget", "plateau_mass"]


def load_fkp(path: Path) -> KnapsackInstance:
    raw = json.loads(path.read_text())
    t = raw["target"]
    return KnapsackInstance(
        instance_id=raw["instance_id"],
        profits=tuple(raw["profits"]), weights=tuple(raw["weights"]),
        capacity=raw["capacity"],
        target=TriangularFuzzyNumber(t["T_L"], t["T_M"], t["T_R"]),
        n=raw["n"], correlation=raw["correlation"], seed=raw["seed"],
        metadata=raw["metadata"])


def make_solver(name):
    return {
        "ils": lambda: KnapILS(guide_mode="mu"),
        "ils_crisp": lambda: KnapILS(guide_mode="crisp"),
        "sim_anneal": lambda: KnapSimAnneal(guide_mode="mu"),
        "sim_anneal_crisp": lambda: KnapSimAnneal(guide_mode="crisp"),
        "random_search": lambda: KnapRandomSearch(),
        "vanilla_ga": lambda: KnapVanillaGA(),
        "greedy": lambda: KnapGreedy(),
    }[name]()


def run_unit(args):
    name, path, seed = args
    inst = load_fkp(path)
    base = {"solver": name, "instance_id": inst.instance_id, "n": inst.n,
            "correlation": inst.correlation,
            "tol_frac": inst.metadata.get("tolerance_fraction_of_mean_profit",
                                          ""),
            "seed": seed, "budget": PROTOCOL.max_evals,
            "plateau_mass": inst.metadata.get("measured_plateau_mass", "")}
    try:
        if name == "dp_exact":
            t0 = time.perf_counter()
            ev = dp_exact_knapsack(inst)
            wall = time.perf_counter() - t0
            if ev is None:
                base.update({"status": "infeasible", "mu": "", "profit": "",
                             "weight": "", "feasible": "", "subset_size": "",
                             "evals_used": "", "wall_time": repr(wall),
                             "stop_reason": "table_too_large"})
                return base
            base.update({"status": "ok", "mu": repr(ev.mu),
                         "profit": ev.profit, "weight": ev.weight,
                         "feasible": int(ev.feasible),
                         "subset_size": len(ev.subset_indices),
                         "evals_used": 1, "wall_time": repr(wall),
                         "stop_reason": "exact"})
            return base
        solver = make_solver(name)
        subset, evals, wall, stop = solver.solve(
            inst, PROTOCOL, np.random.default_rng(seed))
        ev = evaluate_knapsack(inst, subset)
        base.update({"status": "ok", "mu": repr(ev.mu), "profit": ev.profit,
                     "weight": ev.weight, "feasible": int(ev.feasible),
                     "subset_size": len(ev.subset_indices),
                     "evals_used": evals, "wall_time": repr(wall),
                     "stop_reason": stop})
        return base
    except Exception as exc:                        # noqa: BLE001
        base.update({"status": f"error:{type(exc).__name__}", "mu": "",
                     "profit": "", "weight": "", "feasible": "",
                     "subset_size": "", "evals_used": "", "wall_time": "",
                     "stop_reason": ""})
        return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    paths = sorted(p for p in FKP_DIR.glob("*.json") if p.name != "manifest.json")
    if args.limit:
        step = max(1, len(paths) // args.limit)
        paths = paths[::step][: args.limit]
    units = []
    for name in SOLVERS:
        seeds = [0] if name in DETERMINISTIC else list(range(args.seeds))
        units += [(name, p, s) for p in paths for s in seeds]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                done.add((r["solver"], r["instance_id"], int(r["seed"])))
    todo = [(n, p, s) for (n, p, s) in units if (n, p.stem, s) not in done]
    print(f"{len(paths)} instances, {len(units)} units, {len(todo)} to run",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    t0 = time.perf_counter()
    n_done = 0
    if todo:
        with open(CSV_PATH, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            with mp.Pool(args.workers) as pool:
                for row in pool.imap_unordered(run_unit, todo, chunksize=4):
                    w.writerow(row)
                    n_done += 1
                    if n_done % 200 == 0:
                        fh.flush()
                        print(f"  {n_done}/{len(todo)}", flush=True)
                    if (args.max_seconds
                            and time.perf_counter() - t0 > args.max_seconds):
                        fh.flush()
                        print(f"  chunk limit at {n_done}/{len(todo)}",
                              flush=True)
                        pool.terminate()
                        break

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "script": "experiments/30_fkp_run.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "protocol": {"max_evals": PROTOCOL.max_evals,
                     "plateau_tol": PROTOCOL.plateau_tol,
                     "plateau_patience": PROTOCOL.plateau_patience},
        "seeds": args.seeds, "solvers": SOLVERS,
        "n_instances": len(paths),
    }, indent=2))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

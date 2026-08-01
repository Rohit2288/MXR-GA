"""Task 2.6 — baseline fairness re-run: guided SA and ILS (R2.7, R4.3).

Why this experiment exists
--------------------------
In the first main comparison, sim_anneal and ils were steered by the
membership mu alone and returned mu = 0 on 60.2% and 80.2% of Suite T runs
respectively, against 0.3% for random search. That pattern is not a quality
result: mu is flat at exactly zero outside the triangular support, so on
runs whose trajectory lies outside it every neighbour scores 0, ILS's
best-improvement finds no improving flip, and SA's Metropolis test sees
delta = 0 everywhere. The searches stall on a zero plateau.

FH-GA escapes that plateau by construction (hybrid initialisation, repair).
Reporting mu-steered baselines against a plateau-escaping FH-GA is the same
asymmetric instrumentation Reviewer 4 objected to. This experiment gives
sim_anneal and ils the SAME unclamped crisp guide FH-GA-crisp uses --
imported verbatim from src/protocol/guide.py -- and re-runs them under the
identical protocol, budget, seeds and instances.

Both arms are reported in the paper: the gap between them quantifies the
cost of the hard support cut-off (R4.1's third fuzzy-formulation property)
and the guided arm is what the baseline comparison uses.

Usage: .venv/bin/python experiments/18_baseline_fairness.py [--workers 6]
Output: results/baseline_fairness/results.csv + manifest.json
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

# Identical to 04_main_comparison.py / 10_controls.py.
PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)
SEEDS = list(range(20))
VARIANTS = ["sim_anneal_crisp", "ils_crisp"]

OUT_DIR = REPO / "results" / "baseline_fairness"
CSV_PATH = OUT_DIR / "results.csv"
FIELDS = ["solver", "instance_id", "suite", "n", "distribution", "seed",
          "status", "mu", "subset_sum", "subset_size", "evals_used",
          "wall_time", "stop_reason", "budget", "extra"]


def make_solver(name: str):
    from src.solvers.ils import ILS
    from src.solvers.sim_anneal import SimAnneal
    return {
        "sim_anneal_crisp": lambda: SimAnneal(guide_mode="crisp"),
        "ils_crisp": lambda: ILS(guide_mode="crisp"),
    }[name]()


def run_unit(args):
    name, inst_path, seed = args
    inst = load_instance(inst_path)
    solver = make_solver(name)
    res = solver.solve(inst, PROTOCOL, np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    return {
        "solver": name, "instance_id": inst.instance_id,
        "suite": inst.suite or "", "n": inst.n,
        "distribution": inst.distribution, "seed": seed, "status": "ok",
        "mu": repr(ev.mu), "subset_sum": ev.subset_sum,
        "subset_size": len(ev.subset_indices), "evals_used": res.evals_used,
        "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
        "budget": PROTOCOL.max_evals,
        "extra": json.dumps(getattr(solver, "last_stats", {})),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of instances")
    ap.add_argument("--suite", default=None,
                    help="restrict to one sub-suite (T/A/U)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of evaluation seeds (default 20)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this many seconds (chunked runs)")
    args = ap.parse_args()

    seeds = SEEDS[: args.seeds]
    paths = list(iter_instance_paths(FSSP2_DIR))
    if args.suite:
        paths = [p for p in paths
                 if (load_instance(p).suite or "") == args.suite]
    if args.limit:
        paths = paths[: args.limit]
    units = [(n, p, s) for n in VARIANTS for p in paths for s in seeds]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add((row["solver"], row["instance_id"], int(row["seed"])))
    todo = [(n, p, s) for (n, p, s) in units if (n, p.stem, s) not in done]
    print(f"{len(units)} units total, {len(done)} done, {len(todo)} to run",
          flush=True)
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
            it = pool.imap_unordered(run_unit, todo, chunksize=4)
            for row in it:
                writer.writerow(row)
                n_done += 1
                if n_done % 200 == 0:
                    fh.flush()
                    rate = n_done / (time.perf_counter() - t0)
                    eta_h = (len(todo) - n_done) / rate / 3600
                    print(f"  {n_done}/{len(todo)} "
                          f"({rate:.1f}/s, ETA {eta_h:.1f}h)", flush=True)
                if (args.max_seconds is not None
                        and time.perf_counter() - t0 > args.max_seconds):
                    fh.flush()
                    print(f"  chunk limit reached at {n_done}/{len(todo)}; "
                          f"rerun to resume", flush=True)
                    pool.terminate()
                    break

    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/18_baseline_fairness.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": PROTOCOL.max_evals,
                         "plateau_tol": PROTOCOL.plateau_tol,
                         "plateau_patience": PROTOCOL.plateau_patience},
            "seeds": SEEDS, "variants": VARIANTS,
            "comparators": "sim_anneal / ils (mu-guided) from main_comparison",
            "guide": "src/protocol/guide.py:crisp_guide (shared with fhga_crisp)",
            "n_instances": len(paths),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

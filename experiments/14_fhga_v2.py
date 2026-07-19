"""FH-GA v2 — the ablation-guided simplification, full-benchmark run.

v2 keeps the two mechanisms the ablation showed to matter (hybrid
initialization, gradient local search) and the DP-window repair that beat
WBF-Repair; it drops the three mechanisms with no measurable contribution
(Mamdani controller, FB-UX, impact-aware mutation -> plain uniform
crossover and uniform bit-flip mutation, fixed rates). If v2 matches the
full algorithm on all suites, the paper gains an honest simplification
result (answers R2.2 "why this combination" and R4.1 directly).

All FSSP-2 instances, 20 seeds, shared protocol. Run after the controls
sweep (queue5).

Usage: .venv/bin/python experiments/14_fhga_v2.py [--workers 8]
Output: results/fhga_v2/results.csv + manifest.json
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
from src.protocol import units
from src.protocol.runner import git_commit

SEEDS = list(range(20))
OUT_DIR = REPO / "results" / "fhga_v2"
CSV_PATH = OUT_DIR / "results.csv"

V2_OVERRIDES = dict(
    repair_mode="dp_window",       # reviewer-suggested repair (beats WBF)
    uniform_xover=True,            # FB-UX off (inert)
    gamma=0.0,                     # impact-aware mutation off (inert)
    pc_low=0.8, pc_high=0.8,       # controller off (exactly zero effect)
    pm_low_mult=1.0, pm_high_mult=1.0,
)


def run_unit(args):
    inst_path, seed = args
    from src.solvers.fhga import FHGA, FHGAConfig
    inst = load_instance(inst_path)
    solver = FHGA(variant="thorough", config=FHGAConfig(**V2_OVERRIDES),
                  name="fhga_v2")
    res = solver.solve(inst, units.PROTOCOL, np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    return {
        "solver": "fhga_v2", "instance_id": inst.instance_id,
        "suite": inst.suite or "", "n": inst.n,
        "distribution": inst.distribution, "seed": seed, "status": "ok",
        "mu": repr(ev.mu), "subset_sum": ev.subset_sum,
        "subset_size": len(ev.subset_indices), "evals_used": res.evals_used,
        "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
        "budget": units.PROTOCOL.max_evals,
        "extra": json.dumps(solver.last_stats),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = list(iter_instance_paths(FSSP2_DIR))
    all_units = [(p, s) for p in paths for s in SEEDS]
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add((row["instance_id"], int(row["seed"])))
    todo = [(p, s) for (p, s) in all_units if (p.stem, s) not in done]
    print(f"{len(all_units)} units total, {len(done)} done, {len(todo)} to run",
          flush=True)
    rng = np.random.default_rng(0)
    rng.shuffle(todo)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    t0 = time.perf_counter()
    n_done = 0
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=units.FIELDS)
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
            "script": "experiments/14_fhga_v2.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": units.PROTOCOL.max_evals,
                         "plateau_tol": units.PROTOCOL.plateau_tol,
                         "plateau_patience": units.PROTOCOL.plateau_patience},
            "seeds": SEEDS, "overrides": V2_OVERRIDES,
            "n_instances": len(paths),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

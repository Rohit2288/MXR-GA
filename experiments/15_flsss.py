"""Task 1.11 — FLSSS sweep on all FSSP-2 instances. (R1.4, R2.7)

One run per instance (wall-clock-limited published solver, like CP-SAT).
Run after the v2 sweep (queue6). Results merge into the main comparison
at analysis time (05's generators pick it up via a rerun with the merged
CSV — see 16_merge_flsss.py note in the manifest).

Usage: .venv/bin/python experiments/15_flsss.py [--workers 8]
Output: results/flsss/results.csv + manifest.json
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

OUT_DIR = REPO / "results" / "flsss"
CSV_PATH = OUT_DIR / "results.csv"


def run_unit(args):
    (inst_path,) = args
    from src.solvers.flsss_solver import FLSSSSolver
    inst = load_instance(inst_path)
    solver = FLSSSSolver()
    res = solver.solve(inst, units.PROTOCOL, np.random.default_rng(0))
    ev = evaluate(inst, res.subset_indices)
    return {
        "solver": "flsss", "instance_id": inst.instance_id,
        "suite": inst.suite or "", "n": inst.n,
        "distribution": inst.distribution, "seed": 0, "status": "ok",
        "mu": repr(ev.mu), "subset_sum": ev.subset_sum,
        "subset_size": len(ev.subset_indices), "evals_used": res.evals_used,
        "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
        "budget": units.PROTOCOL.max_evals, "extra": "{}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = list(iter_instance_paths(FSSP2_DIR))
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add(row["instance_id"])
    todo = [(p,) for p in paths if p.stem not in done]
    print(f"{len(paths)} instances, {len(done)} done, {len(todo)} to run",
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
            for row in pool.imap_unordered(run_unit, todo, chunksize=2):
                writer.writerow(row)
                n_done += 1
                if n_done % 100 == 0:
                    fh.flush()
                    rate = n_done / (time.perf_counter() - t0)
                    eta_h = (len(todo) - n_done) / rate / 3600
                    print(f"  {n_done}/{len(todo)} "
                          f"({rate:.1f}/s, ETA {eta_h:.1f}h)", flush=True)

    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/15_flsss.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "solver": "FLSSS (CRAN; Liu, arXiv:1612.04484) via certified "
                      "alpha-window probes, tlimit 1s/probe, 60s/instance",
            "n_instances": len(paths),
            "merge_note": "append these rows to main_comparison/results.csv "
                          "copies at analysis time; solver name 'flsss'",
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

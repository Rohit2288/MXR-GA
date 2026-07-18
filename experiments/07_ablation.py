"""Tasks 3.1-3.3 — mechanism ablation + DP-window repair on Suites A and U.

Variants (config overrides exactly as legacy run_ablation_v2.py, plus the
task-3.3 DP-window repair):
  no_init, no_ls, no_controller, no_impact, no_fb_xover, dp_window
The "full" and "no_repair" arms are NOT rerun: they are byte-identical to
the main comparison's fhga_thorough / fhga_fast runs (same protocol, same
seeds, deterministic given seed) and are joined from
results/main_comparison/results.csv by the analysis script.

20 seeds, shared protocol identical to the main comparison. Run AFTER
04_main_comparison.py finishes (shares the machine).

Usage: .venv/bin/python experiments/07_ablation.py [--workers 6]
Output: results/ablation/results.csv + manifest.json
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
SEEDS = list(range(20))
SUITES = ("suite_A", "suite_U")

VARIANTS = {
    "no_init": {"init_greedy_frac": 0.0, "init_alpha_frac": 0.0},
    "no_ls": {"p_ls": 0.0},
    "no_controller": {"pc_low": 0.8, "pc_high": 0.8,
                      "pm_low_mult": 1.0, "pm_high_mult": 1.0},
    "no_impact": {"gamma": 0.0},
    "no_fb_xover": {"uniform_xover": True},
    "dp_window": {"repair_mode": "dp_window"},
}

OUT_DIR = REPO / "results" / "ablation"
CSV_PATH = OUT_DIR / "results.csv"
FIELDS = ["solver", "instance_id", "suite", "n", "distribution", "seed",
          "status", "mu", "subset_sum", "subset_size", "evals_used",
          "wall_time", "stop_reason", "budget", "extra"]


def run_unit(args):
    variant, inst_path, seed = args
    from src.solvers.fhga import FHGA, FHGAConfig
    inst = load_instance(inst_path)
    solver = FHGA(variant="thorough", config=FHGAConfig(**VARIANTS[variant]),
                  name=variant)
    res = solver.solve(inst, PROTOCOL, np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    return {
        "solver": variant, "instance_id": inst.instance_id,
        "suite": inst.suite or "", "n": inst.n,
        "distribution": inst.distribution, "seed": seed,
        "status": "ok", "mu": repr(ev.mu), "subset_sum": ev.subset_sum,
        "subset_size": len(ev.subset_indices), "evals_used": res.evals_used,
        "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
        "budget": PROTOCOL.max_evals, "extra": json.dumps(solver.last_stats),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    paths = [p for p in iter_instance_paths(FSSP2_DIR)
             if any(s in str(p) for s in SUITES)]
    units = [(v, p, s) for v in VARIANTS for p in paths for s in SEEDS]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add((row["solver"], row["instance_id"], int(row["seed"])))
    todo = [(v, p, s) for (v, p, s) in units if (v, p.stem, s) not in done]
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
            "script": "experiments/07_ablation.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": PROTOCOL.max_evals,
                         "plateau_tol": PROTOCOL.plateau_tol,
                         "plateau_patience": PROTOCOL.plateau_patience},
            "seeds": SEEDS, "variants": sorted(VARIANTS),
            "full_and_no_repair": "joined from main_comparison "
                                  "(fhga_thorough / fhga_fast)",
            "n_instances": len(paths),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

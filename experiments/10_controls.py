"""Tasks 4.1 + 4.2 — control runs: crisp objective and local-search variants.

Four controls on ALL FSSP-2 instances, 20 seeds, shared protocol (identical
to the main comparison):
  fhga_crisp_fast, fhga_crisp_thorough   (4.1: fuzzy-vs-crisp, R4.1)
  fhga_ls_bestflip, fhga_ls_swap         (4.2: LS controls, R2.3; on the
                                          Fast config, as is fhga_fast)
Comparators (fhga_fast, fhga_thorough) come from the main comparison —
same seeds, same protocol; not rerun.

Run AFTER the earlier queues (shares the machine).
Usage: .venv/bin/python experiments/10_controls.py [--workers 6]
Output: results/controls/results.csv + manifest.json
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
CONTROLS = ["fhga_crisp_fast", "fhga_crisp_thorough",
            "fhga_ls_bestflip", "fhga_ls_swap"]

OUT_DIR = REPO / "results" / "controls"
CSV_PATH = OUT_DIR / "results.csv"
FIELDS = ["solver", "instance_id", "suite", "n", "distribution", "seed",
          "status", "mu", "subset_sum", "subset_size", "evals_used",
          "wall_time", "stop_reason", "budget", "extra"]


def make_solver(name: str):
    from src.solvers.fhga import FHGA, FHGAConfig
    from src.solvers.fhga_crisp import FHGACrisp
    return {
        "fhga_crisp_fast": lambda: FHGACrisp(variant="fast"),
        "fhga_crisp_thorough": lambda: FHGACrisp(variant="thorough"),
        "fhga_ls_bestflip": lambda: FHGA(
            variant="fast", config=FHGAConfig(ls_mode="best_flip"),
            name="fhga_ls_bestflip"),
        "fhga_ls_swap": lambda: FHGA(
            variant="fast", config=FHGAConfig(ls_mode="swap"),
            name="fhga_ls_swap"),
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
    args = ap.parse_args()

    paths = list(iter_instance_paths(FSSP2_DIR))
    units = [(n, p, s) for n in CONTROLS for p in paths for s in SEEDS]

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
            "script": "experiments/10_controls.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": PROTOCOL.max_evals,
                         "plateau_tol": PROTOCOL.plateau_tol,
                         "plateau_patience": PROTOCOL.plateau_patience},
            "seeds": SEEDS, "controls": CONTROLS,
            "comparators": "fhga_fast / fhga_thorough from main_comparison",
            "n_instances": len(paths),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

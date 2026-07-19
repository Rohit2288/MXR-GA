"""Task 2.2 — FSSP-1 single-pass sanity check. (R4.2)

FSSP-1 is saturated (R2.6, R4.2) and is demoted to a brief appendix note;
this run exists only to confirm the qualitative picture under the fair
protocol. One seed per stochastic solver, same shared protocol as the main
comparison. Run AFTER 04_main_comparison.py finishes (shares the machine).

Usage: .venv/bin/python experiments/06_fssp1_sanity.py [--workers 6]
Output: results/fssp1_sanity/results.csv + manifest.json
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

# Shared worker/protocol live in the importable src package so that
# multiprocessing spawn workers can re-import them (loading the 04 driver
# by file path broke worker pickling — the 2026-07-19 stage failure).
from src.core.instance import FSSP1_DIR, iter_instance_paths
from src.protocol import units as shared
from src.protocol.runner import git_commit

OUT_DIR = REPO / "results" / "fssp1_sanity"
CSV_PATH = OUT_DIR / "results.csv"
SEEDS = [0]  # sanity pass only


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    paths = list(iter_instance_paths(FSSP1_DIR))
    units = []
    for p in paths:
        for name in shared.STOCHASTIC:
            units.extend((name, p, s) for s in SEEDS)
        units.extend((name, p, 0) for name in shared.DETERMINISTIC)

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
        writer = csv.DictWriter(fh, fieldnames=shared.FIELDS)
        if new_file:
            writer.writeheader()
        with mp.Pool(args.workers) as pool:
            for row in pool.imap_unordered(shared.run_unit, todo,
                                           chunksize=4):
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
            "script": "experiments/06_fssp1_sanity.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": shared.PROTOCOL.max_evals,
                         "plateau_tol": shared.PROTOCOL.plateau_tol,
                         "plateau_patience":
                             shared.PROTOCOL.plateau_patience},
            "seeds": SEEDS,
            "n_instances": len(paths),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

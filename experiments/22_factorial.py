"""Task 2.10 — full 2^5 factorial over the FH-GA mechanisms (R2.2).

Why this experiment exists
--------------------------
Reviewer 2 asked why *this exact combination* of mechanisms is necessary.
The leave-one-out ablation in Section 5.5 cannot answer that question: it
varies one factor at a time and is therefore blind to interactions. A
mechanism can measure as inert on its own and still matter in combination
with another, and a pair can be jointly redundant while each looks useful
alone. Removing one at a time cannot distinguish these cases.

This runs the complete 2^5 design over the five membership-aware mechanisms

    init        hybrid initialisation
    xover       fitness-biased uniform crossover (FB-UX)
    impact      impact-aware mutation
    ls          direction-guided local search
    ctrl        Mamdani parameter controller

giving 32 configurations under one protocol, from which main effects AND
all two-way interactions are estimable. The optional repair operator is
held OFF throughout (the Fast configuration), so the design isolates the
five core mechanisms rather than confounding them with repair.

Scale note: the paper reports the confirmatory run on all of Suite U with
20 seeds. This script's `--stratify` option selects a balanced subsample per
(suite, n) stratum so that the design can also be estimated at reduced cost;
whichever slice is used is recorded in the manifest and stated in the paper.

Usage:
  python3 experiments/22_factorial.py --suite U --stratify 24 --seeds 10
  python3 experiments/22_factorial.py --suite U            # full, 20 seeds
Output: results/factorial/results.csv + manifest.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.evaluate import evaluate  # noqa: E402
from src.core.instance import (FSSP2_DIR, iter_instance_paths,  # noqa: E402
                               load_instance)
from src.protocol.budget import ProtocolConfig  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

# Identical to 04_main_comparison.py / 07_ablation.py / 18_baseline_fairness.py
PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)
SEEDS = list(range(20))

FACTORS = ["init", "xover", "impact", "ls", "ctrl"]

# Config overrides applied when a factor is OFF. Identical to the
# leave-one-out settings in 07_ablation.py, so the two studies agree on what
# "disabled" means.
OFF = {
    "init": {"init_greedy_frac": 0.0, "init_alpha_frac": 0.0},
    "xover": {"uniform_xover": True},
    "impact": {"gamma": 0.0},
    "ls": {"p_ls": 0.0},
    "ctrl": {"pc_low": 0.8, "pc_high": 0.8,
             "pm_low_mult": 1.0, "pm_high_mult": 1.0},
}

OUT_DIR = REPO / "results" / "factorial"
CSV_PATH = OUT_DIR / "results.csv"
FIELDS = ["config", "init", "xover", "impact", "ls", "ctrl",
          "instance_id", "suite", "n", "distribution", "seed", "status",
          "mu", "subset_sum", "subset_size", "evals_used", "wall_time",
          "stop_reason", "budget"]


def config_name(bits) -> str:
    """e.g. (1,0,1,1,0) -> 'init+impact+ls'; all-off -> 'none'."""
    on = [f for f, b in zip(FACTORS, bits) if b]
    return "+".join(on) if on else "none"


def make_solver(bits):
    from src.solvers.fhga import FHGA, FHGAConfig
    overrides = {}
    for f, b in zip(FACTORS, bits):
        if not b:
            overrides.update(OFF[f])
    return FHGA(variant="fast", config=FHGAConfig(**overrides),
                name=f"fact_{config_name(bits)}")


def run_unit(args):
    bits, inst_path, seed = args
    inst = load_instance(inst_path)
    solver = make_solver(bits)
    res = solver.solve(inst, PROTOCOL, np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    row = {
        "config": config_name(bits), "instance_id": inst.instance_id,
        "suite": inst.suite or "", "n": inst.n,
        "distribution": inst.distribution, "seed": seed, "status": "ok",
        "mu": repr(ev.mu), "subset_sum": ev.subset_sum,
        "subset_size": len(ev.subset_indices), "evals_used": res.evals_used,
        "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
        "budget": PROTOCOL.max_evals,
    }
    row.update({f: int(b) for f, b in zip(FACTORS, bits)})
    return row


def select_paths(suite, stratify, limit):
    paths = list(iter_instance_paths(FSSP2_DIR))
    if suite:
        paths = [p for p in paths if (load_instance(p).suite or "") == suite]
    if stratify:
        strata = defaultdict(list)
        for p in paths:
            inst = load_instance(p)
            strata[(inst.suite, inst.n)].append(p)
        out = []
        for key in sorted(strata, key=lambda k: (str(k[0]), k[1])):
            out.extend(sorted(strata[key])[:stratify])
        paths = out
    if limit:
        paths = paths[:limit]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--stratify", type=int, default=None,
                    help="instances per (suite, n) stratum")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    seeds = SEEDS[: args.seeds]
    paths = select_paths(args.suite, args.stratify, args.limit)
    designs = list(itertools.product([0, 1], repeat=len(FACTORS)))
    units = [(b, p, s) for b in designs for p in paths for s in seeds]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add((row["config"], row["instance_id"], int(row["seed"])))
    todo = [(b, p, s) for (b, p, s) in units
            if (config_name(b), p.stem, s) not in done]
    print(f"{len(designs)} configs x {len(paths)} instances x {len(seeds)} "
          f"seeds = {len(units)} units; {len(done)} done, {len(todo)} to run",
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
                if n_done % 500 == 0:
                    fh.flush()
                    rate = n_done / (time.perf_counter() - t0)
                    print(f"  {n_done}/{len(todo)} ({rate:.1f}/s, "
                          f"ETA {(len(todo)-n_done)/rate/60:.0f}m)", flush=True)
                if (args.max_seconds is not None
                        and time.perf_counter() - t0 > args.max_seconds):
                    fh.flush()
                    print(f"  chunk limit at {n_done}/{len(todo)}; "
                          f"rerun to resume", flush=True)
                    pool.terminate()
                    break

    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/22_factorial.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": PROTOCOL.max_evals,
                         "plateau_tol": PROTOCOL.plateau_tol,
                         "plateau_patience": PROTOCOL.plateau_patience},
            "factors": FACTORS, "off_settings": OFF,
            "base_variant": "fast (repair disabled throughout)",
            "seeds": seeds, "suite": args.suite, "stratify": args.stratify,
            "n_instances": len(paths), "n_configs": len(designs),
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

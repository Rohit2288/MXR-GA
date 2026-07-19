"""Task 6.5a — FH-GA parameter sensitivity sweep. (R2.2, R3.m3)

One-at-a-time (OAT) ranges around the paper defaults for each FH-GA
parameter, plus a p_c0 x p_m0 interaction grid, on a stratified Suite A/U
slice (2 instances per (suite, n) stratum, n <= 500), 10 seeds per
configuration under the shared protocol. Output feeds the
parameter-justification subsection (6.5) with means, SDs, and error bars.

Run AFTER the main queue (shares the machine).

Usage: .venv/bin/python experiments/08_sensitivity.py [--workers 6]
Output: results/sensitivity/results.csv + manifest.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
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
SEEDS = list(range(10))
MAX_N = 500
PER_STRATUM = 2

# OAT ranges (default marked by the "default" config below). pop_size is a
# multiplier on n (the default rule is max(50, 4n)).
OAT = {
    "pop_mult": [2, 4, 8],
    "p_c0": [0.6, 0.7, 0.8, 0.9, 1.0],
    "p_m0_mult": [0.5, 1.0, 2.0, 4.0],
    "gamma": [0.0, 0.25, 0.5, 0.75, 1.0],
    "p_ls": [0.1, 0.2, 0.3, 0.5],
    "w_max": [6, 8, 10, 12, 14],
    "tournament_size": [2, 3, 5, 7],
    "G_stag": [10, 20, 40],
}
INTERACTION = [(pc, pm) for pc in (0.6, 0.8, 1.0) for pm in (0.5, 1.0, 2.0)]


def configs():
    """Yield (config_id, param, value, overrides_dict)."""
    yield ("default", "default", "", {})
    for param, values in OAT.items():
        for v in values:
            yield (f"{param}={v}", param, v, {param: v})
    for pc, pm in INTERACTION:
        yield (f"inter_pc={pc}_pm={pm}", "interaction", f"{pc}x{pm}",
               {"p_c0": pc, "p_m0_mult": pm})


def tuning_slice():
    strata = defaultdict(list)
    for p in iter_instance_paths(FSSP2_DIR):
        if not any(s in str(p) for s in ("suite_A", "suite_U")):
            continue
        inst = load_instance(p)
        if inst.n <= MAX_N:
            strata[(inst.suite, inst.n)].append(p)
    out = []
    for key in sorted(strata):
        out.extend(strata[key][:PER_STRATUM])
    return out


def run_unit(args):
    config_id, param, value, overrides, inst_path, seed = args
    from src.solvers.fhga import FHGA, FHGAConfig
    inst = load_instance(inst_path)
    ov = dict(overrides)
    kw = {}
    if "pop_mult" in ov:
        kw["pop_size"] = max(50, int(ov.pop("pop_mult") * inst.n))
    if "p_m0_mult" in ov:
        kw["p_m0"] = ov.pop("p_m0_mult") / inst.n
    kw.update(ov)
    solver = FHGA(variant="thorough", config=FHGAConfig(**kw),
                  name=f"sens_{config_id}")
    res = solver.solve(inst, PROTOCOL, np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    return {
        "config_id": config_id, "param": param, "value": value,
        "instance_id": inst.instance_id, "suite": inst.suite or "",
        "n": inst.n, "seed": seed, "mu": repr(ev.mu),
        "evals_used": res.evals_used, "wall_time": repr(res.wall_time),
        "stop_reason": res.stop_reason,
    }


FIELDS = ["config_id", "param", "value", "instance_id", "suite", "n",
          "seed", "mu", "evals_used", "wall_time", "stop_reason"]
OUT_DIR = REPO / "results" / "sensitivity"
CSV_PATH = OUT_DIR / "results.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    paths = tuning_slice()
    units = [(cid, param, v, ov, p, s)
             for (cid, param, v, ov) in configs()
             for p in paths for s in SEEDS]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for row in csv.DictReader(fh):
                done.add((row["config_id"], row["instance_id"],
                          int(row["seed"])))
    todo = [u for u in units if (u[0], u[4].stem, u[5]) not in done]
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
            "script": "experiments/08_sensitivity.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "protocol": {"max_evals": PROTOCOL.max_evals,
                         "plateau_tol": PROTOCOL.plateau_tol,
                         "plateau_patience": PROTOCOL.plateau_patience},
            "seeds": SEEDS, "oat": OAT,
            "interaction_grid": INTERACTION,
            "slice": {"suites": ["A", "U"], "max_n": MAX_N,
                      "per_stratum": PER_STRATUM,
                      "n_instances": len(paths)},
        }, fh, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

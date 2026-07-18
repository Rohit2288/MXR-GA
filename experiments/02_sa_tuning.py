"""Task 1.5 — simulated-annealing parameter tuning, documented. (R2.7)

Grid search over (t0, t_end) on a HELD-OUT tuning slice: FSSP-2 instances
with generator seed s00, stratified-capped per (suite, n), evaluated with
RNG seeds 1000-1001 (disjoint from the evaluation seeds 0-19 used in the
main comparison). The selected defaults are recorded here and in
src/solvers/sim_anneal.py.

Output: results/sa_tuning/grid.csv + manifest.json + chosen.json
Usage: .venv/bin/python experiments/02_sa_tuning.py
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.protocol.runner import git_commit
from src.solvers.sim_anneal import SimAnneal

GRID_T0 = [0.5, 0.1, 0.02]
GRID_TEND = [1e-3, 1e-4, 1e-5]
TUNE_SEEDS = [1000, 1001]
BUDGET = ProtocolConfig(max_evals=10_000)
PER_STRATUM = 2          # instances per (suite, n) stratum
MAX_N = 500              # keep tuning tractable; n=1000 excluded (documented)

OUT = Path(__file__).resolve().parents[1] / "results" / "sa_tuning"


def tuning_slice():
    strata = defaultdict(list)
    for p in iter_instance_paths(FSSP2_DIR):
        inst = load_instance(p)
        if inst.metadata["seed"] == 0 and inst.n <= MAX_N:
            strata[(inst.suite, inst.n)].append(inst)
    out = []
    for key in sorted(strata):
        out.extend(strata[key][:PER_STRATUM])
    return out


def main() -> int:
    instances = tuning_slice()
    print(f"tuning slice: {len(instances)} instances "
          f"(seed s00, n<={MAX_N}, {PER_STRATUM}/stratum)")
    rows = []
    t_start = time.perf_counter()
    for t0 in GRID_T0:
        for t_end in GRID_TEND:
            solver = SimAnneal(t0=t0, t_end=t_end)
            mus = []
            for inst in instances:
                for seed in TUNE_SEEDS:
                    res = solver.solve(inst, BUDGET,
                                       np.random.default_rng(seed))
                    mus.append(evaluate(inst, res.subset_indices).mu)
            rows.append({
                "t0": t0, "t_end": t_end,
                "mean_mu": statistics.mean(mus),
                "sd_mu": statistics.stdev(mus),
                "worst_mu": min(mus),
                "runs": len(mus),
            })
            print(f"  t0={t0:<5} t_end={t_end:<7} mean={rows[-1]['mean_mu']:.4f} "
                  f"worst={rows[-1]['worst_mu']:.4f} "
                  f"({time.perf_counter()-t_start:.0f}s)")

    best = max(rows, key=lambda r: r["mean_mu"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "chosen.json", "w") as fh:
        json.dump(best, fh, indent=2)
    with open(OUT / "manifest.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/02_sa_tuning.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "grid_t0": GRID_T0, "grid_t_end": GRID_TEND,
            "tune_seeds": TUNE_SEEDS, "budget": BUDGET.max_evals,
            "slice": {"generator_seed": 0, "max_n": MAX_N,
                      "per_stratum": PER_STRATUM,
                      "n_instances": len(instances)},
        }, fh, indent=2)
    print(f"\nchosen: t0={best['t0']}, t_end={best['t_end']} "
          f"(mean mu {best['mean_mu']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

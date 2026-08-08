"""Tasks 2.8 + 2.9 — structural instance statistics (R2.6, R2.2, R4.1).

Computes, for every Benchmark-FSSP-2 instance, three solver-independent
quantities by Monte-Carlo sampling of random subsets under the SAME
inclusion density every solver uses for random initialisation
(Bernoulli(T_M / sum(items))):

  plateau_mass   P[mu = 0]        -- the share of the sampled space on which
                                     the membership objective is exactly flat.
                                     This is the quantity that strands a
                                     mu-steered metaheuristic (R2.7/R4.3).
  random_mean_mu E[mu]            -- how well a blind sampler does on average.
  rand_best_mu   max mu over K    -- the best a blind sampler finds with the
                                     same order of effort a solver spends.
  attainability_gap
                 mu* - rand_best_mu, where mu* is the exact optimum from
                 dp_exact/mitm where available. This is the structural
                 difficulty axis (R2.6): it measures how much of the optimum
                 is NOT reachable by blind sampling, i.e. how much genuine
                 search an instance demands. An instance with gap ~ 0 cannot
                 discriminate between solvers no matter how many are run --
                 which is precisely why Benchmark-FSSP-1 was uninformative
                 (R2.6, R4.2).

An earlier version of this script used -log10(P[mu >= 0.99]) as the
difficulty axis. It is recorded here as `hardness` but it saturates at the
sampling floor on most FSSP-2 instances (P[mu >= 0.99] is below 1/K almost
everywhere) and therefore does not discriminate; `attainability_gap` is the
measure the paper uses.

Reviewer 2 objected that the difficulty categories are hand-assigned and
unbalanced, and Reviewer 4 that a corpus random search can break is
uninformative. `hardness` answers both: it is computed from the instance
alone, and Section-level analysis (experiments/21) shows it predicts solver
performance across all 16 solvers.

Sampling is exact-arithmetic (integer sums) and seeded; K = 4000 samples per
instance gives a resolution floor of 2.5e-4 on p_near_opt.

Usage: python3 experiments/20_instance_structure.py [--samples 4000]
Output: results/instance_structure/structure.csv + manifest.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.instance import (FSSP2_DIR, iter_instance_paths,  # noqa: E402
                               load_instance)
from src.protocol.runner import git_commit  # noqa: E402

OUT = REPO / "results" / "instance_structure"
FIELDS = ["instance_id", "suite", "n", "distribution", "spread_ratio",
          "asymmetry", "plateau_mass", "random_mean_mu", "rand_best_mu",
          "p_near_opt", "hardness", "mu_star", "attainability_gap", "samples"]


def exact_optima() -> dict:
    """mu* per instance from the exact solvers, where they are feasible."""
    best = {}
    path = REPO / "results" / "final" / "combined.csv"
    if not path.exists():
        return best
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["solver"] in ("dp_exact", "mitm") and r["mu"] != "":
                v = float(r["mu"])
                k = r["instance_id"]
                if k not in best or v > best[k]:
                    best[k] = v
    return best


def structure(inst, k: int, rng: np.random.Generator) -> dict:
    items = np.asarray(inst.items, dtype=np.int64)
    total = int(items.sum())
    density = min(1.0, inst.target.T_M / max(total, 1))
    # k random subsets under the shared initialisation density.
    mask = rng.random((k, inst.n)) < density
    sums = mask @ items                      # exact integer sums
    mu = np.fromiter((inst.target.mu(int(s)) for s in sums),
                     dtype=float, count=k)
    p_near = float((mu >= 0.99).mean())
    floor = 1.0 / k
    return {
        "plateau_mass": float((mu == 0.0).mean()),
        "random_mean_mu": float(mu.mean()),
        "rand_best_mu": float(mu.max()),
        "p_near_opt": p_near,
        "hardness": -math.log10(max(p_near, floor)),
        "samples": k,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    opt = exact_optima()
    print(f"exact optima available for {len(opt)} instances")
    rows = []
    t0 = time.perf_counter()
    for i, path in enumerate(iter_instance_paths(FSSP2_DIR)):
        inst = load_instance(path)
        # Deterministic per-instance seed (hash() is salted per process).
        seed = int.from_bytes(inst.instance_id.encode()[:8].ljust(8, b"\0"),
                              "little") % (2**32)
        s = structure(inst, args.samples, np.random.default_rng(seed))
        mu_star = opt.get(inst.instance_id)
        s.update({
            "instance_id": inst.instance_id, "suite": inst.suite or "",
            "n": inst.n, "distribution": inst.distribution,
            "spread_ratio": inst.metadata.get("spread_ratio", ""),
            "asymmetry": inst.metadata.get("asymmetry", ""),
            "mu_star": "" if mu_star is None else repr(mu_star),
            "attainability_gap": ("" if mu_star is None
                                  else repr(mu_star - s["rand_best_mu"])),
        })
        rows.append(s)
        if (i + 1) % 200 == 0:
            print(f"  {i+1} instances ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    with open(OUT / "structure.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "manifest.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/20_instance_structure.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "samples_per_instance": args.samples,
            "n_instances": len(rows),
            "density_rule": "Bernoulli(T_M / sum(items)), shared with solver init",
        }, fh, indent=2)
    print(f"wrote {OUT/'structure.csv'} ({len(rows)} instances, "
          f"{time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

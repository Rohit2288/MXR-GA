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

from src.core.instance import (FSSP1_DIR, FSSP2_DIR,  # noqa: E402
                               iter_instance_paths, load_instance)
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


def _mu_vec(target, sums: np.ndarray) -> np.ndarray:
    """Vectorised triangular membership, identical to target.mu elementwise.

    The scalar scorer is called once per instance in `structure()` to assert
    agreement, so this fast path cannot drift from src/core/fuzzy.py silently.
    """
    s = sums.astype(float)
    t_l, t_m, t_r = float(target.T_L), float(target.T_M), float(target.T_R)
    d_l = t_m - t_l
    d_r = t_r - t_m
    left = np.divide(s - t_l, d_l, out=np.zeros_like(s),
                     where=(d_l > 0)) if d_l > 0 else (s >= t_l).astype(float)
    right = np.divide(t_r - s, d_r, out=np.zeros_like(s),
                      where=(d_r > 0)) if d_r > 0 else (s <= t_r).astype(float)
    mu = np.where(s <= t_m, left, right)
    return np.clip(mu, 0.0, 1.0)


def structure(inst, k: int, rng: np.random.Generator) -> dict:
    items = np.asarray(inst.items, dtype=np.int64)
    total = int(items.sum())
    density = min(1.0, inst.target.T_M / max(total, 1))
    # k random subsets under the shared initialisation density.
    mask = rng.random((k, inst.n)) < density
    sums = mask @ items                      # exact integer sums
    mu = _mu_vec(inst.target, sums)
    # Agreement check against the canonical scalar scorer on a small probe.
    probe = sums[:8]
    ref = np.array([inst.target.mu(int(x)) for x in probe])
    if not np.allclose(ref, mu[:8], rtol=0, atol=1e-12):
        raise AssertionError(
            f"vectorised mu disagrees with target.mu on {inst.instance_id}")
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
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after N seconds (chunked runs)")
    ap.add_argument("--corpus", choices=["fsspb2", "fsspb1"],
                    default="fsspb2",
                    help="fsspb1 applies the same measure retroactively to "
                         "the saturated first corpus (R2.6, R4.2)")
    args = ap.parse_args()

    corpus_dir = FSSP2_DIR if args.corpus == "fsspb2" else FSSP1_DIR
    out_name = ("structure.csv" if args.corpus == "fsspb2"
                else "structure_fsspb1.csv")

    OUT.mkdir(parents=True, exist_ok=True)
    opt = exact_optima()
    print(f"corpus={args.corpus}; exact optima available for {len(opt)} "
          f"instances")

    # Resumable: the large corpus does not fit in one bounded run.
    out_path = OUT / out_name
    rows = []
    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add(r["instance_id"])
        print(f"resuming: {len(done)} instances already done")

    t0 = time.perf_counter()
    for i, path in enumerate(iter_instance_paths(corpus_dir)):
        if path.stem in done:
            continue
        if args.max_seconds and time.perf_counter() - t0 > args.max_seconds:
            print(f"  chunk limit reached; rerun to resume", flush=True)
            break
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

    with open(OUT / out_name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / f"manifest_{args.corpus}.json", "w") as fh:
        json.dump({
            "git_commit": git_commit(),
            "script": "experiments/20_instance_structure.py",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "corpus": args.corpus,
            "samples_per_instance": args.samples,
            "n_instances": len(rows),
            "density_rule": "Bernoulli(T_M / sum(items)), shared with solver init",
        }, fh, indent=2)
    print(f"wrote {OUT/out_name} ({len(rows)} instances, "
          f"{time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Task 2.15 — a difficulty-targeted instance generator (R2.6).

Reviewer 2 objected that the difficulty categories of Benchmark-FSSP are
hand-assigned and unbalanced, and Reviewer 4 that a corpus which random
search can break is uninformative. Section~2.8 makes that objection precise
with a solver-independent measure, the attainability gap

    Gamma = mu* - max_K mu_random,

the share of the optimum that blind sampling fails to reach. Measuring the
problem is only half an answer: this script *generates* instances that
maximise it, producing a sub-suite on which solvers must actually differ.

Construction (planted solution, tight tolerance)
------------------------------------------------
1. Draw n items from the requested distribution.
2. Choose a planted subset S* uniformly at random among subsets of a target
   cardinality, and set the modal target T_M = sigma(S*). The optimum is
   therefore exactly 1 and is attained by a known subset -- so mu* needs no
   solver and no enumeration, which is what makes generation cheap.
3. Set the tolerances to a small multiple of the *minimum item value* rather
   than of the target. This is the step that drives Gamma up: a subset whose
   sum misses sigma(S*) by even one item scores near zero, so blind sampling
   essentially never lands inside the support.
4. Reject the instance unless the measured Gamma clears a threshold, and
   unless the planted subset is not trivially discoverable (its cardinality
   is not extreme).

The generated instances are written in the frozen v1.1 schema so that every
existing loader, solver and analysis script consumes them unchanged. They
are a NEW sub-suite (suite tag "H"); nothing in Benchmark-FSSP-2 is altered.

Note on what this does and does not claim: a planted optimum makes mu* known
by construction, and exact methods will solve these instances whenever the
pseudo-polynomial table is affordable. The purpose is to discriminate among
*heuristics* on instances where blind sampling is useless, which is exactly
the regime the existing corpus lacks.

Usage:
  python3 experiments/27_generate_hard.py --count 120 --out-dir <dir>
Output: instance JSONs + manifest.json + a summary of achieved Gamma
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.instance import load_instance  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

DEFAULT_OUT = REPO / "benchmarks" / "fsspb_hard"
DISTRIBUTIONS = ("uniform", "geometric", "bimodal", "heavy_tailed")


def draw_items(dist: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if dist == "uniform":
        a = rng.integers(1_000, 100_000, size=n)
    elif dist == "geometric":
        a = np.maximum(1, (rng.geometric(1 / 20_000, size=n))).astype(np.int64)
    elif dist == "bimodal":
        half = n // 2
        a = np.concatenate([rng.integers(1_000, 5_000, size=half),
                            rng.integers(80_000, 100_000, size=n - half)])
        rng.shuffle(a)
    elif dist == "heavy_tailed":
        a = np.maximum(1, (rng.pareto(1.5, size=n) * 20_000)).astype(np.int64)
    else:
        raise ValueError(dist)
    return a.astype(np.int64)


def measure_gamma(items: np.ndarray, t_l: float, t_m: float, t_r: float,
                  k: int, rng: np.random.Generator) -> tuple[float, float]:
    """(Gamma, plateau mass) under the shared initialisation density."""
    total = int(items.sum())
    p = min(1.0, t_m / max(total, 1))
    mask = rng.random((k, items.size)) < p
    sums = (mask @ items).astype(float)
    d_l, d_r = t_m - t_l, t_r - t_m
    left = (sums - t_l) / d_l if d_l > 0 else (sums >= t_l).astype(float)
    right = (t_r - sums) / d_r if d_r > 0 else (sums <= t_r).astype(float)
    mu = np.clip(np.where(sums <= t_m, left, right), 0.0, 1.0)
    # mu* = 1 by construction (the planted subset attains the modal target).
    return 1.0 - float(mu.max()), float((mu == 0.0).mean())


def make_instance(idx: int, dist: str, n: int, tol_mult: float, seed: int,
                  samples: int, gamma_min: float):
    rng = np.random.default_rng(seed)
    items = draw_items(dist, n, rng)
    # Planted subset: a random subset of moderate cardinality.
    card = int(rng.integers(max(2, n // 4), max(3, (3 * n) // 4)))
    planted = np.sort(rng.choice(n, size=card, replace=False))
    t_m = int(items[planted].sum())
    # Tolerances scaled to the SMALLEST item, so a one-item miss is fatal.
    unit = int(items.min())
    d_l = max(1.0, tol_mult * unit)
    d_r = max(1.0, tol_mult * unit)
    t_l, t_r = t_m - d_l, t_m + d_r
    gamma, plateau = measure_gamma(items, t_l, t_m, t_r, samples, rng)
    if gamma < gamma_min:
        return None
    inst_id = f"fsspb-H-{dist}-n{n}-tol{tol_mult:g}-s{seed:03d}"
    raw = {
        "schema_version": "1.1",
        "instance_id": inst_id,
        "metadata": {
            "n": n, "distribution": dist,
            "spread_ratio": float((d_l + d_r) / max(t_m, 1)),
            "asymmetry": 1.0, "seed": seed,
            "construction": {
                "v2_suite": "H",
                "v2_description": "gamma_targeted_planted_solution",
                "v2_axes": {"v2_distribution": dist},
                "generator": "experiments/27_generate_hard.py",
                "tolerance_multiplier_of_min_item": tol_mult,
                "planted_cardinality": int(card),
                "measured_attainability_gap": gamma,
                "measured_plateau_mass": plateau,
                "gamma_samples": samples,
            },
        },
        "items": [int(x) for x in items],
        "target": {"T_L": float(t_l), "T_M": float(t_m), "T_R": float(t_r)},
        "witness": {
            "subset_indices": [int(i) for i in planted],
            "subset_sum": int(t_m),
            "mu_value": 1.0,
        },
        "difficulty": {"attainability_gap": gamma, "plateau_mass": plateau,
                       "label": "gamma_targeted"},
    }
    payload = json.dumps(raw, sort_keys=True).encode()
    raw["checksum_sha256"] = hashlib.sha256(payload).hexdigest()
    return raw, gamma, plateau


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--gamma-min", type=float, default=0.90)
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100, 200])
    ap.add_argument("--tol-mults", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    made, gammas, plateaus, tried = [], [], [], 0
    seed = 0
    combos = [(d, n, t) for d in DISTRIBUTIONS for n in args.sizes
              for t in args.tol_mults]
    while len(made) < args.count and tried < args.count * 50:
        # Cycle over combinations by ATTEMPT, not by success: keying this on
        # len(made) pins the search to one combination when rejection is high.
        dist, n, tol = combos[tried % len(combos)]
        tried += 1
        seed += 1
        out = make_instance(len(made), dist, n, tol, seed, args.samples,
                            args.gamma_min)
        if out is None:
            continue
        raw, gamma, plateau = out
        path = args.out_dir / f"{raw['instance_id']}.json"
        path.write_text(json.dumps(raw, indent=2))
        # Round-trip through the frozen loader: schema + consistency.
        load_instance(path)
        made.append(raw["instance_id"])
        gammas.append(gamma)
        plateaus.append(plateau)

    manifest = {
        "git_commit": git_commit(),
        "script": "experiments/27_generate_hard.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "count": len(made), "attempts": tried,
        "gamma_min": args.gamma_min, "samples": args.samples,
        "sizes": args.sizes, "tol_mults": args.tol_mults,
        "distributions": list(DISTRIBUTIONS),
        "mean_gamma": st.mean(gammas) if gammas else None,
        "min_gamma": min(gammas) if gammas else None,
        "mean_plateau_mass": st.mean(plateaus) if plateaus else None,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"generated {len(made)} instances in {tried} attempts "
          f"-> {args.out_dir}")
    if gammas:
        print(f"  Gamma: mean {st.mean(gammas):.4f}  min {min(gammas):.4f}  "
              f"max {max(gammas):.4f}")
        print(f"  plateau mass: mean {st.mean(plateaus):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

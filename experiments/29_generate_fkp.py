"""Task 2.16 — generate the Fuzzy Knapsack (FKP) corpus.

Items follow the three standard knapsack correlation structures used in the
subset-sum/knapsack literature (Pisinger): uncorrelated, weakly correlated
and strongly correlated. The capacity is a fixed fraction of the total
weight. The profit target is a triangular fuzzy number centred on a
*reachable* profit -- the profit of a feasible greedy-plus-noise subset --
with tolerances set as a fraction of the mean item profit, so the clamping
regime is comparable to Benchmark-FSSP-2's tight-spread suites.

The corpus deliberately spans loose and tight tolerances: the clipped-object
prediction is that plateau mass, and therefore the gain from the shared
unclamped guide, rises with tightness. A corpus with only tight tolerances
could not test that.

Usage: python3 experiments/29_generate_fkp.py --count 120
Output: benchmarks/fkp/*.json + manifest.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.protocol.runner import git_commit  # noqa: E402

OUT = REPO / "benchmarks" / "fkp"
CORRELATIONS = ("uncorrelated", "weakly", "strongly")


def draw(corr: str, n: int, rng: np.random.Generator):
    R = 1000
    w = rng.integers(1, R + 1, size=n)
    if corr == "uncorrelated":
        p = rng.integers(1, R + 1, size=n)
    elif corr == "weakly":
        p = np.maximum(1, w + rng.integers(-R // 10, R // 10 + 1, size=n))
    elif corr == "strongly":
        p = w + R // 10
    else:
        raise ValueError(corr)
    return p.astype(np.int64), w.astype(np.int64)


def plateau_mass(profits, weights, cap, t_l, t_m, t_r, k, rng) -> tuple:
    n = profits.size
    dens = min(1.0, cap / max(int(weights.sum()), 1))
    mask = rng.random((k, n)) < dens
    wt = mask @ weights
    pf = (mask @ profits).astype(float)
    d_l, d_r = t_m - t_l, t_r - t_m
    left = (pf - t_l) / d_l if d_l > 0 else (pf >= t_l).astype(float)
    right = (t_r - pf) / d_r if d_r > 0 else (pf <= t_r).astype(float)
    mu = np.clip(np.where(pf <= t_m, left, right), 0.0, 1.0)
    mu = np.where(wt <= cap, mu, 0.0)
    return float((mu == 0.0).mean()), float(mu.max()), float(mu.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[30, 60, 100])
    ap.add_argument("--tol-fracs", type=float, nargs="+",
                    default=[0.02, 0.05, 0.15, 0.40])
    ap.add_argument("--cap-frac", type=float, default=0.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    combos = [(c, n, t) for c in CORRELATIONS for n in args.sizes
              for t in args.tol_fracs]
    made, plateaus = [], []
    for k in range(args.count):
        corr, n, tol = combos[k % len(combos)]
        seed = 1000 + k
        rng = np.random.default_rng(seed)
        profits, weights = draw(corr, n, rng)
        cap = int(args.cap_frac * int(weights.sum()))
        # Reachable modal profit: greedy by density to capacity, then jitter.
        order = sorted(range(n), key=lambda i: profits[i] / weights[i],
                       reverse=True)
        chosen, w = [], 0
        for i in order:
            if w + int(weights[i]) <= cap:
                chosen.append(i)
                w += int(weights[i])
        drop = rng.choice(len(chosen), size=max(1, len(chosen) // 10),
                          replace=False)
        chosen = [c for j, c in enumerate(chosen) if j not in set(drop)]
        t_m = int(profits[chosen].sum())
        spread = max(1.0, tol * float(profits.mean()))
        t_l, t_r = t_m - spread, t_m + spread
        pm, best, mean_mu = plateau_mass(profits, weights, cap, t_l, t_m, t_r,
                                         args.samples, rng)
        inst_id = f"fkp-{corr}-n{n}-tol{tol:g}-s{seed}"
        raw = {
            "instance_id": inst_id, "n": n, "correlation": corr, "seed": seed,
            "capacity": cap,
            "profits": [int(x) for x in profits],
            "weights": [int(x) for x in weights],
            "target": {"T_L": float(t_l), "T_M": float(t_m), "T_R": float(t_r)},
            "metadata": {
                "generator": "experiments/29_generate_fkp.py",
                "tolerance_fraction_of_mean_profit": tol,
                "capacity_fraction_of_total_weight": args.cap_frac,
                "measured_plateau_mass": pm,
                "measured_rand_best_mu": best,
                "measured_rand_mean_mu": mean_mu,
                "samples": args.samples,
            },
        }
        (OUT / f"{inst_id}.json").write_text(json.dumps(raw, indent=2))
        made.append(inst_id)
        plateaus.append(pm)

    (OUT / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "script": "experiments/29_generate_fkp.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "count": len(made), "sizes": args.sizes,
        "tolerance_fractions": args.tol_fracs,
        "capacity_fraction": args.cap_frac,
        "correlations": list(CORRELATIONS),
        "mean_plateau_mass": st.mean(plateaus),
    }, indent=2))
    print(f"generated {len(made)} FKP instances -> {OUT}")
    print(f"  plateau mass: mean {st.mean(plateaus):.4f}  "
          f"min {min(plateaus):.4f}  max {max(plateaus):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

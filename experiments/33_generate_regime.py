"""Round-3 pilot: a regime-map instance family (suite "R").

Purpose: answer "when should which solver be used?" (AE, R4 round 2) with
instances laid out on the axes that govern subset-sum difficulty, rather
than on hand-picked distributions.

Axes
----
  n      number of items.
  bits   item magnitude: items uniform in [2^(bits-1), 2^bits). Controls
         whether the pseudo-polynomial DP table (n * T_R bits) fits.
  gap    demanded precision, in bits below the item scale: the window
         width is w = dL + dR = 2^(bits - gap), clamped to w >= 2.
         gap = 8 is coarse, gap = 16 fine, gap = "exact" (w = 2, so with
         integer sums mu is 1 only at T_M) asks for an exact hit.

Two blocks, because density n / bits decides whether solutions are rare:
  dense     n in {100..1000}, bits in {16..40}. Here ~2^(n - bits) subsets
            reach every attainable sum, so solutions are plentiful; the
            difficulty is FINDING one when items are huge relative to the
            window and the DP table does not fit.
  critical  n ~ bits (n in {32, 40, 46}), the classical hardest density;
            at gap = exact the planted subset is typically the only hit.
Sums are kept below 2^53 because targets are stored as JSON floats.

Recorded per instance: kappa = log2 of the expected number of subsets in
the window, n + log2(w) - log2(sigma sqrt(2 pi)), sigma = sqrt(sum a^2)/2
(a normal approximation for a uniformly random subset) -- descriptive only.

Construction: planted subset of random cardinality in [n/4, 3n/4];
T_M = its sum, so mu* = 1 is known without any solver. Asymmetry alternates
1:1 and 2:1 (dL:dR). Written in the frozen v1.1 schema; nothing in
Benchmark-FSSP-2 is touched.

Usage:
  .venv/bin/python experiments/33_generate_regime.py --per-cell 2
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.instance import load_instance  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

DEFAULT_OUT = REPO / "benchmarks" / "fsspb_regime"
DENSE_N = [100, 200, 500, 1000]
DENSE_BITS = [16, 24, 32, 40]
DENSE_GAPS = [8, 16, "exact"]
CRIT_N = [32, 40, 46]
CRIT_GAPS = [16, "exact"]

# Confirmatory profile (task 9.7): the hard region only, FRESH seeds so no
# instance overlaps the pilot. dense: bits = 40; critical: n >= 46.
CONFIRM_DENSE = [(n, 40) for n in DENSE_N]
CONFIRM_CRIT = [(46, 38), (46, 46), (46, 47), (48, 40), (48, 47)]


def make_instance(block: str, n: int, bits: int, gap, rep: int, seed: int):
    rng = np.random.default_rng(seed)
    items = rng.integers(2 ** (bits - 1), 2 ** bits, size=n, dtype=np.int64)
    card = int(rng.integers(max(1, n // 4), max(2, (3 * n) // 4) + 1))
    planted = np.sort(rng.choice(n, size=card, replace=False))
    t_m = int(items[planted].sum())
    sigma = math.sqrt(float(np.sum(items.astype(np.float64) ** 2))) / 2
    w = 2.0 if gap == "exact" else max(2.0, 2.0 ** (bits - gap))
    asym = 1.0 if rep % 2 == 0 else 2.0    # dL : dR
    d_l = max(1, int(round(w * asym / (1 + asym))))
    d_r = max(1, int(round(w - d_l)))
    achieved = n + math.log2(d_l + d_r) - math.log2(sigma * math.sqrt(2 * math.pi))
    inst_id = f"fsspb-R-{block}-n{n}-b{bits}-g{gap}-r{rep}"
    raw = {
        "schema_version": "1.1",
        "instance_id": inst_id,
        "metadata": {
            "n": n, "distribution": f"uniform_b{bits}",
            "spread_ratio": float((d_l + d_r) / t_m),
            "asymmetry": asym, "seed": seed,
            "construction": {
                "v2_suite": "R",
                "v2_description": "regime_map_planted_solution",
                "generator": "experiments/33_generate_regime.py",
                "block": block, "bits": bits, "gap": gap,
                "kappa": achieved,
                "planted_cardinality": card,
            },
        },
        "items": [int(x) for x in items],
        "target": {"T_L": float(t_m - d_l), "T_M": float(t_m),
                   "T_R": float(t_m + d_r)},
        "witness": {"subset_indices": [int(i) for i in planted],
                    "subset_sum": t_m, "mu_value": 1.0},
        "difficulty": {"kappa": achieved, "bits": bits, "gap": gap,
                       "label": "regime_map"},
    }
    raw["checksum_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--profile", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--seed-base", type=int, default=330_000)
    args = ap.parse_args()

    inst_dir = args.out_dir / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seed = 0
    if args.profile == "pilot":
        cells = [("dense", n, b, g) for n, b, g in itertools.product(
            DENSE_N, DENSE_BITS, DENSE_GAPS)]
        for n in CRIT_N:
            for b in sorted({n - 8, n, min(n + 6, 47)}):
                for g in CRIT_GAPS:
                    cells.append(("critical", n, b, g))
    else:
        cells = [("dense", n, b, g) for (n, b) in CONFIRM_DENSE
                 for g in DENSE_GAPS]
        cells += [("critical", n, b, g) for (n, b) in CONFIRM_CRIT
                  for g in CRIT_GAPS]
    for block, n, b, g in cells:
        assert n * 2 ** b < 2 ** 53, (n, b)
        for rep in range(args.per_cell):
            seed += 1
            raw = make_instance(block, n, b, g, rep, args.seed_base + seed)
            path = inst_dir / f"{raw['instance_id']}.json"
            path.write_text(json.dumps(raw, indent=1))
            load_instance(path)  # schema + consistency round-trip
            rows.append((raw["instance_id"],
                         raw["metadata"]["construction"]["kappa"]))
    manifest = {
        "git_commit": git_commit(),
        "script": "experiments/33_generate_regime.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "count": len(rows), "per_cell": args.per_cell,
        "profile": args.profile, "seed_base": args.seed_base,
        "dense": {"n": DENSE_N, "bits": DENSE_BITS, "gaps": DENSE_GAPS},
        "critical": {"n": CRIT_N, "bits": "n-8, n, min(n+6, 47)",
                     "gaps": CRIT_GAPS},
        "kappa": dict(rows),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"generated {len(rows)} instances -> {inst_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

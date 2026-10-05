"""Phase 12.3/12.4: pre-registered confirmation corpus for the robustness claim (suite "S").

The claim under test ("robust across item structure; far better than specialists on structured
distributions") was formed AFTER seeing the FSSP-2 results, so it is confirmed here on FRESH,
independently generated instances with pre-registered criteria (REVISION_PLAN Phase 12).

Axes (one instance per combination and rep):
  distribution  uniform (control), loguniform, geometric, heavytail, bimodal, nearequal, avislai;
                all at large magnitude (max item ~2^40)
  n             200, 1000, 5000
  cardinality   "small" (10 planted items) or "half" (n/4 .. 3n/4)
  gap           "exact" (window width 2) or 16 (graded: width 2^(log2 max item - 16))
  asymmetry     dL:dR rotates over reps through 1:1, 1:20, 20:1
Plus a "parity" variant (12.4): all items even, T_M = planted sum + 1 (odd, unreachable), and
dL >= dR, so the optimum is known exactly: mu* = 1 - 1/dL (attained by the planted subset; no odd
sum exists, so nothing reaches T_M itself).

All sums stay below 2^53. mu* is known for every instance (1, or 1 - 1/dL for parity).
Usage: .venv/bin/python experiments/37_generate_structured.py
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

TOP = 2 ** 40
DISTS = ["uniform", "loguniform", "geometric", "heavytail", "bimodal", "nearequal", "avislai"]
SIZES = [200, 1000, 5000]
CARDS = ["small", "half"]
GAPS = ["exact", 16]
ASYM = [(1, 1), (1, 20), (20, 1)]


def draw(dist, n, rng):
    if dist == "uniform":
        a = rng.integers(TOP // 2, TOP, size=n)
    elif dist == "loguniform":
        a = np.exp(rng.uniform(math.log(2 ** 10), math.log(TOP), size=n))
    elif dist == "geometric":
        r = (2 ** 10 / TOP) ** (1 / max(1, n - 1))        # spans 30 bits
        a = TOP * r ** np.arange(n) * rng.uniform(0.98, 1.02, size=n)
    elif dist == "heavytail":
        a = np.minimum(TOP, 2 ** 20 * (rng.pareto(1.2, size=n) + 1))
    elif dist == "bimodal":
        small = rng.integers(2 ** 19, 2 ** 20, size=n // 2)
        big = rng.integers(TOP // 2, TOP, size=n - n // 2)
        a = np.concatenate([small, big]); rng.shuffle(a)
    elif dist == "nearequal":
        base = 2 ** 38
        a = base + rng.integers(-base // 100, base // 100 + 1, size=n)
    elif dist == "avislai":
        a = 2 ** 36 + rng.integers(-2 ** 16, 2 ** 16 + 1, size=n)
    else:
        raise ValueError(dist)
    a = np.maximum(1, np.round(np.asarray(a, dtype=np.float64))).astype(np.int64)
    rng.shuffle(a)
    return a


def make_asym(dist, n, rep, seed, m=2 ** 20):
    """Asymmetric-tolerance demonstration (R4b.1 / R2b.4). Items are multiples of m, so attainable sums
    are spaced m apart. T_M = S + 0.6m (S = planted sum): the nearest attainable sum ABOVE is S + m
    (0.4m away) but dR = 0.3m puts it outside the support (mu = 0); the nearest BELOW is S (0.6m away),
    inside dL = m (mu = 0.4). No other attainable sum lies in the support, so mu* = 0.4 is certified.
    A crisp closest-sum solver prefers S + m and scores 0; the fuzzy target selects S."""
    rng = np.random.default_rng(seed)
    a = m * np.maximum(1, draw(dist, n, rng) // m)
    k = int(rng.integers(n // 4, 3 * n // 4 + 1))
    planted = np.sort(rng.choice(n, size=k, replace=False))
    s = int(a[planted].sum())
    t_m, d_l, d_r = s + (6 * m) // 10, m, (3 * m) // 10
    mu_star = 1.0 - ((6 * m) // 10) / d_l
    assert int(a.sum()) + d_r < 2 ** 53
    iid = f"fsspb-S-asymgap-{dist}-n{n}-r{rep}"
    raw = {
        "schema_version": "1.1", "instance_id": iid,
        "metadata": {"n": n, "distribution": dist, "spread_ratio": float((d_l + d_r) / t_m),
                     "asymmetry": d_l / d_r, "seed": seed,
                     "construction": {"v2_suite": "S", "asymgap": True, "parity": False,
                                      "generator": "experiments/37_generate_structured.py",
                                      "spacing_m": m, "mu_star": mu_star, "mu_star_certified": True}},
        "items": [int(x) for x in a],
        "target": {"T_L": float(t_m - d_l), "T_M": float(t_m), "T_R": float(t_m + d_r)},
        "witness": {"subset_indices": [int(i) for i in planted], "subset_sum": s, "mu_value": mu_star},
        "difficulty": {"label": "asymmetric_tolerance_demo"},
    }
    raw["checksum_sha256"] = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return raw


def make(dist, n, card, gap, rep, seed, parity=False, asym=None):
    rng = np.random.default_rng(seed)
    a = draw(dist, n, rng)
    if parity:
        a = 2 * np.maximum(1, a // 2)
    k = 10 if card == "small" else int(rng.integers(n // 4, 3 * n // 4 + 1))
    planted = np.sort(rng.choice(n, size=k, replace=False))
    s = int(a[planted].sum())
    bits = int(a.max()).bit_length()
    w = 2.0 if gap == "exact" else max(2.0, 2.0 ** (bits - gap))
    al, ar = asym if asym is not None else ASYM[rep % 3]
    d_l = max(1, int(round(w * al / (al + ar))))
    d_r = max(1, int(round(w - d_l)))
    if parity:                                   # unreachable odd apex, dL >= dR
        d_l, d_r = max(d_l, d_r, 2), min(d_l, d_r)
        t_m = s + 1
        mu_star = 1.0 - 1.0 / d_l
        witness_mu = mu_star
    else:
        t_m = s
        witness_mu = 1.0
    assert int(a.sum()) + d_r < 2 ** 53
    iid = (f"fsspb-S-{'parity-' if parity else ''}{dist}-n{n}-{card}-g{gap}"
           f"-a{al}_{ar}-r{rep}")
    raw = {
        "schema_version": "1.1", "instance_id": iid,
        "metadata": {"n": n, "distribution": dist,
                     "spread_ratio": float((d_l + d_r) / t_m), "asymmetry": d_l / d_r,
                     "seed": seed,
                     "construction": {"v2_suite": "S", "parity": parity,
                                      "generator": "experiments/37_generate_structured.py",
                                      "cardinality": card, "planted_k": k, "gap": gap,
                                      "max_item_bits": bits, "mu_star": witness_mu}},
        "items": [int(x) for x in a],
        "target": {"T_L": float(t_m - d_l), "T_M": float(t_m), "T_R": float(t_m + d_r)},
        "witness": {"subset_indices": [int(i) for i in planted], "subset_sum": s,
                    "mu_value": witness_mu},
        "difficulty": {"label": "structured_confirmation"},
    }
    raw["checksum_sha256"] = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "benchmarks" / "fsspb_structured")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=440_000)
    ap.add_argument("--profile", choices=["confirm", "asym"], default="confirm",
                    help="asym: asymmetric-tolerance instances (see make_asym) showing that the fuzzy target "
                         "changes which subset is optimal")
    args = ap.parse_args()
    inst_dir = args.out_dir / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    seed, ids = 0, []
    if args.profile == "confirm":
        cells = [(d, n, c, g, False, None) for d, n, c, g in itertools.product(DISTS, SIZES, CARDS, GAPS)]
        cells += [(d, n, "half", 16, True, None) for d in ("uniform", "loguniform") for n in SIZES]
    else:
        cells = [(d, n, None, None, None, "asymgap") for d in DISTS for n in SIZES]
    for dist, n, card, gap, parity, asym in cells:
        for rep in range(args.reps):
            seed += 1
            if asym == "asymgap":
                raw = make_asym(dist, n, rep, args.seed_base + seed)
            else:
                raw = make(dist, n, card, gap, rep, args.seed_base + seed, parity, asym)
            path = inst_dir / f"{raw['instance_id']}.json"
            path.write_text(json.dumps(raw, indent=1))
            load_instance(path)
            ids.append(raw["instance_id"])
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit(), "script": "experiments/37_generate_structured.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "count": len(ids),
        "distributions": DISTS, "sizes": SIZES, "cardinalities": CARDS, "gaps": GAPS,
        "asymmetry_rotation": ASYM, "reps": args.reps, "seed_base": args.seed_base}, indent=1))
    print(f"generated {len(ids)} instances -> {inst_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

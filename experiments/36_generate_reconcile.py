"""Phase 11: an application-structured suite, payment reconciliation ("P").

Task modelled: a customer payment arrives; which open invoices does it
settle? The ledger holds n open invoices with amounts in CENTS; the payment
equals the total of a hidden set of 5-40 invoices. Banks deduct fees, so a
short payment is acceptable within a fee allowance while an over-payment
is not. That is a triangular target with asymmetric spreads:

    T_M = planted total,  dL = fee allowance,  dR = 1 cent.

The instances are SYNTHETIC; only the structure follows the real task
(amounts spanning orders of magnitude, few invoices per payment, an
asymmetric tolerance). No real ledger data is used, and the paper must say so.

Amounts: lognormal, median $50,000, sigma 1.2, clipped to [$100, $5,000,000];
in cents this is ~2^13 to 2^29, so the largest totals exceed 2^35 and the
DP table does not fit. Axes: n in {200, 1000, 5000}; fee allowance in
{$1, $100}; payment cardinality in {5-15, 20-40}. mu* = 1 (planted).

Usage: .venv/bin/python experiments/36_generate_reconcile.py --per-cell 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.instance import load_instance  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

SIZES = [200, 1000, 5000]
FEES_CENTS = [100, 10_000]
CARDS = [(5, 15), (20, 40)]


def make(n, fee, card_rng, rep, seed):
    rng = np.random.default_rng(seed)
    dollars = np.clip(rng.lognormal(np.log(50_000), 1.2, size=n), 100, 5e6)
    items = np.maximum(1, np.round(dollars * 100)).astype(np.int64)
    card = int(rng.integers(card_rng[0], card_rng[1] + 1))
    planted = np.sort(rng.choice(n, size=card, replace=False))
    t_m = int(items[planted].sum())
    d_l, d_r = fee, 1
    iid = f"fsspb-P-n{n}-fee{fee}-c{card_rng[0]}_{card_rng[1]}-r{rep}"
    raw = {
        "schema_version": "1.1", "instance_id": iid,
        "metadata": {
            "n": n, "distribution": "lognormal_cents",
            "spread_ratio": float((d_l + d_r) / t_m), "asymmetry": d_l / d_r,
            "seed": seed,
            "construction": {
                "v2_suite": "P", "v2_description": "payment_reconciliation_synthetic",
                "generator": "experiments/36_generate_reconcile.py",
                "fee_allowance_cents": fee, "payment_cardinality": card,
                "cardinality_range": list(card_rng),
                "max_item_bits": int(items.max()).bit_length(),
            },
        },
        "items": [int(x) for x in items],
        "target": {"T_L": float(t_m - d_l), "T_M": float(t_m),
                   "T_R": float(t_m + d_r)},
        "witness": {"subset_indices": [int(i) for i in planted],
                    "subset_sum": t_m, "mu_value": 1.0},
        "difficulty": {"label": "reconciliation"},
    }
    raw["checksum_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "benchmarks" / "fsspb_reconcile")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=660_000)
    args = ap.parse_args()
    inst_dir = args.out_dir / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    seed, ids = 0, []
    for n, fee, cr in itertools.product(SIZES, FEES_CENTS, CARDS):
        for rep in range(args.per_cell):
            seed += 1
            raw = make(n, fee, cr, rep, args.seed_base + seed)
            path = inst_dir / f"{raw['instance_id']}.json"
            path.write_text(json.dumps(raw, indent=1))
            load_instance(path)
            ids.append(raw["instance_id"])
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit(), "script": "experiments/36_generate_reconcile.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "count": len(ids), "sizes": SIZES, "fees_cents": FEES_CENTS,
        "cardinalities": CARDS, "per_cell": args.per_cell,
        "seed_base": args.seed_base, "synthetic": True,
    }, indent=1))
    print(f"generated {len(ids)} instances -> {inst_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

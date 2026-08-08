"""Task 2.11 — is the local search's contribution conditional on repair? (R2.2)

Why this exists
---------------
The leave-one-out ablation (experiments/07, Section 5.5) measured the
direction-guided local search as the largest single contributor on Suite U
(-0.069 when removed). The 2^5 factorial (experiments/22) measures its main
effect as null (+0.0015, p = 1.00). The two studies disagree, and the paper
must not quietly report only one of them.

They differ in the base configuration: the leave-one-out was run on the
Thorough variant, with the windowed repair operator active, whereas the
factorial holds repair OFF throughout so that the design covers exactly the
five membership-aware mechanisms. If the local search only pays off when
repair is available, both results are correct and the disagreement is a
sixth-factor interaction lying outside the 2^5 design.

This script tests exactly that: LS on/off, crossed with repair on/off, on a
stratified Suite U slice under the shared protocol.

Result (32 Suite-U instances, 8 per n-stratum, 10 seeds, 2026-08-03):
    repair on   with LS 0.423519  without 0.413581  +0.009938  p=0.028
    repair off  with LS 0.430210  without 0.431219  -0.001010  p=0.893

The local search's contribution is conditional on the repair operator being
active. Reported as such; the 2^5 factorial's scope (repair held off) is
stated explicitly in its caption, and a confirmatory 2^6 design including
repair as a sixth factor is queued for the author machine
(scripts/run_queue10.sh).

Usage: python3 experiments/24_ls_repair_conditional.py [--per-stratum 8]
Output: results/factorial/ls_repair_conditional.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import multiprocessing as mp
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scipy.stats import wilcoxon  # noqa: E402

from src.core.evaluate import evaluate  # noqa: E402
from src.core.instance import (FSSP2_DIR, iter_instance_paths,  # noqa: E402
                               load_instance)
from src.protocol.budget import ProtocolConfig  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)
OUT = REPO / "results" / "factorial" / "ls_repair_conditional.json"


def unit(a):
    variant, ls_on, path, seed = a
    from src.solvers.fhga import FHGA, FHGAConfig
    cfg = FHGAConfig() if ls_on else FHGAConfig(p_ls=0.0)
    inst = load_instance(path)
    res = FHGA(variant=variant, config=cfg).solve(
        inst, PROTOCOL, np.random.default_rng(seed))
    return (variant, ls_on, inst.instance_id,
            evaluate(inst, res.subset_indices).mu)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    strata = collections.defaultdict(list)
    for p in iter_instance_paths(FSSP2_DIR):
        inst = load_instance(p)
        if (inst.suite or "") == "U":
            strata[inst.n].append(p)
    paths = []
    for k in sorted(strata):
        paths.extend(sorted(strata[k])[: args.per_stratum])

    units = [(v, l, p, s)
             for v in ("thorough", "fast") for l in (1, 0)
             for p in paths for s in range(args.seeds)]
    acc = collections.defaultdict(list)
    with mp.Pool(args.workers) as pool:
        for v, l, iid, mu in pool.imap_unordered(unit, units, chunksize=8):
            acc[(v, l, iid)].append(mu)

    per = collections.defaultdict(dict)
    for (v, l, iid), m in acc.items():
        per[(v, l)][iid] = st.mean(m)
    insts = sorted(per[("fast", 1)])

    out = {"n_instances": len(insts), "seeds": args.seeds,
           "git_commit": git_commit(),
           "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "arms": []}
    print(f"{len(insts)} Suite-U instances x {args.seeds} seeds, LS on vs off\n")
    for v, repair in (("thorough", "on"), ("fast", "off")):
        y = [per[(v, 1)][i] for i in insts]
        x = [per[(v, 0)][i] for i in insts]
        p = wilcoxon(x, y).pvalue
        rec = {"variant": v, "repair": repair, "with_ls": st.mean(y),
               "without_ls": st.mean(x), "delta": st.mean(y) - st.mean(x),
               "p": p}
        out["arms"].append(rec)
        print(f"  repair {repair:<3}  with_LS={rec['with_ls']:.6f}  "
              f"without_LS={rec['without_ls']:.6f}  "
              f"delta={rec['delta']:+.6f}  p={p:.2e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

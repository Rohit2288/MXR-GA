"""Task 1.7 — alpha-cut FPTAS overshoot: before/after on Suite U. (R2.5, R3.1)

BEFORE: a faithful replica of the legacy alpha_cut_fptas (returns mu of the
scaled phantom sum s*K — no real subset). AFTER: the rewritten AlphaCut
solver (certified subsets, harness-scored). Both are compared against the
independently re-verified exact optima from task 0.7.

Outputs:
  results/alpha_cut_overshoot/overshoot.csv + manifest.json
  paper_fragments/alpha_cut_spec.tex  (formal spec + before/after numbers)
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.protocol.runner import git_commit
from src.solvers.alpha_cut import AlphaCut

EPS = 0.01
TOL = 1e-12
OUT = Path(__file__).resolve().parents[1] / "results" / "alpha_cut_overshoot"
FRAG = Path(__file__).resolve().parents[1] / "paper_fragments" / "alpha_cut_spec.tex"


# --- faithful replica of the LEGACY solver (bug preserved, for the "before") --

def _legacy_iss_feasible(items, L, R, eps=EPS):
    if not items or R <= 0:
        return None
    n = len(items)
    K = max(1.0, eps * R / max(n, 1))
    scaled_items = [max(1, int(math.floor(a / K))) for a in items]
    scaled_R = int(math.floor(R / K))
    scaled_L = int(math.ceil(L / K))
    if scaled_R <= 0:
        return None
    reachable = 1
    mask = (1 << (scaled_R + 1)) - 1
    for a in scaled_items:
        reachable |= (reachable << a)
        reachable &= mask
    s = scaled_R
    while s >= max(0, scaled_L):
        if (reachable >> s) & 1:
            return s * K          # <-- the bug: phantom scaled sum, no subset
        s -= 1
    return None


def legacy_reported_mu(items, target, eps=EPS) -> float:
    L1, R1 = target.alpha_cut(1.0)
    s = _legacy_iss_feasible(items, L1, R1, eps)
    if s is not None and target.mu(s) >= 1.0 - eps:
        return target.mu(s)
    lo, hi = 0.0, 1.0
    best_sum = None
    for _ in range(max(1, int(math.ceil(math.log2(1.0 / eps))))):
        mid = 0.5 * (lo + hi)
        L_mid, R_mid = target.alpha_cut(mid)
        s = _legacy_iss_feasible(items, L_mid, R_mid, eps)
        if s is not None:
            best_sum = s
            lo = mid
        else:
            hi = mid
    if best_sum is None:
        L0, R0 = target.alpha_cut(eps)
        s = _legacy_iss_feasible(items, L0, R0, eps)
        best_sum = s if s is not None else 0.0
    return target.mu(best_sum)


def main() -> int:
    paths = [p for p in iter_instance_paths(FSSP2_DIR) if "suite_U" in str(p)]
    cfg = ProtocolConfig(max_evals=1000)
    rows = []
    for p in paths:
        inst = load_instance(p)
        opt = exact_optimum(inst).mu
        before = legacy_reported_mu(list(inst.items), inst.target)
        res = AlphaCut(eps=EPS).solve(inst, cfg, np.random.default_rng(0))
        after = evaluate(inst, res.subset_indices).mu
        rows.append({
            "instance_id": inst.instance_id, "n": inst.n,
            "exact_mu": repr(opt),
            "legacy_mu": repr(before),
            "new_mu": repr(after),
            "legacy_overshoot": before > opt + TOL,
            "new_overshoot": after > opt + TOL,
            "legacy_excess": repr(max(0.0, before - opt)),
        })

    n = len(rows)
    n_before = sum(r["legacy_overshoot"] for r in rows)
    n_after = sum(r["new_overshoot"] for r in rows)
    excesses = [float(r["legacy_excess"]) for r in rows if r["legacy_overshoot"]]
    summary = {
        "instances": n,
        "legacy_overshoot_count": n_before,
        "legacy_overshoot_pct": 100.0 * n_before / n,
        "legacy_overshoot_mean_excess": (statistics.mean(excesses)
                                         if excesses else 0.0),
        "legacy_overshoot_max_excess": max(excesses) if excesses else 0.0,
        "new_overshoot_count": n_after,
        "new_overshoot_pct": 100.0 * n_after / n,
        "new_mean_mu": statistics.mean(float(r["new_mu"]) for r in rows),
        "exact_mean_mu": statistics.mean(float(r["exact_mu"]) for r in rows),
        "eps": EPS,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "overshoot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "manifest.json", "w") as fh:
        json.dump({"git_commit": git_commit(),
                   "script": "experiments/03_alpha_cut_overshoot.py",
                   "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                   "summary": summary}, fh, indent=2)
    write_fragment(summary)
    print(json.dumps(summary, indent=2))
    assert n_after == 0, "new solver overshot — impossible by construction"
    return 0


def write_fragment(s: dict) -> None:
    FRAG.parent.mkdir(parents=True, exist_ok=True)
    FRAG.write_text(f"""% Auto-generated by experiments/03_alpha_cut_overshoot.py — do not edit by hand.
% Task 1.7 (R2.5, R3.1): formal specification of the certified alpha-cut FPTAS
% and the before/after overshoot result on Suite U.

\\subsubsection{{Certified $\\alpha$-cut FPTAS: specification}}
\\label{{sec:alpha-cut-spec}}

The $\\alpha$-cut FPTAS is specified by four components.

\\paragraph{{$\\alpha$-grid.}} A binary search on $\\alpha \\in (0,1]$ is
performed with $\\lceil \\log_2(1/\\varepsilon) \\rceil$ iterations
($\\varepsilon = {s['eps']}$), preceded by a feasibility test at
$\\alpha = 1$ and followed, if every iteration failed, by a fallback test at
$\\alpha = \\varepsilon$.

\\paragraph{{Interval construction.}} At level $\\alpha$ the target's
$\\alpha$-cut $[L_\\alpha, R_\\alpha] = [T^L + \\alpha(T^M - T^L),\\;
T^R - \\alpha(T^R - T^M)]$ is used as the feasibility window.

\\paragraph{{Rounding.}} With $n$ items, the scaling constant is
$K = \\max(1, \\varepsilon R_\\alpha / n)$; item $a_i$ is rounded to
$\\tilde a_i = \\max(1, \\lfloor a_i / K \\rfloor)$ and the window to
$[\\lceil L_\\alpha / K \\rceil, \\lfloor R_\\alpha / K \\rfloor]$.

\\paragraph{{Reconstruction and certification.}} A dynamic program over the
scaled sums stores, for every prefix of the item list, the bitset of
reachable scaled sums. The largest reachable scaled sum inside the scaled
window is traced back through these prefix bitsets to an \\emph{{explicit
index set}}. The membership degree credited to the method is computed by the
shared evaluation routine from the \\emph{{true unscaled sum}} of that index
set --- never from the scaled reconstruction. Since every reported value is
the membership of a subset that actually exists, the reported value cannot
exceed the true optimum.

\\subsubsection{{Effect on the Suite~U overshoot}}

The previously reported implementation returned the scaled sum $sK$ of the
feasibility test directly; that value corresponds to no realizable subset,
and recomputing membership from it exceeded the enumerated exact optimum on
{s['legacy_overshoot_count']} of {s['instances']} Suite~U instances
({s['legacy_overshoot_pct']:.1f}\\%), with mean excess
{s['legacy_overshoot_mean_excess']:.4f} and maximum excess
{s['legacy_overshoot_max_excess']:.4f} in membership. The certified rewrite
reproduces the framework of the same method but reports only
subset-certified memberships: on the same {s['instances']} instances the
overshoot count is {s['new_overshoot_count']}
({s['new_overshoot_pct']:.1f}\\%), which holds by construction rather than
empirically. This resolves the contradiction between Sections~3.2.7
and~5.4.3 of the submitted manuscript: the overshoot was an artifact of the
implementation's solution reconstruction, not an inherent property of the
$\\alpha$-cut framework.
""")


if __name__ == "__main__":
    sys.exit(main())

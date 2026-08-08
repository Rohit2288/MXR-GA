"""Task 2.14 check — validate Proposition 1 against the measured plateau mass.

A bound that is never checked is a liability. This script evaluates the
Berry-Esseen plateau-mass bound

    Pi >= 1 - (delta_L + delta_R)/(s*sqrt(2*pi)) - 2*C*rho/s^3,
    s^2 = p(1-p) sum a_i^2,   rho = p sum a_i^3,   C = 0.56,

on every Benchmark-FSSP-2 instance and compares it with the plateau mass
measured by experiments/20. It reports both the number of violations (there
should be none) and the number of instances on which the bound is
non-vacuous, since a bound that holds only by being negative says nothing.

Recorded result (2026-08-03, 4000 samples per instance):
    866 instances checked, 0 violations
    non-vacuous on 264/360 Suite A, 77/122 Suite T, 0/384 Suite U
The Berry-Esseen term scales as rho/s^3 and degrades at small n, which is
why the bound is uninformative on Suite U (n = 15..22). This is stated in
the paper rather than left for a reader to discover.

Output: results/instance_structure/plateau_bound.json
Usage: python3 experiments/26_plateau_bound_check.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.instance import (FSSP2_DIR, iter_instance_paths,  # noqa: E402
                               load_instance)

STRUCT = REPO / "results" / "instance_structure" / "structure.csv"
OUT = REPO / "results" / "instance_structure" / "plateau_bound.json"
C_BE = 0.56


def main() -> int:
    measured = {r["instance_id"]: (float(r["plateau_mass"]), r["suite"])
                for r in csv.DictReader(open(STRUCT))}
    per_suite = {}
    violations = []
    for path in iter_instance_paths(FSSP2_DIR):
        inst = load_instance(path)
        if inst.instance_id not in measured:
            continue
        a = np.asarray(inst.items, dtype=float)
        total = a.sum()
        t = inst.target
        p = min(1.0, t.T_M / max(total, 1))
        s = math.sqrt(p * (1 - p) * float(np.sum(a ** 2)))
        if s <= 0:
            continue
        rho = p * float(np.sum(a ** 3))
        bound = (1.0
                 - (t.delta_L + t.delta_R) / (s * math.sqrt(2 * math.pi))
                 - 2 * C_BE * rho / s ** 3)
        obs, suite = measured[inst.instance_id]
        rec = per_suite.setdefault(suite, {"n": 0, "nonvacuous": 0,
                                           "bounds": [], "observed": []})
        rec["n"] += 1
        rec["bounds"].append(bound)
        rec["observed"].append(obs)
        if bound > 0:
            rec["nonvacuous"] += 1
        if obs < bound - 1e-9:
            violations.append({"instance_id": inst.instance_id,
                               "bound": bound, "observed": obs})

    summary = {"constant_C": C_BE, "violations": violations,
               "n_violations": len(violations), "per_suite": {}}
    for suite, rec in sorted(per_suite.items()):
        summary["per_suite"][suite] = {
            "n": rec["n"], "nonvacuous": rec["nonvacuous"],
            "mean_bound": st.mean(rec["bounds"]),
            "mean_observed_plateau_mass": st.mean(rec["observed"]),
        }
        print(f"Suite {suite}: n={rec['n']:<4} non-vacuous="
              f"{rec['nonvacuous']:<4} mean bound={st.mean(rec['bounds']):+.3f}"
              f"  mean observed Pi={st.mean(rec['observed']):.3f}")
    print(f"\nviolations: {len(violations)}")
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())

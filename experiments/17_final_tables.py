"""Task 6.9 — final table/figure pack with ALL solvers merged.

Merges the main comparison with the FLSSS, FH-GA v2, and FH-GA v2-crisp
sweeps into results/final/combined.csv, then regenerates the per-suite
stats/tables (05), the runtime table (11), and all figures (13) from the
combined data, plus a dedicated v2/v2crisp verdict block:

  * paired Wilcoxon fhga_v2 vs fhga_thorough (does simplification cost
    anything?) and fhga_v2crisp vs every FH-GA arm, per suite;
  * results/final/v2_verdict.json with the numbers for the narrative.

Runs unattended as queue8 after the v2crisp sweep.

Usage: .venv/bin/python experiments/17_final_tables.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.stats.analysis import cliffs_delta, cliffs_magnitude

FINAL = REPO / "results" / "final"
SOURCES = [
    REPO / "results" / "main_comparison" / "results.csv",
    REPO / "results" / "flsss" / "results.csv",
    REPO / "results" / "fhga_v2" / "results.csv",
    REPO / "results" / "fhga_v2crisp" / "results.csv",
]
EXTENDED_ORDER = ["fhga_fast", "fhga_thorough", "fhga_v2", "fhga_v2crisp",
                  "vanilla_ga", "random_search", "sim_anneal", "ils",
                  "greedy_center", "alpha_cut", "flsss",
                  "dp_exact", "mitm", "cpsat"]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    frames = []
    for src in SOURCES:
        if src.exists():
            frames.append(pd.read_csv(src))
        else:
            print(f"WARNING: missing {src} — final pack incomplete",
                  flush=True)
    df = pd.concat(frames, ignore_index=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    combined = FINAL / "combined.csv"
    df.to_csv(combined, index=False)
    print(f"combined: {len(df)} rows, solvers: {sorted(df.solver.unique())}",
          flush=True)

    # Regenerate per-suite stats/tables and runtime from the combined CSV.
    m05 = load_module("05_stats_report")
    m05.RESULTS = combined
    m05.OUT = FINAL / "analysis"
    m05.SOLVER_ORDER = EXTENDED_ORDER
    m05.main()

    m11 = load_module("11_runtime_report")
    m11.RESULTS = combined
    m11.OUT = FINAL / "analysis"
    m11.SOLVER_ORDER = EXTENDED_ORDER
    m11.main()

    m13 = load_module("13_figures")
    m13.MAIN = combined
    m13.SOLVER_ORDER = EXTENDED_ORDER
    m13.LABELS = {s: s.replace("_", " ").replace("fhga", "FH-GA")
                  for s in EXTENDED_ORDER}
    m13.main()

    # v2 / v2crisp verdicts.
    ok = df[df["status"] == "ok"].copy()
    ok["mu"] = ok["mu"].astype(float)
    seed_mean = (ok.groupby(["suite", "solver", "instance_id"])["mu"]
                 .mean().reset_index())
    verdict = {}
    for suite in ("T", "A", "U"):
        sub = seed_mean[seed_mean["suite"] == suite]
        mats = {s: dict(zip(g["instance_id"], g["mu"]))
                for s, g in sub.groupby("solver")}
        block = {}
        for a, b in [("fhga_v2", "fhga_thorough"),
                     ("fhga_v2", "fhga_fast"),
                     ("fhga_v2crisp", "fhga_v2"),
                     ("fhga_v2crisp", "fhga_fast"),
                     ("fhga_v2crisp", "fhga_thorough")]:
            if a not in mats or b not in mats:
                continue
            common = sorted(set(mats[a]) & set(mats[b]))
            xa = np.array([mats[a][i] for i in common])
            xb = np.array([mats[b][i] for i in common])
            d = xa - xb
            p = 1.0 if np.allclose(d, 0) else float(
                sps.wilcoxon(xa, xb, zero_method="wilcox").pvalue)
            delta = cliffs_delta(xa, xb)
            block[f"{a}_vs_{b}"] = {
                "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
                "mean_diff": float(d.mean()), "p": p,
                "cliffs_delta": delta,
                "magnitude": cliffs_magnitude(delta),
                "n": len(common),
            }
        verdict[suite] = block
    with open(FINAL / "v2_verdict.json", "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

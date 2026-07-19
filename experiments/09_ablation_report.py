"""Tasks 3.1-3.3 — ablation analysis + WBF cost-benefit + DP-window verdict.

Joins results/ablation/results.csv with the main comparison's fhga_thorough
("full") and fhga_fast ("no_repair") rows on Suites A and U, then per suite:

  * per-variant mean/SD/worst of per-instance seed-mean mu + 95% CI
    (t-interval across instances);
  * Wilcoxon signed-rank of each variant vs "full", Bonferroni-Holm across
    variants within a suite, Cliff's delta with magnitude (3.1);
  * WBF-Repair cost-benefit: quality delta (full vs no_repair) against
    repair eval/time overhead, from the per-run instrumentation, plus the
    scaling pattern of repair time by n (3.2);
  * DP-window repair vs full: quality delta and cost (3.3).

Outputs: results/ablation/analysis/*.{csv,json}
Fragments: paper_fragments/table_ablation.tex, table_wbf_cost.tex
Usage: .venv/bin/python experiments/09_ablation_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.stats.analysis import (cliffs_delta, cliffs_magnitude,
                                holm_correction)

ABLATION = REPO / "results" / "ablation" / "results.csv"
MAIN = REPO / "results" / "main_comparison" / "results.csv"
OUT = REPO / "results" / "ablation" / "analysis"
FRAG = REPO / "paper_fragments"
SUITES = ["A", "U"]
VARIANT_ORDER = ["full", "no_init", "no_fb_xover", "no_impact", "no_ls",
                 "no_controller", "no_repair", "dp_window"]


def load() -> pd.DataFrame:
    ab = pd.read_csv(ABLATION)
    main = pd.read_csv(MAIN)
    main = main[(main["status"] == "ok")
                & main["suite"].isin(SUITES)
                & main["solver"].isin(["fhga_thorough", "fhga_fast"])].copy()
    main["solver"] = main["solver"].map(
        {"fhga_thorough": "full", "fhga_fast": "no_repair"})
    df = pd.concat([ab[ab["status"] == "ok"], main], ignore_index=True)
    df["mu"] = df["mu"].astype(float)
    df["wall_time"] = df["wall_time"].astype(float)
    return df


def per_instance(df, suite):
    sub = df[df["suite"] == suite]
    return {v: g.groupby("instance_id")["mu"].mean()
            for v, g in sub.groupby("solver")}


def analyze(df):
    rows, tests = [], []
    for suite in SUITES:
        mats = per_instance(df, suite)
        full = mats["full"]
        raw_p = []
        pending = []
        for v in VARIANT_ORDER:
            if v not in mats:
                continue
            x = mats[v]
            common = full.index.intersection(x.index)
            xv, xf = x.loc[common].values, full.loc[common].values
            ci = sps.t.interval(0.95, len(xv) - 1, loc=xv.mean(),
                                scale=sps.sem(xv)) if len(xv) > 1 else (np.nan,) * 2
            sub = df[(df["suite"] == suite) & (df["solver"] == v)]
            rows.append({
                "suite": suite, "variant": v,
                "n_instances": len(xv),
                "mean_mu": xv.mean(), "sd_mu": xv.std(ddof=1),
                "ci95_lo": ci[0], "ci95_hi": ci[1],
                "worst_instance_mean": xv.min(),
                "worst_single_run": sub["mu"].min(),
                "mean_wall_time": sub["wall_time"].mean(),
                "delta_vs_full": xv.mean() - xf.mean(),
            })
            if v != "full":
                d = xv - xf
                p = 1.0 if np.allclose(d, 0) else float(
                    sps.wilcoxon(xv, xf, zero_method="wilcox").pvalue)
                raw_p.append(p)
                pending.append((suite, v, xv, xf, d))
        for (s, v, xv, xf, d), p, ph in zip(pending, raw_p,
                                            holm_correction(raw_p)):
            delta = cliffs_delta(xv, xf)
            tests.append({
                "suite": s, "variant": v, "p_raw": p, "p_holm": ph,
                "cliffs_delta": delta,
                "magnitude": cliffs_magnitude(delta),
                "median_diff_vs_full": float(np.median(d)),
            })
    return pd.DataFrame(rows), pd.DataFrame(tests)


def wbf_cost(df):
    """3.2/3.3: repair overhead vs quality, per suite and per n."""
    out = []
    for suite in SUITES:
        for variant in ("full", "dp_window"):
            sub = df[(df["suite"] == suite) & (df["solver"] == variant)]
            if sub.empty:
                continue
            extras = sub["extra"].apply(json.loads)
            count_key, time_key = (("wbf_repair_count", "wbf_repair_time_s")
                                   if variant == "full" else
                                   ("dpwin_repair_count", "dpwin_repair_time_s"))
            out.append({
                "suite": suite, "variant": variant,
                "mean_repairs_per_run": float(np.mean(
                    [e.get(count_key, 0) for e in extras])),
                "mean_repair_time_s": float(np.mean(
                    [e.get(time_key, 0.0) for e in extras])),
                "mean_wall_time_s": sub["wall_time"].mean(),
                "repair_time_share": float(
                    np.sum([e.get(time_key, 0.0) for e in extras])
                    / max(sub["wall_time"].sum(), 1e-12)),
                "mean_evals": sub["evals_used"].mean(),
            })
    by_n = []
    full = df[df["solver"] == "full"]
    for (suite, n), sub in full.groupby(["suite", "n"]):
        extras = sub["extra"].apply(json.loads)
        by_n.append({
            "suite": suite, "n": n,
            "mean_repairs_per_run": float(np.mean(
                [e.get("wbf_repair_count", 0) for e in extras])),
            "mean_repair_time_s": float(np.mean(
                [e.get("wbf_repair_time_s", 0.0) for e in extras])),
            "mean_wall_time_s": sub["wall_time"].mean(),
        })
    return pd.DataFrame(out), pd.DataFrame(by_n)


def latex_ablation(rows: pd.DataFrame, tests: pd.DataFrame) -> str:
    lines = [
        r"% Auto-generated by experiments/09_ablation_report.py — do not edit by hand.",
        r"% Task 3.1 (R2.8, R3.5): mechanism ablation on Suites A and U,",
        r"% 20 seeds, shared protocol. p-values Holm-adjusted within suite.",
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Mechanism ablation on Suites A and U. Each variant",
        r"  disables one mechanism of FH-GA-Thorough; dp\_window replaces",
        r"  WBF-Repair's enumeration with an exact window DP. Mean $\mu$",
        r"  over per-instance seed-means with 95\% CI, worst case, and",
        r"  Wilcoxon vs.\ the full configuration (Holm-adjusted) with",
        r"  Cliff's $\delta$.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{llrrrrl}", r"    \toprule",
        r"    Suite & variant & mean $\mu$ [95\% CI] & worst inst. &"
        r" $\Delta$ vs full & $p_{\mathrm{Holm}}$ & $\delta$ (magn.) \\",
        r"    \midrule",
    ]
    for suite in SUITES:
        for v in VARIANT_ORDER:
            r = rows[(rows["suite"] == suite) & (rows["variant"] == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            t = tests[(tests["suite"] == suite) & (tests["variant"] == v)]
            if v == "full":
                ptxt, dtxt = "--", "--"
            else:
                t = t.iloc[0]
                ptxt = f"{t['p_holm']:.2e}"
                dtxt = f"{t['cliffs_delta']:+.3f} ({t['magnitude']})"
            name = v.replace("_", r"\_")
            lines.append(
                f"    {suite} & {name} & {r['mean_mu']:.6f} "
                f"[{r['ci95_lo']:.6f}, {r['ci95_hi']:.6f}] & "
                f"{r['worst_instance_mean']:.6f} & "
                f"{r['delta_vs_full']:+.6f} & {ptxt} & {dtxt} \\\\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_wbf(cost: pd.DataFrame, by_n: pd.DataFrame) -> str:
    lines = [
        r"% Auto-generated by experiments/09_ablation_report.py — do not edit by hand.",
        r"% Task 3.2 (R2.9) + 3.3 (R4.1): repair cost-benefit.",
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Repair cost on Suites A and U under the shared",
        r"  protocol: WBF-Repair (full) vs.\ the exact DP-window variant.",
        r"  Repair-time share is the fraction of total wall-clock spent",
        r"  inside repair.}",
        r"  \label{tab:wbf-cost}",
        r"  \begin{tabular}{llrrrr}", r"    \toprule",
        r"    Suite & repair & fires/run & repair s/run & wall s/run &"
        r" share \\", r"    \midrule",
    ]
    for _, r in cost.iterrows():
        name = "WBF" if r["variant"] == "full" else r"DP\_window"
        lines.append(
            f"    {r['suite']} & {name} & {r['mean_repairs_per_run']:.1f} & "
            f"{r['mean_repair_time_s']:.3f} & {r['mean_wall_time_s']:.3f} & "
            f"{100 * r['repair_time_share']:.1f}\\% \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", "",
              r"% Scaling of WBF repair time by n (for the running text):"]
    for _, r in by_n.iterrows():
        lines.append(f"%   suite {r['suite']} n={int(r['n'])}: "
                     f"{r['mean_repairs_per_run']:.1f} fires/run, "
                     f"{r['mean_repair_time_s']:.3f}s repair, "
                     f"{r['mean_wall_time_s']:.3f}s wall")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tests = analyze(df)
    rows.to_csv(OUT / "ablation_summary.csv", index=False)
    tests.to_csv(OUT / "ablation_tests.csv", index=False)
    cost, by_n = wbf_cost(df)
    cost.to_csv(OUT / "repair_cost.csv", index=False)
    by_n.to_csv(OUT / "repair_cost_by_n.csv", index=False)
    (FRAG / "table_ablation.tex").write_text(latex_ablation(rows, tests))
    (FRAG / "table_wbf_cost.tex").write_text(latex_wbf(cost, by_n))
    print(rows.to_string(index=False))
    print(tests.to_string(index=False))
    print("fragments written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

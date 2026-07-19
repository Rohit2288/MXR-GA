"""Task 6.8 — regenerate all data figures from the new CSVs. (R1.2, R4.4)

Each figure is generated only if its source CSV exists, and every figure
writes an audit JSON with the exact plotted values next to the image
(the "audited against its CSV" requirement). Sizes target the sn-jnl
column (88 mm single, 180 mm double); PDF for the manuscript + PNG preview.

Design rules: identity by position (one bar per solver) with the FH-GA arms
accented; multi-series cases use the colorblind-safe Okabe-Ito subset with
distinct markers and direct labels (grayscale-safe); residuals are shown on
a log scale per R4's requirement, with exact hits clipped to the axis floor
and their count reported in the audit + caption note.

Usage: .venv/bin/python experiments/13_figures.py
Output: paper_fragments/figures/*.{pdf,png} + audit_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAIN = REPO / "results" / "main_comparison" / "results.csv"
ABL_SUM = REPO / "results" / "ablation" / "analysis" / "ablation_summary.csv"
REPAIR_N = REPO / "results" / "ablation" / "analysis" / "repair_cost_by_n.csv"
SENS = REPO / "results" / "sensitivity" / "results.csv"
OUT = REPO / "paper_fragments" / "figures"

SUITES = ["T", "A", "U"]
SOLVER_ORDER = ["fhga_fast", "fhga_thorough", "vanilla_ga", "random_search",
                "sim_anneal", "ils", "greedy_center", "alpha_cut",
                "dp_exact", "mitm", "cpsat"]
LABELS = {s: s.replace("_", " ").replace("fhga", "FH-GA") for s in SOLVER_ORDER}
ACCENT = "#0072B2"      # FH-GA arms
NEUTRAL = "#9a9a9a"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
SINGLE_W, DOUBLE_W = 3.46, 7.08
RESIDUAL_FLOOR = 1e-7

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "figure.dpi": 200,
})


def save(fig, name, audit):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    with open(OUT / f"audit_{name}.json", "w") as fh:
        json.dump(audit, fh, indent=2)
    plt.close(fig)
    print(f"wrote {name} (+audit)")


def per_instance_means(df):
    return (df.groupby(["suite", "solver", "instance_id"])["mu"]
            .mean().reset_index())


def fig_quality(df):
    """Replaces Figure 5: per-suite mean mu per solver, horizontal bars.

    One fixed row order across all panels (missing solver -> gap), no
    in-plot value labels (full precision lives in the table fragment) —
    both direct answers to R4's 'overlapping labels' complaint."""
    pim = per_instance_means(df)
    solvers = [s for s in SOLVER_ORDER if s in set(pim["solver"])]
    y = np.arange(len(solvers))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.8), sharey=True)
    audit = {}
    for ax, suite in zip(axes, SUITES):
        sub = pim[pim["suite"] == suite]
        stats = (sub.groupby("solver")["mu"].agg(["mean", "std"])
                 .reindex(solvers))
        colors = [ACCENT if s.startswith("fhga") else NEUTRAL
                  for s in solvers]
        ax.barh(y, stats["mean"].fillna(0), xerr=stats["std"].fillna(0),
                height=0.62, color=colors,
                error_kw={"lw": 0.7, "capsize": 1.5})
        ax.set_xlim(0, 1.05)
        ax.set_title(f"Suite {suite}")
        audit[suite] = {s: ({"mean": float(m), "sd": float(sd)}
                            if np.isfinite(m) else "absent")
                        for s, m, sd in zip(solvers, stats["mean"],
                                            stats["std"])}
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[s] for s in solvers])
    axes[1].set_xlabel(r"mean $\mu$ (per-instance seed means)")
    save(fig, "fig_quality_by_suite", audit)


def fig_residual_log(df):
    """Replaces Figure 6: log-scale residual (1 - mu) boxplots."""
    pim = per_instance_means(df)
    solvers = [s for s in SOLVER_ORDER if s in set(pim["solver"])]
    positions = np.arange(1, len(solvers) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.8), sharey=True)
    audit = {"note": f"residuals below {RESIDUAL_FLOOR:g} are clipped to the "
                     "axis floor; per-solver counts in n_below_floor. "
                     "State this in the figure caption."}
    for ax, suite in zip(axes, SUITES):
        sub = pim[pim["suite"] == suite]
        audit[suite] = {}
        for pos, s in zip(positions[::-1], solvers):
            r = 1.0 - sub[sub["solver"] == s]["mu"].values
            audit[suite][s] = {
                "median_residual": float(np.median(r)) if len(r) else None,
                "n_below_floor": int((r < RESIDUAL_FLOOR).sum()),
                "n_instances": len(r)}
            if not len(r):
                continue
            ax.boxplot([np.clip(r, RESIDUAL_FLOOR, None)], vert=False,
                       positions=[pos], widths=0.6,
                       flierprops={"markersize": 1.5, "alpha": 0.4},
                       medianprops={"color": ACCENT, "lw": 1.2},
                       boxprops={"lw": 0.7}, whiskerprops={"lw": 0.7},
                       capprops={"lw": 0.7})
        ax.set_xscale("log")
        ax.set_xlim(RESIDUAL_FLOOR * 0.8, 1.5)
        ax.set_title(f"Suite {suite}")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([LABELS[s] for s in solvers[::-1]])
    axes[0].set_ylim(0.4, len(solvers) + 0.6)
    axes[1].set_xlabel(r"residual $1-\mu$ (log scale)")
    save(fig, "fig_residual_log", audit)


def fig_ablation():
    d = pd.read_csv(ABL_SUM)
    variants = [v for v in ["no_init", "no_fb_xover", "no_impact", "no_ls",
                            "no_controller", "no_repair", "dp_window"]
                if v in set(d["variant"])]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.4), sharey=True)
    audit = {}
    for ax, suite in zip(axes, ["A", "U"]):
        sub = d[d["suite"] == suite].set_index("variant")
        y = np.arange(len(variants))[::-1]
        full_mean = sub.loc["full", "mean_mu"]
        deltas = [sub.loc[v, "mean_mu"] - full_mean for v in variants]
        ci_lo = [sub.loc[v, "ci95_lo"] - full_mean for v in variants]
        ci_hi = [sub.loc[v, "ci95_hi"] - full_mean for v in variants]
        ax.errorbar(deltas, y,
                    xerr=[np.array(deltas) - np.array(ci_lo),
                          np.array(ci_hi) - np.array(deltas)],
                    fmt="o", ms=3.5, color=ACCENT, lw=0.9, capsize=2)
        ax.axvline(0, color="#555", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([v.replace("_", " ") for v in variants])
        ax.set_title(f"Suite {suite}")
        ax.set_xlabel(r"$\Delta$ mean $\mu$ vs full (95% CI)")
        audit[suite] = {v: {"delta": float(dl), "ci_lo": float(lo + full_mean),
                            "ci_hi": float(hi + full_mean)}
                        for v, dl, lo, hi in zip(variants, deltas, ci_lo,
                                                 ci_hi)}
    save(fig, "fig_ablation", audit)


def fig_sensitivity():
    d = pd.read_csv(SENS)
    d["mu"] = d["mu"].astype(float)
    oat = d[~d["param"].isin(["default", "interaction"])]
    default_mean = d[d["param"] == "default"]["mu"].mean()
    params = sorted(oat["param"].unique())
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE_W, 1.9 * nrow))
    audit = {"default_mean_mu": float(default_mean)}
    for ax, param in zip(axes.flat, params):
        sub = oat[oat["param"] == param]
        stats = sub.groupby("value")["mu"].agg(["mean", "sem", "count"])
        try:
            xs = stats.index.astype(float)
            order = np.argsort(xs)
        except (TypeError, ValueError):
            xs = np.arange(len(stats))
            order = np.arange(len(stats))
        x = np.asarray(xs)[order]
        m = stats["mean"].values[order]
        e = 1.96 * stats["sem"].values[order]
        ax.errorbar(x, m, yerr=e, fmt="o-", ms=3, lw=1, color=ACCENT,
                    capsize=2)
        ax.axhline(default_mean, color="#999", lw=0.7, ls="--")
        ax.set_title(param)
        audit[param] = {str(v): {"mean": float(mm), "ci95_half": float(ee)}
                        for v, mm, ee in zip(np.asarray(xs)[order], m, e)}
    for ax in axes.flat[len(params):]:
        ax.axis("off")
    fig.supylabel(r"mean $\mu$ (A/U slice)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_sensitivity", audit)


def fig_wbf_share():
    d = pd.read_csv(REPAIR_N)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.3))
    audit = {}
    for (suite, sub), color, marker in zip(d.groupby("suite"), OKABE,
                                           ["o", "s"]):
        sub = sub.sort_values("n")
        share = sub["mean_repair_time_s"] / sub["mean_wall_time_s"].clip(1e-12)
        ax.plot(sub["n"], 100 * share, marker=marker, ms=3.5, lw=1.2,
                color=color, label=f"Suite {suite}")
        ax.annotate(f"Suite {suite}", (sub["n"].iloc[-1],
                                       100 * share.iloc[-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=6,
                    color=color)
        audit[suite] = {str(int(n)): float(s)
                        for n, s in zip(sub["n"], share)}
    ax.set_xscale("log")
    ax.set_xlabel("instance size $n$")
    ax.set_ylabel("WBF repair share of wall-clock (%)")
    ax.legend(frameon=False)
    save(fig, "fig_wbf_share", audit)


def main() -> int:
    made = 0
    if MAIN.exists():
        df = pd.read_csv(MAIN)
        df = df[df["status"] == "ok"].copy()
        df["mu"] = df["mu"].astype(float)
        fig_quality(df)
        fig_residual_log(df)
        made += 2
    if ABL_SUM.exists():
        fig_ablation()
        made += 1
    if SENS.exists():
        fig_sensitivity()
        made += 1
    if REPAIR_N.exists():
        fig_wbf_share()
        made += 1
    print(f"{made} figures generated; missing sources skipped gracefully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

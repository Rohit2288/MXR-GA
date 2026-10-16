"""Generate every table and figure of the new manuscript from the clean (<= 4-worker) runs (Phase 19).

Outputs (paper_fragments/r3/):
  tables/*.tex   booktabs LaTeX for Overleaf (sn-jnl); the method name is the macro \\algname
  figures/*.pdf  vector figures sized for the journal text width (+ .png previews)
  csv/*.csv      the exact numbers behind every table
  review.html    one page with every table and figure, for review before writing
Every number is computed here from results/*/results.csv; nothing is typed by hand.
Success = mu >= certified optimum - 1e-12 (strict). Ties in W/T/L use |delta| <= 1e-9.

Usage: .venv/bin/python experiments/40_tables_figures.py [--name "Proposed"]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.protocol.runner import git_commit  # noqa: E402

R = REPO / "results"
OUT = REPO / "paper_fragments" / "r3"
ALG = "Proposed"

# Display names; the proposed method is \algname in LaTeX and ALG in figures.
NAMES = {
    "ma_final": "PROPOSED", "ma_final_rand": "PROPOSED, random free set", "lns_final": "PROPOSED, no population",
    "ma_final_cpsat": "PROPOSED, CP-SAT sub-solver", "ma_final_sym": "PROPOSED, crisp closest sum",
    "ktree": "k-tree", "lattice": "Lattice (BKZ)", "flsss": "FLSSS", "flsss_exact": "FLSSS-exact",
    "ckk": "CKK", "kk": "KK", "cpsat": "CP-SAT", "dp_exact": "DP (exact)", "mitm": "MITM (exact)",
    "ils_crisp": "ILS (guided)", "sim_anneal_crisp": "SA (guided)", "vanilla_ga_crisp": "GA (guided)",
    "fhga_v2crisp": "FH-GA (prior)", "random_search": "Random search", "alpha_cut": r"$\alpha$-cut FPTAS",
    "greedy_center": "Greedy",
}
ORDER = ["ma_final", "ktree", "lattice", "flsss", "flsss_exact", "ckk", "kk", "cpsat", "dp_exact", "mitm",
         "fhga_v2crisp", "ils_crisp", "sim_anneal_crisp", "vanilla_ga_crisp", "random_search", "alpha_cut",
         "greedy_center"]
STOCH = {"ma_final", "ma_final_rand", "lns_final", "ma_final_cpsat", "ma_final_sym", "ktree", "ils_crisp",
         "sim_anneal_crisp", "vanilla_ga_crisp", "fhga_v2crisp", "random_search"}
# Validated categorical slots (dataviz reference palette, light): 1 blue, 2 orange, 3 aqua; gray for the rest.
C_PROP, C2, C3, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"


def tex_name(s):
    n = NAMES.get(s, s)
    return n.replace("PROPOSED", r"\algname")


def fig_name(s):
    return NAMES.get(s, s).replace("PROPOSED", ALG)


def load(*dirs, drop=("ma_xr", "ma_xr_sym", "ma_rand", "lns_mitm", "ma_xr_cpsat")):
    frames = []
    for d in dirs:
        p = R / d / "results.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    d = pd.concat(frames, ignore_index=True)
    d["solver"] = d["solver"].str.replace(r"@.*", "", regex=True)
    d = d[~d["solver"].isin(drop)]
    d["mu0"] = d["mu"].fillna(0.0)
    d["ok"] = d["status"].eq("ok")
    d["hit"] = (d["known_opt"].notna() & (d["mu0"] >= d["known_opt"] - 1e-12)).astype(float)
    d = d.drop_duplicates(["solver", "instance_id", "seed"])
    return d


def cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def magnitude(x):
    x = abs(x)
    return "negligible" if x < 0.147 else "small" if x < 0.33 else "medium" if x < 0.474 else "large"


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def fmt(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def sci(p):
    if p is None or np.isnan(p):
        return "--"
    if p < 1e-300:
        return r"$<10^{-300}$"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


# ----------------------------------------------------------------------------------------------- tables
TABLES = {}


def save_table(key, df, caption, label, colspec, header, rows_tex, note=""):
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "csv" / f"{key}.csv", index=False)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{adjustbox}{max width=\textwidth}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += rows_tex
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(rf"\par\smallskip\footnotesize {note}")
    lines.append(r"\end{table}")
    (OUT / "tables" / f"{key}.tex").write_text("\n".join(lines) + "\n")
    TABLES[key] = (caption, df, note)


def per_solver_summary(d, opt_known=True):
    """Per solver: mean / SD over instances of seed-mean mu, worst instance, worst run, success, zero-rate,
    coverage, runs, median wall, median evals."""
    rows = []
    n_inst = d["instance_id"].nunique()
    for s, g in d.groupby("solver"):
        gi = g[g["ok"]]
        im = gi.groupby("instance_id")["mu0"].mean()
        hit = g.groupby("instance_id")["hit"].mean()
        known = g[g["known_opt"].notna()]["instance_id"].nunique()
        rows.append({
            "solver": s, "coverage": gi["instance_id"].nunique() / n_inst,
            "mean_mu": im.mean(), "sd": im.std(ddof=1) if len(im) > 1 else 0.0,
            "worst_inst": im.min(), "worst_run": gi["mu0"].min(),
            "success": hit[g[g["known_opt"].notna()]["instance_id"].unique()].mean() if known else np.nan,
            "zero_rate": (gi["mu0"] == 0).mean(), "runs": len(g), "seeds": g["seed"].nunique(),
            "med_wall_s": gi["wall_time"].median(), "med_evals": gi["evals_used"].median()})
    out = pd.DataFrame(rows)
    out["_o"] = out["solver"].map({s: i for i, s in enumerate(ORDER)}).fillna(99)
    return out.sort_values("_o").drop(columns="_o")


def summary_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append(f"{tex_name(r.solver)} & {pct(r.coverage)} & {fmt(r.mean_mu)} & {fmt(r.sd, 3)} & "
                    f"{fmt(r.worst_inst, 3)} & {fmt(r.worst_run, 3)} & {pct(r.success)} & {pct(r.zero_rate)} & "
                    f"{int(r.seeds)} \\\\")
    return rows


SUM_HDR = (r"Method & Cov.\,\% & Mean $\mu$ & SD & Worst inst. & Worst run & Success\,\% & "
           r"$\mu{=}0$\,\% & Seeds")


def t_fssp2():
    d = load("p16_fssp2", "p18_fssp2_final", "p15_fssp2_runtime", "p18_fssp2_kt_ils", "p18_fssp2_meta",
             "p18_fssp2_single")
    allrows, stat_rows, wtl = [], [], []
    for suite in ("T", "A", "U"):
        q = d[d["suite"] == suite]
        s = per_solver_summary(q)
        s.insert(0, "suite", suite)
        allrows.append(s)
        # pairwise vs proposed on per-instance seed means (common instances)
        pim = q[q["ok"]].groupby(["instance_id", "solver"])["mu0"].mean().unstack()
        full = [c for c in pim.columns if pim[c].notna().all()]
        fr = friedmanchisquare(*[pim[c].values for c in full]) if len(full) > 2 else None
        ps, rr = [], []
        for c in [c for c in ORDER if c in pim.columns and c != "ma_final"]:
            common = pim[["ma_final", c]].dropna()
            x = common["ma_final"] - common[c]
            p = float(wilcoxon(x[x.abs() > 1e-12]).pvalue) if (x.abs() > 1e-12).sum() >= 10 else 1.0
            ps.append(p)
            rr.append({"suite": suite, "vs": c, "n": len(common), "delta_mean": float(x.mean()),
                       "cliff": cliff(common["ma_final"], common[c]),
                       "W": int((x > 1e-9).sum()), "T": int((x.abs() <= 1e-9).sum()), "L": int((x < -1e-9).sum())})
        for r_, pa in zip(rr, holm(ps)):
            r_["p_holm"] = pa
            r_["magnitude"] = magnitude(r_["cliff"])
        stat_rows += rr
        stat_rows.append({"suite": suite, "vs": "FRIEDMAN", "n": len(pim),
                          "delta_mean": np.nan, "cliff": np.nan, "W": len(full), "T": 0, "L": 0,
                          "p_holm": float(fr.pvalue) if fr else np.nan, "magnitude": ""})
    summ = pd.concat(allrows)
    rows = []
    for suite in ("T", "A", "U"):
        n = d[d["suite"] == suite]["instance_id"].nunique()
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{Suite {suite}}} ({n} instances)}} \\")
        rows += summary_rows(summ[summ["suite"] == suite])
        if suite != "U":
            rows.append(r"\midrule")
    save_table("t_fssp2_quality", summ,
               r"Benchmark-FSSP-2, per sub-suite. Mean $\mu$ and SD over instances of the per-instance seed mean; "
               r"worst instance (lowest seed mean) and worst single run; success = share of instances whose "
               r"seed-mean run reaches the certified optimum (instances with a certified optimum only); "
               r"coverage = share of instances on which the method returns a subset. Every run under the shared "
               r"protocol (certified scoring, equal wall-clock budget by size, 4 performance cores).",
               "tab:fssp2", "lrrrrrrrr", SUM_HDR, rows,
               r"FLSSS-exact spends its budget on the exact-hit probe and is therefore weak where $\mu^\star<1$ "
               r"(Suites A, U). DP and MITM abstain where their memory envelope is exceeded (coverage).")
    st = pd.DataFrame(stat_rows)
    rows = []
    for suite in ("T", "A", "U"):
        f = st[(st["suite"] == suite) & (st["vs"] == "FRIEDMAN")].iloc[0]
        rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{Suite {suite}}}: Friedman over {int(f.W)} full-coverage "
                    rf"methods, $p$ = {sci(f.p_holm)}}} \\")
        for _, r_ in st[(st["suite"] == suite) & (st["vs"] != "FRIEDMAN")].iterrows():
            rows.append(f"{tex_name(r_.vs)} & {int(r_.n)} & {r_.delta_mean:+.4f} & {r_.cliff:+.3f} & "
                        f"{r_['magnitude']} & {r_['W']}/{r_['T']}/{r_['L']} & {sci(r_['p_holm'])} \\\\")
        if suite != "U":
            rows.append(r"\midrule")
    save_table("t_fssp2_stats", st,
               r"Benchmark-FSSP-2: \algname{} against every method (per-instance seed means, common instances). "
               r"$\Delta$ = mean difference in $\mu$; Cliff's $\delta$ with magnitude; win/tie/loss (tie: "
               r"$|\Delta|\le10^{-9}$); Wilcoxon signed-rank, Holm-adjusted within each sub-suite. Positive "
               r"values favour \algname.",
               "tab:fssp2stats", "lrrrlrr",
               r"vs. & $n$ & $\Delta$ mean $\mu$ & Cliff's $\delta$ & Magnitude & W/T/L & $p_{\mathrm{Holm}}$",
               rows)
    rt = d[d["ok"]].groupby(["suite", "solver"]).agg(med_wall_s=("wall_time", "median"),
                                                     p90_wall_s=("wall_time", lambda x: x.quantile(0.9)),
                                                     med_evals=("evals_used", "median")).reset_index()
    rt["_o"] = rt["solver"].map({s: i for i, s in enumerate(ORDER)}).fillna(99)
    rt = rt.sort_values(["suite", "_o"]).drop(columns="_o")
    rows = []
    for suite in ("T", "A", "U"):
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{Suite {suite}}}}} \\")
        for _, r_ in rt[rt["suite"] == suite].iterrows():
            rows.append(f"{tex_name(r_.solver)} & {r_.med_wall_s:.3f} & {r_.p90_wall_s:.3f} & {int(r_.med_evals)} \\\\")
        if suite != "U":
            rows.append(r"\midrule")
    save_table("t_runtime", rt,
               r"Cost per run on Benchmark-FSSP-2 under the shared protocol: median and 90th-percentile "
               r"wall-clock seconds, and median counted evaluations. Runs stop at the budget or at a certified "
               r"optimum (identical rule for every method). Exact and sub-solver-based methods do uncounted "
               r"internal work, so wall-clock is the comparable cost; hardware: Apple M1, 4 performance cores.",
               "tab:runtime", "lrrr", r"Method & Median s & 90th pct.\ s & Median evals", rows)
    return d


def t_simple(key, dirs, caption, label, note="", by=None):
    d = load(*dirs)
    s = per_solver_summary(d)
    save_table(key, s, caption, label, "lrrrrrrrr", SUM_HDR, summary_rows(s), note)
    return d


def t_regime():
    d = load("p16_regime", "p13_regime", "p13_regime_lattice", "p18_regime")
    x = d["instance_id"].str.extract(r"R-(\w+)-n(\d+)-b(\d+)")
    d["block"], d["n"], d["bits"] = x[0], x[1].astype(int), x[2].astype(int)
    inst = d.groupby(["block", "n", "bits", "instance_id", "solver"])["hit"].mean().reset_index()
    cell = inst.groupby(["block", "n", "bits", "solver"])["hit"].mean().unstack()
    cols = [c for c in ["ma_final", "ktree", "lattice", "flsss_exact", "flsss", "ckk", "kk", "cpsat", "mitm",
                        "dp_exact", "fhga_v2crisp", "ils_crisp", "vanilla_ga_crisp", "sim_anneal_crisp",
                        "random_search"] if c in cell.columns]
    cell = cell[cols]
    show = ["ma_final", "ktree", "lattice", "flsss_exact", "ckk", "cpsat", "mitm", "fhga_v2crisp", "ils_crisp"]
    rows = []
    for (b, n, bits), r_ in cell.iterrows():
        best = r_[show].max()
        vals = []
        for c in show:
            v = r_[c]
            s_ = pct(v)
            vals.append(rf"\textbf{{{s_}}}" if (not np.isnan(v) and v >= best - 1e-9 and best > 0) else s_)
        rows.append(f"{b} & {n} & {bits} & " + " & ".join(vals) + r" \\")
    out = cell.reset_index()
    save_table("t_regime", out,
               r"Regime map: success (\%) per cell (block, $n$, item bits), 10\,s per run, 5 seeds for stochastic "
               r"methods. Best per row in bold. \emph{dense}: $n\gg$ bits; \emph{critical}: $n\approx$ bits.",
               "tab:regime", "lrr" + "r" * len(show),
               "Block & $n$ & Bits & " + " & ".join(tex_name(c) for c in show), rows,
               r"Full table with every method in the CSV. SA, the guided GA and random search succeed almost only on "
               r"the 16-bit dense cells (at most 3.3\% elsewhere).")
    return d, cell


def t_structured():
    out = {}
    for tag, mine, comp in (("A", "p16_structured", "p13_structured"), ("B", "p16_structured_rep", "p14_structured_rep")):
        d = load(mine, comp)
        dn = d[~d["instance_id"].str.contains("parity")]
        inst = dn.groupby(["distribution", "instance_id", "solver"]).agg(mu=("mu0", "mean"), hit=("hit", "mean")).reset_index()
        mu = inst.pivot_table(index="distribution", columns="solver", values="mu")
        hit = inst.pivot_table(index="distribution", columns="solver", values="hit")
        par = d[d["instance_id"].str.contains("parity")]
        par = par.assign(gap=par["known_opt"] - par["mu0"])
        out[tag] = (mu, hit, par.groupby("solver").agg(success=("hit", "mean"), gap=("gap", "mean")))
    comp_cols = ["ktree", "ckk", "kk", "flsss", "flsss_exact", "cpsat", "lattice"]
    rows, recs = [], []
    for tag in ("A", "B"):
        mu, hit, _ = out[tag]
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{Corpus {tag}}} (fresh, {'seed base 440000' if tag == 'A' else 'seed base 123000'})}} \\")
        for dist in [x for x in mu.index if x != "uniform"] + ["uniform"]:
            comp = [c for c in comp_cols if c in mu.columns]
            best = mu.loc[dist, comp].max()
            who = mu.loc[dist, comp].idxmax()
            r1 = mu.loc[dist, "ma_final"] >= best - 0.01
            recs.append({"corpus": tag, "distribution": dist, "proposed_mu": mu.loc[dist, "ma_final"],
                         "proposed_success": hit.loc[dist, "ma_final"], "best_other_mu": best, "best_other": who,
                         "ktree_success": hit.loc[dist, "ktree"], "R1_pass": bool(r1) if dist != "uniform" else None})
            mark = ("pass" if r1 else "fail") if dist != "uniform" else "(control)"
            rows.append(f"{dist} & {fmt(mu.loc[dist, 'ma_final'])} & {pct(hit.loc[dist, 'ma_final'])} & "
                        f"{fmt(best)} ({tex_name(who)}) & {pct(hit.loc[dist, 'ktree'])} & {mark} \\\\")
        st = [x for x in hit.index if x != "uniform"]
        r2 = 100 * (hit.loc[st, "ma_final"] - hit.loc[st, "ktree"]).mean()
        rows.append(rf"\multicolumn{{6}}{{l}}{{R2: \algname{{}} minus k-tree success over the six structured "
                    rf"distributions $= {r2:+.1f}$ points (criterion $\geq 20$)}} \\")
        if tag == "A":
            rows.append(r"\midrule")
    save_table("t_structured", pd.DataFrame(recs),
               r"Pre-registered confirmation on structured item distributions (two fresh corpora, 30\,s, 5 seeds "
               r"for stochastic methods). R1: on every non-uniform distribution \algname's mean $\mu$ is within "
               r"0.01 of the best competitor. R2: \algname{} exceeds k-tree's success by $\geq 20$ points on average.",
               "tab:structured", "lrrlrl",
               r"Distribution & \algname{} $\mu$ & \algname{} succ.\,\% & Best other $\mu$ & k-tree succ.\,\% & R1",
               rows)
    prow, precs = [], []
    for tag in ("A", "B"):
        par = out[tag][2]
        for s_, r_ in par.iterrows():
            precs.append({"corpus": tag, "solver": s_, "success": r_["success"], "mean_gap": r_["gap"]})
    pdf = pd.DataFrame(precs).pivot_table(index="solver", columns="corpus", values=["success", "mean_gap"])
    for s_ in [c for c in ORDER if c in pdf.index]:
        prow.append(f"{tex_name(s_)} & {pct(pdf.loc[s_, ('success', 'A')])} & {fmt(pdf.loc[s_, ('mean_gap', 'A')], 4)} & "
                    f"{pct(pdf.loc[s_, ('success', 'B')])} & {fmt(pdf.loc[s_, ('mean_gap', 'B')], 4)} \\\\")
    save_table("t_parity", pd.DataFrame(precs),
               r"Unreachable apex: parity instances (even items, odd $T^M$), whose optimum $\mu^\star=1-1/\delta_L$ is "
               r"proven. Success = reaching $\mu^\star$; gap = mean $\mu^\star-\mu$.",
               "tab:parity", "lrrrr", r"Method & Succ.\,\% (A) & Gap (A) & Succ.\,\% (B) & Gap (B)", prow)
    return out


def t_gate_and_ablation():
    d = load("p16_gateG", "p16_ablation", "p18_hard_eval", drop=())
    d["block"] = d["instance_id"].str.split("-").str[2]
    d["tto"] = np.where(d["mu0"] >= d["known_opt"] - 1e-12, d["wall_time"], np.inf)
    keep = ["ma_final", "ma_final_rand", "lns_final", "ma_final_cpsat", "ktree", "lattice", "flsss_exact",
            "ckk", "kk", "cpsat"]
    d = d[d["solver"].isin(keep)]
    inst = d.groupby(["instance_id", "solver"]).agg(hit=("hit", "mean"), mu=("mu0", "mean"), tto=("tto", "median")).reset_index()
    H = inst.pivot(index="instance_id", columns="solver", values="hit")
    T = inst.pivot(index="instance_id", columns="solver", values="tto")
    M = inst.pivot(index="instance_id", columns="solver", values="mu")
    blk = d.drop_duplicates("instance_id").set_index("instance_id")["block"]
    rows, recs = [], []
    ctrl = ["ma_final_rand", "lns_final", "ma_final_cpsat"]
    ps = []
    for c in keep:
        if c not in H.columns:
            continue
        rec = {"solver": c, "success": H[c].mean(), "succ_critical": H[c][blk == "critical"].mean(),
               "succ_dense": H[c][blk == "dense"].mean(), "mean_mu": M[c].mean(), "sd": M[c].std(),
               "worst_inst": M[c].min()}
        if c != "ma_final":
            diff = (H["ma_final"] != H[c]) | (T["ma_final"] != T[c])
            rec["tto_cliff"] = cliff(T["ma_final"][diff].replace(np.inf, 1e9), T[c][diff].replace(np.inf, 1e9)) if diff.any() else 0.0
            x = H["ma_final"] - H[c]
            rec["p_raw"] = float(wilcoxon(x[x != 0]).pvalue) if (x != 0).sum() >= 10 else 1.0
            ps.append(rec["p_raw"])
        recs.append(rec)
    adj = iter(holm(ps))
    for rec in recs:
        if rec["solver"] != "ma_final":
            rec["p_holm"] = next(adj)
    for rec in recs:
        rows.append(f"{tex_name(rec['solver'])} & {pct(rec['success'])} & {pct(rec['succ_critical'])} & "
                    f"{pct(rec['succ_dense'])} & {fmt(rec['mean_mu'])} & {fmt(rec['sd'], 3)} & "
                    f"{fmt(rec['worst_inst'], 3)} & {('--' if rec['solver'] == 'ma_final' else f'{rec['tto_cliff']:+.3f}')} & "
                    f"{('--' if rec['solver'] == 'ma_final' else sci(rec['p_holm']))} \\\\")
        if rec["solver"] == "ma_final_cpsat":
            rows.append(r"\midrule")
    save_table("t_gate_ablation", pd.DataFrame(recs),
               r"Fresh hard corpus (88 instances never used for tuning; 30\,s; 5 seeds). Upper block: the "
               r"pre-registered ablation (Gate G: population and exact recombination; sub-solver). Lower block: "
               r"competitors. Time-to-optimum Cliff's $\delta$ (negative = \algname{} faster) on instances where the "
               r"pair differs; Wilcoxon on per-instance success, Holm-adjusted. Lattice reduction runs for $n\le150$ only "
               r"(dense cells with $n>150$ count as unsolved). MITM is omitted: every instance "
               r"exceeds its memory envelope ($n>44$), so it abstains rather than fails.",
               "tab:gate", "lrrrrrrrr",
               r"Method & Succ.\,\% & Critical & Dense & Mean $\mu$ & SD & Worst & TTO $\delta$ & $p_{\mathrm{Holm}}$",
               rows)
    return d


def t_tuning():
    d = load("p16_tuning", drop=())
    d["k"] = d["solver"].str.extract(r"k=(\d+)")[0]
    x = pd.read_csv(R / "p16_tuning" / "results.csv")
    x["hit"] = (x["mu"].fillna(0) >= x["known_opt"] - 1e-12).astype(float)
    x["k"] = x["solver"].str.extract(r"k=(\d+)")[0].astype(int)
    x["init"] = x["solver"].str.extract(r"init=(\d)")[0].astype(int)
    x["move"] = x["solver"].str.extract(r"move=(\d)")[0].astype(int)
    x["set"] = np.where(x["instance_id"].str.contains("-S-"), "structured", "hard")
    inst = x.groupby(["k", "init", "move", "set", "instance_id"])["hit"].mean().reset_index()
    g = inst.groupby(["k", "init", "move", "set"])["hit"].mean().unstack()
    g["overall"] = inst.groupby(["k", "init", "move"])["hit"].mean()
    g["ci95"] = inst.groupby(["k", "init", "move"])["hit"].apply(lambda v: 1.96 * v.std() / np.sqrt(len(v)))
    g = g.reset_index().sort_values(["k", "init", "move"])
    rows = []
    for _, r_ in g.iterrows():
        sel = r"\textbf{selected}" if (r_.k == 36 and r_.init == 0 and r_.move == 0) else ""
        rows.append(f"{int(r_.k)} & {'yes' if r_.init else 'no'} & {'yes' if r_.move else 'no'} & {pct(r_.hard)} & "
                    f"{pct(r_.structured)} & {pct(r_.overall)} $\\pm$ {100 * r_.ci95:.1f} & {sel} \\\\")
    save_table("t_tuning", g,
               r"Design choices, decided on a separate tuning set (88 hard + 90 structured instances, never used "
               r"for evaluation; 3 seeds; 30\,s). Pre-registered rule: highest mean success; configurations within "
               r"2 points go to the simplest (fewer components, then smaller $k$).",
               "tab:tuning", "rllrrrl",
               r"$k$ & Greedy start & Post-move & Hard \% & Structured \% & Overall \% (95\% CI) & ", rows)
    p = pd.read_csv(R / "p17_popsens" / "results.csv")
    p["hit"] = (p["mu"].fillna(0) >= p["known_opt"] - 1e-12).astype(float)
    p["pop"] = p["solver"].str.extract(r"pop=(\d+)")[0].astype(int)
    pi = p.groupby(["pop", "instance_id"])["hit"].mean().reset_index()
    pg = pi.groupby("pop")["hit"].agg(["mean", "std", "count"]).reset_index()
    pg["ci95"] = 1.96 * pg["std"] / np.sqrt(pg["count"])
    pg["med_wall_s"] = p.groupby("pop")["wall_time"].median().values
    rows = [f"{int(r_['pop'])} & {pct(r_['mean'])} $\\pm$ {100 * r_['ci95']:.1f} & {r_['med_wall_s']:.2f} \\\\"
            for _, r_ in pg.iterrows()]
    save_table("t_popsize", pg,
               r"Population-size sensitivity of the final configuration (tuning set, 3 seeds, 30\,s).",
               "tab:pop", "rrr", r"Population & Success \% (95\% CI) & Median s", rows)
    return g, pg


def t_asym():
    d = load("p16_asym", "p15_asym")
    s = per_solver_summary(d)
    save_table("t_asym", s,
               r"Asymmetric-tolerance instances (42; certified $\mu^\star=0.4$): the nearest attainable sum lies "
               r"above $T^M$ but outside the narrow excess tolerance, so a crisp closest-sum objective lands on "
               r"$\mu=0$. \algname, crisp closest sum: the same algorithm minimising $|\sigma-T^M|$.",
               "tab:asym", "lrrrrrrrr", SUM_HDR, summary_rows(s))
    return s


def t_speed():
    d = load("p18_speed120")
    d["grp"] = np.where(d["instance_id"].str.contains("geometric"), "structured geometric",
                        np.where(d["instance_id"].str.contains("scale"), "scale n5000 b32", "dense n1000 b32"))
    ref = load("p16_structured", "p16_regime", "p16_scale")
    ref = ref[(ref["solver"] == "ma_final") & ref["instance_id"].isin(d["instance_id"].unique())].copy()
    ref["grp"] = np.where(ref["instance_id"].str.contains("geometric"), "structured geometric",
                          np.where(ref["instance_id"].str.contains("scale"), "scale n5000 b32", "dense n1000 b32"))
    t = pd.concat([d, ref]).groupby(["solver", "grp"])["hit"].mean().unstack()
    t = t.reindex([c for c in ORDER if c in t.index])
    rows = [f"{tex_name(s_)} & " + " & ".join(pct(t.loc[s_, g]) for g in t.columns) + r" \\" for s_ in t.index]
    save_table("t_speed", t.reset_index(),
               r"Speed equalisation: success (\%) of the Python-implemented baselines with 120\,s per run (4--12$\times$ "
               r"their main-experiment budgets), against \algname{} at its main budget (10--30\,s). If slowness "
               r"explained the gaps, the baselines would close them here.",
               "tab:speed", "l" + "r" * len(t.columns), "Method & " + " & ".join(t.columns), rows)
    return t


# ----------------------------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans"})


def savefig(fig, name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f_regime(cell):
    show = ["ma_final", "ktree", "lattice", "flsss_exact", "ckk", "cpsat", "mitm", "fhga_v2crisp", "ils_crisp"]
    m = cell[show].copy()
    labels = [f"{b} n={n} {bits}b" for (b, n, bits) in m.index]
    fig, ax = plt.subplots(figsize=(6.5, 7.2))
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(show)), [fig_name(c) for c in show], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("success (share of instances at the certified optimum)")
    savefig(fig, "fig_regime_map")


def f_structured(out):
    mu, hit, _ = out["A"]
    dists = [x for x in hit.index if x != "uniform"] + ["uniform"]
    series = [("ma_final", C_PROP, None), ("ktree", C2, "//"), ("ckk", C3, ".."), ("flsss_exact", C_GRAY, "xx")]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    w = 0.2
    for i, (s_, col, hatch) in enumerate(series):
        xs = np.arange(len(dists)) + (i - 1.5) * w
        vals_ = [hit.loc[dd, s_] for dd in dists]
        ax.bar(xs, vals_, width=w - 0.02, color=col, hatch=hatch,
               edgecolor="white", linewidth=0.4, label=fig_name(s_))
        for x_, v_ in zip(xs, vals_):
            if v_ < 0.005:
                ax.text(x_, 0.01, "0", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(dists)), dists)
    ax.set_ylabel("success")
    ax.set_ylim(0, 1.05)
    ax.axvline(len(dists) - 1.5, color=INK2, lw=0.6, ls=":")
    ax.text(len(dists) - 1.45, 1.0, "control", fontsize=7, color=INK2)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    savefig(fig, "fig_structured")


def f_anytime():
    d = load("p16_gateG", "p18_hard_eval", "p16_structured", "p13_structured", drop=())
    keep = {"ma_final": (C_PROP, "-"), "ktree": (C2, "--"), "lattice": (C3, "-."), "flsss_exact": (C_GRAY, ":")}
    d = d[d["solver"].isin(keep)]
    d["panel"] = np.where(d["instance_id"].str.contains("critical"), "hard corpus: critical",
                          np.where(d["instance_id"].str.contains("dense"), "hard corpus: dense",
                                   np.where(d["instance_id"].str.contains("-S-geometric|-S-bimodal|-S-nearequal"),
                                            "structured (geometric, bimodal, near-equal)", None)))
    d = d[d["panel"].notna()]
    grid = np.logspace(-3, np.log10(30), 60)
    panels = ["hard corpus: dense", "hard corpus: critical", "structured (geometric, bimodal, near-equal)"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for ax, pnl in zip(axes, panels):
        for s_, (col, ls) in keep.items():
            g = d[(d["panel"] == pnl) & (d["solver"] == s_)]
            if g.empty:
                continue
            curves = []
            for _, r_ in g.iterrows():
                tr = json.loads(r_["extra"]).get("trace", []) if isinstance(r_["extra"], str) else []
                ts = np.array([t[0] for t in tr]) if tr else np.array([np.inf])
                ms = np.array([t[2] for t in tr]) if tr else np.array([0.0])
                opt = r_["known_opt"]
                hitc = ms >= opt - 1e-12
                first = ts[hitc].min() if hitc.any() else np.inf
                curves.append(grid >= first)
            ax.plot(grid, np.mean(curves, axis=0), color=col, ls=ls, lw=1.6, label=fig_name(s_))
        ax.set_xscale("log")
        ax.set_title(pnl, fontsize=7, loc="left", color=INK)
        ax.set_xlabel("seconds")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("share at certified optimum")
    axes[0].legend(fontsize=6.5, loc="upper left")
    savefig(fig, "fig_anytime")


def f_residual(fd):
    show = ["ma_final", "flsss", "fhga_v2crisp", "ils_crisp", "ckk", "ktree", "cpsat", "random_search"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True)
    for ax, suite in zip(axes, ("T", "A", "U")):
        q = fd[(fd["suite"] == suite) & fd["ok"]]
        im = q.groupby(["instance_id", "solver"])["mu0"].mean().unstack()
        opt = q.groupby("instance_id")["known_opt"].first()
        data = []
        for s_ in show:
            if s_ not in im:
                data.append([])
                continue
            ref = opt.reindex(im.index).fillna(1.0)
            r_ = (ref - im[s_]).clip(lower=0).dropna() if suite == "U" else (1 - im[s_]).dropna()
            data.append(np.maximum(r_.values, 1e-7))
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 2, "markeredgecolor": C_GRAY},
                        medianprops={"color": INK, "linewidth": 1.2})
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(C_PROP if show[i] == "ma_final" else "#d9d8d2")
            b.set_edgecolor(INK2)
        ax.set_xscale("log")
        ax.set_xlim(5e-8, 1.5)
        ax.set_title(f"Suite {suite}", loc="left", fontsize=8, color=INK)
        ax.set_xlabel("gap to optimum" if suite == "U" else "residual 1 - mu")
    axes[0].set_yticks(range(1, len(show) + 1), [fig_name(s_) for s_ in show])

    savefig(fig, "fig_fssp2_residual")


def f_tuning(g, pg):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [2, 1]})
    labels, vals, errs, cols = [], [], [], []
    for _, r_ in g.iterrows():
        labels.append(f"{'G' if r_.init else '-'}{'M' if r_.move else '-'}")
        vals.append(r_.overall)
        errs.append(r_.ci95)
        cols.append(C_PROP if (r_.k == 36 and r_.init == 0 and r_.move == 0) else "#bdbcb5")
    a1.bar(range(len(vals)), vals, yerr=errs, color=cols, edgecolor="white", capsize=1.5,
           error_kw={"elinewidth": 0.6, "ecolor": INK2})
    a1.set_xticks(range(len(vals)), labels, fontsize=6.5)
    for j, kk_ in enumerate(sorted(g["k"].unique())):
        a1.text(4 * j + 1.5, 0.672, f"k = {int(kk_)}", ha="center", va="top", fontsize=7, color=INK,
                transform=a1.transData, clip_on=False)
        if j:
            a1.axvline(4 * j - 0.5, color=INK2, lw=0.5, ls=":")
    a1.set_ylim(0.7, 1.0)
    a1.set_ylabel("tuning-set success")

    a2.errorbar(pg["pop"], pg["mean"], yerr=pg["ci95"], color=C_PROP, marker="o", ms=4, lw=1.6, capsize=2)
    a2.set_xscale("log", base=2)
    a2.set_xticks(pg["pop"], [str(p) for p in pg["pop"]])
    a2.set_ylim(0.7, 1.0)
    a2.set_xlabel("population size")
    a2.set_title("Population (k = 36)", loc="left", fontsize=7, color=INK)
    savefig(fig, "fig_tuning")


# ----------------------------------------------------------------------------------------------- review page
def review_page():
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>R3 tables and figures</title>"
             "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;"
             "color:#0b0b0b;background:#fcfcfb}table{border-collapse:collapse;font-size:12px;margin:8px 0 4px}"
             "th,td{border-bottom:1px solid #ddd;padding:3px 7px;text-align:right}th:first-child,td:first-child"
             "{text-align:left}caption{caption-side:top;text-align:left;font-size:13px;color:#52514e;"
             "padding-bottom:6px}h2{margin-top:36px}img{max-width:100%;border:1px solid #eee}.note{font-size:12px;"
             "color:#52514e}</style></head><body>",
             f"<h1>Round-3 tables and figures (review draft)</h1><p class='note'>Generated by "
             f"experiments/40_tables_figures.py at commit {git_commit()[:10]}. Method shown as "
             f"&ldquo;{html.escape(ALG)}&rdquo; until named. Numbers from clean runs only.</p>"]
    for key, (cap, df, note) in TABLES.items():
        d2 = df.copy()
        for c in d2.columns:
            if d2[c].dtype.kind == "f":
                d2[c] = d2[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        if "solver" in d2.columns:
            d2["solver"] = d2["solver"].map(fig_name)
        clean_cap = cap.replace(r"\algname", ALG).replace("{}", "").replace("$", "").replace("\\", "")
        parts.append(f"<h2>{key}</h2>" + d2.to_html(index=False, border=0).replace(
            "<table", f"<table><caption>{html.escape(clean_cap)}</caption", 1).replace("<table><caption",
                                                                                     "<table><caption", 1))
        if note:
            parts.append(f"<p class='note'>{html.escape(note.replace(chr(92), '').replace('$', ''))}</p>")
    for f in sorted((OUT / "figures").glob("*.png")):
        parts.append(f"<h2>{f.stem}</h2><img src='figures/{f.name}'>")
    parts.append("</body></html>")
    (OUT / "review.html").write_text("\n".join(parts))


def main() -> int:
    global ALG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="MXR-GA")
    args = ap.parse_args()
    ALG = args.name
    style()
    fd = t_fssp2()
    t_simple("t_suiteH", ("p16_hard", "p18_hard"),
             r"Suite H (97 difficulty-targeted instances, planted optimum; 20 seeds for stochastic methods).",
             "tab:suiteH", r"Small magnitudes: every strong method solves it; it separates blind and plain-GA search.")
    t_simple("t_reconcile", ("p16_reconcile", "p18_reconcile"),
             r"Payment reconciliation (48 synthetic instances; amounts in cents; 5--40 invoices per payment; "
             r"asymmetric fee tolerance).", "tab:reconcile",
             r"Very dense, few items per payment: CKK and k-tree are as accurate and faster here.")
    _, cell = t_regime()
    t_simple("t_scale", ("p16_scale", "p13_scale", "p18_scale"),
             r"Very large instances ($n=2000$--$10000$, 32/39-bit items; 30\,s).", "tab:scale",
             r"Lattice reduction declines $n>150$; population-based GA baselines exceed memory at this size.")
    out = t_structured()
    t_gate_and_ablation()
    g, pg = t_tuning()
    t_asym()
    t_simple("t_beyond", ("p18_beyond",),
             r"Limits: $n=56$--$80$ at 46--47-bit items, just beyond meet-in-the-middle's memory reach (30\,s).",
             "tab:beyond", r"k-tree and lattice reduction lead here; stated as a limitation of \algname.")
    t_speed()
    t_simple("t_fssp1", ("p18_fssp1",), r"Benchmark-FSSP-1 (3{,}238 instances), sanity check (3 seeds).",
             "tab:fssp1", r"Saturated: most methods reach $\mu=1$; reported for completeness only.")
    f_regime(cell)
    f_structured(out)
    f_anytime()
    f_residual(fd)
    f_tuning(g, pg)
    (OUT / "tables" / "algname.tex").write_text("\\providecommand{\\algname}{" + ALG + "}\n")
    review_page()
    (OUT / "manifest.json").write_text(json.dumps({"git_commit": git_commit(), "script": "experiments/40_tables_figures.py",
                                                   "tables": list(TABLES), "name": ALG}, indent=1))
    print(f"wrote {len(TABLES)} tables, figures -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

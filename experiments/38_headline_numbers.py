"""Phase 14.5: recompute every headline number of the new manuscript from the raw CSVs.

Single source of truth for writing: results/headline_numbers.json. Every number quoted in the paper
must be read from this file (or from a table generated from the same CSVs), never typed by hand.

Success = mu >= certified optimum - 1e-12 (strict; see load()).
Phase-14 directories are read only once scripts/run_r3_queue9.sh has logged completion.

Sources: Phase 13 (performance cores only) wherever wall-clock matters; Phase 10-11 for FSSP-2, Suite H
and reconciliation (outcomes reached in milliseconds, unaffected by core contention); Phase 14 for the
replications. Success = reaching the certified optimum (planted mu* = 1, parity mu* = 1 - 1/dL, Suite U
enumeration, exact DP); where no optimum is certified, only mean mu is reported.

Usage: .venv/bin/python experiments/38_headline_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.protocol.runner import git_commit  # noqa: E402

R = REPO / "results"


QUEUE9_DONE = (R / "r3_queue9.log").exists() and "queue9 done" in (R / "r3_queue9.log").read_text()


def load(*paths):
    if any(p.startswith("p14_") for p in paths) and not QUEUE9_DONE:
        return None
    frames = [pd.read_csv(R / p / "results.csv") for p in paths if (R / p / "results.csv").exists()]
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["solver"] = d["solver"].str.replace(r"@.*", "", regex=True)
    d["mu0"] = d["mu"].fillna(0.0)
    # Strict: an answer one unit off the optimum inside a 2^32-wide window has mu = 1 - 2^-32, which a
    # 1e-9 tolerance would wrongly count as optimal. 1e-12 admits only floating-point noise.
    d["hit"] = (d["known_opt"].notna() & (d["mu0"] >= d["known_opt"] - 1e-12)).astype(float)
    return d


def cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def boot_ci(x, reps=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def per_instance(d, by=()):
    return d.groupby(list(by) + ["instance_id", "solver"]).agg(
        hit=("hit", "mean"), mu=("mu0", "mean")).reset_index()


def gate_g(d):
    d = d.copy()
    d["tto"] = np.where(d["mu0"] >= 1 - 1e-12, d["wall_time"], np.inf)
    inst = d.groupby(["instance_id", "solver"]).agg(hit=("hit", "mean"), tto=("tto", "median")).reset_index()
    H = inst.pivot(index="instance_id", columns="solver", values="hit")
    T = inst.pivot(index="instance_id", columns="solver", values="tto")
    out = {"success": {c: round(float(H[c].mean()), 4) for c in H.columns}}
    passed = []
    for c in ("ma_rand", "lns_mitm"):
        diff = (H["ma_xr"] != H[c]) | (T["ma_xr"] != T[c])
        cd = cliff(T["ma_xr"][diff].replace(np.inf, 1e9), T[c][diff].replace(np.inf, 1e9))
        x = H["ma_xr"] - H[c]
        p = float(wilcoxon(x[x != 0]).pvalue) if (x != 0).sum() else 1.0
        pts = 100 * float(H["ma_xr"].mean() - H[c].mean())
        ok = pts >= 10 or cd <= -0.147
        passed.append(ok)
        out[f"vs_{c}"] = {"points": round(pts, 1), "tto_cliff": round(cd, 3), "n_differ": int(diff.sum()),
                          "wilcoxon_p": p, "pass": bool(ok)}
    out["gate_met"] = bool(all(passed))
    return out


def r1r2(d, single=None):
    d = d[~d["instance_id"].str.contains("parity")]
    if single is not None:
        single = single[~single["instance_id"].str.contains("parity")]
        d = pd.concat([d, single[~single["solver"].isin(d["solver"].unique())]])
    inst = per_instance(d, ["distribution"])
    mu = inst.pivot_table(index="distribution", columns="solver", values="mu")
    hit = inst.pivot_table(index="distribution", columns="solver", values="hit")
    struct = [x for x in mu.index if x != "uniform"]
    comp = [c for c in mu.columns if c != "ma_xr"]
    r1 = {x: {"ma_xr": round(float(mu.loc[x, "ma_xr"]), 4),
              "best_other": round(float(mu.loc[x, comp].max()), 4),
              "best_other_solver": str(mu.loc[x, comp].idxmax()),
              "pass": bool(mu.loc[x, "ma_xr"] >= mu.loc[x, comp].max() - 0.01)} for x in struct}
    r2 = 100 * float((hit.loc[struct, "ma_xr"] - hit.loc[struct, "ktree"]).mean()) if "ktree" in hit else None
    return {"R1": r1, "R1_pass": bool(all(v["pass"] for v in r1.values())),
            "R2_points": None if r2 is None else round(r2, 1),
            "R2_pass": None if r2 is None else bool(r2 >= 20),
            "success_by_distribution": hit.round(4).to_dict()}


def main() -> int:
    H = {"git_commit": git_commit(), "script": "experiments/38_headline_numbers.py"}

    g13 = load("p13_gateG")
    H["gate_G_clean"] = gate_g(g13)
    g14 = load("p14_gateG")
    if g14 is not None:
        H["gate_G_replication_seeds5_9"] = gate_g(g14)

    for tag, abl, ref in (("clean", "p13_ablation", g13), ("replication", "p14_ablation", g14)):
        a = load(abl)
        if a is None or ref is None:
            continue
        a["blk"] = a["instance_id"].str.split("-").str[2]
        rr = ref[ref["solver"] == "ma_xr"].copy()
        rr["blk"] = rr["instance_id"].str.split("-").str[2]
        H[f"ablation_cpsat_subsolver_{tag}"] = {
            "ma_xr_cpsat": a.groupby("blk")["hit"].mean().round(4).to_dict(),
            "ma_xr": rr.groupby("blk")["hit"].mean().round(4).to_dict()}

    s13 = load("p13_structured")
    H["structured_clean"] = r1r2(s13)
    par = s13[s13["instance_id"].str.contains("parity")].copy()
    par["gap"] = par["known_opt"] - par["mu0"]
    H["parity_clean"] = par.groupby("solver").agg(success=("hit", "mean"), mean_gap=("gap", "mean")).round(4).to_dict("index")
    s14 = load("p14_structured_seeds")
    if s14 is not None:
        H["structured_replication_seeds5_9"] = r1r2(s14, single=s13)
    srep = load("p14_structured_rep")
    if srep is not None:
        H["structured_second_corpus"] = r1r2(srep)

    reg = load("p13_regime", "p13_regime_lattice")
    x = reg["instance_id"].str.extract(r"R-(\w+)-n(\d+)-b(\d+)")
    reg["cell"] = x[0] + "-n" + x[1].str.zfill(4) + "-b" + x[2]
    inst = per_instance(reg, ["cell"])
    H["regime_map_clean_success"] = inst.pivot_table(index="cell", columns="solver", values="hit").round(4).to_dict()
    H["regime_map_ci_ma_xr"] = {c: boot_ci(g[g["solver"] == "ma_xr"]["hit"]) for c, g in inst.groupby("cell")}

    sc = load("p13_scale")
    sc["cell"] = sc["instance_id"].str.extract(r"(n\d+-b\d+)")[0]
    H["scale_clean_success"] = per_instance(sc, ["cell"]).pivot_table(index="cell", columns="solver", values="hit").round(4).to_dict()

    m = load("r3_main", "r3_main_ktree")
    pim = m.groupby(["suite", "instance_id", "solver"])["mu0"].mean().unstack()
    H["fssp2_mean_mu"] = {s: pim.loc[s].mean().round(4).dropna().to_dict() for s in ("T", "A", "U")}
    wtl = {}
    for s in ("T", "A", "U"):
        q = pim.loc[s]
        wtl[s] = {c: [int((q.ma_xr - q[c] > 1e-9).sum()), int(((q.ma_xr - q[c]).abs() <= 1e-9).sum()),
                      int((q.ma_xr - q[c] < -1e-9).sum())]
                  for c in q.columns if c != "ma_xr" and q[c].notna().all()}
    H["fssp2_wtl_ma_xr_vs"] = wtl

    for name, dirs in (("suite_H", ("r3_hard", "r3_hard_ktree")),
                       ("reconciliation", ("r3_reconcile", "r3_reconcile_ktree"))):
        d = load(*dirs)
        H[f"{name}_success"] = per_instance(d).groupby("solver")["hit"].mean().round(4).to_dict()

    out = R / "headline_numbers.json"
    out.write_text(json.dumps(H, indent=1, default=float))
    print(f"wrote {out}")
    for k in ("gate_G_clean", "gate_G_replication_seeds5_9"):
        if k in H:
            print(k, "met:", H[k]["gate_met"])
    for k in ("structured_clean", "structured_replication_seeds5_9", "structured_second_corpus"):
        if k in H:
            print(k, "R1:", H[k]["R1_pass"], "R2:", H[k]["R2_points"], H[k]["R2_pass"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

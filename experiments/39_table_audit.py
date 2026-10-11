"""Data-completeness audit for every table and figure of the new manuscript (Phase 18).

Each planned table lists its corpus, the solvers it reports and the minimum number of seeds for stochastic
solvers. Data counts only if it comes from a CLEAN run (manifest "workers" <= 4, i.e. performance cores
only; see Phase 12.6 / 13) and covers EVERY instance of the corpus with at least the required seeds.

Usage: .venv/bin/python experiments/39_table_audit.py   (exit code 1 if any gap remains)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.core.instance import FSSP1_DIR, FSSP2_DIR, iter_instance_paths  # noqa: E402

R = REPO / "results"
STOCH = {"ma_final", "ma_final_rand", "lns_final", "ma_final_cpsat", "ma_final_sym", "ktree", "ils_crisp",
         "sim_anneal_crisp", "vanilla_ga_crisp", "fhga_v2crisp", "random_search"}
META = ["ils_crisp", "sim_anneal_crisp", "vanilla_ga_crisp", "fhga_v2crisp", "random_search"]
EXACT_ETC = ["dp_exact", "mitm", "cpsat", "flsss", "flsss_exact", "lattice", "ckk", "kk", "alpha_cut",
             "greedy_center"]


def n_instances(corpus: str) -> int:
    if corpus == "fssp2":
        return len(list(iter_instance_paths(FSSP2_DIR)))
    if corpus == "fssp1":
        return len(list(iter_instance_paths(FSSP1_DIR)))
    if corpus == "hard":
        return len(list((REPO / "benchmarks" / "fsspb_hard").glob("fsspb-H-*.json")))
    return len(list((REPO / corpus / "instances").glob("*.json")))


TABLES = {
    "T-main FSSP-2 (mean/SD/worst/success, stats, W/T/L, runtime)":
        ("fssp2", ["ma_final", "ktree"] + META, EXACT_ETC, 20),
    "T-suiteH": ("hard", ["ma_final", "ktree"] + META, EXACT_ETC, 20),
    "T-reconciliation": ("benchmarks/fsspb_reconcile", ["ma_final", "ktree"] + META, EXACT_ETC, 20),
    "T-regime map (+ heat-map figure)":
        ("benchmarks/fsspb_regime_final", ["ma_final", "ktree"] + META,
         ["dp_exact", "mitm", "cpsat", "flsss", "flsss_exact", "lattice", "ckk", "kk"], 5),
    "T-scale": ("benchmarks/fsspb_regime_scale", ["ma_final", "ktree", "ils_crisp", "sim_anneal_crisp",
                                                  "random_search"],
                ["cpsat", "flsss", "flsss_exact", "ckk", "kk"], 5),
    "T-hard (fresh) vs competitors": ("benchmarks/fsspb_hard_eval", ["ma_final", "ktree"],
                                      ["lattice", "flsss_exact", "ckk", "kk", "cpsat", "mitm"], 5),
    "T-GateG/ablation": ("benchmarks/fsspb_hard_eval", ["ma_final", "ma_final_rand", "lns_final",
                                                         "ma_final_cpsat"], [], 5),
    "T-structured A (R1/R2)": ("benchmarks/fsspb_structured", ["ma_final", "ktree"],
                               ["ckk", "kk", "flsss", "flsss_exact", "cpsat", "lattice"], 5),
    "T-structured B (R1/R2)": ("benchmarks/fsspb_structured_rep", ["ma_final", "ktree"],
                               ["ckk", "kk", "flsss", "flsss_exact", "cpsat", "lattice"], 5),
    "T-asymmetry": ("benchmarks/fsspb_asym", ["ma_final", "ma_final_sym", "ktree"],
                    ["ckk", "kk", "flsss_exact", "cpsat"], 5),
    "T-beyond MITM (limits)": ("benchmarks/fsspb_regime_beyond", ["ma_final", "ktree"],
                               ["lattice", "ckk", "kk", "flsss_exact", "cpsat"], 5),
    "T-speed equalisation (120 s)": ("benchmarks/subset_speed", ["ktree"] + META, ["ckk"], 1),
    "T-FSSP-1 sanity (appendix)": ("fssp1", ["ma_final", "ktree", "random_search"],
                                   ["dp_exact", "flsss"], 3),
}


def clean_runs():
    rows = []
    for m in sorted(R.glob("*/manifest.json")):
        try:
            man = json.loads(m.read_text())
        except Exception:
            continue
        w = man.get("workers")
        if w is None or w > 4 or not (m.parent / "results.csv").exists():
            continue
        d = pd.read_csv(m.parent / "results.csv", usecols=["solver", "instance_id", "seed"])
        d["solver"] = d["solver"].str.replace(r"@max_n=\d+", "", regex=True)
        d = d[~d["solver"].str.contains("@")]
        d["corpus"] = man.get("corpus")
        d["dir"] = m.parent.name
        if man.get("wall") == 120:
            d["corpus"] = "benchmarks/subset_speed"
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    runs = clean_runs()
    gaps = 0
    for name, (corpus, stoch, single, min_seeds) in TABLES.items():
        n = n_instances(corpus)
        sub = runs[runs["corpus"] == corpus]
        if name.startswith("T-speed"):
            sub = runs[runs["dir"].str.contains("speed")]
        print(f"\n== {name}  [{corpus}, {n} instances]")
        for s in stoch + single:
            q = sub[sub["solver"] == s]
            need = min_seeds if s in STOCH else 1
            cnt = q.groupby("instance_id")["seed"].nunique()
            full = int((cnt >= need).sum())
            ok = full >= n
            gaps += not ok
            dirs = ",".join(sorted(q["dir"].unique()))[:60]
            print(f"   {'OK ' if ok else 'GAP'} {s:17s} instances with >= {need} seeds: {full}/{n}  {dirs}")
    print(f"\nTOTAL GAPS: {gaps}")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())

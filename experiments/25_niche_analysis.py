"""Task 2.13 — does FH-GA have a structurally characterised niche? (R4.1)

The submitted paper claimed a practical niche on large, huge-value instances
where exact methods are infeasible. That claim was withdrawn: on the Suite-A
instances DP cannot solve, random search matches FH-GA to within 0.0002.

This script asks the question properly rather than abandoning it. Using the
solver-independent instance descriptors from experiments/20 -- the
attainability gap Gamma, the plateau mass Pi, n, sub-suite and item
distribution -- it searches for any region in which

    (a) FH-GA v2-crisp beats the strongest available heuristic (FLSSS), and
    (b) the exact methods are infeasible or absent,

i.e. a region where a practitioner would actually be right to reach for
FH-GA. Regions are reported with their size and the margin, so that a null
result is as legible as a positive one.

The honest outcome is reported either way: if no such region exists, the
paper says so and rests the algorithmic claim on reliability (coverage and
zero-rate) alone.

Output: results/instance_structure/niche.json + paper_fragments/table_niche.tex
Usage: python3 experiments/25_niche_analysis.py
"""

from __future__ import annotations

import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scipy.stats import wilcoxon  # noqa: E402

from src.stats.analysis import cliffs_delta, cliffs_magnitude  # noqa: E402

STRUCT = REPO / "results" / "instance_structure" / "structure.csv"
FINAL = REPO / "results" / "final" / "combined.csv"
FAIR = REPO / "results" / "baseline_fairness" / "results.csv"
FRAG = REPO / "paper_fragments"
OUT = REPO / "results" / "instance_structure" / "niche.json"

FHGA = "fhga_v2crisp"
RIVALS = ["flsss", "ils_crisp", "random_search", "vanilla_ga"]
EXACT = ["dp_exact", "mitm"]


def load():
    per = collections.defaultdict(dict)
    acc = collections.defaultdict(list)
    exact_ok = collections.defaultdict(set)
    for path in (FINAL, FAIR):
        if not path.exists():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["mu"] == "":
                    if r["solver"] in EXACT:
                        pass          # infeasible: absent from exact_ok
                    continue
                acc[(r["solver"], r["instance_id"])].append(float(r["mu"]))
                if r["solver"] in EXACT:
                    exact_ok[r["solver"]].add(r["instance_id"])
    for (sv, i), v in acc.items():
        per[sv][i] = st.mean(v)
    return per, exact_ok


def region_report(name, ids, per, rival):
    """Paired comparison of FH-GA vs one rival on a set of instances."""
    ids = [i for i in ids if i in per[FHGA] and i in per[rival]]
    if len(ids) < 8:
        return None
    y = [per[FHGA][i] for i in ids]
    x = [per[rival][i] for i in ids]
    if all(a == b for a, b in zip(x, y)):
        return {"region": name, "rival": rival, "n": len(ids),
                "delta": 0.0, "p": 1.0, "cliffs_delta": 0.0,
                "magnitude": "negligible", "wins": 0, "losses": 0}
    p = wilcoxon(x, y).pvalue
    d = cliffs_delta(y, x)
    return {"region": name, "rival": rival, "n": len(ids),
            "delta": st.mean(y) - st.mean(x), "p": p, "cliffs_delta": d,
            "magnitude": cliffs_magnitude(d),
            "wins": sum(1 for a, b in zip(x, y) if b > a),
            "losses": sum(1 for a, b in zip(x, y) if b < a)}


def main() -> int:
    struct = {r["instance_id"]: r for r in csv.DictReader(open(STRUCT))}
    per, exact_ok = load()
    solved_exact = set().union(*exact_ok.values()) if exact_ok else set()

    all_ids = [i for i in struct if i in per[FHGA]]
    no_exact = [i for i in all_ids if i not in solved_exact]
    print(f"{len(all_ids)} instances; exact methods infeasible on "
          f"{len(no_exact)}")

    def gam(i):
        v = struct[i]["attainability_gap"]
        return float(v) if v != "" else None

    # Candidate regions, each defined only by solver-independent descriptors.
    regions = {
        "all instances": all_ids,
        "exact infeasible": no_exact,
        "high Gamma (>=0.05)": [i for i in all_ids
                                if (gam(i) or 0) >= 0.05],
        "high Gamma + exact infeasible": [i for i in no_exact
                                          if (gam(i) or 0) >= 0.05],
        "high plateau (Pi>=0.9)": [i for i in all_ids
                                   if float(struct[i]["plateau_mass"]) >= 0.9],
        "large n (>=500)": [i for i in all_ids if int(struct[i]["n"]) >= 500],
        "large n + exact infeasible": [i for i in no_exact
                                       if int(struct[i]["n"]) >= 500],
    }
    for suite in ("T", "A", "U"):
        regions[f"suite {suite}"] = [i for i in all_ids
                                     if struct[i]["suite"] == suite]

    results, rows = [], []
    for name, ids in regions.items():
        for rival in RIVALS:
            rec = region_report(name, ids, per, rival)
            if rec:
                results.append(rec)
    for rec in results:
        if rec["rival"] != "flsss":
            continue
        rows.append(
            rf"    {rec['region']} & {rec['n']} & {rec['delta']:+.6f} & "
            rf"{rec['p']:.2e} & {rec['cliffs_delta']:+.3f} & "
            rf"{rec['magnitude']} \\")

    # A niche requires beating the strongest rival with a non-negligible
    # effect on a region of meaningful size.
    niche = [r for r in results
             if r["rival"] == "flsss" and r["delta"] > 0 and r["p"] < 0.05
             and r["magnitude"] not in ("negligible",) and r["n"] >= 30]

    summary = {"n_instances": len(all_ids),
               "n_exact_infeasible": len(no_exact),
               "comparisons": results,
               "qualifying_niche_regions": niche}
    OUT.write_text(json.dumps(summary, indent=2))

    tbl = "\n".join([
        r"% Auto-generated by experiments/25_niche_analysis.py.",
        r"% Task 2.13 (R4.1): search for a structurally defined region where",
        r"% FH-GA is the right choice. Comparator: FLSSS (strongest heuristic).",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{FH-GA v2-crisp against FLSSS, the strongest heuristic "
        r"baseline, on regions defined purely by solver-independent instance "
        r"descriptors: the attainability gap $\Gamma$, the plateau mass "
        r"$\Pi$, $n$, and whether the exact methods are feasible. A practical "
        r"niche would require a positive difference with a non-negligible "
        r"effect size on a region of meaningful size. Wilcoxon signed-rank "
        r"over per-instance seed means; Cliff's $\delta$ alongside.}",
        r"  \label{tab:niche}",
        r"  \begin{tabular}{lrrrrl}",
        r"    \toprule",
        r"    Region & $N$ & $\Delta$ mean $\mu$ & $p$ & Cliff's $\delta$ & "
        r"magnitude \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    (FRAG / "table_niche.tex").write_text(tbl)

    print(f"\n{'region':<32}{'rival':<16}{'N':>5}{'delta':>11}{'p':>10}"
          f"{'d':>8}  magnitude")
    for r in results:
        print(f"{r['region']:<32}{r['rival']:<16}{r['n']:>5}"
              f"{r['delta']:>+11.6f}{r['p']:>10.2e}{r['cliffs_delta']:>+8.3f}"
              f"  {r['magnitude']}")
    print(f"\nqualifying niche regions vs FLSSS: {len(niche)}")
    for r in niche:
        print(f"  {r['region']}  N={r['n']}  delta={r['delta']:+.6f}  "
              f"{r['magnitude']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

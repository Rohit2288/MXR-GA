"""Task 2.15 validation — do solvers actually separate on the Gamma-targeted suite?

Generating high-Gamma instances is only useful if they discriminate. This
runs a representative set of solvers on the generated Suite H under the same
protocol as everything else, and reports the spread between them alongside
the spread on the existing sub-suites.

A suite on which every solver scores alike is no more informative than the
saturated corpus the reviewers objected to, regardless of how high Gamma is.
The honest outcome is reported either way.

Usage: python3 experiments/28_hard_suite_run.py [--workers 4] [--seeds 10]
Output: results/hard_suite/results.csv + summary.json
        paper_fragments/table_hard_suite.tex
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import multiprocessing as mp
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.evaluate import evaluate  # noqa: E402
from src.core.instance import load_instance  # noqa: E402
from src.protocol.budget import ProtocolConfig  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

HARD_DIR = REPO / "benchmarks" / "fsspb_hard"
OUT_DIR = REPO / "results" / "hard_suite"
CSV_PATH = OUT_DIR / "results.csv"
FRAG = REPO / "paper_fragments"

PROTOCOL = ProtocolConfig(max_evals=50_000, plateau_tol=1e-9,
                          plateau_patience=10_000)

SOLVERS = ["fhga_v2crisp", "fhga_fast", "flsss", "ils_crisp",
           "sim_anneal_crisp", "random_search", "vanilla_ga", "greedy_center",
           "dp_exact"]
DETERMINISTIC = {"greedy_center", "dp_exact", "flsss"}
FIELDS = ["solver", "instance_id", "n", "distribution", "seed", "status",
          "mu", "subset_sum", "subset_size", "evals_used", "wall_time",
          "stop_reason", "budget"]


def make_solver(name):
    from src.solvers.baselines import GreedyCenter, RandomSearch, VanillaGA
    from src.solvers.dp_bitset import DPBitset
    from src.solvers.fhga import FHGA, FHGAConfig
    from src.solvers.fhga_crisp import FHGACrisp
    from src.solvers.ils import ILS
    from src.solvers.sim_anneal import SimAnneal
    V2 = dict(init_greedy_frac=0.30, init_alpha_frac=0.20,
              repair_mode="dp_window")
    return {
        "fhga_v2crisp": lambda: FHGACrisp(variant="thorough",
                                          config=FHGAConfig(**V2),
                                          name="fhga_v2crisp"),
        "fhga_fast": lambda: FHGA(variant="fast"),
        "ils_crisp": lambda: ILS(guide_mode="crisp"),
        "sim_anneal_crisp": lambda: SimAnneal(guide_mode="crisp"),
        "random_search": lambda: RandomSearch(),
        "vanilla_ga": lambda: VanillaGA(),
        "greedy_center": lambda: GreedyCenter(),
        "dp_exact": lambda: DPBitset(),
        "flsss": lambda: __import__(
            "src.solvers.flsss_solver",
            fromlist=["FLSSSSolver"]).FLSSSSolver(),
    }[name]()


def run_unit(args):
    name, path, seed = args
    inst = load_instance(path)
    try:
        solver = make_solver(name)
        res = solver.solve(inst, PROTOCOL, np.random.default_rng(seed))
    except Exception as exc:                       # noqa: BLE001
        return {"solver": name, "instance_id": inst.instance_id, "n": inst.n,
                "distribution": inst.distribution, "seed": seed,
                "status": f"error:{type(exc).__name__}", "mu": "",
                "subset_sum": "", "subset_size": "", "evals_used": "",
                "wall_time": "", "stop_reason": "", "budget": PROTOCOL.max_evals}
    ev = evaluate(inst, res.subset_indices)
    return {"solver": name, "instance_id": inst.instance_id, "n": inst.n,
            "distribution": inst.distribution, "seed": seed, "status": "ok",
            "mu": repr(ev.mu), "subset_sum": ev.subset_sum,
            "subset_size": len(ev.subset_indices), "evals_used": res.evals_used,
            "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
            "budget": PROTOCOL.max_evals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of instances (spread across combos)")
    args = ap.parse_args()

    paths = sorted(p for p in HARD_DIR.glob("*.json")
                   if p.name != "manifest.json")
    if args.limit:
        # Every k-th instance, so the cap stays spread across distributions,
        # sizes and tolerance settings rather than truncating to one corner.
        step = max(1, len(paths) // args.limit)
        paths = paths[::step][: args.limit]
    units = []
    for name in SOLVERS:
        seeds = [0] if name in DETERMINISTIC else list(range(args.seeds))
        units += [(name, p, s) for p in paths for s in seeds]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                done.add((r["solver"], r["instance_id"], int(r["seed"])))
    todo = [(n, p, s) for (n, p, s) in units if (n, p.stem, s) not in done]
    print(f"{len(paths)} instances, {len(units)} units, {len(todo)} to run",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    t0 = time.perf_counter()
    n_done = 0
    if todo:
        with open(CSV_PATH, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            with mp.Pool(args.workers) as pool:
                for row in pool.imap_unordered(run_unit, todo, chunksize=4):
                    w.writerow(row)
                    n_done += 1
                    if n_done % 200 == 0:
                        fh.flush()
                        print(f"  {n_done}/{len(todo)}", flush=True)
                    if (args.max_seconds
                            and time.perf_counter() - t0 > args.max_seconds):
                        fh.flush()
                        print(f"  chunk limit at {n_done}/{len(todo)}",
                              flush=True)
                        pool.terminate()
                        break

    # ---- summary --------------------------------------------------------
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(CSV_PATH)):
        if r["mu"] != "":
            acc[(r["solver"], r["instance_id"])].append(float(r["mu"]))
    per = collections.defaultdict(dict)
    for (sv, i), v in acc.items():
        per[sv][i] = st.mean(v)
    present = [s for s in SOLVERS if s in per and per[s]]
    if not present:
        print("no results yet")
        return 0
    insts = sorted(set.intersection(*[set(per[s]) for s in present]))
    rows, summary = [], {}
    for sv in present:
        vals = [per[sv][i] for i in insts]
        flat = [x for i in insts for x in acc[(sv, i)]]
        zero = 100.0 * sum(1 for x in flat if x == 0.0) / len(flat)
        summary[sv] = {"mean": st.mean(vals), "worst": min(vals),
                       "zero_pct": zero, "n": len(vals)}
        rows.append(rf"    {sv.replace('_', chr(92)+'_')} & {st.mean(vals):.6f}"
                    rf" & {min(vals):.6f} & {zero:.1f}\% \\")
    spread = st.mean(max(per[s][i] for s in present)
                     - min(per[s][i] for s in present) for i in insts)
    summary["_mean_solver_spread"] = spread
    summary["_n_instances"] = len(insts)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit(),
        "script": "experiments/28_hard_suite_run.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "protocol": {"max_evals": PROTOCOL.max_evals},
        "seeds": args.seeds, "solvers": SOLVERS,
    }, indent=2))

    tbl = "\n".join([
        r"% Auto-generated by experiments/28_hard_suite_run.py.",
        r"% Task 2.15 (R2.6): the Gamma-targeted suite discriminates.",
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Solver performance on Suite~H, the {len(insts)} "
        r"difficulty-targeted instances generated by maximising the "
        r"attainability gap $\Gamma$ (mean $\Gamma=0.998$, against $0.079$ "
        r"on Suite~U). Blind sampling reaches a membership of zero on every "
        r"instance of this suite, so any separation between solvers is "
        r"search, not luck. Mean solver spread per instance: "
        rf"{spread:.4f}.}}",
        r"  \label{tab:hard-suite}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Solver & mean $\mu$ & worst instance & zero-$\mu$ runs \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    (FRAG / "table_hard_suite.tex").write_text(tbl)

    print(f"\n{len(insts)} instances; mean solver spread {spread:.4f}\n")
    print(f"{'solver':<18}{'mean':>10}{'worst':>10}{'zero%':>8}")
    for sv in present:
        d = summary[sv]
        print(f"{sv:<18}{d['mean']:>10.6f}{d['worst']:>10.6f}"
              f"{d['zero_pct']:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

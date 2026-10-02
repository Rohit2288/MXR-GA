"""Round-3 final runs: one driver for every corpus (Phase 10).

Solvers and protocol come from src/protocol/r3.py (identical for all). A
solver spec may carry parameters for the sensitivity study, e.g.
"ma_xr@k=24,pop=8".

Corpora: "fssp2" (Benchmark-FSSP-2, 866), "hard" (Suite H), or a path to a
corpus root containing instances/ (e.g. benchmarks/fsspb_regime_final).

Usage examples:
  .venv/bin/python experiments/35_r3_runs.py --corpus fssp2 --out-dir results/r3_main
  .venv/bin/python experiments/35_r3_runs.py --corpus benchmarks/fsspb_regime_final \
      --out-dir results/r3_regime --wall 10 --seeds-new 5 --seeds-base 5
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.evaluate import evaluate  # noqa: E402
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance  # noqa: E402
from src.protocol import r3  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

FIELDS = ["solver", "instance_id", "suite", "n", "distribution", "seed",
          "status", "mu", "known_opt", "subset_sum", "evals_used",
          "wall_time", "budget_s", "stop_reason", "extra"]


def corpus_paths(spec: str):
    if spec == "fssp2":
        return list(iter_instance_paths(FSSP2_DIR))
    if spec == "hard":
        return sorted((REPO / "benchmarks" / "fsspb_hard").glob("fsspb-H-*.json"))
    return list(iter_instance_paths(REPO / spec))


def parse_spec(spec: str):
    base, _, params = spec.partition("@")
    kw = {}
    for part in filter(None, params.split(",")):
        key, val = part.split("=")
        kw[key] = int(val)
    return base, kw


def suite_of(inst) -> str:
    return inst.suite or inst.metadata.get("construction", {}).get("v2_suite", "")


def run_unit(args):
    spec, path, seed, wall_fixed = args
    inst = load_instance(path)
    base_name, kw = parse_spec(spec)
    wall = wall_fixed if wall_fixed else r3.wall_budget(inst.n)
    opt = r3.known_optimum(inst)
    row = {"solver": spec, "instance_id": inst.instance_id,
           "suite": suite_of(inst), "n": inst.n,
           "distribution": inst.distribution, "seed": seed,
           "known_opt": "" if opt is None else repr(opt + 1e-12),
           "budget_s": wall}
    solver = r3.make_solver(base_name, wall, **kw)
    if hasattr(solver, "feasible"):
        ok, reason = solver.feasible(inst)
        if not ok:
            return {**row, "status": "infeasible", "mu": "", "subset_sum": "",
                    "evals_used": 0, "wall_time": "", "stop_reason": "",
                    "extra": json.dumps({"reason": reason})}
    res = solver.solve(inst, r3.protocol(wall, opt), np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    extra = dict(getattr(solver, "last_stats", {}) or {})
    if getattr(solver, "last_status", ""):
        extra["cpsat_status"] = solver.last_status
    trace = getattr(solver, "last_trace", [])
    if len(trace) > 200:   # keep first, last and an even thinning between
        step = len(trace) / 199
        trace = [trace[int(i * step)] for i in range(199)] + [trace[-1]]
    extra["trace"] = trace
    return {**row, "status": "ok", "mu": repr(ev.mu),
            "subset_sum": ev.subset_sum, "evals_used": res.evals_used,
            "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
            "extra": json.dumps(extra, default=float)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--solvers", nargs="+", default=r3.STOCHASTIC + r3.SINGLE)
    ap.add_argument("--seeds-new", type=int, default=20)
    ap.add_argument("--seeds-base", type=int, default=10)
    ap.add_argument("--wall", type=float, default=None,
                    help="fixed wall budget (s); default: r3.wall_budget(n)")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    out_dir = REPO / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "results.csv"
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["solver"], r["instance_id"], int(r["seed"]))
                    for r in csv.DictReader(fh)}
    paths = corpus_paths(args.corpus)
    units = []
    for p in paths:
        iid = load_instance(p).instance_id
        for spec in args.solvers:
            base_name, _ = parse_spec(spec)
            if base_name in r3.NEW or base_name in r3.ABLATION:
                seeds = range(args.seeds_new)
            elif base_name in r3.BASE_STOCHASTIC:
                seeds = range(args.seeds_base)
            else:
                seeds = [0]
            for seed in seeds:
                if (spec, iid, seed) not in done:
                    units.append((spec, str(p), seed, args.wall))
    units.sort(key=lambda u: (u[2], u[1], u[0]))
    (out_dir / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit(), "script": "experiments/35_r3_runs.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "corpus": args.corpus, "instances": len(paths),
        "solvers": args.solvers, "seeds_new": args.seeds_new,
        "seeds_base": args.seeds_base,
        "wall": args.wall or "r3.wall_budget(n): 2 s (n<=100), 5 s (<=500), 10 s",
        "stop_rule": "known certified optimum (r3.known_optimum)",
        "workers": args.workers,
    }, indent=1))
    print(f"{len(units)} units ({len(done)} already done)", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh, Pool(args.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_unit, units), 1):
            w.writerow(row)
            fh.flush()
            if i % 500 == 0:
                print(f"  {i}/{len(units)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

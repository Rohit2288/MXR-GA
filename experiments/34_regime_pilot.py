"""Round-3 pilot: which solver wins where on the regime map (suite R)?

Decision experiment for the resubmission (AE / R4 round 2: "show when and
why the proposed method should be used"). Every solver runs under ONE
protocol: equal wall-clock budget, equal evaluation cap, protocol-owned
stop at a proven optimum (mu = 1, the planted apex). Wall-clock is the
binding budget because several solvers do uncounted internal work (DP,
MITM, CP-SAT, FLSSS, and the exact moves of lns_mitm / ma_xr); the
evaluation cap is set high enough never to bind; budget-fraction schedules
(SA cooling, FH-GA annealing) follow the protocol's progress(), i.e. the
wall-clock fraction here. Which limit fired is logged.

Usage:
  .venv/bin/python experiments/34_regime_pilot.py --workers 6
Output: results/regime_pilot/{results.csv, manifest.json}
"""

from __future__ import annotations

import argparse
import math
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
from src.core.instance import iter_instance_paths, load_instance  # noqa: E402
from src.protocol.budget import ProtocolConfig  # noqa: E402
from src.protocol.runner import git_commit  # noqa: E402

CORPUS = REPO / "benchmarks" / "fsspb_regime"
OUT_DIR = REPO / "results" / "regime_pilot"
WALL_S = 10.0


def protocol(wall_s: float) -> ProtocolConfig:
    return ProtocolConfig(max_evals=10_000_000, plateau_tol=1e-9,
                          plateau_patience=None, max_wall_s=wall_s,
                          stop_at_mu=1.0)


PROTOCOL = protocol(WALL_S)
STOCHASTIC = ["ma_xr", "ma_rand", "lns_mitm", "fhga_v2crisp", "ils_crisp",
              "sim_anneal_crisp", "random_search"]
SINGLE = ["dp_exact", "mitm", "cpsat", "flsss"]
SEEDS = [0, 1, 2]
FIELDS = ["solver", "instance_id", "block", "n", "bits", "gap", "kappa",
          "seed", "status", "mu", "subset_sum", "evals_used", "wall_time",
          "stop_reason", "extra"]

V2 = dict(repair_mode="dp_window", uniform_xover=True, gamma=0.0,
          pc_low=0.8, pc_high=0.8, pm_low_mult=1.0, pm_high_mult=1.0)


def make_solver(name: str, wall_s: float = WALL_S, flsss_eps: float = 0.01):
    from src.solvers.baselines import RandomSearch
    from src.solvers.cpsat import CPSAT
    from src.solvers.dp_bitset import DPBitset
    from src.solvers.exact_lns import LNSMITM, MAXR
    from src.solvers.fhga import FHGAConfig
    from src.solvers.fhga_crisp import FHGACrisp
    from src.solvers.flsss_solver import FLSSSSolver
    from src.solvers.ils import ILS
    from src.solvers.mitm import MITM
    from src.solvers.sim_anneal import SimAnneal
    return {
        "ma_xr": lambda: MAXR(recomb="exact"),
        "ma_rand": lambda: MAXR(recomb="random"),
        "lns_mitm": LNSMITM,
        "fhga_v2crisp": lambda: FHGACrisp(variant="thorough",
                                          config=FHGAConfig(**V2),
                                          name="fhga_v2crisp"),
        "ils_crisp": lambda: ILS(guide_mode="crisp"),
        "sim_anneal_crisp": lambda: SimAnneal(guide_mode="crisp"),
        "random_search": RandomSearch,
        "dp_exact": DPBitset,
        "mitm": MITM,
        "cpsat": lambda: CPSAT(max_time_s=wall_s),
        # 1 + ceil(log2(1/eps)) probes share the wall budget; total capped
        # near it (R start-up is the +5 s).
        "flsss": lambda: FLSSSSolver(
            eps=flsss_eps,
            tlimit_per_probe=wall_s / (2 + math.ceil(math.log2(1 / flsss_eps))),
            total_timeout=wall_s + 5),
    }[name]()


def run_unit(args):
    solver_name, path, seed, wall_s, flsss_eps = args
    inst = load_instance(path)
    c = inst.metadata["construction"]
    base = {"solver": solver_name, "instance_id": inst.instance_id,
            "block": c["block"], "n": inst.n, "bits": c["bits"],
            "gap": c["gap"], "kappa": round(c["kappa"], 2), "seed": seed}
    solver = make_solver(solver_name, wall_s, flsss_eps)
    if hasattr(solver, "feasible"):
        ok, reason = solver.feasible(inst)
        if not ok:
            return {**base, "status": "infeasible", "mu": "", "subset_sum": "",
                    "evals_used": 0, "wall_time": "", "stop_reason": "",
                    "extra": json.dumps({"reason": reason})}
    res = solver.solve(inst, protocol(wall_s), np.random.default_rng(seed))
    ev = evaluate(inst, res.subset_indices)
    extra = dict(getattr(solver, "last_stats", {}) or {})
    if getattr(solver, "last_status", ""):
        extra["cpsat_status"] = solver.last_status
    return {**base, "status": "ok", "mu": repr(ev.mu),
            "subset_sum": ev.subset_sum, "evals_used": res.evals_used,
            "wall_time": repr(res.wall_time), "stop_reason": res.stop_reason,
            "extra": json.dumps(extra, default=float)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--solvers", nargs="+", default=STOCHASTIC + SINGLE)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default="results.csv")
    ap.add_argument("--wall", type=float, default=WALL_S)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--flsss-eps", type=float, default=0.01)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.out_name
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["solver"], r["instance_id"], int(r["seed"]))
                    for r in csv.DictReader(fh)}
    paths = list(iter_instance_paths(args.corpus))
    units = []
    for p in paths:
        iid = p.stem
        for s in args.solvers:
            for seed in (args.seeds if s in STOCHASTIC else [0]):
                if (s, iid, seed) not in done:
                    units.append((s, str(p), seed, args.wall, args.flsss_eps))
    # Interleave slow single-shot solvers with fast ones.
    units.sort(key=lambda u: (u[2], u[1], u[0]))
    (args.out_dir / args.out_name.replace(".csv", "_manifest.json")).write_text(json.dumps({
        "git_commit": git_commit(), "script": "experiments/34_regime_pilot.py",
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "protocol": protocol(args.wall).__dict__, "seeds": args.seeds,
        "corpus": str(args.corpus.relative_to(REPO)), "solvers": args.solvers,
        "flsss_eps": args.flsss_eps,
        "stochastic": STOCHASTIC, "single": SINGLE,
        "workers": args.workers, "instances": len(paths),
    }, indent=1, default=str))
    print(f"{len(units)} units to run ({len(done)} already done)", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh, Pool(args.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_unit, units), 1):
            w.writerow(row)
            fh.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(units)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

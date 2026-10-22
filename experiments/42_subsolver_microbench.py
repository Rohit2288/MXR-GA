"""Sub-solver micro-benchmark: one exact move (k = 36) solved by meet-in-the-middle and by CP-SAT.

For each sampled instance of the hard evaluation corpus, a state is drawn as in MXR-GA's initial
population (Bernoulli(T_M / A) inclusion) with a random free set of k = 36 items. The induced
closest-sum sub-problem is solved by closest_subset (meet-in-the-middle, exact) and by CP-SAT with
the model of the CP-SAT sub-solver (src/solvers/exact_lns.py, _cpsat_closest): one worker, current
assignment as hint, objective w_l * under + w_r * over. CP-SAT gets up to --cap seconds; the time at
which its incumbent first matches the meet-in-the-middle optimum is recorded (inf if never).

Output: results/p20_subsolver_microbench/results.csv and manifest.json.
Usage: .venv/bin/python experiments/42_subsolver_microbench.py [--n-inst 40] [--cap 30] [--workers 4]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.core.instance import load_instance  # noqa: E402
from src.solvers.exact_lns import closest_subset  # noqa: E402

K = 36


def one(args):
    path, seed, cap = args
    from ortools.sat.python import cp_model
    inst = load_instance(path)
    items = np.asarray(inst.items, dtype=np.int64)
    t = inst.target
    t_m = int(round(t.T_M))
    d_l = t.delta_L if t.delta_L > 0 else 1.0
    d_r = t.delta_R if t.delta_R > 0 else 1.0
    rng = np.random.default_rng(seed)
    p = min(1.0, t_m / max(1, int(items.sum())))
    chrom = rng.random(len(items)) < p
    free = rng.choice(len(items), size=min(K, len(items)), replace=False)
    fixed = chrom.copy()
    fixed[free] = False
    r = t_m - int(items[fixed].sum())
    vals = items[free]

    t0 = time.perf_counter()
    d_mitm, mask = closest_subset(vals, r, d_l, d_r)
    t_mitm = time.perf_counter() - t0

    import math
    wl = max(1, int(round(d_r)))
    wr = max(1, int(round(d_l)))
    g = math.gcd(wl, wr)
    wl, wr = wl // g, wr // g          # same weights and pre-scaling as the CP-SAT sub-solver
    scale = 1
    while (int(vals.sum()) + abs(r)) // scale * max(wl, wr) > 2 ** 60:
        scale *= 2
    vals_m = [max(1, int(v) // scale) for v in vals]
    r_m = r // scale
    true_vals = [int(v) for v in vals]

    def true_d(s):
        u = r - s
        return u / d_l if u >= 0 else -u / d_r

    m = cp_model.CpModel()
    x = [m.new_bool_var(f"x{i}") for i in range(len(vals_m))]
    hi = sum(vals_m) + abs(r_m)
    under = m.new_int_var(0, hi, "u")
    over = m.new_int_var(0, hi, "o")
    m.add(sum(v * xi for v, xi in zip(vals_m, x)) + under - over == r_m)
    m.minimize(wl * under + wr * over)
    for xi, c in zip(x, chrom[free]):
        m.add_hint(xi, bool(c))
    assert not m.validate(), m.validate()[:200]

    class CB(cp_model.CpSolverSolutionCallback):
        """Judges every incumbent on TRUE values: matched when its true d reaches the MITM optimum."""
        def __init__(self):
            super().__init__()
            self.t_hit = float("inf")
            self.best_d = float("inf")
            self.start = time.perf_counter()

        def on_solution_callback(self):
            s = sum(v for v, xi in zip(true_vals, x) if self.Value(xi))
            d = true_d(s)
            self.best_d = min(self.best_d, d)
            if d <= d_mitm + 1e-12 and self.t_hit == float("inf"):
                self.t_hit = time.perf_counter() - self.start

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = cap
    solver.parameters.random_seed = seed + 1
    cb = CB()
    st = solver.solve(m, cb)
    return {"instance_id": inst.instance_id, "seed": seed, "n": inst.n, "bits": int(items.max()).bit_length(),
            "t_mitm_s": t_mitm, "d_mitm": float(d_mitm), "scale": scale, "cpsat_best_true_d": cb.best_d,
            "cpsat_status": solver.status_name(st), "cpsat_t_match_s": cb.t_hit,
            "cpsat_wall_s": solver.wall_time}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inst", type=int, default=40)
    ap.add_argument("--cap", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", REPO / "experiments" / "35_r3_runs.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    power = runner.power_state()
    if not runner.power_ok(power):
        print(f"refusing to start: power state {power} (needs mains power, Low Power Mode off)")
        return 2
    import os
    subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    paths = sorted(glob.glob(str(REPO / "benchmarks/fsspb_hard_eval/instances/*.json")))
    rng = np.random.default_rng(20261020)
    pick = sorted(rng.choice(len(paths), size=min(a.n_inst, len(paths)), replace=False))
    jobs = [(paths[i], int(i), a.cap) for i in pick]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs))
    import pandas as pd
    out = REPO / "results" / "p20_subsolver_microbench"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    (out / "manifest.json").write_text(json.dumps({
        "git_commit": commit, "script": "experiments/42_subsolver_microbench.py",
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "corpus": "benchmarks/fsspb_hard_eval",
        "n_inst": len(jobs), "k": K, "power_at_start": power, "cpsat_cap_s": a.cap, "cpsat_workers": 1, "workers": a.workers}, indent=1))
    within = lambda s: float((df["cpsat_t_match_s"] <= s).mean())
    print(f"subproblems {len(df)}; MITM median {1000 * df.t_mitm_s.median():.1f} ms (max {1000 * df.t_mitm_s.max():.1f})")
    print(f"CP-SAT matched MITM optimum within 0.2 s: {within(0.2):.3f}, 2 s: {within(2):.3f}, {a.cap:g} s: {within(a.cap):.3f}")
    for tag, g in (("unscaled", df[df["scale"] == 1]), ("pre-scaled", df[df["scale"] > 1])):
        if len(g):
            print(f"  {tag}: {len(g)} sub-problems; matched within {a.cap:g} s: {float((g.cpsat_t_match_s <= a.cap).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

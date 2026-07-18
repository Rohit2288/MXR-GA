"""Multi-seed experiment runner (task 0.5).

Runs solvers × instances × seeds under ONE shared ProtocolConfig, recomputes
quality through the single scorer, and writes results.csv + manifest.json
(git commit, seeds, date, script) into a run directory.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from src.core.evaluate import evaluate
from src.core.instance import Instance
from src.protocol.budget import ProtocolConfig
from src.solvers.base import Solver

CSV_FIELDS = [
    "solver", "instance_id", "suite", "n", "distribution", "seed",
    "mu", "subset_sum", "subset_size", "evals_used", "wall_time",
    "stop_reason", "budget",
]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2], check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_experiment(
    solvers: Sequence[Solver],
    instances: Iterable[Instance],
    seeds: Sequence[int],
    config: ProtocolConfig,
    out_dir: Path | str,
    script_name: str,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Run the full grid and write results.csv + manifest.json to out_dir.

    Returns the path to results.csv. Every row's `mu` is recomputed by the
    single scorer from the returned subset — never taken from a solver.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = list(instances)

    rows: List[dict] = []
    for solver in solvers:
        for inst in instances:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                res = solver.solve(inst, config, rng)
                ev = evaluate(inst, res.subset_indices)
                rows.append({
                    "solver": solver.name,
                    "instance_id": inst.instance_id,
                    "suite": inst.suite or "",
                    "n": inst.n,
                    "distribution": inst.distribution,
                    "seed": seed,
                    "mu": repr(ev.mu),  # full precision (R3.m2)
                    "subset_sum": ev.subset_sum,
                    "subset_size": len(ev.subset_indices),
                    "evals_used": res.evals_used,
                    "wall_time": repr(res.wall_time),
                    "stop_reason": res.stop_reason,
                    "budget": config.max_evals,
                })
            if progress:
                progress(f"{solver.name} / {inst.instance_id}: done")

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "git_commit": git_commit(),
        "script": script_name,
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seeds": list(seeds),
        "solvers": [s.name for s in solvers],
        "n_instances": len(instances),
        "protocol": {
            "max_evals": config.max_evals,
            "plateau_tol": config.plateau_tol,
            "plateau_patience": config.plateau_patience,
        },
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return csv_path

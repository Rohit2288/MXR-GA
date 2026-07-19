"""FLSSS baseline — published tolerance-based subset-sum solver (task 1.11).

Wraps the CRAN package FLSSS (Liu, arXiv:1612.04484): an exact,
tolerance-window subset-sum-family solver with maintained public code —
the "tolerance-based subset-sum" method of R2.7, included per R1.4
(decision 1.11, Option A).

Integration mirrors the certified alpha-cut FPTAS: a binary search on the
alpha level; at each level FLSSS is asked for a subset whose sum lies in
the alpha-cut window (subset size freed by zero-padding); every subset it
returns is scored through the shared protocol objective on its true item
sum. Certified by construction — FLSSS's own reported values are never
used. Per-probe wall-clock is capped (`tlimit`); like CP-SAT it is a
wall-clock-limited method, so it is marked non-deterministic.

Requires Rscript + FLSSS on the machine (install:
  brew install r && Rscript -e 'install.packages("FLSSS")').
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver

_PROBE_R = Path(__file__).resolve().parent / "flsss_probe.R"


class FLSSSSolver(Solver):
    name = "flsss"
    deterministic = False  # per-probe wall-clock limit

    def __init__(self, eps: float = 0.01, tlimit_per_probe: float = 1.0,
                 total_timeout: float = 60.0):
        self.eps = eps
        self.tlimit_per_probe = tlimit_per_probe
        self.total_timeout = total_timeout
        self.rscript = shutil.which("Rscript") or "/opt/homebrew/bin/Rscript"

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        t = instance.target
        payload = (f"{t.T_L} {t.T_M} {t.T_R} {self.eps} "
                   f"{self.tlimit_per_probe}\n"
                   + " ".join(str(a) for a in instance.items) + "\n")
        try:
            proc = subprocess.run(
                [self.rscript, "--vanilla", str(_PROBE_R)],
                input=payload, capture_output=True, text=True,
                timeout=self.total_timeout,
            )
            out = proc.stdout
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")

        evaluated_any = False
        for line in out.splitlines():
            if line.startswith("SUBSET"):
                parts = line.split()[1:]
                subset = sorted(int(x) for x in parts)
                objective.evaluate(subset)
                evaluated_any = True
        if not evaluated_any:
            objective.evaluate([])

"""Regression: the CP-SAT sub-solver must build a valid model on large item values.

Its first version cross-multiplied the spreads into the objective weights, which overflowed CP-SAT's
64-bit bound on 38-47-bit instances (MODEL_INVALID): every call returned the current assignment
without solving, and the ablation measured nothing. Every call must now be OPTIMAL or FEASIBLE."""
import glob

import numpy as np
import pytest

from src.core.instance import load_instance
from src.protocol import r3

pytest.importorskip("ortools")
PATHS = sorted(glob.glob("benchmarks/fsspb_hard_eval/instances/*.json"))


@pytest.mark.skipif(not PATHS, reason="hard evaluation corpus not present")
@pytest.mark.parametrize("pattern", ["dense-n100-b40", "critical-n46-b47"])
def test_cpsat_subsolver_never_invalid(pattern):
    path = next(p for p in PATHS if pattern in p)
    inst = load_instance(path)
    solver = r3.make_solver("ma_final_cpsat", 2.0)
    solver.solve(inst, r3.protocol(2.0, r3.known_optimum(inst)), np.random.default_rng(0))
    status = solver.last_stats.get("cpsat", {}).get("status", {})
    assert status, "no CP-SAT call was made"
    assert set(status) <= {"OPTIMAL", "FEASIBLE"}, status

"""Tests for the CP-SAT solver (task 1.4)."""

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.cpsat import CPSAT

CFG = ProtocolConfig(max_evals=10)


def suite_u_instances(k=4):
    out = []
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            out.append(load_instance(p))
            if len(out) == k:
                break
    return out


@pytest.mark.parametrize("inst", suite_u_instances(),
                         ids=lambda i: i.instance_id)
def test_cpsat_matches_enumeration_optimum_when_optimal(inst):
    solver = CPSAT(max_time_s=30.0)
    res = solver.solve(inst, CFG, np.random.default_rng(0))
    mu = evaluate(inst, res.subset_indices).mu
    opt = exact_optimum(inst).mu
    assert mu <= opt + 1e-12
    if solver.last_status == "OPTIMAL":
        assert mu == pytest.approx(opt, abs=1e-9)

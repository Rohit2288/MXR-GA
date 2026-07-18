"""Tests for meet-in-the-middle (task 1.3)."""

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.mitm import MITM

CFG = ProtocolConfig(max_evals=10)


def suite_u_instances(k=8):
    out = []
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            out.append(load_instance(p))
            if len(out) == k:
                break
    return out


@pytest.mark.parametrize("inst", suite_u_instances(),
                         ids=lambda i: i.instance_id)
def test_mitm_matches_enumeration_optimum(inst):
    res = MITM().solve(inst, CFG, np.random.default_rng(0))
    mu = evaluate(inst, res.subset_indices).mu
    assert mu == pytest.approx(exact_optimum(inst).mu, abs=1e-12)
    assert res.evals_used == 1


def test_over_ceiling_returns_empty():
    inst = suite_u_instances(k=1)[0]
    res = MITM(max_n=10).solve(inst, CFG, np.random.default_rng(0))
    assert res.subset_indices == ()

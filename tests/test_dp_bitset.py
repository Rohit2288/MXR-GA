"""Tests for the exact bitset DP (task 1.2)."""

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.dp_bitset import DPBitset

CFG = ProtocolConfig(max_evals=10)


def small_u_instances(k=6, max_n=18):
    out = []
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            inst = load_instance(p)
            if inst.n <= max_n:
                out.append(inst)
            if len(out) == k:
                break
    return out


@pytest.mark.parametrize("inst", small_u_instances(),
                         ids=lambda i: i.instance_id)
def test_dp_matches_enumeration_optimum(inst):
    res = DPBitset().solve(inst, CFG, np.random.default_rng(0))
    mu = evaluate(inst, res.subset_indices).mu
    assert mu == pytest.approx(exact_optimum(inst).mu, abs=1e-12)
    assert res.evals_used == 1


def test_infeasible_instance_returns_empty():
    inst = small_u_instances(k=1)[0]
    res = DPBitset(max_tr=1).solve(inst, CFG, np.random.default_rng(0))
    assert res.subset_indices == ()

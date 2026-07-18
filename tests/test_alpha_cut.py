"""Tests for the rewritten alpha-cut FPTAS (task 1.7)."""

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.alpha_cut import AlphaCut, _scaled_dp_subset

CFG = ProtocolConfig(max_evals=1000)


def suite_u_instances(k=10):
    out = []
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            out.append(load_instance(p))
            if len(out) == k:
                break
    return out


@pytest.mark.parametrize("inst", suite_u_instances(),
                         ids=lambda i: i.instance_id)
def test_no_overshoot_by_construction(inst):
    res = AlphaCut().solve(inst, CFG, np.random.default_rng(0))
    mu = evaluate(inst, res.subset_indices).mu
    assert mu <= exact_optimum(inst).mu + 1e-12


def test_scaled_dp_returns_real_subset():
    inst = suite_u_instances(k=1)[0]
    L, R = inst.target.alpha_cut(0.5)
    subset = _scaled_dp_subset(inst.items, L, R, eps=0.01)
    if subset is not None:
        ev = evaluate(inst, subset)  # valid index set, true unscaled sum
        assert ev.subset_sum == sum(inst.items[i] for i in subset)


def test_probe_count_bounded():
    inst = suite_u_instances(k=1)[0]
    res = AlphaCut(eps=0.01).solve(inst, CFG, np.random.default_rng(0))
    assert res.evals_used <= 10  # 1 + ceil(log2(100)) + fallback

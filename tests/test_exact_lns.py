"""Tests for the exact closest-sum move and the wall-clock protocol options."""

import itertools
import time

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import BudgetedObjective, ProtocolConfig, StopSearch
from src.solvers.exact_lns import LNSMITM, MAXR, closest_subset


def _brute(vals, r, d_l, d_r):
    best = np.inf
    for bits in itertools.product((0, 1), repeat=len(vals)):
        u = r - int(np.dot(bits, vals))
        best = min(best, u / d_l if u >= 0 else -u / d_r)
    return best


@pytest.mark.parametrize("seed", range(25))
def test_closest_subset_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 13))
    vals = rng.integers(1, 10 ** int(rng.integers(2, 12)), size=k).astype(np.int64)
    r = int(rng.integers(-5, int(vals.sum()) + 5))
    d_l, d_r = float(rng.integers(1, 50)), float(rng.integers(1, 50))
    d, mask = closest_subset(vals, r, d_l, d_r)
    u = r - int(vals[mask].sum())
    assert d == pytest.approx(u / d_l if u >= 0 else -u / d_r)
    assert d == pytest.approx(_brute(vals, r, d_l, d_r))


def _inst():
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            return load_instance(p)


def test_wall_clock_budget_stops_run():
    inst = _inst()
    cfg = ProtocolConfig(max_evals=10 ** 9, max_wall_s=0.3)
    t0 = time.perf_counter()
    res = LNSMITM(k=8).solve(inst, cfg, np.random.default_rng(0))
    assert time.perf_counter() - t0 < 2.0
    assert res.stop_reason in ("wall_clock", "optimum")


def test_stop_at_mu_is_protocol_owned():
    obj = BudgetedObjective(_inst(), ProtocolConfig(max_evals=10, stop_at_mu=0.0))
    with pytest.raises(StopSearch):
        obj.evaluate([])
    assert obj.stop_reason == "optimum"


@pytest.mark.parametrize("make", [lambda: LNSMITM(k=64),
                                  lambda: MAXR(k=64),
                                  lambda: MAXR(k=64, recomb="random")])
def test_k_at_least_n_reaches_exact_optimum(make):
    """With k >= n a single exact move solves the instance outright."""
    inst = _inst()
    res = make().solve(inst, ProtocolConfig(max_evals=50),
                       np.random.default_rng(1))
    assert evaluate(inst, res.subset_indices).mu == pytest.approx(
        exact_optimum(inst).mu, abs=1e-12)

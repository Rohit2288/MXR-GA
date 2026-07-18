"""Tests for the exhaustive-enumeration reference (task 0.6)."""

import itertools

import pytest

from src.core.brute_force import all_subset_sums, exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance


def small_u_instance(n_max=15):
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            inst = load_instance(p)
            if inst.n <= n_max:
                return inst
    raise RuntimeError("no small Suite U instance found")


def test_subset_sums_match_naive():
    items = [3, 5, 9, 14]
    sums = all_subset_sums(items)
    assert len(sums) == 16
    for k in range(16):
        assert sums[k] == sum(items[i] for i in range(4) if (k >> i) & 1)


def test_matches_itertools_enumeration_on_tiny_instance():
    inst = small_u_instance()
    best = 0.0
    for r in range(inst.n + 1):
        for combo in itertools.combinations(range(inst.n), r):
            best = max(best, evaluate(inst, combo).mu)
    assert exact_optimum(inst).mu == pytest.approx(best, abs=1e-12)


def test_reproduces_generator_recorded_optimum():
    inst = small_u_instance()
    recorded = inst.metadata["construction"]["v2_axes"]["true_optimum_mu"]
    assert exact_optimum(inst).mu == pytest.approx(recorded, abs=1e-12)


def test_optimum_at_least_witness():
    inst = small_u_instance()
    assert exact_optimum(inst).mu >= inst.witness["mu_value"] - 1e-12

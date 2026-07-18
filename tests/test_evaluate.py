"""Tests for the single scorer (task 0.3)."""

import pytest

from src.core.evaluate import InvalidSubsetError, evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance


@pytest.fixture(scope="module")
def inst():
    return load_instance(next(iter(iter_instance_paths(FSSP2_DIR))))


def test_witness_reproduces_recorded_mu(inst):
    ev = evaluate(inst, inst.witness["subset_indices"])
    assert ev.subset_sum == inst.witness["subset_sum"]
    assert ev.mu == pytest.approx(inst.witness["mu_value"], abs=1e-12)


def test_empty_subset(inst):
    ev = evaluate(inst, [])
    assert ev.subset_sum == 0
    assert ev.mu == 0.0  # targets are far above 0


def test_indices_canonically_sorted(inst):
    ev = evaluate(inst, [3, 0, 1])
    assert ev.subset_indices == (0, 1, 3)
    assert ev.subset_sum == inst.items[0] + inst.items[1] + inst.items[3]


@pytest.mark.parametrize("bad", [[-1], [10**9], [0, 0], [1.5], [True]])
def test_invalid_subsets_rejected(inst, bad):
    with pytest.raises(InvalidSubsetError):
        evaluate(inst, bad)


def test_mu_in_unit_interval(inst):
    ev = evaluate(inst, range(inst.n))  # full set
    assert 0.0 <= ev.mu <= 1.0

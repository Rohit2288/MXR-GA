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


def _reference_evaluate(instance, subset_indices):
    """The original element-wise formulation, kept as the oracle."""
    idx = tuple(subset_indices)
    n = instance.n
    seen = set()
    for i in idx:
        if isinstance(i, bool) or not isinstance(i, int):
            raise InvalidSubsetError(f"non-integer index {i!r}")
        if not 0 <= i < n:
            raise InvalidSubsetError(f"index {i} out of range for n={n}")
        if i in seen:
            raise InvalidSubsetError(f"duplicate index {i}")
        seen.add(i)
    return (tuple(sorted(idx)), sum(instance.items[i] for i in idx))


def test_fast_path_matches_reference_on_random_subsets(inst):
    """The set/min/max validation must accept and score exactly what the
    original per-element loop did (speed change only, task 2.6 profiling)."""
    import random
    rng = random.Random(12345)
    for _ in range(200):
        k = rng.randrange(0, min(inst.n, 64) + 1)
        idx = rng.sample(range(inst.n), k)
        ev = evaluate(inst, idx)
        ref_idx, ref_sum = _reference_evaluate(inst, idx)
        assert ev.subset_indices == ref_idx
        assert ev.subset_sum == ref_sum
        assert ev.mu == inst.target.mu(ref_sum)


@pytest.mark.parametrize("bad", [[-1], [10**9], [0, 0], [1.5], [True],
                                [0, -1], [2, 2, 3], [0, 1.5], [False, 1],
                                [0, 10**9]])
def test_fast_path_rejects_exactly_what_reference_rejects(inst, bad):
    with pytest.raises(InvalidSubsetError):
        _reference_evaluate(inst, bad)
    with pytest.raises(InvalidSubsetError):
        evaluate(inst, bad)

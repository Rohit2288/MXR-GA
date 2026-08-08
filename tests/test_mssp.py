"""Tests for the MSSP third-domain core (task 2.17): exact DP vs brute force."""

import itertools

import numpy as np
import pytest

from src.core.fuzzy import TriangularFuzzyNumber
from src.core.mssp import MSSPInstance, exact_optimum, membership


def brute_force(inst: MSSPInstance):
    best = -1.0
    for assign in itertools.product((0, 1, 2), repeat=inst.n):
        best = max(best, membership(inst, assign))
    return best


def random_instance(rng, n):
    items = [int(rng.integers(5, 40)) for _ in range(n)]
    # plant two disjoint subsets as the targets so mu*=1 exists
    perm = list(rng.permutation(n))
    s1 = perm[: n // 3]
    s2 = perm[n // 3: 2 * n // 3]
    tm1 = sum(items[i] for i in s1) or 1
    tm2 = sum(items[i] for i in s2) or 1
    d = max(1, min(items) // 2)
    return MSSPInstance(
        instance_id=f"mssp-test-n{n}",
        items=tuple(items),
        target1=TriangularFuzzyNumber(tm1 - d, tm1, tm1 + d),
        target2=TriangularFuzzyNumber(tm2 - d, tm2, tm2 + d),
        n=n,
    )


@pytest.mark.parametrize("seed", range(12))
def test_exact_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    inst = random_instance(rng, n=rng.integers(6, 11))
    mu_dp, assign = exact_optimum(inst)
    assert mu_dp == pytest.approx(brute_force(inst), abs=1e-12)
    # reconstructed assignment actually attains the reported mu
    assert membership(inst, assign) == pytest.approx(mu_dp, abs=1e-12)


def test_planted_optimum_reached():
    rng = np.random.default_rng(0)
    inst = random_instance(rng, n=9)
    mu_dp, _ = exact_optimum(inst)
    assert mu_dp == pytest.approx(1.0, abs=1e-9)  # planted mu*=1 exists

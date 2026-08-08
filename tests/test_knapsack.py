"""Tests for the Fuzzy Knapsack domain (task 2.16)."""

import itertools
import random

import numpy as np
import pytest

from src.core.fuzzy import TriangularFuzzyNumber
from src.core.knapsack import (InvalidSubsetError, KnapsackInstance,
                               crisp_guide, dp_exact_knapsack,
                               evaluate_knapsack)
from src.protocol.budget import ProtocolConfig
from src.protocol.knapsack import (KnapGreedy, KnapILS, KnapRandomSearch,
                                   KnapSimAnneal, KnapVanillaGA)


def make(n=8, seed=0):
    rng = random.Random(seed)
    profits = tuple(rng.randint(1, 20) for _ in range(n))
    weights = tuple(rng.randint(1, 15) for _ in range(n))
    cap = max(5, sum(weights) // 2)
    t_m = max(2, sum(profits) // 2)
    return KnapsackInstance("t", profits, weights, cap,
                            TriangularFuzzyNumber(t_m - 5, t_m, t_m + 5),
                            n, "uncorrelated", seed)


def test_infeasible_subset_scores_zero():
    inst = make()
    ev = evaluate_knapsack(inst, list(range(inst.n)))   # everything
    assert ev.weight > inst.capacity
    assert not ev.feasible
    assert ev.mu == 0.0


def test_empty_subset_valid():
    inst = make()
    ev = evaluate_knapsack(inst, [])
    assert ev.profit == 0 and ev.weight == 0 and ev.feasible


@pytest.mark.parametrize("bad", [[-1], [10 ** 9], [0, 0], [1.5], [True]])
def test_invalid_subsets_rejected(bad):
    with pytest.raises(InvalidSubsetError):
        evaluate_knapsack(make(), bad)


def test_dp_matches_brute_force():
    """The exact solver is cited as ground truth; pin it to brute force."""
    rng = random.Random(11)
    for _ in range(25):
        n = rng.randint(4, 11)
        inst = KnapsackInstance(
            "t", tuple(rng.randint(1, 20) for _ in range(n)),
            tuple(rng.randint(1, 15) for _ in range(n)),
            rng.randint(5, 40),
            TriangularFuzzyNumber(rng.randint(1, 10), rng.randint(11, 30),
                                  rng.randint(31, 45)),
            n, "uncorrelated", 0)
        best = max(evaluate_knapsack(inst, list(c)).mu
                   for r in range(n + 1)
                   for c in itertools.combinations(range(n), r))
        got = dp_exact_knapsack(inst)
        assert got is not None
        assert got.mu == pytest.approx(best, abs=1e-12)
        assert got.feasible


def test_guide_equals_mu_inside_support_and_capacity():
    inst = make()
    g = crisp_guide(inst)
    # Empty set is feasible; pick a subset inside the support if one exists.
    for r in range(inst.n + 1):
        for c in itertools.combinations(range(inst.n), r):
            ev = evaluate_knapsack(inst, list(c))
            if ev.feasible and ev.mu > 0:
                assert g(ev.profit, ev.weight, ev.mu) == pytest.approx(
                    ev.mu, abs=1e-12)
                return
    pytest.skip("no in-support feasible subset on this instance")


@pytest.mark.parametrize("solver_factory", [
    lambda: KnapRandomSearch(),
    lambda: KnapGreedy(),
    lambda: KnapVanillaGA(pop=20),
    lambda: KnapILS(guide_mode="mu"),
    lambda: KnapILS(guide_mode="crisp"),
    lambda: KnapSimAnneal(guide_mode="mu"),
    lambda: KnapSimAnneal(guide_mode="crisp"),
])
def test_solver_contract(solver_factory):
    """Every solver returns a valid subset and respects the budget."""
    inst = make(n=10, seed=3)
    cfg = ProtocolConfig(max_evals=500, plateau_tol=1e-9,
                         plateau_patience=200)
    subset, evals, wall, stop = solver_factory().solve(
        inst, cfg, np.random.default_rng(0))
    ev = evaluate_knapsack(inst, subset)          # raises if invalid
    assert evals <= cfg.max_evals
    assert wall >= 0
    exact = dp_exact_knapsack(inst)
    assert ev.mu <= exact.mu + 1e-12              # never beats the optimum


def test_determinism_given_seed():
    inst = make(n=10, seed=5)
    cfg = ProtocolConfig(max_evals=400, plateau_tol=1e-9,
                         plateau_patience=200)
    a = KnapILS(guide_mode="crisp").solve(inst, cfg, np.random.default_rng(1))
    b = KnapILS(guide_mode="crisp").solve(inst, cfg, np.random.default_rng(1))
    assert a[0] == b[0] and a[1] == b[1]

"""Tests for the statistics pipeline (task 2.3)."""

import numpy as np
import pytest

from src.stats.analysis import (analyze_suite, cliffs_delta, cliffs_magnitude,
                                holm_correction, win_tie_loss)


class TestCliffsDelta:
    def test_disjoint_distributions(self):
        assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
        assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)

    def test_identical_distributions(self):
        assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_magnitudes(self):
        assert cliffs_magnitude(0.1) == "negligible"
        assert cliffs_magnitude(0.2) == "small"
        assert cliffs_magnitude(0.4) == "medium"
        assert cliffs_magnitude(0.9) == "large"


class TestHolm:
    def test_monotone_and_bounded(self):
        adj = holm_correction([0.01, 0.04, 0.03, 0.005])
        assert all(0 <= p <= 1 for p in adj)
        # Smallest raw p gets the largest multiplier.
        assert adj[3] == pytest.approx(0.02)
        assert adj[0] == pytest.approx(0.03)

    def test_all_ones(self):
        assert holm_correction([1.0, 1.0]) == [1.0, 1.0]


class TestAnalyzeSuite:
    def make_matrix(self, rng, shift_b=0.0, n=40):
        base = rng.uniform(0.3, 0.9, size=n)
        noise = lambda: rng.normal(0, 0.01, size=n)
        return {
            "A": dict(zip(map(str, range(n)), base + noise())),
            "B": dict(zip(map(str, range(n)),
                          np.clip(base + shift_b + noise(), 0, 1))),
            "C": dict(zip(map(str, range(n)), base + noise())),
        }

    def test_detects_clear_difference(self):
        rng = np.random.default_rng(0)
        res = analyze_suite(self.make_matrix(rng, shift_b=0.08),
                            reference="B")
        assert res.friedman_p < 0.01
        for pw in res.pairwise:
            assert pw.p_holm < 0.05
            assert pw.delta != 0
        assert res.mean_ranks["B"] < res.mean_ranks["A"]

    def test_null_case_not_significant(self):
        rng = np.random.default_rng(1)
        res = analyze_suite(self.make_matrix(rng, shift_b=0.0))
        # No systematic difference: Holm-adjusted p-values should mostly
        # be non-significant.
        assert min(pw.p_holm for pw in res.pairwise) > 0.01

    def test_paired_on_common_instances_only(self):
        m = {"A": {"i1": 0.5, "i2": 0.6, "i3": 0.7},
             "B": {"i1": 0.5, "i2": 0.7}}
        res = analyze_suite(m)
        assert res.n_instances == 2


class TestWinTieLoss:
    def test_counts(self):
        m = {"A": {"i1": 0.9, "i2": 0.5, "i3": 0.7},
             "B": {"i1": 0.8, "i2": 0.5, "i3": 0.9}}
        wtl = win_tie_loss(m)
        assert wtl[("A", "B")] == (1, 1, 1)
        assert wtl[("B", "A")] == (1, 1, 1)

"""Unit tests for TriangularFuzzyNumber (task 0.1)."""

import math

import pytest

from src.core.fuzzy import TriangularFuzzyNumber


@pytest.fixture
def tfn():
    return TriangularFuzzyNumber(T_L=10.0, T_M=20.0, T_R=40.0)


class TestMembership:
    def test_mu_at_anchors(self, tfn):
        assert tfn.mu(tfn.T_L) == 0.0
        assert tfn.mu(tfn.T_M) == 1.0
        assert tfn.mu(tfn.T_R) == 0.0

    def test_mu_outside_support_is_zero(self, tfn):
        assert tfn.mu(9.999) == 0.0
        assert tfn.mu(40.001) == 0.0
        assert tfn.mu(-1e9) == 0.0
        assert tfn.mu(1e9) == 0.0

    def test_mu_left_slope_linear(self, tfn):
        assert tfn.mu(15.0) == pytest.approx(0.5)
        assert tfn.mu(12.5) == pytest.approx(0.25)

    def test_mu_right_slope_linear(self, tfn):
        assert tfn.mu(30.0) == pytest.approx(0.5)
        assert tfn.mu(35.0) == pytest.approx(0.25)

    def test_mu_bounded_in_unit_interval(self, tfn):
        for x in [t / 10 for t in range(0, 501)]:
            assert 0.0 <= tfn.mu(x) <= 1.0

    def test_degenerate_left_spread(self):
        t = TriangularFuzzyNumber(20.0, 20.0, 40.0)
        assert t.mu(20.0) == 1.0
        assert t.mu(19.999) == 0.0
        assert t.mu(30.0) == pytest.approx(0.5)

    def test_degenerate_right_spread(self):
        t = TriangularFuzzyNumber(10.0, 20.0, 20.0)
        assert t.mu(20.0) == 1.0
        assert t.mu(20.001) == 0.0
        assert t.mu(15.0) == pytest.approx(0.5)

    def test_crisp_singleton(self):
        t = TriangularFuzzyNumber(20.0, 20.0, 20.0)
        assert t.mu(20.0) == 1.0
        assert t.mu(19.999) == 0.0
        assert t.mu(20.001) == 0.0


class TestAlphaCut:
    def test_alpha_one_is_peak(self, tfn):
        lo, hi = tfn.alpha_cut(1.0)
        assert lo == pytest.approx(tfn.T_M)
        assert hi == pytest.approx(tfn.T_M)

    def test_alpha_half(self, tfn):
        lo, hi = tfn.alpha_cut(0.5)
        assert lo == pytest.approx(15.0)
        assert hi == pytest.approx(30.0)

    def test_alpha_cut_endpoints_have_membership_alpha(self, tfn):
        for alpha in (0.1, 0.25, 0.5, 0.75, 0.9):
            lo, hi = tfn.alpha_cut(alpha)
            assert tfn.mu(lo) == pytest.approx(alpha)
            assert tfn.mu(hi) == pytest.approx(alpha)

    def test_alpha_cut_nested(self, tfn):
        lo1, hi1 = tfn.alpha_cut(0.3)
        lo2, hi2 = tfn.alpha_cut(0.7)
        assert lo1 <= lo2 and hi2 <= hi1

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0001, math.nan])
    def test_invalid_alpha_rejected(self, tfn, alpha):
        with pytest.raises(ValueError):
            tfn.alpha_cut(alpha)


class TestValidationAndProperties:
    @pytest.mark.parametrize("L,M,R", [(2, 1, 3), (1, 3, 2), (3, 2, 1)])
    def test_unordered_anchors_rejected(self, L, M, R):
        with pytest.raises(ValueError):
            TriangularFuzzyNumber(L, M, R)

    def test_spreads(self, tfn):
        assert tfn.delta_L == pytest.approx(10.0)
        assert tfn.delta_R == pytest.approx(20.0)
        assert tfn.spread == pytest.approx(30.0)

    def test_dict_round_trip(self, tfn):
        assert TriangularFuzzyNumber.from_dict(tfn.to_dict()) == tfn

    def test_frozen(self, tfn):
        with pytest.raises(Exception):
            tfn.T_M = 25.0

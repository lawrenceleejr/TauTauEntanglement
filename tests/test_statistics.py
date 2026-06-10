"""Tests for the statistical machinery in entanglement.py.

These tests verify the exact analytic results quoted in the paper:
  * SM correlation matrix C = diag(1, 1, -1) for any beta (3P0 argument)
  * CP-mixed correlation matrix = transverse rotation by 2*delta
  * per-event LLR moments: E_SM[t] = ln2 - 1/2, Var_SM = 1/4,
                           E_0[t] = ln2 - 1,   Var_0 = 1
  * m12 noise bias under the null: E[m12_hat | C=0] = 78.5/N
  * CHSH S and concurrence witness W estimators close on SM values
"""
import numpy as np
import pytest

from entanglement import (
    sm_correlation_matrix, sample_sm_pairs, llr_per_event,
    chsh_test, concurrence_witness, permutation_test,
    m12_null_expectation, M12_NULL_BIAS_CONST,
    LLR_MEAN_SM, LLR_VAR_SM, LLR_MEAN_NULL, LLR_VAR_NULL,
    compute_m12, expected_significance_no_correlation, expected_vpsi_reach,
    SM_CORRELATION_MATRIX,
)
from spin_analysis import extract_correlation_matrix


def _uniform_sphere(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestSMExpectation:
    def test_sm_matrix(self):
        np.testing.assert_allclose(sm_correlation_matrix(0.0),
                                   np.diag([1.0, 1.0, -1.0]))

    def test_cp_odd_is_singlet(self):
        """delta = pi/2 (pure CP-odd) gives the singlet C = -identity."""
        np.testing.assert_allclose(sm_correlation_matrix(np.pi / 2),
                                   -np.eye(3), atol=1e-12)

    def test_m12_invariant_under_cp_mixing(self):
        """C^T C = identity for any delta, so m12 = 2 for any CP mixture."""
        for delta in [0.0, 0.3, np.pi / 4, 1.2]:
            C = sm_correlation_matrix(delta)
            m12, _ = compute_m12(C)
            assert abs(m12 - 2.0) < 1e-12

    def test_sm_m12_is_2(self):
        m12, eig = compute_m12(SM_CORRELATION_MATRIX)
        assert abs(m12 - 2.0) < 1e-12


class TestLLRMoments:
    """Verify the exact closed-form per-event LLR moments by Monte Carlo."""

    def test_sm_moments(self):
        rng = np.random.default_rng(11)
        hp, hm = sample_sm_pairs(400000, rng=rng)
        t = llr_per_event(hp, hm)
        assert abs(np.mean(t) - LLR_MEAN_SM) < 0.005     # ln2 - 1/2
        assert abs(np.var(t) - LLR_VAR_SM) < 0.005       # exactly 1/4

    def test_null_moments(self):
        rng = np.random.default_rng(12)
        hp = _uniform_sphere(400000, rng)
        hm = _uniform_sphere(400000, rng)
        t = llr_per_event(hp, hm)
        assert abs(np.mean(t) - LLR_MEAN_NULL) < 0.005   # ln2 - 1
        assert abs(np.var(t) - LLR_VAR_NULL) < 0.01      # exactly 1


class TestM12NullBias:
    def test_wishart_constant(self):
        """E[m12_hat | C=0] = 9 E[lam1+lam2]/N with E[lam1+lam2] = 8.72
        for the 3x3 Wishart with 3 degrees of freedom."""
        rng = np.random.default_rng(5)
        N, trials = 60, 1500
        vals = []
        for _ in range(trials):
            hp = _uniform_sphere(N, rng)
            hm = _uniform_sphere(N, rng)
            C, _, _ = extract_correlation_matrix(hp, hm)
            m12, _ = compute_m12(C)
            vals.append(m12)
        expected = m12_null_expectation(N)
        assert abs(np.mean(vals) - expected) / expected < 0.10

    def test_constant_value(self):
        assert abs(M12_NULL_BIAS_CONST - 78.5) < 1.0


class TestLinearObservables:
    def test_chsh_sm_closure(self):
        """CHSH S with fixed axes closes on 2 sqrt(2) for SM events."""
        rng = np.random.default_rng(21)
        hp, hm = sample_sm_pairs(200000, rng=rng)
        res = chsh_test(hp, hm)
        assert abs(res['S'] - 2.0 * np.sqrt(2.0)) < 0.05
        assert res['z_bell'] > 5.0

    def test_chsh_null_is_zero(self):
        rng = np.random.default_rng(22)
        hp = _uniform_sphere(100000, rng)
        hm = _uniform_sphere(100000, rng)
        res = chsh_test(hp, hm)
        assert abs(res['S']) < 0.1

    def test_witness_sm_closure(self):
        """Concurrence witness W closes on 1 for SM events."""
        rng = np.random.default_rng(23)
        hp, hm = sample_sm_pairs(200000, rng=rng)
        res = concurrence_witness(hp, hm)
        assert abs(res['W'] - 1.0) < 0.05
        assert res['z_sep'] > 5.0

    def test_witness_null_is_minus_half(self):
        """Uncorrelated taus have C = 0, so W = (0+0-0-1)/2 = -1/2."""
        rng = np.random.default_rng(24)
        hp = _uniform_sphere(100000, rng)
        hm = _uniform_sphere(100000, rng)
        res = concurrence_witness(hp, hm)
        assert abs(res['W'] + 0.5) < 0.1


class TestPermutationTest:
    def test_null_p_value_uniformish(self):
        """Under the null, the permutation p-value should not be small."""
        rng = np.random.default_rng(31)
        pvals = []
        for _ in range(20):
            hp = _uniform_sphere(60, rng)
            hm = _uniform_sphere(60, rng)
            res = permutation_test(hp, hm, statistic='llr', n_perm=300,
                                   rng=rng)
            pvals.append(res['p_value'])
        # Mean p-value ~ 0.5 under the null
        assert 0.25 < np.mean(pvals) < 0.75

    def test_sm_detected(self):
        """SM correlations at N=100 should give a small p-value, with
        z near the expected sqrt(N)/2 = 5."""
        rng = np.random.default_rng(32)
        hp, hm = sample_sm_pairs(100, rng=rng)
        res = permutation_test(hp, hm, statistic='llr', n_perm=1000, rng=rng)
        assert res['p_value'] < 0.01
        assert res['z_score'] > 3.0

    def test_m12_statistic_supported(self):
        rng = np.random.default_rng(33)
        hp, hm = sample_sm_pairs(60, rng=rng)
        res = permutation_test(hp, hm, statistic='m12', n_perm=200, rng=rng)
        assert res['p_value'] < 0.2


class TestProjections:
    def test_expected_significance(self):
        assert abs(expected_significance_no_correlation(100) - 5.0) < 1e-12

    def test_expected_reach_scales_linearly(self):
        r1 = expected_vpsi_reach(1000)
        r2 = expected_vpsi_reach(2000)
        assert abs(r2 / r1 - 2.0) < 1e-12

    def test_expected_reach_value(self):
        # v95 = N beta / (4 z^2) with z = 1.96
        assert abs(expected_vpsi_reach(1000) - 1000 / (4 * 1.96**2)) < 1e-9

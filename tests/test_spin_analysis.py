"""Tests for spin analysis in spin_analysis.py."""
import numpy as np
import pytest
from spin_analysis import _define_basis, extract_correlation_matrix


class TestBasis:
    """Tests for the {n, r, k} basis definition."""

    def test_orthonormality(self):
        """Basis vectors should be orthonormal."""
        p_tau = np.array([62.5, 0.0, 0.0, 62.0])  # tau- along z
        p_beam = np.array([120.0, 0.0, 0.0, 120.0])
        k, r, n = _define_basis(p_tau, p_beam)

        # Orthogonality
        assert abs(np.dot(k, r)) < 1e-10
        assert abs(np.dot(k, n)) < 1e-10
        assert abs(np.dot(r, n)) < 1e-10

        # Normalisation
        assert abs(np.linalg.norm(k) - 1) < 1e-10
        assert abs(np.linalg.norm(r) - 1) < 1e-10
        assert abs(np.linalg.norm(n) - 1) < 1e-10

    def test_k_along_tau(self):
        """k should point along the tau- direction."""
        p_tau = np.array([62.5, 10.0, 20.0, 55.0])
        p_beam = np.array([120.0, 0.0, 0.0, 120.0])
        k, r, n = _define_basis(p_tau, p_beam)

        # k should be parallel to tau 3-momentum
        tau_hat = p_tau[1:4] / np.linalg.norm(p_tau[1:4])
        assert abs(np.dot(k, tau_hat) - 1.0) < 1e-10

    def test_right_handed(self):
        """n = k x r (right-handed system)."""
        p_tau = np.array([62.5, 5.0, 10.0, 60.0])
        p_beam = np.array([120.0, 0.0, 0.0, 120.0])
        k, r, n = _define_basis(p_tau, p_beam)

        n_cross = np.cross(k, r)
        np.testing.assert_allclose(n, n_cross, atol=1e-10)

    def test_degenerate_along_beam(self):
        """Graceful handling when tau is exactly along beam."""
        p_tau = np.array([62.5, 0.0, 0.0, 62.0])
        p_beam = np.array([120.0, 0.0, 0.0, 120.0])
        k, r, n = _define_basis(p_tau, p_beam)

        # Should still be orthonormal
        assert abs(np.dot(k, r)) < 1e-10
        assert abs(np.dot(k, n)) < 1e-10
        assert abs(np.linalg.norm(k) - 1) < 1e-10


class TestCorrelationMatrix:
    """Tests for C_ij extraction."""

    def test_uniform_distribution(self):
        """Uniform random directions should give C_ij ~ 0."""
        rng = np.random.default_rng(123)
        N = 50000
        # Random unit vectors on the sphere
        cos_p = 2 * rng.random((N, 3)) - 1
        cos_m = 2 * rng.random((N, 3)) - 1

        C, B_plus, B_minus = extract_correlation_matrix(cos_p, cos_m)

        # C should be near zero (within statistical noise)
        assert np.all(np.abs(C) < 0.1)
        assert np.all(np.abs(B_plus) < 0.05)
        assert np.all(np.abs(B_minus) < 0.05)

    def test_perfect_correlation(self):
        """If cos+ = cos- exactly, C_ii = 9 * <cos^2> = 3 for uniform."""
        rng = np.random.default_rng(42)
        N = 100000
        cos_vals = 2 * rng.random((N, 3)) - 1
        C, _, _ = extract_correlation_matrix(cos_vals, cos_vals)

        # <x^2> for Uniform(-1,1) = 1/3, so C_ii = 9 * 1/3 = 3
        for i in range(3):
            assert abs(C[i, i] - 3.0) < 0.1

    def test_empty_input(self):
        """Empty arrays should return zeros."""
        cos_p = np.empty((0, 3))
        cos_m = np.empty((0, 3))
        C, B_plus, B_minus = extract_correlation_matrix(cos_p, cos_m)
        np.testing.assert_allclose(C, np.zeros((3, 3)))

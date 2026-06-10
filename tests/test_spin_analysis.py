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
        """If cos+ = cos- exactly, C_ii = -9 * <cos^2> = -3 for uniform.

        The extraction uses C_ij = -9 <cos+ cos->, the minus sign coming
        from the tau+ analysing power alpha+ = -1 (alpha+ alpha- = -1).
        """
        rng = np.random.default_rng(42)
        N = 100000
        cos_vals = 2 * rng.random((N, 3)) - 1
        C, _, _ = extract_correlation_matrix(cos_vals, cos_vals)

        # <x^2> for Uniform(-1,1) = 1/3, so C_ii = -9 * 1/3 = -3
        for i in range(3):
            assert abs(C[i, i] + 3.0) < 0.1

    def test_sm_closure(self):
        """Events sampled from the exact SM density close on C = diag(1,1,-1)."""
        from entanglement import sample_sm_pairs
        rng = np.random.default_rng(7)
        h_plus, h_minus = sample_sm_pairs(200000, rng=rng)
        C, B_plus, B_minus = extract_correlation_matrix(h_plus, h_minus)
        np.testing.assert_allclose(C, np.diag([1.0, 1.0, -1.0]), atol=0.05)
        assert np.all(np.abs(B_plus) < 0.05)
        assert np.all(np.abs(B_minus) < 0.05)

    def test_empty_input(self):
        """Empty arrays should return zeros."""
        cos_p = np.empty((0, 3))
        cos_m = np.empty((0, 3))
        C, B_plus, B_minus = extract_correlation_matrix(cos_p, cos_m)
        np.testing.assert_allclose(C, np.zeros((3, 3)))


class TestPolarimeter:
    """Tests for the hadronic polarimeter vector."""

    def test_pi_nu_reduces_to_pion_direction(self):
        """For tau -> pi nu with consistent kinematics, h = q_hat."""
        from spin_analysis import polarimeter_direction
        from config import M_TAU, M_PI
        rng = np.random.default_rng(3)
        for _ in range(50):
            # Two-body decay in the tau rest frame
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            p_pi_mag = (M_TAU**2 - M_PI**2) / (2 * M_TAU)
            E_pi = np.sqrt(p_pi_mag**2 + M_PI**2)
            q = np.array([E_pi, *(p_pi_mag * d)])
            N = np.array([p_pi_mag, *(-p_pi_mag * d)])
            h = polarimeter_direction(q, N)
            np.testing.assert_allclose(h, d, atol=1e-10)

    def test_rho_polarimeter_is_lightlike(self):
        """H = 2(q.N)q - q^2 N satisfies H.H = 0 for massless N,
        so |H_vec| = H^0 (unit analysing power)."""
        rng = np.random.default_rng(4)
        for _ in range(100):
            # Random massive q and lightlike N
            q3 = rng.normal(size=3)
            q2_target = rng.uniform(0.1, 1.0)  # q^2 (rho-like, GeV^2)
            E_q = np.sqrt(q2_target + q3 @ q3)
            q = np.array([E_q, *q3])
            n3 = rng.normal(size=3)
            N = np.array([np.linalg.norm(n3), *n3])
            qN = q[0]*N[0] - q[1:] @ N[1:]
            q2 = q[0]**2 - q[1:] @ q[1:]
            H = 2*qN*q - q2*N
            H2 = H[0]**2 - H[1:] @ H[1:]
            scale = max(abs(H[0]), 1e-12)**2
            assert abs(H2) / scale < 1e-9
            # |H_vec| = H^0
            assert abs(np.linalg.norm(H[1:]) - H[0]) / max(H[0], 1e-12) < 1e-9

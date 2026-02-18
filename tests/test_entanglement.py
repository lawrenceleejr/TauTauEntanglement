"""Tests for entanglement measures in entanglement.py."""
import numpy as np
import pytest
from entanglement import (
    compute_m12, compute_bell_score, compute_concurrence,
    compute_density_matrix, compute_concurrence_from_C,
    locality_rejection_sigma, entanglement_rejection_sigma,
)


class TestM12:
    """Tests for the Horodecki parameter m12."""

    def test_sm_correlation_matrix(self):
        """SM H->tautau: C = diag(+1, +1, -1) -> m12 = 2."""
        C = np.diag([1.0, 1.0, -1.0])
        m12, eigvals = compute_m12(C)
        assert abs(m12 - 2.0) < 1e-10

    def test_zero_correlation(self):
        """No correlations: C = 0 -> m12 = 0."""
        C = np.zeros((3, 3))
        m12, eigvals = compute_m12(C)
        assert abs(m12) < 1e-10

    def test_classical_correlation(self):
        """Classical correlations: C = diag(0.5, 0.5, 0) -> m12 = 0.5."""
        C = np.diag([0.5, 0.5, 0.0])
        m12, eigvals = compute_m12(C)
        assert abs(m12 - 0.5) < 1e-10

    def test_bell_boundary(self):
        """At Bell boundary: C = diag(1, 0, 0) -> m12 = 1."""
        C = np.diag([1.0, 0.0, 0.0])
        m12, eigvals = compute_m12(C)
        assert abs(m12 - 1.0) < 1e-10

    def test_eigenvalues_sorted(self):
        """Eigenvalues should be sorted descending."""
        C = np.diag([0.3, 0.8, 0.5])
        m12, eigvals = compute_m12(C)
        assert eigvals[0] >= eigvals[1] >= eigvals[2]


class TestBellScore:
    """Tests for the CHSH Bell score."""

    def test_sm_score(self):
        """SM: 2*sqrt(2) ~ 2.828."""
        score = compute_bell_score(2.0)
        assert abs(score - 2 * np.sqrt(2)) < 1e-10

    def test_classical_limit(self):
        """Classical limit: m12 = 1 -> score = 2."""
        score = compute_bell_score(1.0)
        assert abs(score - 2.0) < 1e-10

    def test_zero(self):
        """No correlations: score = 0."""
        score = compute_bell_score(0.0)
        assert abs(score) < 1e-10

    def test_negative_clipped(self):
        """Negative m12 (unphysical) gives 0."""
        score = compute_bell_score(-0.5)
        assert abs(score) < 1e-10


class TestConcurrence:
    """Tests for concurrence."""

    def test_maximally_entangled(self):
        """SM H->tautau (pure state): concurrence = 1."""
        C = np.diag([1.0, 1.0, -1.0])
        B_plus = np.zeros(3)
        B_minus = np.zeros(3)
        conc = compute_concurrence_from_C(C, B_plus, B_minus)
        assert abs(conc - 1.0) < 0.05  # small tolerance for numerical rounding

    def test_separable_state(self):
        """Product state (C = 0, no polarisation): concurrence = 0."""
        C = np.zeros((3, 3))
        B_plus = np.zeros(3)
        B_minus = np.zeros(3)
        conc = compute_concurrence_from_C(C, B_plus, B_minus)
        assert abs(conc) < 1e-8

    def test_concurrence_non_negative(self):
        """Concurrence should always be >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            C = rng.uniform(-1, 1, (3, 3))
            B_p = rng.uniform(-0.5, 0.5, 3)
            B_m = rng.uniform(-0.5, 0.5, 3)
            conc = compute_concurrence_from_C(C, B_p, B_m)
            assert conc >= -1e-10


class TestDensityMatrix:
    """Tests for density matrix construction."""

    def test_trace_one(self):
        """Density matrix should have trace 1."""
        C = np.diag([0.8, 0.6, -0.4])
        B_plus = np.array([0.1, 0.0, 0.0])
        B_minus = np.array([0.0, 0.1, 0.0])
        rho = compute_density_matrix(C, B_plus, B_minus)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_hermitian(self):
        """Density matrix should be Hermitian."""
        C = np.diag([1.0, 1.0, -1.0])
        B_plus = np.zeros(3)
        B_minus = np.zeros(3)
        rho = compute_density_matrix(C, B_plus, B_minus)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)

    def test_identity_for_zero(self):
        """C=0, B=0 should give rho = I/4."""
        C = np.zeros((3, 3))
        rho = compute_density_matrix(C, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(rho, np.eye(4) / 4, atol=1e-10)


class TestSignificance:
    """Tests for rejection significance calculations."""

    def test_locality_above_threshold(self):
        """m12 > 1 with small error gives large sigma."""
        sig = locality_rejection_sigma(2.0, 0.1)
        assert abs(sig - 10.0) < 1e-10

    def test_locality_at_threshold(self):
        """m12 = 1 gives 0 sigma."""
        sig = locality_rejection_sigma(1.0, 0.1)
        assert abs(sig) < 1e-10

    def test_entanglement_positive(self):
        """Concurrence > 0 gives positive sigma."""
        sig = entanglement_rejection_sigma(0.5, 0.1)
        assert abs(sig - 5.0) < 1e-10

    def test_zero_error_gives_inf(self):
        """Zero error with m12 > 1 gives infinity."""
        sig = locality_rejection_sigma(2.0, 0.0)
        assert sig == np.inf

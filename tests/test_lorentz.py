"""Tests for Lorentz algebra utilities in tau_reconstruction.py."""
import numpy as np
import pytest
from tau_reconstruction import mass, mass2, boost, beta_vec, p3mag, p3hat, p3vec


def test_mass_at_rest():
    """Particle at rest should return its rest mass."""
    p = np.array([1.77686, 0.0, 0.0, 0.0])
    assert abs(mass(p) - 1.77686) < 1e-10


def test_mass_moving():
    """Mass should be Lorentz invariant."""
    m = 1.77686
    px, py, pz = 10.0, 5.0, 30.0
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    p = np.array([E, px, py, pz])
    assert abs(mass(p) - m) < 1e-8


def test_mass_photon():
    """Massless particle."""
    p = np.array([10.0, 0.0, 0.0, 10.0])
    assert abs(mass(p)) < 1e-10


def test_boost_to_rest():
    """Boosting a particle by its own velocity should give rest frame."""
    m = 1.77686
    px, py, pz = 3.0, 4.0, 5.0
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    p = np.array([E, px, py, pz])

    beta = beta_vec(p)
    p_rest = boost(p, beta)

    assert abs(p_rest[0] - m) < 1e-8
    assert abs(p_rest[1]) < 1e-8
    assert abs(p_rest[2]) < 1e-8
    assert abs(p_rest[3]) < 1e-8


def test_boost_preserves_mass():
    """Boost should preserve invariant mass."""
    m = 0.13957
    p = np.array([10.0, 3.0, 4.0, 8.0])
    m_orig = mass(p)

    beta = np.array([0.3, 0.1, -0.2])
    p_boosted = boost(p, beta)
    m_boosted = mass(p_boosted)

    assert abs(m_orig - m_boosted) < 1e-8


def test_boost_zero_velocity():
    """Boost with zero velocity should be identity."""
    p = np.array([10.0, 3.0, 4.0, 5.0])
    beta = np.array([0.0, 0.0, 0.0])
    p_boosted = boost(p, beta)
    np.testing.assert_allclose(p_boosted, p, atol=1e-10)


def test_p3mag():
    p = np.array([5.0, 3.0, 4.0, 0.0])
    assert abs(p3mag(p) - 5.0) < 1e-10


def test_p3hat_unit():
    """p3hat should return a unit vector."""
    p = np.array([100.0, 3.0, 4.0, 5.0])
    hat = p3hat(p)
    assert abs(np.linalg.norm(hat) - 1.0) < 1e-10


def test_beta_vec_magnitude():
    """Beta should be less than 1 for massive particles."""
    m = 1.77686
    p = np.array([np.sqrt(100 + m**2), 10.0, 0.0, 0.0])
    beta = beta_vec(p)
    assert np.linalg.norm(beta) < 1.0

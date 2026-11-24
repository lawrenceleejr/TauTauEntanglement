"""Unit tests for the version-2 analysis code (v2/)."""
import numpy as np
import pytest

from v2 import kin, reco, analysis as an
import beam_parameters as bp


def test_boost_round_trip():
    rng = np.random.default_rng(0)
    p = kin.four(np.full(100, 50.0), rng.normal(size=(100, 3)) * 10)
    beta = rng.uniform(-0.5, 0.5, size=(100, 3)) / np.sqrt(3)
    back = kin.boost_from_rest(kin.boost_to_rest(p, beta), beta)
    np.testing.assert_allclose(back, p, atol=1e-9)
    np.testing.assert_allclose(kin.mass2(kin.boost_to_rest(p, beta)), kin.mass2(p), atol=1e-8)


def test_vmin_spacelike_pair():
    c = kin.C_LIGHT
    dv_m = np.array([[1e-3 / c, 1e-3, 0, 0]])
    dv_p = np.array([[2e-3 / c, -2e-3, 0, 0]])
    assert kin.vmin(dv_m, dv_p)[0] == pytest.approx(3.0)


def test_cone_solution_recovers_true_direction():
    """With an exact Higgs four-momentum, one of the two cone solutions is
    the true τ- direction."""
    rng = np.random.default_rng(1)
    n = 200
    k = kin.unit(rng.normal(size=(n, 3)))
    e = reco.M_H / 2
    p = np.sqrt(e**2 - reco.M_TAU**2)
    ph = kin.four(np.full(n, np.hypot(reco.M_H, 50.0)), kin.unit(rng.normal(size=(n, 3))) * 50.0)
    bh = kin.beta_of(ph)
    pis = []
    for sign in (1, -1):
        q = kin.unit(rng.normal(size=(n, 3)))
        ps = (reco.M_TAU**2 - reco.M_PI**2) / (2 * reco.M_TAU)
        es = (reco.M_TAU**2 + reco.M_PI**2) / (2 * reco.M_TAU)
        pi_rf = kin.four(np.full(n, es), ps * q)
        pi_h = kin.boost_from_rest(pi_rf, (sign * p / e) * k)
        pis.append(kin.boost_from_rest(pi_h, bh))
    (n1, n2), resid, _ = reco.cone_solutions(pis[0], pis[1], ph)
    err = np.minimum(np.arccos(np.clip(kin.dot3(n1, k), -1, 1)),
                     np.arccos(np.clip(kin.dot3(n2, k), -1, 1)))
    assert np.median(err) < 1e-6
    assert np.all(resid < 1e-9)


def test_density_ratio_recovers_R_at_truth():
    """Moment matching returns D = diag(1, 1, -1), κ = 1 for the exact SM
    density against the uncorrelated one."""
    rng = np.random.default_rng(2)
    R = np.array([1.0, 1.0, -1.0])

    def iso(m):
        return kin.unit(rng.normal(size=(m, 3)))
    a, b = iso(800000), iso(800000)
    keep = rng.uniform(0, 1, len(a)) < 0.5 * (1 - np.sum(b * a * R, axis=1))
    q_sm = (a[keep], b[keep])
    q_u = (iso(400000), iso(400000))
    kappa, D = an.fit_density_ratio(q_sm, q_u)
    np.testing.assert_allclose(D.reshape(3, 3), np.diag(R), atol=0.03)
    assert kappa == pytest.approx(1.0, abs=0.01)


def test_cls_crossing_interpolates():
    v = np.array([1.0, 2.0, 4.0, 8.0])
    cls = np.array([1e-4, 1e-3, 0.1, 0.5])
    x = an.crossing(v, cls)
    assert 2.0 < x < 4.0


def test_luminous_region_derivation():
    # Piwinski-shortened luminous length is well below the bunch length
    assert bp.LUMINOUS_REGION_MM[2] < bp.SIGMA_Z_BUNCH_M * 1e3 / np.sqrt(2)
    assert bp.LUMINOUS_REGION_MM[0] == pytest.approx(13e-3 / np.sqrt(2))

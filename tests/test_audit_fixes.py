"""Regression tests for the defects found in the audit of arXiv:2603.28868."""
import os

import numpy as np
import pytest

import config
from parse_hepmc import check_sqrts, sample_sqrts, parse_events
from spin_analysis import extract_correlation_matrix

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "EventSample100.hepmc")


@pytest.fixture
def restore_sqrts():
    before = config.SQRT_S
    yield
    config.set_sqrts(before)


def test_luminosity_units():
    """1 ab^-1 = 1000 fb^-1: 4462 pi-pi events need ~750 ab^-1, not 0.75."""
    sigma_eff_fb = (config.SIGMA_ZH_FB * config.BR_H_TAUTAU * config.BR_Z_MUMU
                    * config.BR_TAU_PI_NU**2)
    lumi_fb, lumi_ab = config.lumi_from_events(4462, sigma_eff_fb)
    assert lumi_ab == pytest.approx(lumi_fb / 1000.0)
    assert lumi_ab == pytest.approx(752.5, rel=1e-3)
    # events per ab^-1 in the mu mu x pi pi channel at 100% efficiency
    assert sigma_eff_fb * 1000.0 == pytest.approx(5.93, rel=1e-2)


def test_bundled_sample_is_250_gev():
    assert sample_sqrts(SAMPLE) == pytest.approx(250.0, abs=0.01)


def test_sqrts_mismatch_is_fatal(restore_sqrts):
    config.set_sqrts(240.0)
    with pytest.raises(RuntimeError, match="250"):
        check_sqrts(SAMPLE)
    with pytest.raises(RuntimeError):
        parse_events(SAMPLE)
    config.set_sqrts(250.0)
    assert check_sqrts(SAMPLE) == pytest.approx(250.0, abs=0.01)


def test_set_sqrts_updates_beam_momenta(restore_sqrts):
    config.set_sqrts(250.0)
    np.testing.assert_allclose(config.P_BEAM_TOTAL, [250.0, 0, 0, 0])
    np.testing.assert_allclose(config.P_BEAM_MINUS, [125.0, 0, 0, 125.0])


def test_recoil_mass_uses_runtime_sqrts(restore_sqrts):
    """The reconstructed Higgs must be built from the configured beams."""
    from tau_reconstruction import reconstruct_event, mass
    config.set_sqrts(250.0)
    events = parse_events(SAMPLE, allowed_modes=["pi_nu", "rho_nu"])
    masses = [mass(config.P_BEAM_TOTAL - e.mu_plus_p4 - e.mu_minus_p4)
              for e in events]
    assert 124.0 < np.percentile(masses, 25) < 126.0


def test_acoplanarity_coefficient():
    assert config.B_ACOPLANARITY_SM == pytest.approx(-0.6169, abs=1e-4)


def test_polarisation_signs():
    """tau- -> pi- nu has analysing power +1, tau+ -> pi+ nu has -1."""
    rng = np.random.default_rng(1)
    n = 200_000
    # fully polarised along +k for both: pi- follows the spin, pi+ opposes it
    def sample(alpha):
        c = rng.uniform(-1, 1, 4 * n)
        c = c[rng.uniform(0, 2, 4 * n) < 1 + alpha * c][:n]
        ph = rng.uniform(0, 2 * np.pi, n)
        s = np.sqrt(1 - c**2)
        return np.stack([s * np.cos(ph), s * np.sin(ph), c], axis=1)
    _, b_plus, b_minus = extract_correlation_matrix(sample(-1.0), sample(+1.0))
    assert b_plus[2] == pytest.approx(1.0, abs=0.02)
    assert b_minus[2] == pytest.approx(1.0, abs=0.02)

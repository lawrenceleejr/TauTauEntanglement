"""
ILC detector resolution model for smearing truth-level quantities.

Simulates the effect of finite detector resolution on:
  - Charged track momenta (pT, theta, phi)
  - Track positions / impact parameters (d0, z0)

Resolution model based on the ILD detector concept:
  - sigma(1/pT) = sqrt((2e-5)^2 + (1e-3/(pT sin theta))^2) GeV^-1
  - sigma(d0) = sigma(z0) = sqrt((5 um)^2 + (10 um GeV/(p sin^{3/2} theta))^2)
  - sigma(theta) = sigma(phi) = 0.1 mrad

Reference: ILD Interim Design Report, arXiv:2003.01116
"""
import numpy as np
from copy import deepcopy
from config import M_PI, M_MU

# ---------------------------------------------------------------------------
# ILD resolution parameters
# ---------------------------------------------------------------------------
_A_PT = 2e-5            # sigma(1/pT) asymptotic term [GeV^-1]
_B_PT = 1e-3            # sigma(1/pT) MS term [GeV^-1], divide by (pT sinθ)
_A_D0 = 5e-6            # sigma(d0) constant term [m] = 5 um
_B_D0 = 10e-6           # sigma(d0) MS term [m GeV] = 10 um GeV, divide by (p sin^{3/2} theta)
_SIGMA_THETA = 1e-4     # angular resolution [rad] = 0.1 mrad
_SIGMA_PHI = 1e-4        # angular resolution [rad] = 0.1 mrad


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------

def sigma_inv_pt(pT, theta):
    """Track momentum resolution sigma(1/pT) [GeV^-1]."""
    sin_th = max(abs(np.sin(theta)), 0.1)
    return np.sqrt(_A_PT**2 + (_B_PT / (pT * sin_th))**2)


def sigma_d0(p, theta):
    """Impact parameter resolution sigma(d0) [m]."""
    sin_th = max(abs(np.sin(theta)), 0.1)
    return np.sqrt(_A_D0**2 + (_B_D0 / (p * sin_th**1.5))**2)


# ---------------------------------------------------------------------------
# Smearing functions
# ---------------------------------------------------------------------------

def smear_p4(p4, particle_mass, rng):
    """Smear a charged particle 4-momentum using ILD tracking resolution.

    Smears (1/pT, theta, phi) independently with Gaussian noise,
    then reconstructs the on-shell 4-vector.

    Parameters
    ----------
    p4 : np.ndarray, shape (4,)
        Truth 4-momentum (E, px, py, pz) [GeV]
    particle_mass : float
        Particle mass [GeV]
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (4,)
        Smeared 4-momentum (on-shell)
    """
    px, py, pz = p4[1], p4[2], p4[3]
    pT = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pT, pz)
    phi = np.arctan2(py, px)

    if pT < 1e-6:
        return p4.copy()

    # Smear 1/pT
    sig = sigma_inv_pt(pT, theta)
    inv_pT_s = (1.0 / pT) + rng.normal(0, sig)
    if inv_pT_s <= 1e-6:
        inv_pT_s = 1.0 / pT
    pT_s = 1.0 / inv_pT_s

    # Smear angles
    theta_s = theta + rng.normal(0, _SIGMA_THETA)
    theta_s = np.clip(theta_s, 0.01, np.pi - 0.01)
    phi_s = phi + rng.normal(0, _SIGMA_PHI)

    # Reconstruct on-shell 4-vector
    px_s = pT_s * np.cos(phi_s)
    py_s = pT_s * np.sin(phi_s)
    pz_s = pT_s / np.tan(theta_s)
    E_s = np.sqrt(px_s**2 + py_s**2 + pz_s**2 + particle_mass**2)

    return np.array([E_s, px_s, py_s, pz_s])


def _smear_track_point(decay_vtx_xyz, pi_hat, sigma, rng):
    """Shift a track point perpendicular to the track direction.

    Simulates the d0/z0 resolution by displacing the track line
    transversely.  The Jeans method sees a shifted impact parameter.

    Parameters
    ----------
    decay_vtx_xyz : np.ndarray, shape (3,)
        Truth point on the track [m]
    pi_hat : np.ndarray, shape (3,)
        (Smeared) track direction unit vector
    sigma : float
        Impact parameter resolution [m]
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (3,)
        Shifted point on the track
    """
    # Two orthogonal directions perpendicular to pi_hat
    if abs(pi_hat[2]) < 0.9:
        arb = np.array([0.0, 0.0, 1.0])
    else:
        arb = np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(pi_hat, arb)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(pi_hat, perp1)

    shift = rng.normal(0, sigma) * perp1 + rng.normal(0, sigma) * perp2
    return decay_vtx_xyz + shift


def smear_event(event, rng):
    """Apply ILD-like detector smearing to a truth-level event.

    Smears:
      1. Muon momenta (for Z/Higgs reconstruction)
      2. Charged pion momenta (for spin analysis and Jeans reconstruction)
      3. Track positions (impact parameter resolution for Jeans reconstruction)

    Truth tau_p4, neutrino_p4, and production_vertex are preserved unchanged.

    Parameters
    ----------
    event : EventRecord
        Truth-level event
    rng : np.random.Generator

    Returns
    -------
    EventRecord
        New event with smeared "measured" quantities.
    """
    evt = deepcopy(event)

    # 1. Smear muon momenta
    evt.mu_minus_p4 = smear_p4(event.mu_minus_p4, M_MU, rng)
    evt.mu_plus_p4 = smear_p4(event.mu_plus_p4, M_MU, rng)

    # 2. Smear pion momenta and track positions
    for tau_s, tau_t in [(evt.tau_minus, event.tau_minus),
                         (evt.tau_plus, event.tau_plus)]:
        if tau_s is None:
            continue

        p_pi = tau_t.charged_pion_p4

        # Smear pion 4-momentum
        tau_s.charged_pion_p4 = smear_p4(p_pi, M_PI, rng)
        if tau_s.decay_mode == "pi_nu":
            tau_s.visible_p4 = tau_s.charged_pion_p4.copy()

        # Smear track position (decay vertex used as point on track)
        dv_xyz = tau_t.decay_vertex[1:4]
        pi_3 = tau_s.charged_pion_p4[1:4]
        pi_hat = pi_3 / max(np.linalg.norm(pi_3), 1e-30)

        p_mag = np.linalg.norm(p_pi[1:4])
        pT = np.sqrt(p_pi[1]**2 + p_pi[2]**2)
        theta = np.arctan2(pT, p_pi[3])
        sig = sigma_d0(p_mag, theta)

        tau_s.decay_vertex = tau_s.decay_vertex.copy()
        tau_s.decay_vertex[1:4] = _smear_track_point(dv_xyz, pi_hat, sig, rng)

    return evt


def print_resolution_summary():
    """Print the detector resolution model parameters and typical values."""
    print("\n  ILD-like detector resolution model:")
    print(f"    Momentum: sigma(1/pT) = ({_A_PT:.0e}) + ({_B_PT:.0e}/(pT sin theta)) GeV^-1")
    print(f"    Impact:   sigma(d0) = ({_A_D0*1e6:.0f} um) + ({_B_D0*1e6:.0f} um GeV/(p sin^3/2 theta))")
    print(f"    Angular:  sigma(theta,phi) = {_SIGMA_THETA*1e3:.1f} mrad")

    # Typical values for tau->pi nu at ILC
    pT_typ = 20.0    # GeV
    theta_typ = np.pi / 4
    p_typ = pT_typ / np.sin(theta_typ)

    sig_pt_rel = sigma_inv_pt(pT_typ, theta_typ) * pT_typ
    sig_d = sigma_d0(p_typ, theta_typ)

    print(f"\n    Typical pi+/- (pT={pT_typ:.0f} GeV, theta=45 deg):")
    print(f"      sigma(pT)/pT = {sig_pt_rel*100:.3f}%")
    print(f"      sigma(d0)    = {sig_d*1e6:.1f} um")
    print(f"      sigma(theta) = {_SIGMA_THETA*1e6:.0f} urad")

    # Impact parameter significance
    L_typ = 3e-3    # 3 mm typical tau decay length
    alpha_typ = 8e-4  # typical tau-pion opening angle
    d_typ = L_typ * alpha_typ
    print(f"\n    Typical tau IP ~ {d_typ*1e6:.1f} um, "
          f"sigma(d0) ~ {sig_d*1e6:.1f} um, "
          f"S/N ~ {d_typ/sig_d:.2f}")

"""
Spin correlation matrix extraction for H -> tau+ tau-.

For tau -> pi nu, the pion direction in the tau rest frame is a perfect
spin analyser (analysing power = 1). The spin correlation matrix C_ij
is extracted from the angular distributions:

  C_ij = -9 * <cos(theta_i^+) * cos(theta_j^-)>

where theta_i^pm is the angle of the pi^pm momentum (in the tau^pm rest
frame) projected onto axis i of the {n, r, k} basis defined in the
tau-pair rest frame (= Higgs rest frame for H -> tau tau).

Sign convention: with analysing powers alpha_- = +1 (tau- -> pi- nu:
pion preferentially along the tau- spin) and alpha_+ = -1 (CP), the
joint distribution is
  (1/sigma) d^2 sigma / d(cos theta_i^+) d(cos theta_j^-) =
      (1/4)(1 - C_ij cos theta_i^+ cos theta_j^-)

This gives C_ij = -9 * <cos theta_i^+ cos theta_j^->

For pi-nu mode with unit analysing power, the factor 9 comes from:
  <cos^2 theta> = 1/3 for a uniform distribution on the sphere.

The single-tau polarisation B_i is extracted from:
  B_i^- = +3 * <cos theta_i^->     (alpha_- = +1: pion along tau- spin)
  B_i^+ = -3 * <cos theta_i^+>     (alpha_+ = -1, from CP)
Both vanish for a spin-0 parent; they serve as null diagnostics here.
"""
import numpy as np
from config import P_BEAM_MINUS, P_BEAM_TOTAL, M_TAU
from tau_reconstruction import boost, p3hat, p3vec, p3mag, beta_vec, mass, mass2


# ---------------------------------------------------------------------------
# Hadronic polarimeter vectors
# ---------------------------------------------------------------------------

def polarimeter_direction(q_rf, N_rf):
    """Polarimeter direction for a hadronic tau decay, in the tau rest frame.

    For a real hadronic current J ~ q the polarimeter 4-vector is
        H^mu = 2 (q.N) q^mu - q^2 N^mu ,
    with q the hadronic momentum (q = p_pi for pi_nu; q = p_pi - p_pi0 for
    rho_nu) and N the neutrino 4-momentum [Kuhn, Phys. Lett. B 313 (1993)].
    Since N^2 = 0, H is exactly lightlike (H.H = 0), so |H_vec| = H^0 and
    the polarimeter DIRECTION carries unit analysing power:
        dGamma ~ 1 + s . h,   h = H_vec / |H_vec| .
    H is quadratic in q, so the pi/pi0 ordering sign is irrelevant.
    For pi_nu this reduces to the pion direction.

    Parameters are 4-vectors in the tau rest frame.
    """
    qN = q_rf[0] * N_rf[0] - np.dot(q_rf[1:4], N_rf[1:4])
    q2 = mass2(q_rf)
    H_vec = 2.0 * qN * q_rf[1:4] - q2 * N_rf[1:4]
    norm = np.linalg.norm(H_vec)
    if norm < 1e-30:
        return np.array([0.0, 0.0, 1.0])
    return H_vec / norm


def _tau_polarimeter(tau_info, p_pi_H, pi0s_H, beta_tau, nu_H=None):
    """Compute the polarimeter direction h-hat for one tau.

    All inputs are 4-vectors in the Higgs rest frame; beta_tau boosts into
    the tau rest frame.  If nu_H (the neutrino 4-momentum in the Higgs
    frame) is given it is used directly (truth mode); otherwise the
    neutrino is inferred from the tau-mass constraint in the tau rest
    frame, N = (m_tau, 0) - p_vis (reco mode).
    """
    p_pi_rf = boost(p_pi_H, beta_tau)

    if tau_info.decay_mode == "rho_nu" and len(pi0s_H) >= 1:
        p_pi0_rf = boost(pi0s_H[0], beta_tau)
        q_rf = p_pi_rf - p_pi0_rf
        if nu_H is not None:
            N_rf = boost(nu_H, beta_tau)
        else:
            p_vis_rf = p_pi_rf + p_pi0_rf
            N_rf = np.array([M_TAU, 0.0, 0.0, 0.0]) - p_vis_rf
        return polarimeter_direction(q_rf, N_rf)

    # pi_nu: the pion direction is the polarimeter (alpha = 1)
    return p3hat(p_pi_rf)


def _define_basis(p_tau_minus, p_beam_minus):
    """Define the {n, r, k} orthonormal basis in the tau-pair rest frame.

    k = tau- flight direction (helicity axis)
    r = component of beam direction perpendicular to k, normalised
    n = k x r (normal to the scattering plane)

    Parameters
    ----------
    p_tau_minus : np.ndarray
        tau- 4-momentum in the tau-pair rest frame (Higgs rest frame)
    p_beam_minus : np.ndarray
        e- beam 4-momentum in the tau-pair rest frame

    Returns
    -------
    k_hat, r_hat, n_hat : np.ndarray (3,)
    """
    k_hat = p3hat(p_tau_minus)

    # Beam direction in the Higgs rest frame
    beam_hat = p3hat(p_beam_minus)

    # r is the component of beam perpendicular to k, normalised
    cos_theta = np.dot(beam_hat, k_hat)
    r_vec = beam_hat - cos_theta * k_hat
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-10:
        # Tau along beam -- pick arbitrary perpendicular
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arb, k_hat)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        r_vec = arb - np.dot(arb, k_hat) * k_hat
        r_norm = np.linalg.norm(r_vec)
    r_hat = r_vec / r_norm

    # n = k x r
    n_hat = np.cross(k_hat, r_hat)

    return k_hat, r_hat, n_hat


def compute_spin_observables(event, reco, use_reco_tau=True):
    """Compute the spin-sensitive angular observables for a single event.

    Parameters
    ----------
    event : EventRecord
        The parsed event (for truth tau 4-momenta and pion 4-momenta)
    reco : dict
        The reconstruction result (for reconstructed tau 4-momenta)
    use_reco_tau : bool
        If True, use reconstructed tau momenta for the boost.
        If False, use truth tau momenta.

    Returns
    -------
    dict with:
        'cos_theta_plus'  : array of 3 (n, r, k projections for tau+)
        'cos_theta_minus' : array of 3 (n, r, k projections for tau-)
        'acoplanarity'    : delta phi between the two decay planes
    """
    # Decay-product momenta (the "measured" tracks/clusters)
    p_pi_m = event.tau_minus.charged_pion_p4
    p_pi_p = event.tau_plus.charged_pion_p4
    pi0s_m = event.tau_minus.neutral_pions_p4
    pi0s_p = event.tau_plus.neutral_pions_p4

    if use_reco_tau and reco is not None:
        # CRITICAL: Use the Higgs 4-momentum from beam - Z (measured muons),
        # NOT from the reco tau sum, which has large errors and corrupts the
        # Higgs rest frame boost.
        p_H = reco['p_H']
        beta_H = beta_vec(p_H)
        m_H = mass(p_H)

        # Boost BOTH reco taus independently to Higgs RF to get directions.
        # Using independent directions avoids correlated errors: the
        # back-to-back constraint locks tau+ and tau- direction errors
        # to be anti-correlated, which introduces a systematic negative
        # bias in the transverse C_ij (C_nn, C_rr).
        p_tau_m_H_reco = boost(reco['tau_minus']['p_tau_reco'], beta_H)
        p_tau_p_H_reco = boost(reco['tau_plus']['p_tau_reco'], beta_H)
        tau_m_dir = p3hat(p_tau_m_H_reco)
        tau_p_dir = p3hat(p_tau_p_H_reco)

        # Apply kinematic constraints: energy E = m_H/2 and
        # momentum |p| = sqrt(E^2 - m_tau^2) are fixed by the Higgs mass.
        # The Jeans reco gives the direction; kinematics fixes the magnitude.
        E_tau = m_H / 2.0
        p_tau_mag = np.sqrt(max(E_tau**2 - M_TAU**2, 0.0))
        p_tau_m_H = np.array([E_tau, *(p_tau_mag * tau_m_dir)])
        p_tau_p_H = np.array([E_tau, *(p_tau_mag * tau_p_dir)])

        # Boost decay products to this (correct) Higgs rest frame
        p_pi_m_H = boost(p_pi_m, beta_H)
        p_pi_p_H = boost(p_pi_p, beta_H)
        pi0s_m_H = [boost(p, beta_H) for p in pi0s_m]
        pi0s_p_H = [boost(p, beta_H) for p in pi0s_p]
        p_beam_m_H = boost(P_BEAM_MINUS, beta_H)
        # Neutrinos inferred from the tau-mass constraint in reco mode
        nu_m_H = None
        nu_p_H = None
    else:
        # Truth: use exact tau momenta and the truth neutrinos
        p_tau_m = event.tau_minus.tau_p4
        p_tau_p = event.tau_plus.tau_p4
        p_H = p_tau_m + p_tau_p
        beta_H = beta_vec(p_H)
        p_tau_m_H = boost(p_tau_m, beta_H)
        p_tau_p_H = boost(p_tau_p, beta_H)
        p_pi_m_H = boost(p_pi_m, beta_H)
        p_pi_p_H = boost(p_pi_p, beta_H)
        pi0s_m_H = [boost(p, beta_H) for p in pi0s_m]
        pi0s_p_H = [boost(p, beta_H) for p in pi0s_p]
        p_beam_m_H = boost(P_BEAM_MINUS, beta_H)
        nu_m_H = boost(event.tau_minus.neutrino_p4, beta_H)
        nu_p_H = boost(event.tau_plus.neutrino_p4, beta_H)

    # Define the {n, r, k} basis using tau- direction in Higgs rest frame
    k_hat, r_hat, n_hat = _define_basis(p_tau_m_H, p_beam_m_H)
    basis = np.array([n_hat, r_hat, k_hat])  # shape (3, 3), rows are basis vecs

    # Compute the polarimeter direction in each tau rest frame.
    #
    # WHY TAU REST FRAMES? The polarimeter direction h-hat in the tau rest
    # frame is the spin analyser: dGamma ~ (1 + s . h) with unit analysing
    # power for fully reconstructed pi_nu and rho_nu decays (see
    # polarimeter_direction).  The spin information lives in the tau RF.
    #
    # WHY IS THE {n,r,k} BASIS STILL VALID? The boost from Higgs RF to each
    # tau RF is along k-hat (the tau flight direction). The transverse axes
    # {n, r} are perpendicular to the boost and therefore identical in both
    # frames. The k-hat direction is parallel to the boost and also unchanged.
    # So projecting h-hat (computed in the tau RF) onto {n, r, k}
    # (defined in Higgs RF) is correct — the basis vectors are the same in
    # both frames.
    beta_tau_m = beta_vec(p_tau_m_H)
    beta_tau_p = beta_vec(p_tau_p_H)

    h_m_hat = _tau_polarimeter(event.tau_minus, p_pi_m_H, pi0s_m_H,
                               beta_tau_m, nu_H=nu_m_H)
    h_p_hat = _tau_polarimeter(event.tau_plus, p_pi_p_H, pi0s_p_H,
                               beta_tau_p, nu_H=nu_p_H)

    # Project onto the {n, r, k} basis (same in Higgs RF and tau RF, see above)
    cos_theta_minus = basis @ h_m_hat  # shape (3,): (n, r, k) components
    cos_theta_plus = basis @ h_p_hat

    # Acoplanarity angle
    # phi of each pion in the tau rest frame, w.r.t. the {n, r} plane
    phi_minus = np.arctan2(cos_theta_minus[0], cos_theta_minus[1])  # atan2(n, r)
    phi_plus = np.arctan2(cos_theta_plus[0], cos_theta_plus[1])
    delta_phi = phi_plus - phi_minus
    # Wrap to [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi

    return {
        'cos_theta_plus': cos_theta_plus,
        'cos_theta_minus': cos_theta_minus,
        'acoplanarity': delta_phi,
    }


def extract_correlation_matrix(cos_theta_plus_arr, cos_theta_minus_arr):
    """Extract the spin correlation matrix C_ij from arrays of angular observables.

    Parameters
    ----------
    cos_theta_plus_arr : np.ndarray, shape (N, 3)
        Pion projections for tau+ events (n, r, k)
    cos_theta_minus_arr : np.ndarray, shape (N, 3)
        Pion projections for tau- events (n, r, k)

    Returns
    -------
    C_ij : np.ndarray, shape (3, 3)
        Spin correlation matrix
    B_plus : np.ndarray, shape (3,)
        tau+ polarisation vector
    B_minus : np.ndarray, shape (3,)
        tau- polarisation vector
    """
    N = len(cos_theta_plus_arr)
    if N == 0:
        return np.zeros((3, 3)), np.zeros(3), np.zeros(3)

    # C_ij = -9 * <cos_theta_i^+ * cos_theta_j^->
    # The sign accounts for the tau+ analysing power alpha_+ = -1
    # (pion emitted opposite to spin for tau+), so
    # alpha_+ * alpha_- = (-1)(+1) = -1.
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = -9.0 * np.mean(cos_theta_plus_arr[:, i] *
                                       cos_theta_minus_arr[:, j])

    # Single-tau polarisations.  With alpha_- = +1 and alpha_+ = -1
    # (dGamma ~ 1 + alpha s.h), <h_i^-> = +B_i^-/3 and <h_i^+> = -B_i^+/3.
    # For H -> tautau both vanish (spin-0 parent); these are diagnostics.
    B_plus = -3.0 * np.mean(cos_theta_plus_arr, axis=0)
    B_minus = +3.0 * np.mean(cos_theta_minus_arr, axis=0)

    return C, B_plus, B_minus

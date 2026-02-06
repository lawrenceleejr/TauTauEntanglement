"""
Spin correlation matrix extraction for H -> tau+ tau-.

For tau -> pi nu, the pion direction in the tau rest frame is a perfect
spin analyser (analysing power = 1). The spin correlation matrix C_ij
is extracted from the angular distributions:

  C_ij = -9 * <cos(theta_i^+) * cos(theta_j^-)>

where theta_i^pm is the angle of the pi^pm momentum (in the tau^pm rest
frame) projected onto axis i of the {n, r, k} basis defined in the
tau-pair rest frame (= Higgs rest frame for H -> tau tau).

Sign convention: we use the convention from arXiv:2602.03960 where
the distribution is
  (1/sigma) d^2 sigma / d(cos theta_i^+) d(cos theta_j^-) =
      (1/4)(1 + C_ij cos theta_i^+ cos theta_j^-)

This gives C_ij = 9 * <cos theta_i^+ cos theta_j^->

For pi-nu mode with unit analysing power, the factor 9 comes from:
  <cos^2 theta> = 1/3 for a uniform distribution on the sphere.

The single-tau polarisation B_i is extracted from:
  B_i^- = -3 * <cos theta_i^->
  B_i^+ = +3 * <cos theta_i^+>
(sign from the tau+/tau- convention in arXiv:2602.03960)
"""
import numpy as np
from config import P_BEAM_MINUS
from tau_reconstruction import boost, p3hat, p3vec, p3mag, beta_vec


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
    # Get tau momenta
    if use_reco_tau and reco is not None:
        p_tau_m = reco['tau_minus']['p_tau_reco']
        p_tau_p = reco['tau_plus']['p_tau_reco']
    else:
        p_tau_m = event.tau_minus.tau_p4
        p_tau_p = event.tau_plus.tau_p4

    # Pion momenta (always from truth -- these are the "measured" decay products)
    p_pi_m = event.tau_minus.charged_pion_p4
    p_pi_p = event.tau_plus.charged_pion_p4

    # Higgs rest frame = tau-pair rest frame
    p_H = p_tau_m + p_tau_p
    beta_H = beta_vec(p_H)

    # Boost everything to Higgs rest frame
    p_tau_m_H = boost(p_tau_m, beta_H)
    p_tau_p_H = boost(p_tau_p, beta_H)
    p_pi_m_H = boost(p_pi_m, beta_H)
    p_pi_p_H = boost(p_pi_p, beta_H)
    p_beam_m_H = boost(P_BEAM_MINUS, beta_H)

    # Define the {n, r, k} basis using tau- direction in Higgs rest frame
    k_hat, r_hat, n_hat = _define_basis(p_tau_m_H, p_beam_m_H)
    basis = np.array([n_hat, r_hat, k_hat])  # shape (3, 3), rows are basis vecs

    # Boost pions to their parent tau rest frames.
    #
    # WHY TAU REST FRAMES? The pion direction in the tau rest frame is the
    # spin analyser: dGamma/d(cos theta) ~ (1 + alpha_pi * P * cos theta)
    # with alpha_pi = 1 for tau -> pi nu. The spin information lives in
    # the tau RF, not the Higgs RF.
    #
    # WHY IS THE {n,r,k} BASIS STILL VALID? The boost from Higgs RF to each
    # tau RF is along k-hat (the tau flight direction). The transverse axes
    # {n, r} are perpendicular to the boost and therefore identical in both
    # frames. The k-hat direction is parallel to the boost and also unchanged.
    # So projecting the pion direction (measured in tau RF) onto {n, r, k}
    # (defined in Higgs RF) is correct — the basis vectors are the same in
    # both frames.
    beta_tau_m = beta_vec(p_tau_m_H)
    beta_tau_p = beta_vec(p_tau_p_H)

    p_pi_m_taurf = boost(p_pi_m_H, beta_tau_m)
    p_pi_p_taurf = boost(p_pi_p_H, beta_tau_p)

    # Unit vectors of pion momenta in tau rest frames
    pi_m_hat = p3hat(p_pi_m_taurf)
    pi_p_hat = p3hat(p_pi_p_taurf)

    # Project onto the {n, r, k} basis (same in Higgs RF and tau RF, see above)
    cos_theta_minus = basis @ pi_m_hat  # shape (3,): (n, r, k) components
    cos_theta_plus = basis @ pi_p_hat

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

    # C_ij = 9 * <cos_theta_i^+ * cos_theta_j^->
    # Using outer products and averaging
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = 9.0 * np.mean(cos_theta_plus_arr[:, i] *
                                      cos_theta_minus_arr[:, j])

    # Single-tau polarisations
    B_plus = 3.0 * np.mean(cos_theta_plus_arr, axis=0)
    B_minus = -3.0 * np.mean(cos_theta_minus_arr, axis=0)

    return C, B_plus, B_minus

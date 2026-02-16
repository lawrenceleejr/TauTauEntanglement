"""
Tau kinematic reconstruction using the Jeans impact parameter method.

For tau -> pi nu (single-prong, no neutrals), the reconstruction is
particularly clean:

  1. We know the beam 4-momentum exactly => reconstruct pH = pbeam - pZ
  2. The pion track defines a line in 3D.  Together with the production
     vertex (PV), the PCA of the track to PV gives an impact parameter
     vector d.  The tau momentum is constrained to lie in the plane
     spanned by d and the pion direction (the "track plane").
  3. Within this plane, the tau direction is parameterised by an angle alpha
     relative to the pion direction.
  4. The tau mass constraint m_tau^2 = (p_pi + p_nu)^2 provides a
     quadratic equation for the tau momentum magnitude |p_tau|.
  5. The decay length is determined geometrically: L = |d| / sin(alpha).
  6. The decay vertex position is PV + L * tau_hat.
  7. The decay time is t = L / (beta * c).
  8. Total missing momentum (p_H - p_vis1 - p_vis2 = p_nu1 + p_nu2)
     provides additional constraints to resolve the two-fold ambiguity.

Reference: arXiv:1507.01700 (Jeans), arXiv:1804.01241 (Jeans & Wilson)
"""
import numpy as np
from config import M_TAU, M_PI, M_HIGGS, CTAU_TAU, C_LIGHT, SQRT_S, P_BEAM_TOTAL
from parse_hepmc import EventRecord
from smearing import sigma_d0


# ---------------------------------------------------------------------------
# Lorentz algebra utilities
# ---------------------------------------------------------------------------
def mass2(p):
    """Invariant mass squared of a 4-vector (E, px, py, pz)."""
    return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2


def mass(p):
    """Invariant mass of a 4-vector."""
    m2 = mass2(p)
    return np.sqrt(max(m2, 0.0))


def p3mag(p):
    """3-momentum magnitude."""
    return np.sqrt(p[1]**2 + p[2]**2 + p[3]**2)


def p3vec(p):
    """Spatial 3-vector from a 4-vector."""
    return p[1:4]


def p3hat(p):
    """Unit 3-vector in the direction of the 3-momentum."""
    v = p3vec(p)
    mag = np.linalg.norm(v)
    if mag < 1e-30:
        return np.array([0.0, 0.0, 1.0])
    return v / mag


def beta_vec(p):
    """Velocity vector beta = p/E for a 4-vector."""
    if p[0] < 1e-30:
        return np.zeros(3)
    return p3vec(p) / p[0]


def boost(p4, beta):
    """Boost a 4-vector p4 = (E, px, py, pz) by velocity beta (3-vector).

    Boosts INTO the rest frame of a particle with velocity beta.
    (i.e., if beta = p_parent/E_parent, this goes to parent rest frame)
    """
    b2 = np.dot(beta, beta)
    if b2 < 1e-30:
        return p4.copy()
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.dot(beta, p4[1:4])
    p4_new = np.empty(4)
    p4_new[0] = gamma * (p4[0] - bp)
    p4_new[1:4] = p4[1:4] + (((gamma - 1.0) * bp / b2) - gamma * p4[0]) * beta
    return p4_new


# ---------------------------------------------------------------------------
# Track geometry
# ---------------------------------------------------------------------------

def compute_impact_parameter(pv_xyz, track_point_xyz, track_dir):
    """Compute the impact parameter geometry for a track relative to PV.

    The track is a line through `track_point_xyz` in direction `track_dir`.
    The impact parameter vector d points from PV to the PCA on the track.

    Parameters
    ----------
    pv_xyz : np.ndarray, shape (3,)
        Production vertex position (x, y, z)
    track_point_xyz : np.ndarray, shape (3,)
        Any point on the track (e.g., truth decay vertex)
    track_dir : np.ndarray, shape (3,)
        Unit vector along the track direction

    Returns
    -------
    d_vec : np.ndarray, shape (3,)
        Impact parameter vector (PV → PCA), perpendicular to track_dir
    d_mag : float
        Impact parameter magnitude |d|
    pca_xyz : np.ndarray, shape (3,)
        Point of closest approach on the track
    """
    # Vector from track point to PV
    w = pv_xyz - track_point_xyz

    # Project onto track direction to find PCA
    s_pca = np.dot(w, track_dir)
    pca_xyz = track_point_xyz + s_pca * track_dir

    # Impact parameter vector: PV to PCA (perpendicular to track)
    d_vec = pca_xyz - pv_xyz
    d_mag = np.linalg.norm(d_vec)

    return d_vec, d_mag, pca_xyz


# ---------------------------------------------------------------------------
# Jeans reconstruction for tau -> pi nu
# ---------------------------------------------------------------------------

def _solve_tau_momentum(p_pi, tau_direction):
    """Solve for the tau momentum magnitude given the pion 4-momentum and
    a candidate tau flight direction.

    For tau -> pi nu with m_nu = 0:
      p_tau = p_pi + p_nu,  m_tau^2 = (p_pi + p_nu)^2
      p_nu lightlike => E_nu = |vec_p_nu|

    Parameterising p_tau = t * tau_hat (3-momentum), E_tau = sqrt(t^2 + m_tau^2):
      The massless neutrino constraint gives a quadratic in t:

      (4 E_pi^2 - 4 B^2) t^2 - 4 A B t + (4 E_pi^2 m_tau^2 - A^2) = 0

    where A = m_tau^2 + m_pi^2, B = tau_hat . vec_p_pi.

    Returns list of (t, p_tau_4vec, p_nu_4vec) solutions with t > 0.
    """
    d = tau_direction
    k = p3vec(p_pi)
    E_k = p_pi[0]

    A = M_TAU**2 + M_PI**2
    B = np.dot(d, k)

    aa = 4.0 * (E_k**2 - B**2)
    bb = -4.0 * A * B
    cc = 4.0 * E_k**2 * M_TAU**2 - A**2

    if abs(aa) < 1e-30:
        return []

    disc = bb**2 - 4.0 * aa * cc
    if disc < 0:
        return []

    solutions = []
    sqrt_disc = np.sqrt(disc)

    for sign in [+1, -1]:
        t = (-bb + sign * sqrt_disc) / (2.0 * aa)
        if t <= 0:
            continue

        E_tau = np.sqrt(t**2 + M_TAU**2)

        # Check the un-squared equation: 2 E_pi sqrt(t^2 + m_tau^2) = A + 2tB
        lhs = A + 2.0 * t * B
        if lhs < 0:
            continue

        p_tau = np.array([E_tau, t * d[0], t * d[1], t * d[2]])
        p_nu = p_tau - p_pi

        # Neutrino must have positive energy
        if p_nu[0] < -0.01:
            continue

        solutions.append((t, p_tau, p_nu))

    return solutions


# Pre-compute the alpha scan grid once at module load
_ALPHA_VALUES = np.concatenate([
    np.logspace(-5, -2, 200),    # 1e-5 to 0.01: dense at small angles
    np.linspace(0.01, 0.5, 100), # 0.01 to 0.5: linear at larger angles
])
_COS_ALPHA = np.cos(_ALPHA_VALUES)
_SIN_ALPHA = np.sin(_ALPHA_VALUES)
_A_CONST = M_TAU**2 + M_PI**2


def _solve_alpha_scan_vectorised(p_pi, pi_hat, d_hat, d_mag):
    """Vectorised alpha scan: solve the mass constraint for all alpha at once.

    Instead of looping over 300 alpha values in Python, this computes all
    tau directions, quadratic coefficients, and solutions in numpy arrays.

    Returns list of dicts (same format as the scalar version).
    """
    k = p_pi[1:4]
    E_k = p_pi[0]
    A = _A_CONST

    # tau_dirs: (N_alpha, 3) — tau direction for each alpha
    # tau_dir = cos(alpha) * pi_hat + sin(alpha) * d_hat  (already unit)
    tau_dirs = _COS_ALPHA[:, None] * pi_hat[None, :] + _SIN_ALPHA[:, None] * d_hat[None, :]
    # Normalise rows (should be ~1 already since pi_hat . d_hat ~ 0)
    norms = np.linalg.norm(tau_dirs, axis=1, keepdims=True)
    tau_dirs /= norms

    # B(alpha) = tau_dir . k   — shape (N_alpha,)
    B = tau_dirs @ k

    # Quadratic coefficients — shape (N_alpha,)
    aa = 4.0 * (E_k**2 - B**2)
    bb = -4.0 * A * B
    cc = 4.0 * E_k**2 * M_TAU**2 - A**2

    disc = bb**2 - 4.0 * aa * cc

    # Mask: valid if |aa| > 0 and disc >= 0
    valid = (np.abs(aa) > 1e-30) & (disc >= 0)

    results = []
    if not np.any(valid):
        return results

    # Work only on valid alphas
    idx_valid = np.where(valid)[0]
    aa_v = aa[idx_valid]
    bb_v = bb[idx_valid]
    disc_v = disc[idx_valid]
    B_v = B[idx_valid]
    sqrt_disc_v = np.sqrt(disc_v)

    # Solve for both roots: t_plus and t_minus
    t_plus = (-bb_v + sqrt_disc_v) / (2.0 * aa_v)
    t_minus = (-bb_v - sqrt_disc_v) / (2.0 * aa_v)

    # Process both roots
    for t_arr, sign_label in [(t_plus, '+'), (t_minus, '-')]:
        # Physical constraints: t > 0, A + 2tB > 0, E_nu > 0
        ok = (t_arr > 0) & (A + 2.0 * t_arr * B_v > 0)
        if not np.any(ok):
            continue

        idx_ok = idx_valid[ok]
        t_ok = t_arr[ok]
        E_tau = np.sqrt(t_ok**2 + M_TAU**2)

        # Build p_tau 4-vectors: (N_ok, 4)
        dirs_ok = tau_dirs[idx_ok]  # (N_ok, 3)
        p_tau_3 = t_ok[:, None] * dirs_ok  # (N_ok, 3)

        # p_nu energy check
        E_nu = E_tau - E_k
        p_nu_3 = p_tau_3 - k[None, :]
        nu_ok = E_nu > -0.01

        alpha_ok = _ALPHA_VALUES[idx_ok]
        sin_alpha_ok = _SIN_ALPHA[idx_ok]
        L_ok = d_mag / sin_alpha_ok

        # Filter by nu energy and positive L
        mask = nu_ok & (L_ok > 0)
        for j in np.where(mask)[0]:
            i_alpha = idx_ok[j]
            t_val = t_ok[j]
            E_val = E_tau[j]
            tau_dir_j = dirs_ok[j]
            L_j = L_ok[j]
            p_tau_j = np.array([E_val, p_tau_3[j, 0], p_tau_3[j, 1], p_tau_3[j, 2]])
            p_nu_j = p_tau_j - p_pi

            beta_j = t_val / E_val
            results.append({
                'p_tau': p_tau_j,
                'p_nu': p_nu_j,
                'tau_dir': tau_dir_j,
                'alpha': alpha_ok[j],
                'decay_length_m': L_j,
                'decay_vertex_xyz': None,   # deferred — computed only for best solution
                'decay_vertex_t': None,
                'tau_momentum_mag': t_val,
                '_beta': beta_j,
            })

    return results


def reconstruct_single_tau(p_pi, pv_xyz, decay_vtx_xyz_truth):
    """Reconstruct a single tau -> pi nu using the Jeans impact parameter method.

    Uses the truth decay vertex position ONLY to define the pion track
    geometry (simulating a measured track from the detector). The reconstruction
    then proceeds using only the track + PV + mass constraint.

    Parameters
    ----------
    p_pi : np.ndarray
        Charged pion 4-momentum (E, px, py, pz)
    pv_xyz : np.ndarray
        Production vertex position (x, y, z) in metres
    decay_vtx_xyz_truth : np.ndarray
        Truth decay vertex (x, y, z) in metres — used only to define
        the pion track line (surrogate for a measured detector track)

    Returns
    -------
    list of dict, each with:
        'p_tau', 'p_nu', 'tau_dir', 'alpha',
        'decay_length_m', 'decay_vertex_xyz', 'decay_vertex_t'
    """
    pi_hat = p3hat(p_pi)

    # Step 1: Compute impact parameter geometry
    # The pion track passes through the truth decay vertex in direction pi_hat
    d_vec, d_mag, pca = compute_impact_parameter(pv_xyz, decay_vtx_xyz_truth, pi_hat)

    # Handle degenerate case: pion track passes through PV (d = 0)
    if d_mag < 1e-12:
        # Tau and pion exactly collinear from PV — no track plane defined
        # Fall back to collinear approximation: tau_dir = pi_hat
        solutions = _solve_tau_momentum(p_pi, pi_hat)
        results = []
        for (t, p_tau, p_nu) in solutions:
            # Decay length indeterminate when d=0; use truth as fallback
            L = np.linalg.norm(decay_vtx_xyz_truth - pv_xyz)
            beta = t / p_tau[0]
            results.append({
                'p_tau': p_tau,
                'p_nu': p_nu,
                'tau_dir': pi_hat,
                'alpha': 0.0,
                'decay_length_m': L,
                'decay_vertex_xyz': pv_xyz + L * pi_hat,
                'decay_vertex_t': L / (beta * C_LIGHT),
            })
        return results

    d_hat = d_vec / d_mag

    # Step 2: Define the track plane
    # The tau direction lies in the plane spanned by pi_hat and d_hat.
    # Parameterise: tau_hat = cos(alpha) * pi_hat + sin(alpha) * d_hat
    # where alpha is the opening angle between tau and pion directions.
    #
    # The decay length is then: L = |d| / sin(alpha)
    # (geometric: the perpendicular distance from PV to the pion line
    #  equals L * sin(alpha), which is |d|)

    # Step 3: Vectorised alpha scan — solve mass constraint for all alpha at once
    return _solve_alpha_scan_vectorised(p_pi, pi_hat, d_hat, d_mag)


def reconstruct_event(event: EventRecord):
    """Full event reconstruction using the Jeans method + Higgs constraint.

    For e+e- -> ZH -> mu+mu- tau+tau-, we know:
      p_Z = p_mu+ + p_mu-
      p_H = p_beam - p_Z
      p_tau1 + p_tau2 = p_H

    For each tau -> pi nu, the Jeans method gives candidate (p_tau, decay_vtx)
    solutions. We select the combination that best satisfies the missing
    momentum constraint: p_nu1 + p_nu2 = p_H - p_pi1 - p_pi2.

    Returns dict with reconstructed quantities, or None if reconstruction fails.
    """
    if event.tau_minus is None or event.tau_plus is None:
        return None

    # Reconstruct Higgs 4-momentum
    p_Z = event.mu_plus_p4 + event.mu_minus_p4
    p_H = P_BEAM_TOTAL - p_Z

    # Missing momentum should equal total neutrino momentum
    p_pi_minus = event.tau_minus.charged_pion_p4
    p_pi_plus = event.tau_plus.charged_pion_p4
    p_miss = p_H - p_pi_minus - p_pi_plus

    # Production vertex (PV) — both taus produced at the Higgs decay vertex
    pv_xyz = event.tau_minus.production_vertex[1:4]  # (x, y, z) in metres

    # Truth decay vertices — used only to define the pion tracks
    dv_minus_truth = event.tau_minus.decay_vertex[1:4]
    dv_plus_truth = event.tau_plus.decay_vertex[1:4]

    # Get candidate solutions for each tau
    sols_minus = reconstruct_single_tau(p_pi_minus, pv_xyz, dv_minus_truth)
    sols_plus = reconstruct_single_tau(p_pi_plus, pv_xyz, dv_plus_truth)

    if not sols_minus or not sols_plus:
        return None

    # Select the best combination using the missing momentum constraint.
    # Vectorised: build (N_minus, 4) and (N_plus, 4) p_nu arrays, then
    # broadcast to (N_minus, N_plus, 4) and find the minimum chi2.
    n_m = len(sols_minus)
    n_p = len(sols_plus)

    pnu_m = np.array([s['p_nu'] for s in sols_minus])   # (n_m, 4)
    pnu_p = np.array([s['p_nu'] for s in sols_plus])     # (n_p, 4)

    # dp[i, j, :] = pnu_m[i] + pnu_p[j] - p_miss
    dp = pnu_m[:, None, :] + pnu_p[None, :, :] - p_miss[None, None, :]
    chi2_grid = np.sum(dp**2, axis=2)                     # (n_m, n_p)

    flat_idx = np.argmin(chi2_grid)
    best_i, best_j = divmod(flat_idx, n_p)
    best_chi2 = chi2_grid[best_i, best_j]

    sol_minus = sols_minus[best_i]
    sol_plus = sols_plus[best_j]

    # Materialise deferred vertex/time for the two winning solutions
    for sol in (sol_minus, sol_plus):
        if sol['decay_vertex_xyz'] is None:
            L = sol['decay_length_m']
            sol['decay_vertex_xyz'] = pv_xyz + L * sol['tau_dir']
            sol['decay_vertex_t'] = L / (sol['_beta'] * C_LIGHT)

    result = {
        'p_H': p_H,
        'p_Z': p_Z,
        'p_miss': p_miss,
        'chi2': best_chi2,
    }

    for label, sol, tau_info in [('tau_minus', sol_minus, event.tau_minus),
                                  ('tau_plus', sol_plus, event.tau_plus)]:
        p_tau = sol['p_tau']
        pmag = p3mag(p_tau)
        E = p_tau[0]
        beta = pmag / E
        gamma = E / M_TAU
        beta_gamma = pmag / M_TAU

        result[label] = {
            'p_tau_reco': p_tau,
            'p_nu_reco': sol['p_nu'],
            'tau_dir': sol['tau_dir'],
            'alpha': sol['alpha'],
            'decay_length_m': sol['decay_length_m'],
            'decay_vertex_xyz': sol['decay_vertex_xyz'],
            'decay_vertex_t': sol['decay_vertex_t'],
            'production_vertex_xyz': pv_xyz,
            'beta': beta,
            'gamma': gamma,
            'beta_gamma': beta_gamma,
        }

        # Impact parameter significance for quality cuts
        p_pi = tau_info.charged_pion_p4
        pi_hat = p3hat(p_pi)
        dv_truth = tau_info.decay_vertex[1:4]
        _, d_mag, _ = compute_impact_parameter(pv_xyz, dv_truth, pi_hat)
        p_pi_mag = p3mag(p_pi)
        theta_pi = np.arccos(np.clip(p_pi[3] / p_pi_mag, -1, 1))
        sig_d = sigma_d0(p_pi_mag, theta_pi)
        result[label]['d_mag'] = d_mag
        result[label]['ip_significance'] = d_mag / sig_d if sig_d > 0 else 0.0

        # Truth values for comparison
        truth_decay_xyz = tau_info.decay_vertex[1:4]
        truth_decay_t = tau_info.decay_vertex[0]
        truth_disp = truth_decay_xyz - pv_xyz
        truth_L = np.linalg.norm(truth_disp)

        result[label]['truth_p_tau'] = tau_info.tau_p4
        result[label]['truth_decay_vertex_xyz'] = truth_decay_xyz
        result[label]['truth_decay_vertex_t'] = truth_decay_t
        result[label]['truth_decay_length_m'] = truth_L

    return result

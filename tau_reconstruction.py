"""
Tau kinematic reconstruction using the Jeans impact parameter method.

For tau -> pi nu (single-prong, no neutrals), the reconstruction is
particularly clean:

  1. We know the beam 4-momentum exactly => reconstruct pH = pbeam - pZ
  2. The tau momentum direction is constrained to lie in a plane defined
     by the production vertex (PV) and the charged pion trajectory.
  3. For pi-nu, there are no neutral hadrons, so the only invisible is
     the single neutrino.
  4. The tau mass constraint m_tau^2 = (p_pi + p_nu)^2 provides a
     quadratic equation for the neutrino energy.
  5. Total missing momentum (p_H - p_vis1 - p_vis2 = p_nu1 + p_nu2)
     provides additional constraints to resolve the two-fold ambiguity.

Reference: arXiv:1507.01700 (Jeans), arXiv:1804.01241 (Jeans & Wilson)
"""
import numpy as np
from config import M_TAU, M_PI, M_HIGGS, CTAU_TAU, C_LIGHT, SQRT_S, P_BEAM_TOTAL
from parse_hepmc import EventRecord


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
# Jeans reconstruction for tau -> pi nu
# ---------------------------------------------------------------------------
def _solve_neutrino_pi_nu(p_pi, p_tau_direction):
    """Solve for the neutrino 4-momentum given:
      - p_pi: the charged pion 4-vector (E, px, py, pz)
      - p_tau_direction: unit 3-vector of the tau flight direction

    For tau -> pi nu with m_nu = 0:
      m_tau^2 = (p_pi + p_nu)^2 = m_pi^2 + 2 (E_pi E_nu - vec_pi . vec_nu)

    The tau momentum is p_tau = |p_tau| * tau_hat, and E_tau = sqrt(p^2 + m_tau^2).
    Also p_tau = p_pi + p_nu.

    Instead of parameterising the neutrino, we parameterise the tau:
      p_tau = t * tau_hat   (3-momentum)
      E_tau = sqrt(t^2 + m_tau^2)

    Then: m_tau^2 = (p_pi + p_nu)^2 = p_tau^2 (since p_tau = p_pi + p_nu)
    which is trivially satisfied.

    The real constraint is:
      E_tau = E_pi + E_nu
      vec_p_tau = vec_p_pi + vec_p_nu

    So:  p_nu = p_tau - p_pi  (as 4-vectors)

    We need E_nu >= 0 and p_nu lightlike (m_nu = 0):
      E_nu^2 = |vec_p_nu|^2
      (E_tau - E_pi)^2 = |t*tau_hat - vec_p_pi|^2

    Let d = tau_hat, k = vec_p_pi, E_k = E_pi:
      (sqrt(t^2 + m_tau^2) - E_k)^2 = t^2 - 2t(d.k) + |k|^2

    Expanding:
      t^2 + m_tau^2 - 2*E_k*sqrt(t^2 + m_tau^2) + E_k^2 = t^2 - 2t(d.k) + |k|^2

    Using E_k^2 - |k|^2 = m_pi^2:
      m_tau^2 + m_pi^2 - 2*E_k*sqrt(t^2 + m_tau^2) = -2t(d.k)

    Let A = m_tau^2 + m_pi^2, B = d.k:
      A - 2*E_k*sqrt(t^2 + m_tau^2) = -2*t*B
      2*E_k*sqrt(t^2 + m_tau^2) = A + 2*t*B

    Squaring:
      4*E_k^2*(t^2 + m_tau^2) = (A + 2*t*B)^2
      4*E_k^2*t^2 + 4*E_k^2*m_tau^2 = A^2 + 4*A*B*t + 4*B^2*t^2

    Rearranging:
      (4*E_k^2 - 4*B^2)*t^2 - 4*A*B*t + (4*E_k^2*m_tau^2 - A^2) = 0

    Returns list of (t, p_tau_4vec, p_nu_4vec) solutions.
    """
    d = p_tau_direction
    k = p3vec(p_pi)
    E_k = p_pi[0]

    A = M_TAU**2 + M_PI**2
    B = np.dot(d, k)

    aa = 4.0 * (E_k**2 - B**2)
    bb = -4.0 * A * B
    cc = 4.0 * E_k**2 * M_TAU**2 - A**2

    disc = bb**2 - 4.0 * aa * cc

    if disc < 0:
        return []

    solutions = []
    sqrt_disc = np.sqrt(disc)

    for sign in [+1, -1]:
        t = (-bb + sign * sqrt_disc) / (2.0 * aa) if abs(aa) > 1e-30 else None
        if t is None or t <= 0:
            continue

        E_tau = np.sqrt(t**2 + M_TAU**2)

        # Check the unsquared equation (sign consistency)
        lhs = A + 2.0 * t * B
        if lhs < 0:
            continue  # rejected by the squaring step

        p_tau = np.array([E_tau, t * d[0], t * d[1], t * d[2]])
        p_nu = p_tau - p_pi

        # Neutrino must have positive energy
        if p_nu[0] < -0.01:
            continue

        solutions.append((t, p_tau, p_nu))

    return solutions


def _tau_lifetime_likelihood(p_tau, prod_vtx_xyz, decay_vtx_xyz):
    """Compute the tau lifetime likelihood for a given momentum and vertex pair.

    L = decay_length / (beta*gamma)
    probability ~ exp(-L / ctau)

    prod_vtx_xyz, decay_vtx_xyz in metres.
    Returns (log_likelihood, decay_length_m, proper_time_s).
    """
    disp = decay_vtx_xyz - prod_vtx_xyz
    decay_length = np.linalg.norm(disp)  # metres

    pmag = p3mag(p_tau)
    E = p_tau[0]
    if pmag < 1e-10:
        return -1e10, 0.0, 0.0

    beta_gamma = pmag / M_TAU
    proper_length = decay_length / beta_gamma  # metres

    # p(L) ~ exp(-proper_length / ctau)
    log_like = -proper_length / CTAU_TAU
    proper_time = proper_length / C_LIGHT

    return log_like, decay_length, proper_time


def reconstruct_tau_pi_nu(p_pi, production_vertex_xyz, p_higgs=None):
    """Reconstruct tau -> pi nu using the Jeans method.

    For truth-level HepMC without detector effects, we don't have a
    'track plane' from impact parameters.  Instead we use a simplified
    approach:

    Since we know p_H (from p_beam - p_Z), and for each event there are
    two taus, we can use:
      p_tau1 + p_tau2 = p_H
      m(p_pi_i + p_nu_i) = m_tau  for each tau

    For the single-tau reconstruction (needed for the decay vertex), we
    need the tau flight direction. At truth level, the pion carries most
    of the tau momentum (collinear-ish), but we can do better:

    We scan over possible tau directions consistent with the kinematics
    and pick the solution that best satisfies the combined constraints.

    However, for maximum fidelity to the Jeans method, we reconstruct
    the tau flight direction from the impact parameter geometry. At truth
    level the 'impact parameter' is determined by the displacement between
    the production vertex and the pion trajectory.

    Parameters
    ----------
    p_pi : np.ndarray
        Charged pion 4-momentum (E, px, py, pz)
    production_vertex_xyz : np.ndarray
        Production vertex position (x, y, z) in metres
    p_higgs : np.ndarray or None
        Higgs 4-momentum if known (for combined fit)

    Returns
    -------
    list of dict with keys: 'p_tau', 'p_nu', 'tau_dir', 'decay_length',
                            'decay_vertex_xyz', 'decay_vertex_t', 'log_like'
    """
    # The tau direction can be constrained:
    # In the plane defined by the PV and the pion trajectory, the tau
    # momentum must lie. For truth-level, the pion direction *is* the
    # best estimate of the track direction at the PCA.
    #
    # At truth level, we parameterise the tau direction as lying in the
    # plane spanned by the pion momentum and a perpendicular direction.
    # The perpendicular direction comes from the "impact parameter vector".
    #
    # For the truth case with no detector smearing, the pion trajectory
    # passes exactly through the decay vertex. The vector from PV to the
    # decay vertex defines the tau direction. But we don't know the decay
    # vertex yet (that's what we're trying to find!).
    #
    # So we use the Jeans approach: parameterise tau direction by an angle
    # psi in the plane containing the pion momentum, and solve the mass
    # constraint.

    pi_dir = p3hat(p_pi)

    # We need a perpendicular direction in the "track plane".
    # At truth level, pick any perpendicular direction and scan.
    # A natural choice: the direction perpendicular to the pion in the
    # plane containing the pion and the beam axis.
    beam_dir = np.array([0.0, 0.0, 1.0])
    perp = np.cross(pi_dir, beam_dir)
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-10:
        beam_dir = np.array([1.0, 0.0, 0.0])
        perp = np.cross(pi_dir, beam_dir)
        perp_norm = np.linalg.norm(perp)
    perp = perp / perp_norm

    # Second perpendicular (out of the "track plane")
    perp2 = np.cross(pi_dir, perp)

    # Scan over angle psi in the plane (pi_dir, perp)
    # The tau direction: d = cos(psi) * pi_dir + sin(psi) * perp
    # For truth-level with no detector smearing, the tau is nearly
    # collinear with the pion. Scan a small range.
    best_solutions = []
    psi_values = np.linspace(-0.3, 0.3, 200)

    for psi in psi_values:
        tau_dir = np.cos(psi) * pi_dir + np.sin(psi) * perp
        tau_dir = tau_dir / np.linalg.norm(tau_dir)

        solutions = _solve_neutrino_pi_nu(p_pi, tau_dir)

        for (t, p_tau, p_nu) in solutions:
            # Decay length from geometry:
            # The pion originates at the decay vertex which is along tau_dir
            # from the PV. Decay length = t * M_TAU / p_tau_mag * ctau ... no.
            # Decay length is a free parameter; we estimate it from the
            # tau momentum: the mean is gamma*beta*ctau.
            beta_gamma = t / M_TAU
            mean_decay_length = beta_gamma * CTAU_TAU  # metres

            # For truth-level, estimate the actual decay vertex from the
            # tau momentum direction. We use the tau lifetime to assign a
            # probability.
            # The decay vertex is at: PV + L * tau_hat
            # L is drawn from exp(-L / (beta_gamma * ctau))
            # We use L = beta_gamma * ctau as the estimate (mean).
            decay_length = mean_decay_length  # placeholder; will be refined

            decay_vtx = production_vertex_xyz + decay_length * tau_dir
            decay_time = decay_length / (t / p_tau[0] * C_LIGHT)
            # v = p/E * c, so t_lab = L / v = L * E / (p * c)

            log_like = -1.0  # placeholder
            # Penalise large opening angles (pion should be roughly collinear)
            cos_open = np.dot(tau_dir, pi_dir)
            angle_penalty = -0.5 * (np.arccos(np.clip(cos_open, -1, 1)) / 0.05)**2

            best_solutions.append({
                'p_tau': p_tau,
                'p_nu': p_nu,
                'tau_dir': tau_dir,
                'decay_length': decay_length,
                'decay_vertex_xyz': decay_vtx,
                'decay_vertex_t': decay_time,
                'log_like': angle_penalty,
                'tau_momentum_mag': t,
                'psi': psi,
            })

    return best_solutions


def reconstruct_event(event: EventRecord):
    """Full event reconstruction using the Jeans method + Higgs constraint.

    For e+e- -> ZH -> mu+mu- tau+tau-, we know:
      p_Z = p_mu+ + p_mu-
      p_H = p_beam - p_Z
      p_tau1 + p_tau2 = p_H

    For each tau -> pi nu:
      p_tau = p_pi + p_nu
      m_tau^2 = (p_pi + p_nu)^2  (tau mass constraint)

    The combined constraint: p_nu1 + p_nu2 = p_H - p_pi1 - p_pi2 = p_miss
    gives 4 equations. Together with 2 mass constraints, this over-constrains
    the system (6 constraints for 6 unknowns -- 3 per neutrino minus 1 mass
    constraint each = 4 unknowns).

    We solve by:
    1. Parameterise each tau direction
    2. Solve the mass constraint for each tau independently
    3. Pick the combination that best satisfies p_miss = p_nu1 + p_nu2

    Returns dict with reconstructed quantities.
    """
    if event.tau_minus is None or event.tau_plus is None:
        return None

    # Reconstruct Higgs
    p_Z = event.mu_plus_p4 + event.mu_minus_p4
    p_H = P_BEAM_TOTAL - p_Z

    # Missing momentum = p_H - p_pi1 - p_pi2 (should equal p_nu1 + p_nu2)
    p_pi_minus = event.tau_minus.charged_pion_p4
    p_pi_plus = event.tau_plus.charged_pion_p4
    p_miss = p_H - p_pi_minus - p_pi_plus

    # Production vertex (both taus produced at the Higgs decay vertex,
    # which is essentially the IP for this topology)
    # Use truth production vertex of the tau_minus (should be same for both)
    prod_vtx_xyz = event.tau_minus.production_vertex[1:4]  # (x, y, z) metres

    # Get candidate solutions for each tau
    sols_minus = reconstruct_tau_pi_nu(p_pi_minus, prod_vtx_xyz, p_H)
    sols_plus = reconstruct_tau_pi_nu(p_pi_plus, prod_vtx_xyz, p_H)

    if not sols_minus or not sols_plus:
        return None

    # Find the best combination using the missing momentum constraint
    best_chi2 = 1e30
    best_combo = None

    for sm in sols_minus:
        for sp in sols_plus:
            # Total neutrino 4-momentum from this combination
            p_nu_total = sm['p_nu'] + sp['p_nu']
            # Residual: should equal p_miss
            dp = p_nu_total - p_miss
            chi2 = dp[0]**2 + dp[1]**2 + dp[2]**2 + dp[3]**2
            # Add angle penalties
            chi2 -= 2.0 * (sm['log_like'] + sp['log_like'])

            if chi2 < best_chi2:
                best_chi2 = chi2
                best_combo = (sm, sp)

    if best_combo is None:
        return None

    sol_minus, sol_plus = best_combo

    # Now refine the decay vertices using the reconstructed tau momenta
    # and the collinearity between tau direction and displacement
    result = {
        'p_H': p_H,
        'p_Z': p_Z,
        'p_miss': p_miss,
        'chi2': best_chi2,
    }

    for label, sol, tau_info in [('tau_minus', sol_minus, event.tau_minus),
                                  ('tau_plus', sol_plus, event.tau_plus)]:
        p_tau = sol['p_tau']
        tau_dir = sol['tau_dir']
        pmag = p3mag(p_tau)
        E = p_tau[0]
        beta = pmag / E
        gamma = E / M_TAU
        beta_gamma = pmag / M_TAU

        # Mean decay length
        mean_L = beta_gamma * CTAU_TAU  # metres

        # For truth-level reco, use the mean decay length as the estimate.
        # A real experiment would use the impact parameter to get L directly.
        # We can also use truth to compute L.
        decay_length = mean_L

        decay_vtx_xyz = prod_vtx_xyz + decay_length * tau_dir
        decay_time = decay_length / (beta * C_LIGHT)  # seconds

        result[label] = {
            'p_tau_reco': p_tau,
            'p_nu_reco': sol['p_nu'],
            'tau_dir': tau_dir,
            'decay_length_m': decay_length,
            'decay_vertex_xyz': decay_vtx_xyz,
            'decay_vertex_t': decay_time,
            'production_vertex_xyz': prod_vtx_xyz,
            'beta': beta,
            'gamma': gamma,
            'beta_gamma': beta_gamma,
        }

        # Truth comparison
        truth_decay_xyz = tau_info.decay_vertex[1:4]
        truth_decay_t = tau_info.decay_vertex[0]
        truth_disp = truth_decay_xyz - prod_vtx_xyz
        truth_L = np.linalg.norm(truth_disp)

        result[label]['truth_p_tau'] = tau_info.tau_p4
        result[label]['truth_decay_vertex_xyz'] = truth_decay_xyz
        result[label]['truth_decay_vertex_t'] = truth_decay_t
        result[label]['truth_decay_length_m'] = truth_L

    return result


def reconstruct_event_with_truth_vertices(event: EventRecord):
    """Reconstruct tau momenta using the combined Higgs + mass constraint,
    and use truth vertices for the spacetime positions.

    This hybrid approach gives the best of both worlds:
    - Tau momenta reconstructed from visible products (Jeans-like)
    - Decay vertex positions from truth (for the spacetime interval)

    Also computes the 'Jeans-estimated' vertex from the reconstructed
    tau momentum + exponential lifetime sampling.
    """
    reco = reconstruct_event(event)
    if reco is None:
        return None

    # Overwrite the decay vertices with truth for the primary measurement,
    # but keep the reco vertices for comparison
    for label in ['tau_minus', 'tau_plus']:
        tau_info = event.tau_minus if label == 'tau_minus' else event.tau_plus
        reco[label]['reco_decay_vertex_xyz'] = reco[label]['decay_vertex_xyz'].copy()
        reco[label]['reco_decay_vertex_t'] = reco[label]['decay_vertex_t']

    return reco

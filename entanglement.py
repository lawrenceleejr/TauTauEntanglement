"""
Quantum entanglement observables for tau-tau spin correlations.

==========================================================================
FROM COLLIDER ANGLES TO ENTANGLEMENT — A GUIDE FOR EXPERIMENTALISTS
==========================================================================

WHAT WE MEASURE (kinematic space)
---------------------------------
For each event with tau- -> pi- nu and tau+ -> pi+ nu:

  1. Boost pions to their parent tau rest frames.
  2. Project each pion direction onto the {n, r, k} basis defined in the
     Higgs rest frame (k = tau- flight dir, r = beam component perp to k,
     n = k x r).
  3. This gives 6 numbers per event:
       cos_theta_minus = (n-, r-, k-)   projections of pi- in tau- RF
       cos_theta_plus  = (n+, r+, k+)   projections of pi+ in tau+ RF

     These are just direction cosines of the pion momentum in the tau
     rest frame, projected onto a fixed coordinate system.

HOW WE BUILD THE SPIN CORRELATION MATRIX C_ij
----------------------------------------------
C_ij is a 3x3 matrix extracted from the angular distributions:

  C_ij = 9 * <cos_theta_i^+  *  cos_theta_j^->      (average over events)

  B_i^+ = +3 * <cos_theta_i^+>      (tau+ polarisation)
  B_i^- = -3 * <cos_theta_i^->      (tau- polarisation)

The factor 9 arises because for tau -> pi nu (unit analysing power),
the angular distribution is:
  (1/sigma) d^2sigma / d(cos_i+) d(cos_j-)  =  (1/4)(1 + C_ij cos_i+ cos_j-)
and <cos^2> = 1/3 for a uniform distribution, so C_ij = 9 * <cos_i+ cos_j->.

For H -> tau+tau- (SM, CP-even), the prediction in the helicity basis is:
  C = diag(2*beta^2 - 1,  1,  1 - 2*beta^2)
    ~ diag(+1, +1, -1)     for beta ~ 1

where beta = |p_tau|/E_tau.  The diagonal structure means: the tau spins
are anti-correlated along k (helicity axis), and correlated transversely.

WHAT THE ENTANGLEMENT QUANTITIES MEAN
--------------------------------------
All entanglement quantities are derived from C_ij (and B_i):

  * DENSITY MATRIX rho (4x4):
      Built from C_ij and B_i using Pauli matrices.
      rho = (1/4)[I*I + B_i^+(sigma_i*I) + B_j^-(I*sigma_j) + C_ij(sigma_i*sigma_j)]
      This is the quantum state of the two-tau spin system.

  * m12 (Horodecki parameter):
      - Form the matrix M = C^T C  (3x3, positive semi-definite)
      - m1, m2 = two largest eigenvalues of M
      - m12 = m1 + m2
      - m12 > 1  =>  Bell nonlocality (CHSH inequality violation)
      - For SM H->tautau: m12 ~ 2 (maximal)
      - For uncorrelated taus (C=0): m12 = 0
      - For classical correlations: m12 <= 1

  * CHSH Bell score = 2*sqrt(m12):
      - Classical limit: score <= 2
      - Tsirelson (QM) bound: score <= 2*sqrt(2) ~ 2.83
      - For SM H->tautau: score ~ 2.83

  * CONCURRENCE:
      - 0 = no entanglement (separable state)
      - 1 = maximal entanglement
      - Computed from the eigenvalues of an auxiliary matrix built from rho
      - For SM H->tautau: concurrence = 1 (exact, for any beta)

PHYSICAL PICTURE
----------------
The Higgs is spin-0, so the tau+tau- pair is produced in a definite
(pure) quantum state. The two tau spins are maximally entangled —
measuring one instantly determines the other, regardless of how far
apart the two taus have traveled before decaying. This is the EPR
paradox realized with tau leptons.

The pion direction in the tau rest frame is a perfect spin analyser
(analysing power = 1) because tau -> pi nu is a two-body decay of a
spin-1/2 particle, and angular momentum conservation forces the pion
to carry full spin information.

References:
  - arXiv:2602.03960 (tau pair entanglement at FCC-ee)
  - Horodecki, Horodecki, Horodecki, Phys. Lett. A 200 (1995) 340
  - Fabbrichesi et al., arXiv:2208.11723 (entanglement in H->tautau)
"""
import numpy as np
from config import N_BOOTSTRAP


# ---------------------------------------------------------------------------
# Core entanglement measures
# ---------------------------------------------------------------------------

def compute_m12(C):
    """Compute the Horodecki value m12 = m1 + m2 from the correlation matrix.

    m1, m2 are the two largest eigenvalues of M = C^T C.
    m12 > 1 implies Bell nonlocality (CHSH violation).

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Spin correlation matrix

    Returns
    -------
    m12 : float
    eigenvalues : np.ndarray, shape (3,)
        Sorted eigenvalues of M (descending)
    """
    M = C.T @ C
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(eigvals)[::-1]  # descending
    m12 = eigvals[0] + eigvals[1]
    return m12, eigvals


def compute_bell_score(m12):
    """CHSH Bell score = 2 * sqrt(m12).

    Violation of CHSH inequality requires score > 2.
    Tsirelson bound: score <= 2*sqrt(2) ~ 2.83.
    """
    if m12 <= 0:
        return 0.0
    return 2.0 * np.sqrt(m12)


def compute_density_matrix(C, B_plus, B_minus):
    """Construct the 4x4 density matrix from C_ij and polarisation vectors.

    rho = (1/4) [I x I + sum_i B_i^+(sigma_i x I)
                       + sum_j B_j^-(I x sigma_j)
                       + sum_ij C_ij (sigma_i x sigma_j)]

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
    B_plus : np.ndarray, shape (3,)
    B_minus : np.ndarray, shape (3,)

    Returns
    -------
    rho : np.ndarray, shape (4, 4), complex
    """
    # Pauli matrices
    sigma = np.array([
        [[0, 1], [1, 0]],       # sigma_1 (x)
        [[0, -1j], [1j, 0]],    # sigma_2 (y)
        [[1, 0], [0, -1]],      # sigma_3 (z)
    ], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    rho = np.kron(I2, I2).astype(complex)

    for i in range(3):
        rho += B_plus[i] * np.kron(sigma[i], I2)
        rho += B_minus[i] * np.kron(I2, sigma[i])
        for j in range(3):
            rho += C[i, j] * np.kron(sigma[i], sigma[j])

    rho *= 0.25
    return rho


def _physicalize_density_matrix(rho):
    """Project a density matrix onto the nearest physical (PSD, trace-1) state.

    With finite statistics, the C_ij extraction can yield a density matrix
    with small negative eigenvalues.  We clip these to zero and renormalize
    so that the concurrence calculation remains meaningful.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.real(eigvals)
    eigvals = np.maximum(eigvals, 0.0)
    s = eigvals.sum()
    if s > 0:
        eigvals /= s
    return (eigvecs * eigvals) @ eigvecs.conj().T


def compute_concurrence(rho):
    """Compute the concurrence of a 4x4 density matrix.

    C = max(0, r1 - r2 - r3 - r4)
    where r_i are the square roots of the eigenvalues (in decreasing order)
    of the matrix R = rho * (sigma_y x sigma_y) * rho* * (sigma_y x sigma_y)

    The density matrix is first projected onto the nearest physical state
    (positive semi-definite, trace 1) to handle statistical noise.

    Parameters
    ----------
    rho : np.ndarray, shape (4, 4), complex

    Returns
    -------
    concurrence : float
    """
    # Ensure physical density matrix
    rho = _physicalize_density_matrix(rho)

    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Y = np.kron(sigma_y, sigma_y)

    R = rho @ Y @ rho.conj() @ Y

    eigvals = np.linalg.eigvals(R)
    # Take real parts (should be real up to numerical noise)
    eigvals = np.real(eigvals)
    eigvals = np.maximum(eigvals, 0.0)
    r = np.sqrt(eigvals)
    r = np.sort(r)[::-1]  # descending

    concurrence = max(0.0, r[0] - r[1] - r[2] - r[3])
    return concurrence


def compute_concurrence_from_C(C, B_plus, B_minus):
    """Convenience: compute concurrence from C_ij and polarisations."""
    rho = compute_density_matrix(C, B_plus, B_minus)
    return compute_concurrence(rho)


# ---------------------------------------------------------------------------
# Bootstrap uncertainty estimation
# ---------------------------------------------------------------------------

def bootstrap_entanglement(cos_theta_plus, cos_theta_minus,
                            n_bootstrap=N_BOOTSTRAP, rng=None):
    """Bootstrap the C_ij extraction and entanglement observables.

    Parameters
    ----------
    cos_theta_plus : np.ndarray, shape (N, 3)
    cos_theta_minus : np.ndarray, shape (N, 3)
    n_bootstrap : int
    rng : np.random.Generator or None

    Returns
    -------
    dict with central values and uncertainties for m12, concurrence, bell_score
    """
    from spin_analysis import extract_correlation_matrix

    if rng is None:
        rng = np.random.default_rng(42)

    N = len(cos_theta_plus)
    if N == 0:
        return {
            'm12': 0.0, 'm12_err': 0.0,
            'concurrence': 0.0, 'concurrence_err': 0.0,
            'bell_score': 0.0, 'bell_score_err': 0.0,
            'C': np.zeros((3, 3)), 'C_err': np.zeros((3, 3)),
            'B_plus': np.zeros(3), 'B_minus': np.zeros(3),
        }

    # Central values
    C, B_plus, B_minus = extract_correlation_matrix(cos_theta_plus, cos_theta_minus)
    m12, _ = compute_m12(C)
    bell = compute_bell_score(m12)
    conc = compute_concurrence_from_C(C, B_plus, B_minus)

    # Bootstrap
    m12_samples = np.empty(n_bootstrap)
    conc_samples = np.empty(n_bootstrap)
    bell_samples = np.empty(n_bootstrap)
    C_samples = np.empty((n_bootstrap, 3, 3))

    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        C_b, Bp_b, Bm_b = extract_correlation_matrix(
            cos_theta_plus[idx], cos_theta_minus[idx])
        m12_b, _ = compute_m12(C_b)
        m12_samples[b] = m12_b
        bell_samples[b] = compute_bell_score(m12_b)
        conc_samples[b] = compute_concurrence_from_C(C_b, Bp_b, Bm_b)
        C_samples[b] = C_b

    return {
        'm12': m12,
        'm12_err': np.std(m12_samples),
        'm12_samples': m12_samples,
        'concurrence': conc,
        'concurrence_err': np.std(conc_samples),
        'concurrence_samples': conc_samples,
        'bell_score': bell,
        'bell_score_err': np.std(bell_samples),
        'C': C,
        'C_err': np.std(C_samples, axis=0),
        'B_plus': B_plus,
        'B_minus': B_minus,
    }


# ---------------------------------------------------------------------------
# Locality hypothesis test
# ---------------------------------------------------------------------------

def locality_rejection_sigma(m12, m12_err):
    """Number of sigma at which we reject m12 <= 1 (locality hypothesis).

    Under the null hypothesis H0: m12 <= 1, the significance is:
      n_sigma = (m12 - 1) / sigma(m12)
    """
    if m12_err <= 0:
        return np.inf if m12 > 1 else 0.0
    return (m12 - 1.0) / m12_err


def entanglement_rejection_sigma(concurrence, concurrence_err):
    """Number of sigma at which we reject C <= 0 (separability)."""
    if concurrence_err <= 0:
        return np.inf if concurrence > 0 else 0.0
    return concurrence / concurrence_err


# ---------------------------------------------------------------------------
# v_psi hypothesis test
# ---------------------------------------------------------------------------

def vpsi_rejection_significance(cos_theta_plus, cos_theta_minus,
                                 v_signal_arr, v_psi,
                                 n_bootstrap=N_BOOTSTRAP, rng=None):
    """Test the hypothesis that correlations are mediated by a signal at speed v_psi.

    Under this hypothesis, events with v_signal > v_psi should show NO
    correlations (C_ij = 0, m12 = 0) because the signal cannot connect
    the two tau decays.

    We select events with v_signal > v_psi, take the statistical uncertainty
    from bootstrap, and assume the central value agrees with the SM prediction
    (m12 = 2.0).  This gives a projected sensitivity: how significantly can
    we reject the v_psi hypothesis if the SM is correct?

    Parameters
    ----------
    cos_theta_plus, cos_theta_minus : np.ndarray, shape (N, 3)
    v_signal_arr : np.ndarray, shape (N,)
        Signal speed for each event (in units of c)
    v_psi : float
        Hypothesised signal speed (in units of c)
    n_bootstrap : int
    rng : np.random.Generator or None

    Returns
    -------
    dict with:
        'n_events'      : number of events with v_signal > v_psi
        'm12'           : measured m12 for those events
        'm12_err'       : bootstrap uncertainty
        'sigma_vs_0'    : significance of rejecting m12 = 0, assuming SM central value (m12=2)
        'sigma_vs_1'    : significance of rejecting m12 <= 1, assuming SM central value (m12=2)
    """
    mask = v_signal_arr > v_psi
    n_sel = np.sum(mask)

    if n_sel < 10:
        return {
            'n_events': n_sel,
            'm12': np.nan,
            'm12_err': np.nan,
            'sigma_vs_0': 0.0,
            'sigma_vs_1': 0.0,
            'concurrence': np.nan,
            'concurrence_err': np.nan,
        }

    result = bootstrap_entanglement(
        cos_theta_plus[mask], cos_theta_minus[mask],
        n_bootstrap=n_bootstrap, rng=rng,
    )

    m12 = result['m12']
    m12_err = result['m12_err']

    # Use SM central value (m12 = 2.0) with the measured uncertainty to give
    # projected sensitivity: how significantly can we reject each hypothesis
    # if the SM is correct?
    M12_SM = 2.0
    sigma_vs_0 = M12_SM / m12_err if m12_err > 0 else np.inf
    sigma_vs_1 = (M12_SM - 1.0) / m12_err if m12_err > 0 else np.inf

    return {
        'n_events': n_sel,
        'm12': m12,
        'm12_err': m12_err,
        'sigma_vs_0': sigma_vs_0,
        'sigma_vs_1': sigma_vs_1,
        'concurrence': result['concurrence'],
        'concurrence_err': result['concurrence_err'],
    }


def scan_vpsi(cos_theta_plus, cos_theta_minus, v_signal_arr,
              v_psi_values, n_bootstrap=N_BOOTSTRAP):
    """Scan over v_psi hypotheses and compute rejection significance.

    Returns a list of dicts (one per v_psi value).
    """
    rng = np.random.default_rng(42)
    results = []
    for v_psi in v_psi_values:
        res = vpsi_rejection_significance(
            cos_theta_plus, cos_theta_minus, v_signal_arr,
            v_psi, n_bootstrap=n_bootstrap, rng=rng,
        )
        res['v_psi'] = v_psi
        results.append(res)
    return results

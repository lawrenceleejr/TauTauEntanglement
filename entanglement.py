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

  C_ij = -9 * <cos_theta_i^+  *  cos_theta_j^->     (average over events;
         the minus sign is the analysing-power product alpha+ alpha- = -1)

  B_i^+ = -3 * <cos_theta_i^+>      (tau+ polarisation, alpha_+ = -1)
  B_i^- = +3 * <cos_theta_i^->      (tau- polarisation, alpha_- = +1)

The factor 9 arises because for tau -> pi nu (unit analysing power),
the angular distribution is:
  (1/sigma) d^2sigma / d(cos_i+) d(cos_j-)  =  (1/4)(1 + C_ij cos_i+ cos_j-)
and <cos^2> = 1/3 for a uniform distribution, so C_ij = -9 * <cos_i+ cos_j->
(with the alpha+ alpha- = -1 sign and density 1 - C_ij cos_i+ cos_j-).

For H -> tau+tau- (SM, CP-even), the prediction in the helicity basis is
EXACT and velocity-independent (3P0 partial wave; see below):
  C = diag(+1, +1, -1)     for any beta

The diagonal structure means: the tau spins are anti-correlated along k
(helicity axis), and correlated transversely.

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
# Standard Model expectations (exact)
# ---------------------------------------------------------------------------
#
# A CP-even scalar decaying to two spin-1/2 fermions can only do so in the
# 3P0 partial wave (J = 0 and parity (+) force L = S = 1).  Along the decay
# axis the orbital wavefunction Y_1^{m_L}(theta=0) selects m_L = 0, so the
# spin state is exactly the triplet
#
#     |psi+> = (|up,down> + |down,up>) / sqrt(2)        (quantised along k)
#
# for ANY tau velocity beta.  Hence, exactly:
#
#     C_SM = diag(+1, +1, -1)   in the {n, r, k} basis,
#     m12 = 2,  CHSH S = 2 sqrt(2)  (Tsirelson bound),  concurrence = 1.
#
# For a CP-mixed coupling  ~ tau-bar (cos D + i g5 sin D) tau, the transverse
# block rotates by 2D:  C_nn = C_rr = cos 2D, C_nr = -C_rn = sin 2D,
# C_kk = -1; m12 = 2 for all D.  The pi-pi acoplanarity follows
#     dGamma/dphi  ~  1 - (pi^2/16) cos(phi - 2D),
# i.e. the SM modulation coefficient is B = -pi^2/16 ~= -0.6169 (NOT -0.5):
# the transverse correlation C_nn = C_rr = 1 enters multiplied by
# <sin theta>^2 = (pi/4)^2 from the polar-angle average on each side.

SM_CORRELATION_MATRIX = np.diag([1.0, 1.0, -1.0])
M12_SM = 2.0
CHSH_SM = 2.0 * np.sqrt(2.0)
CONCURRENCE_SM = 1.0
ACOPLANARITY_B_SM = -np.pi**2 / 16.0

# Per-event log-likelihood-ratio moments for the joint pi+/pi- angular
# density  p1 = (1 - h+ . C_SM . h-) / (4pi)^2  vs  p0 = 1/(4pi)^2.
# With t = ln(1 - h+ . C_SM . h-) one finds, EXACTLY (see paper):
#   E_SM[t]   = ln 2 - 1/2,   Var_SM[t]   = 1/4
#   E_null[t] = ln 2 - 1,     Var_null[t] = 1
LLR_MEAN_SM = np.log(2.0) - 0.5
LLR_VAR_SM = 0.25
LLR_MEAN_NULL = np.log(2.0) - 1.0
LLR_VAR_NULL = 1.0

# Small-sample noise bias of the moment estimator of m12 under the
# no-correlation null:  Chat ~ (3/sqrt(N)) G with G_ij iid N(0,1), so
# m12_hat = (9/N) (lam1 + lam2) of the Wishart matrix G^T G, giving
#   E[m12_hat | C=0] = 9 E[lam1+lam2] / N = M12_NULL_BIAS_CONST / N.
# E[lam1+lam2] = 8.724 for Wishart_3(I, 3) (verified in tests/).
M12_NULL_BIAS_CONST = 78.5


def sm_correlation_matrix(delta_cp=0.0):
    """Exact spin correlation matrix for H -> tau tau with CP-mixing angle.

    delta_cp = 0 is the SM (CP-even); delta_cp = pi/2 is pure CP-odd.
    The {n, r} block is a rotation by 2*delta_cp; C_kk = -1 always.
    """
    c, s = np.cos(2.0 * delta_cp), np.sin(2.0 * delta_cp)
    return np.array([[c,  s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, -1.0]])


def m12_null_expectation(n_events):
    """Expected value of the m12 moment estimator for UNCORRELATED taus.

    This is pure noise bias: with finite statistics the eigenvalue sum of
    Chat^T Chat is positive even when the true C vanishes.
    """
    if n_events <= 0:
        return np.inf
    return M12_NULL_BIAS_CONST / n_events


# ---------------------------------------------------------------------------
# Unbiased linear observables: CHSH (fixed axes) and concurrence witness
# ---------------------------------------------------------------------------
#
# The Horodecki m12 involves eigenvalues of Chat^T Chat and is therefore a
# nonlinear, noise-biased statistic.  For hypothesis tests we instead use
# observables LINEAR in C (hence unbiased, with exact per-event variances):
#
#  * CHSH with axes fixed a priori from the SM prediction:
#      a = n, a' = r, b = (n+r)/sqrt2, b' = (n-r)/sqrt2
#      <S> = sqrt(2) (C_nn + C_rr);  SM: 2 sqrt(2).  Local realism: S <= 2.
#
#  * Fully-entangled-fraction witness: the overlap with the Bell state
#      F = <psi+|rho|psi+> = (1 + C_nn + C_rr - C_kk)/4
#    bounds the concurrence from below [Bennett et al., PRA 54, 3824 (1996)]:
#      concurrence >= W = 2F - 1 = (C_nn + C_rr - C_kk - 1)/2.
#    SM: W = 1.  Any separable state has W <= 0.

def chsh_test(cos_theta_plus, cos_theta_minus):
    """Unbiased CHSH estimator with fixed (SM-optimal) measurement axes.

    Per-event statistic: s_e = -9 sqrt(2) (h+_n h-_n + h+_r h-_r),
    whose mean is sqrt(2) (C_nn + C_rr).  Returns S, its standard error,
    and the significance of rejecting the local-realist bound S <= 2.
    """
    s_e = -9.0 * np.sqrt(2.0) * (
        cos_theta_plus[:, 0] * cos_theta_minus[:, 0]
        + cos_theta_plus[:, 1] * cos_theta_minus[:, 1])
    n = len(s_e)
    if n < 2:
        return {'S': np.nan, 'S_err': np.nan, 'z_bell': 0.0, 'n_events': n}
    S = float(np.mean(s_e))
    S_err = float(np.std(s_e, ddof=1) / np.sqrt(n))
    z_bell = (S - 2.0) / S_err if S_err > 0 else 0.0
    return {'S': S, 'S_err': S_err, 'z_bell': z_bell, 'n_events': n}


def concurrence_witness(cos_theta_plus, cos_theta_minus):
    """Lower bound on the concurrence from the fully entangled fraction.

    Per-event statistic: w_e = -(9/2)(h+_n h-_n + h+_r h-_r - h+_k h-_k) - 1/2,
    whose mean is W = (C_nn + C_rr - C_kk - 1)/2 <= concurrence.
    W > 0 certifies entanglement; SM expectation W = 1.
    """
    w_e = -4.5 * (cos_theta_plus[:, 0] * cos_theta_minus[:, 0]
                  + cos_theta_plus[:, 1] * cos_theta_minus[:, 1]
                  - cos_theta_plus[:, 2] * cos_theta_minus[:, 2]) - 0.5
    n = len(w_e)
    if n < 2:
        return {'W': np.nan, 'W_err': np.nan, 'z_sep': 0.0, 'n_events': n}
    W = float(np.mean(w_e))
    W_err = float(np.std(w_e, ddof=1) / np.sqrt(n))
    z_sep = W / W_err if W_err > 0 else 0.0
    return {'W': W, 'W_err': W_err, 'z_sep': z_sep, 'n_events': n}


# ---------------------------------------------------------------------------
# Optimal likelihood-ratio test (Neyman-Pearson) with permutation calibration
# ---------------------------------------------------------------------------

def llr_per_event(cos_theta_plus, cos_theta_minus, C=None):
    """Per-event log-likelihood ratio ln[p_SM / p_uncorrelated].

    The joint decay density for unit analysing power is
      p(h+, h- | C) = (1 - h+^T C h-) / (4 pi)^2
    (the minus sign matches the alpha+ alpha- = -1 convention used in the
    C_ij extraction).  The LR against the uncorrelated null p0 = 1/(4pi)^2
    is the Neyman-Pearson optimal statistic for "SM correlations vs none".
    """
    if C is None:
        C = SM_CORRELATION_MATRIX
    arg = 1.0 - np.einsum('ni,ij,nj->n', cos_theta_plus, C, cos_theta_minus)
    return np.log(np.maximum(arg, 1e-12))


def permutation_test(cos_theta_plus, cos_theta_minus, statistic='llr',
                     n_perm=5000, rng=None, C=None):
    """Calibrate a correlation statistic against the no-correlation null.

    The null hypothesis (taus uncorrelated, e.g. because a finite-speed
    signal could not connect the two decays) is realised EXACTLY in data
    by permuting the tau+ <-> tau- event pairing: marginal single-tau
    distributions are preserved while all correlations are destroyed.
    This yields a distribution-free, finite-N-exact p-value, immune to
    the noise bias of eigenvalue-based statistics.

    Parameters
    ----------
    statistic : 'llr' (Neyman-Pearson optimal, default) or 'm12'
    n_perm : number of pairing permutations

    Returns
    -------
    dict with t_obs, p_value (one-sided), z_score (studentized against the
    permutation distribution), null_mean, null_std.
    """
    from spin_analysis import extract_correlation_matrix

    if rng is None:
        rng = np.random.default_rng(42)
    n = len(cos_theta_plus)

    def _stat(cp, cm):
        if statistic == 'llr':
            return float(np.sum(llr_per_event(cp, cm, C=C)))
        elif statistic == 'm12':
            C_hat, _, _ = extract_correlation_matrix(cp, cm)
            m12, _ = compute_m12(C_hat)
            return m12
        raise ValueError(f"unknown statistic '{statistic}'")

    t_obs = _stat(cos_theta_plus, cos_theta_minus)

    t_null = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(n)
        t_null[k] = _stat(cos_theta_plus[idx], cos_theta_minus)

    null_mean = float(np.mean(t_null))
    null_std = float(np.std(t_null, ddof=1))
    # One-sided p-value with the +1 convention (never exactly zero)
    p_value = (1.0 + np.sum(t_null >= t_obs)) / (n_perm + 1.0)
    z_score = (t_obs - null_mean) / null_std if null_std > 0 else 0.0

    return {'t_obs': t_obs, 'p_value': float(p_value), 'z_score': float(z_score),
            'null_mean': null_mean, 'null_std': null_std,
            'n_perm': n_perm, 'statistic': statistic}


def sample_sm_pairs(n, rng=None, delta_cp=0.0):
    """Sample n (h+, h-) pairs from the exact H->tautau angular density.

    Used for closure tests and parametric calibration of estimator bias.
    Sampling: h- uniform on the sphere; then h+ has density
    (1 - u)/2 in u = h+ . (C h-), i.e. u = 1 - 2 sqrt(1 - r), r ~ U(0,1).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    C = sm_correlation_matrix(delta_cp)

    def _unit(m):
        v = rng.normal(size=(m, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    h_minus = _unit(n)
    a = h_minus @ C.T  # axis with |a| = 1 (C is orthogonal up to sign)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    u = 1.0 - 2.0 * np.sqrt(1.0 - rng.random(n))
    # Orthonormal frame around each a
    tmp = _unit(n)
    e1 = np.cross(a, tmp)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(a, e1)
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.maximum(1.0 - u**2, 0.0))
    h_plus = (u[:, None] * a
              + s[:, None] * (np.cos(phi)[:, None] * e1
                              + np.sin(phi)[:, None] * e2))
    return h_plus, h_minus


def expected_significance_no_correlation(n_events):
    """Expected (median) Z for rejecting 'no correlation' with the LR test.

    Z = sqrt(N) (E_SM[t] - E_null[t]) / sqrt(Var_null[t]) = sqrt(N)/2,
    using the exact moments above.  Valid for unit analysing power and
    perfect reconstruction; detector effects dilute this.
    """
    return np.sqrt(max(n_events, 0)) / 2.0


def expected_vpsi_reach(n_total, z_threshold=1.96, beta=1.0):
    """Analytic v_psi exclusion reach (in units of c).

    For back-to-back taus with i.i.d. exponential decay lengths the
    causal-signal speed in the Higgs frame is v = beta (L1+L2)/|L1-L2|
    and (L1-L2)/(L1+L2) is exactly uniform on [-1,1], giving a Pareto
    tail  P(v_sig > v) = beta/v.  With N(v_psi) = N_total beta/v_psi
    events surviving the causal-disconnection cut and Z = sqrt(N)/2:
        v_psi^95 = N_total beta / (4 z^2).
    """
    return n_total * beta / (4.0 * z_threshold**2)


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
            'm12': 0.0, 'm12_err': 0.0, 'm12_bc': 0.0, 'm12_null_exp': np.inf,
            'concurrence': 0.0, 'concurrence_err': 0.0,
            'bell_score': 0.0, 'bell_score_err': 0.0,
            'chsh_S': np.nan, 'chsh_S_err': np.nan, 'chsh_z_bell': 0.0,
            'witness_W': np.nan, 'witness_W_err': np.nan, 'witness_z_sep': 0.0,
            'C': np.zeros((3, 3)), 'C_err': np.zeros((3, 3)),
            'B_plus': np.zeros(3), 'B_minus': np.zeros(3),
        }

    # Central values
    C, B_plus, B_minus = extract_correlation_matrix(cos_theta_plus, cos_theta_minus)
    m12, _ = compute_m12(C)
    bell = compute_bell_score(m12)
    conc = compute_concurrence_from_C(C, B_plus, B_minus)

    # Unbiased linear observables with exact per-event standard errors
    chsh = chsh_test(cos_theta_plus, cos_theta_minus)
    witness = concurrence_witness(cos_theta_plus, cos_theta_minus)

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

    # Bootstrap bias correction: bias ~= mean(boot) - m12_hat, so
    # m12_bc = 2 m12_hat - mean(boot).  Removes the leading O(1/N)
    # noise inflation of the eigenvalue-sum estimator.
    m12_bc = 2.0 * m12 - float(np.mean(m12_samples))

    return {
        'm12': m12,
        'm12_err': np.std(m12_samples),
        'm12_samples': m12_samples,
        'm12_bc': m12_bc,
        'm12_null_exp': m12_null_expectation(N),
        'concurrence': conc,
        'concurrence_err': np.std(conc_samples),
        'concurrence_samples': conc_samples,
        'bell_score': bell,
        'bell_score_err': np.std(bell_samples),
        'chsh_S': chsh['S'],
        'chsh_S_err': chsh['S_err'],
        'chsh_z_bell': chsh['z_bell'],
        'witness_W': witness['W'],
        'witness_W_err': witness['W_err'],
        'witness_z_sep': witness['z_sep'],
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
                                 n_bootstrap=N_BOOTSTRAP, n_perm=5000,
                                 rng=None):
    """Test the hypothesis that correlations are mediated by a signal at speed v_psi.

    Under this hypothesis, events with v_signal > v_psi are causally
    disconnected and must show NO spin correlations.  Two complementary
    tests are run on the surviving subsample:

    1. NO-CORRELATION test ('sigma_vs_0'): the Neyman-Pearson optimal
       per-event likelihood ratio  t = ln(1 - h+ . C_SM . h-), summed
       over events and calibrated with the exact permutation null
       (tau+/tau- pairing shuffled).  This replaces the naive m12/sigma
       counting, which is invalid: the moment estimator of m12 has a
       positive noise bias E[m12_hat | C=0] = 78.5/N.

    2. LOCAL-REALISM test ('sigma_vs_1'): the CHSH score with fixed
       a-priori axes, S = sqrt(2)(C_nn + C_rr), a linear (unbiased)
       statistic tested against the local-hidden-variable bound S <= 2.
       Reported as the EXPECTED sensitivity: the central value is held at
       the SM prediction (S = 2 sqrt(2), i.e. m12 = 2) and only the
       statistical uncertainty scales with the surviving sample size,
       z = (S_SM - 2) / sigma_S(N).  This is the median significance with
       which an SM signal would exclude S <= 2 at each v_psi, and is
       monotonic in N (unlike the per-subsample measured score, which
       fluctuates and is kept as 'sigma_vs_1_measured' for reference).

    Returns
    -------
    dict with n_events, m12 (raw and bias-corrected, for display),
    LR-test p-value/z, CHSH S and significances.
    """
    mask = v_signal_arr > v_psi
    n_sel = int(np.sum(mask))

    if n_sel < 6:
        return {
            'n_events': n_sel,
            'm12': np.nan, 'm12_err': np.nan, 'm12_bc': np.nan,
            'm12_null_exp': np.nan,
            'sigma_vs_0': 0.0, 'sigma_vs_1': 0.0, 'sigma_vs_1_measured': 0.0,
            'p_llr': np.nan, 'z_llr': 0.0,
            'chsh_S': np.nan, 'chsh_S_err': np.nan,
            'concurrence': np.nan, 'concurrence_err': np.nan,
            'witness_W': np.nan, 'witness_W_err': np.nan,
        }

    cp, cm = cos_theta_plus[mask], cos_theta_minus[mask]

    result = bootstrap_entanglement(cp, cm, n_bootstrap=n_bootstrap, rng=rng)

    # Optimal LR test, permutation-calibrated (exact at finite N)
    perm = permutation_test(cp, cm, statistic='llr', n_perm=n_perm, rng=rng)

    # Expected CHSH sensitivity: hold the central value at the SM prediction
    # (S = 2 sqrt(2)) and scale only the measured statistical error with N.
    # Gives the median significance for excluding S <= 2, monotonic in N.
    s_err = result['chsh_S_err']
    if np.isfinite(s_err) and s_err > 0:
        sigma_vs_1_proj = (CHSH_SM - 2.0) / s_err
    else:
        sigma_vs_1_proj = 0.0

    return {
        'n_events': n_sel,
        'm12': result['m12'],
        'm12_err': result['m12_err'],
        'm12_bc': result['m12_bc'],
        'm12_null_exp': m12_null_expectation(n_sel),
        'sigma_vs_0': perm['z_score'],
        'sigma_vs_1': sigma_vs_1_proj,
        'sigma_vs_1_measured': result['chsh_z_bell'],
        'p_llr': perm['p_value'],
        'z_llr': perm['z_score'],
        'chsh_S': result['chsh_S'],
        'chsh_S_err': result['chsh_S_err'],
        'concurrence': result['concurrence'],
        'concurrence_err': result['concurrence_err'],
        'witness_W': result['witness_W'],
        'witness_W_err': result['witness_W_err'],
    }


def scan_vpsi(cos_theta_plus, cos_theta_minus, v_signal_arr,
              v_psi_values, n_bootstrap=N_BOOTSTRAP, n_perm=5000):
    """Scan over v_psi hypotheses and compute rejection significance.

    Returns a list of dicts (one per v_psi value).
    """
    rng = np.random.default_rng(42)
    results = []
    for v_psi in v_psi_values:
        res = vpsi_rejection_significance(
            cos_theta_plus, cos_theta_minus, v_signal_arr,
            v_psi, n_bootstrap=n_bootstrap, n_perm=n_perm, rng=rng,
        )
        res['v_psi'] = v_psi
        results.append(res)
    return results

"""Vectorised relativistic kinematics.  Four-vectors are (..., 4) = (E, px, py, pz)."""
import numpy as np

C_LIGHT = 2.99792458e8


def p3(p):
    return p[..., 1:4]


def mag(v):
    return np.linalg.norm(v, axis=-1)


def unit(v):
    m = mag(v)[..., None]
    return v / np.where(m > 0, m, 1.0)


def dot3(a, b):
    return np.sum(a * b, axis=-1)


def mass2(p):
    return p[..., 0]**2 - dot3(p3(p), p3(p))


def mass(p):
    return np.sqrt(np.maximum(mass2(p), 0.0))


def beta_of(p):
    return p3(p) / p[..., 0:1]


def boost_to_rest(p, beta):
    """Lorentz-transform p into the frame moving with velocity beta.
    Works for momenta (E, p) and for four-positions (ct, x)."""
    b2 = dot3(beta, beta)
    g = 1.0 / np.sqrt(1.0 - b2)
    bp = dot3(beta, p3(p))
    safe = np.where(b2 > 0, b2, 1.0)
    coef = (g - 1.0) * bp / safe - g * p[..., 0]
    out = np.empty(np.broadcast(p, beta[..., :1]).shape[:-1] + (4,))
    out[..., 0] = g * (p[..., 0] - bp)
    out[..., 1:] = p3(p) + coef[..., None] * beta
    return out


def boost_from_rest(p, beta):
    return boost_to_rest(p, -beta)


def four(e, v):
    return np.concatenate([np.asarray(e)[..., None], v], axis=-1)


def nrk_basis(k_hat, beam_hat):
    """{n, r, k}: k = τ- direction, r = beam direction minus its k component."""
    r = beam_hat - dot3(beam_hat, k_hat)[..., None] * k_hat
    r = unit(r)
    n = np.cross(k_hat, r)
    return n, r, k_hat


def spin_angles(pi_m, pi_p, p_h, k_hat_h, beam_minus, e_tau=None, m_tau=1.77693,
                tau_p_hat_h=None, tau_h=None):
    """Pion directions in the τ rest frames, as (n, r, k) components.

    pi_m, pi_p : lab pion four-momenta
    p_h        : lab Higgs (ττ system) four-momentum
    k_hat_h    : τ- direction in the Higgs frame
    tau_p_hat_h: τ+ direction in the Higgs frame (default -k_hat_h)
    e_tau      : τ energy in the Higgs frame (default m(p_h)/2)
    tau_h      : optional (τ-, τ+) four-momenta in the Higgs frame; if given
                 they define the τ rest frames (truth level)
    Returns q_minus, q_plus (N, 3) and the acoplanarity angle.
    """
    bh = beta_of(p_h)
    if e_tau is None:
        e_tau = mass(p_h) / 2.0
    e_tau = np.broadcast_to(e_tau, k_hat_h.shape[:-1])
    pt = np.sqrt(np.maximum(e_tau**2 - m_tau**2, 0.0))
    if tau_p_hat_h is None:
        tau_p_hat_h = -k_hat_h
    beam_h = unit(p3(boost_to_rest(beam_minus, bh)))
    n, r, k = nrk_basis(k_hat_h, beam_h)
    out = []
    for j, (pi, d) in enumerate(((pi_m, k_hat_h), (pi_p, tau_p_hat_h))):
        pih = boost_to_rest(pi, bh)
        bt = beta_of(tau_h[j]) if tau_h is not None else (pt / e_tau)[..., None] * d
        q = unit(p3(boost_to_rest(pih, bt)))
        out.append(np.stack([dot3(q, n), dot3(q, r), dot3(q, k)], axis=-1))
    qm, qp = out
    dphi = np.arctan2(qp[..., 0], qp[..., 1]) - np.arctan2(qm[..., 0], qm[..., 1])
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return qm, qp, dphi


def c_matrix(qm, qp, w=None):
    """C_ij = -9 <q+_i q-_j> and its per-element standard error."""
    x = -9.0 * qp[:, :, None] * qm[:, None, :]
    if w is None:
        return x.mean(0), x.std(0) / np.sqrt(len(x))
    w = w / w.sum()
    m = np.tensordot(w, x, 1)
    v = np.tensordot(w, (x - m)**2, 1)
    return m, np.sqrt(v / (1.0 / np.sum(w**2)))


def acoplanarity_b(dphi):
    """B in dN/dφ ∝ 1 + B cos φ, unbinned moment estimator B = 2<cos φ>."""
    c = np.cos(dphi)
    return 2 * c.mean(), 2 * c.std() / np.sqrt(len(c))


def vmin(dv_m, dv_p):
    """Lab-frame v_min/c = |Δx| / (c |Δt|) from four-positions (t[s], x, y, z [m])."""
    dx = mag(dv_p[..., 1:] - dv_m[..., 1:])
    cdt = C_LIGHT * np.abs(dv_p[..., 0] - dv_m[..., 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cdt > 0, dx / cdt, np.inf)

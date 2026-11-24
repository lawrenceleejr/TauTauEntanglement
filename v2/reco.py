"""
Vectorised H -> τ τ reconstruction for e+e- -> ZH with both τ -> π ν.

1. Recoil: p_H = p_beams - p_Z - p_photons(detected).  If the recoil mass
   exceeds m_H, one undetected initial-state photon collinear with either
   beam is fitted from the m_H constraint (closed form).
2. τ direction: in the Higgs rest frame both τ have E = m_H/2 and are back to
   back, so the τ- direction n satisfies n·π̂- = cos θ-, -n·π̂+ = cos θ+ with
   cos θ fixed by the τ mass.  This has two solutions (mirror images in the
   plane of the two pions); if measurement errors leave no real solution the
   point of closest approach is used.
3. The ambiguity is resolved with the tracks: for each solution the τ line
   from the primary vertex should meet the pion track.  The solution with the
   smaller miss-distance χ² is kept.
4. The decay vertex is the point on the τ line closest to the pion track.
   Decay times follow from the τ velocity.  v_min = |Δx| / (c |Δt|), lab frame.

This replaces the α-grid scan of tau_reconstruction.py, which had 5 mrad
spacing where the τ-π opening angle peaks.
"""
import numpy as np

from v2 import kin

M_H = 125.0
M_TAU = 1.77693
M_PI = 0.13957


def recoil(meas):
    return meas["beam_minus"] + meas["beam_plus"] - meas["p_z"] - meas["p_photons"]


def isr_candidates(p_rec, beam_minus, beam_plus):
    """Recoil corrected for one collinear photon along either beam.
    Returns list of (p_H, E_gamma) for: no photon, along e-, along e+."""
    out = [(p_rec, np.zeros(len(p_rec)))]
    m2 = kin.mass2(p_rec)
    for b in (beam_minus, beam_plus):
        u = kin.unit(kin.p3(b))
        den = 2 * (p_rec[:, 0] - kin.dot3(u, kin.p3(p_rec)))
        eg = np.where(m2 > M_H**2, (m2 - M_H**2) / np.maximum(den, 1e-9), 0.0)
        eg = np.clip(eg, 0, None)
        out.append((p_rec - kin.four(eg, eg[:, None] * u), eg))
    return out


def cone_cos(p_pi_h, e_tau):
    """cos of the τ-π angle in the frame where the τ has energy e_tau."""
    p_tau = np.sqrt(np.maximum(e_tau**2 - M_TAU**2, 1e-12))
    e_pi = p_pi_h[:, 0]
    p_pi = kin.mag(kin.p3(p_pi_h))
    return (2 * e_tau * e_pi - M_TAU**2 - M_PI**2) / (2 * p_tau * p_pi)


def cone_solutions(pi_m, pi_p, p_h):
    """Two candidate τ- directions in the Higgs frame, and the cone residual
    (0 if the cones intersect, else the squared shortfall)."""
    ok = (kin.mass2(p_h) > (2 * M_TAU + 1.0)**2) & (p_h[:, 0] > kin.mag(kin.p3(p_h)))
    p_h = np.where(ok[:, None], p_h, np.array([M_H, 0, 0, 0.0]))
    bh = kin.beta_of(p_h)
    e_tau = kin.mass(p_h) / 2
    a4, b4 = kin.boost_to_rest(pi_m, bh), kin.boost_to_rest(pi_p, bh)
    a, b = kin.unit(kin.p3(a4)), kin.unit(kin.p3(b4))
    c1 = np.clip(cone_cos(a4, e_tau), -1, 1)
    c2 = np.clip(cone_cos(b4, e_tau), -1, 1)
    g = kin.dot3(a, b)
    den = np.maximum(1 - g**2, 1e-12)
    x = (c1 + c2 * g) / den
    y = (-c2 - c1 * g) / den
    w = x[:, None] * a + y[:, None] * b
    axb = np.cross(a, b)
    z2 = (1 - kin.dot3(w, w)) / np.maximum(kin.dot3(axb, axb), 1e-12)
    resid = np.where(z2 < 0, -z2, 0.0)
    z = np.sqrt(np.maximum(z2, 0))
    n1 = kin.unit(w + z[:, None] * axb)
    n2 = kin.unit(w - z[:, None] * axb)
    return (n1, n2), resid, e_tau


def tau_lab(n_h, p_h, e_tau, sign):
    p = np.sqrt(np.maximum(e_tau**2 - M_TAU**2, 1e-12))
    t_h = kin.four(e_tau, (sign * p)[:, None] * n_h)
    return kin.boost_from_rest(t_h, kin.beta_of(p_h))


def line_pca(pv, t_hat, pt, u):
    """Closest approach between the τ line pv + L t̂ and the track pt + s û.
    Returns L, s, miss vector (track point minus τ point)."""
    w0 = pv - pt
    b = kin.dot3(t_hat, u)
    d = kin.dot3(t_hat, w0)
    e = kin.dot3(u, w0)
    den = np.maximum(1 - b**2, 1e-12)
    L = (b * e - d) / den
    s = (e - b * d) / den
    miss = (pt + s[:, None] * u) - (pv + L[:, None] * t_hat)
    return L, s, miss


def _miss_chi2(miss, meas, lab):
    var = meas[f"pi_{lab}_sd0"]**2 + meas[f"pi_{lab}_sz0"]**2
    # the PV error projected on the miss direction
    md = kin.unit(miss)
    var = var / 2 + np.einsum("ni,nij,nj->n", md, meas["pv_cov"], md)
    return kin.dot3(miss, miss) / var


SIGMA_CONE = 0.25e-3    # rad; 68% half-width of the Higgs-frame cone-angle residual
                        # for events without radiation, beam energies known,
                        # on an independent 10^5-event test sample: 0.23 mrad


def _hframe(p_h, e_tau_fixed=None):
    e_tau = kin.mass(p_h) / 2 if e_tau_fixed is None else e_tau_fixed
    return kin.beta_of(p_h), e_tau


def _residuals(par, n0, e1, e2, p_rec, meas, pv, sig_beam):
    """Residuals of the τ-direction fit in units of their errors.

    par = (α, β, δ-, δ+): τ- Higgs-frame direction n = unit(n0 + α e1 + β e2)
    and the beam-energy corrections (GeV) applied to the recoil.
    Residuals: two Higgs-frame cone residuals, two signed track-miss
    distances, two beam-energy pulls."""
    n_h = kin.unit(n0 + par[:, 0:1] * e1 + par[:, 1:2] * e2)
    p_h = p_rec + par[:, 2:3] * meas["_u_minus"] + par[:, 3:4] * meas["_u_plus"]
    bh, e_tau = _hframe(p_h)
    a4 = kin.boost_to_rest(meas["pi_m"], bh)
    b4 = kin.boost_to_rest(meas["pi_p"], bh)
    th1 = np.arccos(np.clip(cone_cos(a4, e_tau), -1, 1))
    th2 = np.arccos(np.clip(cone_cos(b4, e_tau), -1, 1))
    r = [(np.arccos(np.clip(kin.dot3(n_h, kin.unit(kin.p3(a4))), -1, 1)) - th1) / SIGMA_CONE,
         (np.arccos(np.clip(-kin.dot3(n_h, kin.unit(kin.p3(b4))), -1, 1)) - th2) / SIGMA_CONE]
    extra = {"n_h": n_h, "p_h": p_h, "e_tau": e_tau}
    for lab, sign in (("m", 1.0), ("p", -1.0)):
        t = tau_lab(n_h, p_h, e_tau, sign)
        th = kin.unit(kin.p3(t))
        u = meas[f"pi_{lab}_dir"]
        L, s, miss = line_pca(pv, th, meas[f"pi_{lab}_pt"], u)
        nrm = kin.unit(np.cross(th, u))
        var = meas[f"pi_{lab}_sd0"]**2 / 2 + meas[f"pi_{lab}_sz0"]**2 / 2 \
            + np.einsum("ni,nij,nj->n", nrm, meas["pv_cov"], nrm)
        r.append(kin.dot3(miss, nrm) / np.sqrt(var))
    r.append(par[:, 2] / sig_beam)
    r.append(par[:, 3] / sig_beam)
    return np.stack(r, axis=1), extra


def fit_direction(n0, p_rec, meas, pv, sig_beam, n_iter=8):
    """Gauss-Newton fit of the τ- direction and the two beam energies."""
    one = np.ones(len(n0))
    meas["_u_minus"] = kin.four(one, kin.unit(kin.p3(meas["beam_minus"])))
    meas["_u_plus"] = kin.four(one, kin.unit(kin.p3(meas["beam_plus"])))
    ref = np.where(np.abs(n0[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    e1 = kin.unit(np.cross(n0, ref))
    e2 = np.cross(n0, e1)
    par = np.zeros((len(n0), 4))
    steps = np.array([1e-6, 1e-6, 1e-4, 1e-4])
    lim = np.array([0.05, 0.05, 1.0, 1.0])
    for _ in range(n_iter):
        r0, _ = _residuals(par, n0, e1, e2, p_rec, meas, pv, sig_beam)
        J = np.empty(r0.shape + (4,))
        for k in range(4):
            dp = par.copy()
            dp[:, k] += steps[k]
            r1, _ = _residuals(dp, n0, e1, e2, p_rec, meas, pv, sig_beam)
            J[..., k] = (r1 - r0) / steps[k]
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        r0 = np.nan_to_num(r0, nan=0.0, posinf=0.0, neginf=0.0)
        JtJ = np.einsum("nki,nkj->nij", J, J)
        JtJ = JtJ + (1e-9 * np.trace(JtJ, axis1=1, axis2=2)[:, None, None] + 1e-12) * np.eye(4)
        Jtr = np.einsum("nki,nk->ni", J, r0)
        step = -np.linalg.solve(JtJ, Jtr[..., None])[..., 0]
        par = par + np.clip(step, -lim, lim)
    r, extra = _residuals(par, n0, e1, e2, p_rec, meas, pv, sig_beam)
    return extra, np.sum(r**2, axis=1)


def reconstruct(meas, isr=True, fit=True, sig_beam=None):
    """Return a dict of reconstructed quantities for every event.

    sig_beam: r.m.s. of each beam energy about its nominal value, GeV
    (default: beam_energy_spread x E_beam)."""
    if sig_beam is None:
        import beam_parameters as bp
        sig_beam = max(bp.BEAM_ENERGY_SPREAD * meas["beam_minus"][0, 0], 1e-3)
    pv = meas["pv"]
    p_rec = recoil(meas)
    cands = isr_candidates(p_rec, meas["beam_minus"], meas["beam_plus"]) if isr \
        else [(p_rec, np.zeros(len(p_rec)))]
    best = None
    for ci, (p_h, eg) in enumerate(cands):
        (n1, n2), cres, e_tau = cone_solutions(meas["pi_m"], meas["pi_p"], p_h)
        for si, n_start in enumerate((n1, n2)):
            if fit:
                extra, chi2 = fit_direction(n_start, p_h, meas, pv, sig_beam)
                n_h, p_fit, e_fit = extra["n_h"], extra["p_h"], extra["e_tau"]
            else:
                n_h, p_fit, e_fit = n_start, p_h, e_tau
                chi2 = np.zeros(len(pv))
            sol = {"p_h": p_fit, "e_gamma": eg, "n_h": n_h, "e_tau": e_fit,
                   "cone_resid": cres, "isr_choice": np.full(len(pv), ci)}
            for lab, sign in (("m", 1.0), ("p", -1.0)):
                t = tau_lab(n_h, p_fit, e_fit, sign)
                th = kin.unit(kin.p3(t))
                L, s, miss = line_pca(pv, th, meas[f"pi_{lab}_pt"], meas[f"pi_{lab}_dir"])
                sol[f"tau_{lab}"] = t
                sol[f"L_{lab}"] = L
                sol[f"dv_{lab}_xyz"] = pv + L[:, None] * th
                sol[f"miss_{lab}"] = miss
                if not fit:
                    chi2 = chi2 + _miss_chi2(miss, meas, lab)
                # an upstream decay is unphysical; penalise it by its significance
                chi2 = chi2 + np.where(L < 0, (L / 50e-6)**2, 0.0)
            if not fit:
                chi2 = chi2 + cres / 1e-4
            sol["chi2"] = chi2
            if best is None:
                best = sol
            else:
                take = chi2 < best["chi2"]
                for k in best:
                    v = sol[k]
                    best[k] = np.where(take[:, None], v, best[k]) if np.ndim(v) == 2 \
                        else np.where(take, v, best[k])
    # validity: physical Higgs candidate and τ energy above the τ mass
    m_h = kin.mass2(best["p_h"])
    best["ok"] = (m_h > (2 * M_TAU + 1.0)**2) & (best["p_h"][:, 0] > kin.mag(kin.p3(best["p_h"])))
    for k in ("p_h", "n_h", "tau_m", "tau_p"):
        best[k] = np.where(best["ok"][:, None], best[k], _SAFE[k](len(m_h)))
    best["e_tau"] = np.where(best["ok"], best["e_tau"], M_H / 2)
    # decay times (relative to the common production time)
    for lab in ("m", "p"):
        t = best[f"tau_{lab}"]
        beta = kin.mag(kin.p3(t)) / t[:, 0]
        best[f"t_{lab}"] = best[f"L_{lab}"] / (beta * kin.C_LIGHT)
    dx = kin.mag(best["dv_p_xyz"] - best["dv_m_xyz"])
    cdt = kin.C_LIGHT * np.abs(best["t_p"] - best["t_m"])
    best["v_min"] = np.where(cdt > 0, dx / np.where(cdt > 0, cdt, 1), np.inf)
    # spin observables
    qm, qp, dphi = kin.spin_angles(meas["pi_m"], meas["pi_p"], best["p_h"], best["n_h"],
                                   meas["beam_minus"], e_tau=best["e_tau"], m_tau=M_TAU)
    best["q_m"], best["q_p"], best["dphi"] = qm, qp, dphi
    # impact-parameter significance of each pion w.r.t. the fitted vertex
    for lab in ("m", "p"):
        u = meas[f"pi_{lab}_dir"]
        r = meas[f"pi_{lab}_pt"] - pv
        d = r - kin.dot3(r, u)[:, None] * u
        dhat = kin.unit(d)
        var = meas[f"pi_{lab}_sd0"]**2 * kin.dot3(dhat, _e1(u))**2 \
            + meas[f"pi_{lab}_sz0"]**2 * kin.dot3(dhat, np.cross(u, _e1(u)))**2 \
            + np.einsum("ni,nij,nj->n", dhat, meas["pv_cov"], dhat)
        best[f"ip_sig_{lab}"] = kin.mag(d) / np.sqrt(var)
    # neutrino-sum residual (closes by construction when the cones intersect)
    p_nu = best["tau_m"] + best["tau_p"] - meas["pi_m"] - meas["pi_p"]
    p_miss = best["p_h"] - meas["pi_m"] - meas["pi_p"]
    best["nu_residual"] = kin.mag(kin.p3(p_nu - p_miss))
    return best


_SAFE = {
    "p_h": lambda n: np.tile([M_H, 0, 0, 0.0], (n, 1)),
    "n_h": lambda n: np.tile([0, 0, 1.0], (n, 1)),
    "tau_m": lambda n: np.tile([M_H / 2, 0, 0, 62.47], (n, 1)),
    "tau_p": lambda n: np.tile([M_H / 2, 0, 0, -62.47], (n, 1)),
}


def _e1(u):
    return kin.unit(np.cross(np.array([0.0, 0.0, 1.0]), u))

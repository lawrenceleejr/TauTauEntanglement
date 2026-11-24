"""
Version-2 analysis: selection, response, test statistic and CL_s limits.

Hypothesis H(v_ψ, λ): a τ pair whose TRUE lab-frame v_min is below v_ψ is
correlated as in the SM; above v_ψ its density is (1-λ) W_SM + λ W_U, with
W_U the uncorrelated density.  λ = 1 is "no correlation"; λ = 1 - 1/√2
scales C by 1/√2, the local (CHSH = 2) bound, because W is linear in C.

Every event exists in two versions with identical kinematics, decay vertices
and detector random numbers: the SM version, and the U version in which the
τ+ decay is taken from another event.  Pseudo-experiments under H draw events
from the pool and pick the version per event, so selection efficiencies and
detector effects are exact.
"""
import numpy as np

import beam_parameters as bp
from v2 import kin, detector, reco
from v2.truth_gates import truth_observables

M_PI = 0.13957


# ---------------------------------------------------------------------------
# uncorrelated version of a sample
# ---------------------------------------------------------------------------
def swap_tau_plus(a, perm, sqrts):
    """Return a copy of sample `a` in which event i's τ+ decays like event
    perm[i]'s: the π+ direction in the τ+ rest frame, expressed in the
    {n, r, k} basis, is transplanted.  Decay vertices are unchanged."""
    qm, qp, _ = truth_observables(a, sqrts)
    ph = a["higgs"]
    bh = kin.beta_of(ph)
    tm_h = kin.boost_to_rest(a["tau_m"], bh)
    tp_h = kin.boost_to_rest(a["tau_p"], bh)
    from v2.truth_gates import nominal_beam_minus
    beam = np.broadcast_to(nominal_beam_minus(sqrts), ph.shape)
    if a["beams"][:, 1].std() == 0 and abs(a["beams"][0, 1]) < 1e-9:
        beam = np.broadcast_to(np.array([sqrts / 2, 0, 0, sqrts / 2]), ph.shape)
    beam_h = kin.unit(kin.p3(kin.boost_to_rest(beam, bh)))
    n, r, k = kin.nrk_basis(kin.unit(kin.p3(tm_h)), beam_h)
    c = qp[perm]
    q_new = c[:, 0:1] * n + c[:, 1:2] * r + c[:, 2:3] * k
    m_tau = kin.mass(a["tau_p"])
    p_star = (m_tau**2 - M_PI**2) / (2 * m_tau)
    e_star = (m_tau**2 + M_PI**2) / (2 * m_tau)
    pi_rf = kin.four(e_star, p_star[:, None] * q_new)
    pi_h = kin.boost_from_rest(pi_rf, kin.beta_of(tp_h))
    pi_lab = kin.boost_from_rest(pi_h, bh)
    b = dict(a)
    b["pi_p"] = pi_lab
    b["nu_p"] = a["tau_p"] - pi_lab
    return b


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
SELECTION = {
    "track_cos_max": 0.98,     # ASSUMPTION: tracking acceptance |cos θ| < 0.98
    "pion_p_min": 1.0,         # ASSUMPTION: pion identification above 1 GeV
    "recoil_window": (115.0, 160.0),   # GeV, recoil mass before the ISR fit
    "ip_sig_min": None,        # set by optimise_ip_cut() and frozen in results
}


def preselect(a, meas, r, zmode):
    """Selection before the vertex-quality cut.  Returns dict of step masks."""
    steps = {}
    cosmax = SELECTION["track_cos_max"]
    acc = np.ones(len(r["ok"]), bool)
    for lab in ("m", "p"):
        u = meas[f"pi_{lab}_dir"]
        acc &= np.abs(u[:, 2]) < cosmax
        acc &= kin.mag(kin.p3(meas[f"pi_{lab}"])) > SELECTION["pion_p_min"]
        if zmode in ("mumu", "ee"):
            acc &= np.abs(kin.unit(kin.p3(meas[f"lep_{lab}"]))[:, 2]) < cosmax
    steps["acceptance"] = acc
    steps["pi_nu_id"] = meas["pass_id"]
    m_rec = kin.mass(reco.recoil(meas))
    lo, hi = SELECTION["recoil_window"]
    steps["recoil_mass"] = (m_rec > lo) & (m_rec < hi)
    steps["reco_ok"] = r["ok"] & np.isfinite(r["v_min"])
    return steps


def ip_min(r):
    return np.minimum(r["ip_sig_m"], r["ip_sig_p"])


# ---------------------------------------------------------------------------
# reconstruction of both versions
# ---------------------------------------------------------------------------
def process(a, meta, seed, smear=True):
    """Measure and reconstruct the SM and U versions of a sample.

    Returns dict with per-event arrays for both versions."""
    sqrts, zmode = meta["sqrts"], meta["zmode"]
    beam_effects = meta.get("beam_effects", True)
    n = len(a["pi_m"])
    perm = np.random.default_rng(seed + 7).permutation(n)
    out = {"v_true": kin.vmin(a["dv_m"], a["dv_p"])}
    for ver, sample in (("sm", a), ("u", swap_tau_plus(a, perm, sqrts))):
        meas = detector.measure(sample, np.random.default_rng(seed), sqrts, zmode,
                                smear=smear, beam_effects=beam_effects)
        r = chunked_reco(meas)
        steps = preselect(sample, meas, r, zmode)
        qm_t, qp_t, dphi_t = truth_observables(sample, sqrts)
        out[ver] = {
            "q_m": r["q_m"], "q_p": r["q_p"], "dphi": r["dphi"], "v_reco": r["v_min"],
            "ip_min": ip_min(r), "steps": steps,
            "q_m_true": qm_t, "q_p_true": qp_t, "dphi_true": dphi_t,
            "tau_dir_err": np.arccos(np.clip(kin.dot3(kin.unit(kin.p3(r["tau_m"])),
                                                      kin.unit(kin.p3(sample["tau_m"]))), -1, 1)),
            "L_ratio": r["L_m"] / kin.mag(sample["dv_m"][:, 1:] - sample["pv"][:, 1:]),
            "m_recoil": kin.mass(reco.recoil(meas)),
            "e_gamma_fit": r["e_gamma"], "nu_residual": r["nu_residual"],
        }
    return out


def chunked_reco(meas, size=50000):
    n = len(meas["pv"])
    parts = []
    for i in range(0, n, size):
        sub = {k: (v[i:i + size] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
               for k, v in meas.items()}
        parts.append(reco.reconstruct(sub))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------------------
# reco-level density ratio  f_U / f_SM  after selection
# ---------------------------------------------------------------------------
def features(q_m, q_p):
    return (q_p[:, :, None] * q_m[:, None, :]).reshape(len(q_m), 9)


def fit_density_ratio(q_sm, q_u):
    """Model f_SM/f_U = κ (1 - x·D), x_ij = q+_i q-_j, by matching the
    moments E_SM[x] = E_U[x f_SM/f_U] (closed form).  Returns (κ, D)."""
    x_sm = features(*q_sm)
    x_u = features(*q_u)
    m_u = x_u.mean(0)
    m_sm = x_sm.mean(0)
    M_u = x_u.T @ x_u / len(x_u)
    A = M_u - np.outer(m_sm, m_u)
    D = np.linalg.solve(A, m_u - m_sm)
    kappa = 1.0 / (1.0 - D @ m_u)
    return float(kappa), D


def log_ratio_u_over_sm(q_m, q_p, kappa, D, floor=0.02):
    """ln f_U/f_SM per event."""
    s = 1.0 - features(q_m, q_p) @ D
    return -np.log(kappa * np.maximum(s, floor))


# ---------------------------------------------------------------------------
# v_min response
# ---------------------------------------------------------------------------
LNV_EDGES = np.concatenate([[-50.0], np.linspace(0.0, np.log(3e3), 61), [50.0]])


def response(v_reco, v_true, v_psi_grid):
    """w[k, b] = P(v_true > v_psi_grid[k] | ln v_reco in bin b)."""
    lv = np.log(np.clip(v_reco, 1e-12, 1e12))
    b = np.clip(np.digitize(lv, LNV_EDGES) - 1, 0, len(LNV_EDGES) - 2)
    nb = len(LNV_EDGES) - 1
    tot = np.bincount(b, minlength=nb).astype(float)
    w = np.zeros((len(v_psi_grid), nb))
    for k, v in enumerate(v_psi_grid):
        hi = np.bincount(b, weights=(v_true > v).astype(float), minlength=nb)
        # empty bins: fall back to the step function at the bin centre
        centre = np.exp(0.5 * (LNV_EDGES[:-1] + LNV_EDGES[1:]))
        w[k] = np.where(tot > 0, hi / np.maximum(tot, 1), (centre > v).astype(float))
    return w


def response_weight(w_row, v_reco):
    lv = np.log(np.clip(v_reco, 1e-12, 1e12))
    b = np.clip(np.digitize(lv, LNV_EDGES) - 1, 0, len(LNV_EDGES) - 2)
    return w_row[b]


# ---------------------------------------------------------------------------
# pseudo-experiments and CL_s
# ---------------------------------------------------------------------------
# Expected-limit bands: quantiles p of the summed statistic under the SM.
# Small p is the less SM-like outcome (weaker exclusion).
QUANTILES = {"p02.3": 0.02275, "p15.9": 0.15866, "p50": 0.5, "p84.1": 0.84134,
             "p97.7": 0.97725}

def per_event_t(lnr, w, lam):
    """-2 ln [ f_H / f_SM ] for one event, f_H/f_SM = 1 + λ w (r - 1)."""
    return -2.0 * np.log(np.maximum(1.0 + lam * w * (np.exp(lnr) - 1.0), 1e-300))


def toys(pool, n_exp, v_psi, lam, w_row, kappa, D, n_toys, rng, hyp):
    """Summed statistic for n_toys pseudo-experiments drawn from `pool`.

    pool: dict with versions 'sm' and 'u', each with 'sel', 'q_m', 'q_p',
    'v_reco', and the shared 'v_true'.  hyp: 'sm' or 'h'."""
    tv = {}
    for ver in ("sm", "u"):
        p = pool[ver]
        lnr = log_ratio_u_over_sm(p["q_m"], p["q_p"], kappa, D)
        w = response_weight(w_row, p["v_reco"])
        tv[ver] = np.where(p["sel"], per_event_t(lnr, w, lam), 0.0)
    n_pool = len(pool["v_true"])
    above = pool["v_true"] > v_psi
    out = np.empty(n_toys)
    counts = rng.poisson(n_exp, n_toys)
    for i in range(n_toys):
        idx = rng.integers(0, n_pool, counts[i])
        if hyp == "sm":
            out[i] = tv["sm"][idx].sum()
        else:
            use_u = above[idx] & (rng.random(counts[i]) < lam)
            out[i] = np.where(use_u, tv["u"][idx], tv["sm"][idx]).sum()
    return out


def cls_expected(t_sm, t_h):
    """Expected CL_s = P_H(t >= t_obs) / P_SM(t >= t_obs), with t_obs at the
    SM quantiles in QUANTILES.  Large t is SM-like."""
    t_h = np.sort(t_h)
    res = {}
    for name, p in QUANTILES.items():
        t_obs = np.quantile(t_sm, p)
        cl_h = 1.0 - np.searchsorted(t_h, t_obs, side="left") / len(t_h)
        res[name] = float(cl_h / (1.0 - p))
    return res


def crossing(v_grid, cls, level=0.05):
    """Largest v_ψ with CL_s < level, interpolated in (ln v, ln CL_s)."""
    v_grid = np.asarray(v_grid)
    cls = np.clip(np.asarray(cls), 1e-12, 1)
    below = cls < level
    if not below[0]:
        return float(v_grid[0]) if below.any() else 0.0
    k = np.argmax(~below) if (~below).any() else len(v_grid)
    if k == len(v_grid):
        return float(v_grid[-1])
    x0, x1 = np.log(v_grid[k - 1]), np.log(v_grid[k])
    y0, y1 = np.log(cls[k - 1]), np.log(cls[k])
    x = x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0)
    return float(np.exp(x))

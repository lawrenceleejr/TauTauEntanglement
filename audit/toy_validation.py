#!/usr/bin/env python3
"""
Independent toy-MC cross-checks for the audit of arXiv:2603.28868.

Generates e+e- -> ZH -> (mu mu)(tau tau), tau -> pi nu from first principles
(exact SM spin density for a CP-even scalar, exponential proper lifetimes),
without using any of the analysis code, and then:

  Part 1  truth-level facts (no reconstruction)
          - fraction of timelike-separated decay pairs
          - P(v_min > V) in the Higgs frame and in the lab frame
          - size of the pion impact parameter relative to sigma(d0)
          - acoplanarity coefficient B
          - luminosity arithmetic

  Part 2  behaviour of the m12 estimator at finite N
          - bias under the uncorrelated null and under the SM
          - significance the paper quotes vs a calibrated test
          - the optimal likelihood-ratio test and an unbiased linear CHSH test

  Part 3  toy events pushed through the repository's own smearing and
          reconstruction chain, with sqrt(s) of the sample equal to (240) and
          different from (250) the value assumed by the analysis (240)

Usage:
    python audit/toy_validation.py [--n-reco 6000] [--out audit/toy_results.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

M_TAU = 1.77686
M_PI = 0.13957
M_MU = 0.10566
M_H = 125.0
M_Z = 91.1876
G_Z = 2.4952
CTAU = 87.03e-6       # m
C_LIGHT = 2.99792458e8
P_STAR = (M_TAU**2 - M_PI**2) / (2 * M_TAU)
E_STAR = (M_TAU**2 + M_PI**2) / (2 * M_TAU)
B_SM = -np.pi**2 / 16


# ---------------------------------------------------------------------------
# kinematics helpers (vectorised; 4-vectors are (..., 4) = (E, px, py, pz))
# ---------------------------------------------------------------------------
def boost_from_rest(p4, beta):
    """Active boost: p4 given in the rest frame of a parent that moves with
    velocity `beta` (..., 3) in the target frame.  Returns p4 in that frame."""
    b2 = np.sum(beta**2, axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * p4[..., 1:], axis=-1)
    safe = np.where(b2 > 0, b2, 1.0)
    coef = (gamma - 1.0) * bp / safe + gamma * p4[..., 0]
    out = np.empty_like(p4)
    out[..., 0] = gamma * (p4[..., 0] + bp)
    out[..., 1:] = p4[..., 1:] + coef[..., None] * beta
    return out


def iso(n, rng):
    c = rng.uniform(-1, 1, n)
    s = np.sqrt(1 - c**2)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([s * np.cos(ph), s * np.sin(ph), c], axis=-1)


def sample_spin(n, rng, correlated=True):
    """Pion directions (q_minus, q_plus) in the tau rest frames, components
    along (n, r, k).  SM CP-even scalar: W = (1 - q+ . R q-)/2 with
    R = diag(1, 1, -1), i.e. C = diag(+1, +1, -1) in the convention
    C_ij = -9 <q+_i q-_j>."""
    qm = np.empty((0, 3))
    qp = np.empty((0, 3))
    R = np.array([1.0, 1.0, -1.0])
    while len(qm) < n:
        a = iso(2 * n, rng)
        b = iso(2 * n, rng)
        if correlated:
            w = 0.5 * (1.0 - np.sum(b * (a * R), axis=1))
            keep = rng.uniform(0, 1, 2 * n) < w
            a, b = a[keep], b[keep]
        qm = np.concatenate([qm, a])
        qp = np.concatenate([qp, b])
    return qm[:n], qp[:n]


def generate(n, sqrts, rng, correlated=True):
    """Generate truth-level events.  Returns a dict of arrays."""
    s = sqrts**2
    # Z lineshape (truncated Breit-Wigner), Higgs narrow
    mz = M_Z + 0.5 * G_Z * np.tan(rng.uniform(-1.3, 1.3, n))
    e_h = (s + M_H**2 - mz**2) / (2 * sqrts)
    p_h = np.sqrt(e_h**2 - M_H**2)
    dir_h = iso(n, rng)
    beta_h = (p_h / e_h)[:, None] * dir_h
    beta_z = -(p_h / (sqrts - e_h))[:, None] * dir_h

    # Z -> mu mu
    dmu = iso(n, rng)
    pmu = np.sqrt((mz / 2)**2 - M_MU**2)
    mu_m = np.concatenate([(mz / 2)[:, None], pmu[:, None] * dmu], axis=1)
    mu_p = np.concatenate([(mz / 2)[:, None], -pmu[:, None] * dmu], axis=1)
    mu_m = boost_from_rest(mu_m, beta_z)
    mu_p = boost_from_rest(mu_p, beta_z)

    # H -> tau tau in the Higgs frame; k = tau- direction
    k = iso(n, rng)
    e_t = M_H / 2
    p_t = np.sqrt(e_t**2 - M_TAU**2)
    beta_t = p_t / e_t
    # {n, r, k} basis exactly as the analysis defines it: r = beam component
    # perpendicular to k, with the e- beam direction boosted to the H frame.
    beam = np.broadcast_to(np.array([sqrts / 2, 0, 0, sqrts / 2]), (n, 4)).copy()
    beam_h = boost_from_rest(beam, -beta_h)
    bh = beam_h[:, 1:] / np.linalg.norm(beam_h[:, 1:], axis=1, keepdims=True)
    r = bh - np.sum(bh * k, axis=1, keepdims=True) * k
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    nn = np.cross(k, r)

    qm, qp = sample_spin(n, rng, correlated)

    def to_vec(q):
        return q[:, 0:1] * nn + q[:, 1:2] * r + q[:, 2:3] * k

    def pion_lab(q, sign):
        p4 = np.concatenate([np.full((n, 1), E_STAR), P_STAR * to_vec(q)], axis=1)
        p4 = boost_from_rest(p4, sign * beta_t * k)
        return boost_from_rest(p4, beta_h)

    pi_m = pion_lab(qm, +1)
    pi_p = pion_lab(qp, -1)
    tau_m_h = np.concatenate([np.full((n, 1), e_t), p_t * k], axis=1)
    tau_p_h = np.concatenate([np.full((n, 1), e_t), -p_t * k], axis=1)
    tau_m = boost_from_rest(tau_m_h, beta_h)
    tau_p = boost_from_rest(tau_p_h, beta_h)

    # decays: proper times in units of the lifetime
    tm = rng.exponential(1.0, n)
    tp = rng.exponential(1.0, n)

    def vertex(tau, t):
        pmag = np.linalg.norm(tau[:, 1:], axis=1)
        L = pmag / M_TAU * CTAU * t                 # m
        xyz = L[:, None] * tau[:, 1:] / pmag[:, None]
        t_lab = L / (pmag / tau[:, 0]) / C_LIGHT    # s
        return L, xyz, t_lab

    L_m, x_m, t_m = vertex(tau_m, tm)
    L_p, x_p, t_p = vertex(tau_p, tp)

    return dict(mu_m=mu_m, mu_p=mu_p, pi_m=pi_m, pi_p=pi_p, tau_m=tau_m,
                tau_p=tau_p, x_m=x_m, x_p=x_p, t_m=t_m, t_p=t_p,
                L_m=L_m, L_p=L_p, qm=qm, qp=qp, tm=tm, tp=tp,
                beta_t=beta_t, gamma_t=e_t / M_TAU)


# ---------------------------------------------------------------------------
# estimators (independent re-implementations)
# ---------------------------------------------------------------------------
def c_matrix(qp, qm):
    return -9.0 * (qp.T @ qm) / len(qp)


def m12_of(C):
    ev = np.sort(np.linalg.eigvalsh(C.T @ C))[::-1]
    return ev[0] + ev[1]


def m12_hat(qp, qm):
    return m12_of(c_matrix(qp, qm))


def acoplanarity_B(qp, qm):
    """B in dN/dphi ~ 1 + B cos(dphi), dphi = phi+ - phi-, phi = atan2(n, r).
    Unbinned moment estimator B = 2 <cos dphi>."""
    dphi = np.arctan2(qp[:, 0], qp[:, 1]) - np.arctan2(qm[:, 0], qm[:, 1])
    c = np.cos(dphi)
    return 2 * c.mean(), 2 * c.std() / np.sqrt(len(c))


def sigma_d0(p, theta):
    st = np.maximum(np.abs(np.sin(theta)), 0.1)
    return np.sqrt((5e-6)**2 + (10e-6 / (p * st**1.5))**2)


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------
def part1(rng, n=2_000_000):
    out = {}
    for sqrts in (240.0, 250.0):
        ev = generate(n, sqrts, rng)
        dr = np.linalg.norm(ev['x_p'] - ev['x_m'], axis=1)
        cdt = C_LIGHT * np.abs(ev['t_p'] - ev['t_m'])
        v_lab = dr / cdt
        # Higgs frame: back-to-back, equal speeds
        b = ev['beta_t']
        v_h = b * (ev['tm'] + ev['tp']) / np.abs(ev['tm'] - ev['tp'])
        res = {
            'timelike_fraction': float(np.mean(dr < cdt)),
            'timelike_fraction_analytic': float((1 - b) / (1 + b) * 2 /
                                                (1 + (1 - b) / (1 + b))),
            'P_vmin_gt_V_lab': {str(V): float(np.mean(v_lab > V))
                                for V in (2, 5, 10, 20, 50, 100)},
            'P_vmin_gt_V_higgs': {str(V): float(np.mean(v_h > V))
                                  for V in (2, 5, 10, 20, 50, 100)},
        }
        # impact parameters (3D distance of the pion line from the PV)
        for lab, pi, x in (('m', ev['pi_m'], ev['x_m']),):
            ph = pi[:, 1:] / np.linalg.norm(pi[:, 1:], axis=1, keepdims=True)
            d = np.linalg.norm(x - np.sum(x * ph, axis=1, keepdims=True) * ph,
                               axis=1)
            pmag = np.linalg.norm(pi[:, 1:], axis=1)
            theta = np.arccos(pi[:, 3] / pmag)
            sig = sigma_d0(pmag, theta)
            alpha = np.arccos(np.clip(np.sum(
                ph * ev['tau_m'][:, 1:], axis=1) /
                np.linalg.norm(ev['tau_m'][:, 1:], axis=1), -1, 1))
            res['ip_um_percentiles_16_50_84'] = [
                float(x_) for x_ in np.percentile(d * 1e6, [16, 50, 84])]
            res['ip_over_sigma_median'] = float(np.median(d / sig))
            res['frac_ip_sig_gt3_single'] = float(np.mean(d / sig > 3))
            res['tau_pi_angle_mrad_median'] = float(np.median(alpha) * 1e3)
            res['decay_length_mm_mean'] = float(ev['L_m'].mean() * 1e3)
        B, Be = acoplanarity_B(ev['qp'], ev['qm'])
        res['B_truth'] = [float(B), float(Be)]
        res['B_SM_analytic'] = float(B_SM)
        res['m12_truth_full_sample'] = float(m12_hat(ev['qp'], ev['qm']))
        out[f'sqrts_{int(sqrts)}'] = res

    # luminosity arithmetic
    sig_eff = 240.0 * 0.0627 * 0.03366 * 0.1082**2      # fb
    out['luminosity'] = {
        'sigma_eff_fb': sig_eff,
        'events_per_ab': sig_eff * 1000.0,
        'lumi_for_4462_events_ab': 4462 / sig_eff / 1000.0,
        'events_at_0p75_ab': 0.75 * 1000 * sig_eff,
        'events_at_2_ab_ILC': 2.0 * 1000 * sig_eff,
        'events_at_10p8_ab_FCCee': 10.8 * 1000 * sig_eff,
        'events_at_20_ab_CEPC': 20.0 * 1000 * sig_eff,
    }
    return out


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------
def part2(rng, n_pseudo=2000):
    out = {}
    R = np.array([1.0, 1.0, -1.0])
    for N in (30, 100, 300, 1000, 4462):
        qm0, qp0 = sample_spin(N * n_pseudo, rng, correlated=False)
        qm1, qp1 = sample_spin(N * n_pseudo, rng, correlated=True)
        qm0 = qm0.reshape(n_pseudo, N, 3)
        qp0 = qp0.reshape(n_pseudo, N, 3)
        qm1 = qm1.reshape(n_pseudo, N, 3)
        qp1 = qp1.reshape(n_pseudo, N, 3)
        C0 = -9.0 * np.einsum('pni,pnj->pij', qp0, qm0) / N
        C1 = -9.0 * np.einsum('pni,pnj->pij', qp1, qm1) / N
        m0 = np.array([m12_of(c) for c in C0])
        m1 = np.array([m12_of(c) for c in C1])
        # what the arXiv version quotes: Z = 2 / sigma(m12) and 1 / sigma(m12)
        z_paper_0 = 2.0 / m1.std()
        z_paper_1 = 1.0 / m1.std()
        # calibrated: where does the median SM outcome sit in the true null?
        p_emp = float(np.mean(m0 >= np.median(m1)))
        # optimal likelihood ratio for SM vs no correlation
        t0 = np.log(1.0 - np.sum(qp0 * qm0 * R, axis=2)).sum(axis=1)
        t1 = np.log(1.0 - np.sum(qp1 * qm1 * R, axis=2)).sum(axis=1)
        z_lr = (t1.mean() - t0.mean()) / t0.std()
        # unbiased linear CHSH with a-priori axes: S = sqrt2 (C_nn + C_rr) <= 2
        s1 = np.sqrt(2) * (C1[:, 0, 0] + C1[:, 1, 1])
        z_chsh = (s1.mean() - 2.0) / s1.std()
        out[str(N)] = {
            'm12_null_mean': float(m0.mean()), 'm12_null_std': float(m0.std()),
            'm12_null_95pct': float(np.percentile(m0, 95)),
            'frac_null_above_1': float(np.mean(m0 > 1.0)),
            'm12_SM_mean': float(m1.mean()), 'm12_SM_std': float(m1.std()),
            'frac_SM_above_2': float(np.mean(m1 > 2.0)),
            'Z_paper_reject_0': float(z_paper_0),
            'Z_paper_reject_1': float(z_paper_1),
            'null_pvalue_of_median_SM_outcome': p_emp,
            'Z_likelihood_ratio_vs_0': float(z_lr),
            'Z_linear_CHSH_vs_local_bound': float(z_chsh),
            'S_linear_mean': float(s1.mean()),
            'S_linear_var_times_N': float(s1.var() * N),
        }
    return out


# ---------------------------------------------------------------------------
# Part 3: repository reconstruction chain
# ---------------------------------------------------------------------------
def to_records(ev):
    from parse_hepmc import EventRecord, TauDecayInfo
    recs = []
    n = len(ev['tm'])
    zero4 = np.zeros(4)
    for i in range(n):
        def tau(pid, p_tau, p_pi, x, t):
            return TauDecayInfo(
                tau_p4=p_tau[i].copy(), tau_pdgid=pid, decay_mode="pi_nu",
                charged_pion_p4=p_pi[i].copy(), neutral_pions_p4=[],
                neutrino_p4=p_tau[i] - p_pi[i], visible_p4=p_pi[i].copy(),
                production_vertex=zero4.copy(),
                decay_vertex=np.array([t[i], *x[i]]))
        recs.append(EventRecord(
            event_number=i, mu_plus_p4=ev['mu_p'][i].copy(),
            mu_minus_p4=ev['mu_m'][i].copy(),
            tau_minus=tau(15, ev['tau_m'], ev['pi_m'], ev['x_m'], ev['t_m']),
            tau_plus=tau(-15, ev['tau_p'], ev['pi_p'], ev['x_p'], ev['t_p']),
            is_pi_pi=True))
    return recs


def part3(rng, n, sqrts_sample):
    """Run the repository chain (which assumes config.SQRT_S) on a toy sample
    generated at `sqrts_sample`."""
    import config
    from smearing import smear_event
    from tau_reconstruction import reconstruct_event, mass
    from spacetime import compute_reco_intervals, compute_truth_intervals
    from spin_analysis import compute_spin_observables

    ev = generate(n, sqrts_sample, rng)
    recs = to_records(ev)
    srng = np.random.default_rng(123)

    rows = []
    for rec in recs:
        meas = smear_event(rec, srng)
        reco = reconstruct_event(meas)
        if reco is None:
            continue
        sp = compute_spin_observables(meas, reco, use_reco_tau=True)
        iv = compute_reco_intervals(reco)
        tv = compute_truth_intervals(rec)
        ipmin = min(reco['tau_minus']['ip_significance'],
                    reco['tau_plus']['ip_significance'])
        k_true = rec.tau_minus.tau_p4[1:] / np.linalg.norm(rec.tau_minus.tau_p4[1:])
        ang = np.arccos(np.clip(np.dot(reco['tau_minus']['tau_dir'], k_true), -1, 1))
        rows.append((iv['v_signal_c'], tv['v_signal_c'], ipmin,
                     float(iv['is_spacelike']), mass(reco['p_H']),
                     reco['tau_minus']['decay_length_m'] /
                     reco['tau_minus']['truth_decay_length_m'], ang,
                     *sp['cos_theta_plus'], *sp['cos_theta_minus']))
    a = np.array(rows)
    v_reco, v_true, ipmin, spacelike, m_h, l_ratio, ang = a[:, :7].T
    qp, qm = a[:, 7:10], a[:, 10:13]
    sel = ipmin >= 3.0

    res = {
        'sqrts_sample': sqrts_sample, 'sqrts_assumed': float(config.SQRT_S),
        'n_generated': n, 'n_reconstructed': int(len(a)),
        'frac_pass_ip_cut': float(sel.mean()),
        'reco_higgs_mass_median': float(np.median(m_h)),
        'reco_timelike_fraction_after_ip_cut': float(1 - spacelike[sel].mean()),
        'tau_direction_error_mrad_median': float(np.median(ang) * 1e3),
        'L_reco_over_true_percentiles_16_50_84': [
            float(x) for x in np.percentile(l_ratio[sel], [16, 50, 84])],
        'B_reco_all': [float(x) for x in acoplanarity_B(qp, qm)],
        'B_reco_after_ip_cut': [float(x) for x in acoplanarity_B(qp[sel], qm[sel])],
        'C_diag_reco_all': [float(x) for x in np.diag(c_matrix(qp, qm))],
        'm12_reco_all': float(m12_hat(qp, qm)),
    }
    # response of v_min: reco bins vs truth
    edges = [1, 2, 4, 8, 16, 32, 1e9]
    binned = []
    vr, vt = v_reco[sel], v_true[sel]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vr >= lo) & (vr < hi)
        if m.sum() < 30:
            continue
        B, Be = acoplanarity_B(qp[sel][m], qm[sel][m])
        binned.append({
            'v_reco_bin': [lo, hi], 'n': int(m.sum()),
            'B': float(B), 'B_err': float(Be),
            'm12_hat': float(m12_hat(qp[sel][m], qm[sel][m])),
            'median_true_v': float(np.median(vt[m])),
            'frac_true_v_below_bin_lo': float(np.mean(vt[m] < lo)),
            'frac_true_v_below_half_bin_lo': float(np.mean(vt[m] < lo / 2)),
        })
    res['binned_vs_reco_vmin'] = binned
    # cumulative selection used for the exclusion: purity of "v_min > v_psi"
    cum = []
    for V in (2, 5, 9, 16, 30):
        m = vr > V
        if m.sum() < 30:
            continue
        cum.append({'v_psi': V, 'n': int(m.sum()),
                    'frac_truly_below_v_psi': float(np.mean(vt[m] < V)),
                    'efficiency_for_true_above': float(
                        np.sum(m & (vt > V)) / max(np.sum(vt > V), 1))})
    res['cumulative_selection'] = cum
    lr = np.log(np.clip(v_reco[sel] / v_true[sel], 1e-6, 1e6))
    res['ln_vreco_over_vtrue_percentiles_2.5_16_50_84_97.5'] = [
        float(x) for x in np.percentile(lr, [2.5, 16, 50, 84, 97.5])]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-reco', type=int, default=6000)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'toy_results.json'))
    args = ap.parse_args()
    rng = np.random.default_rng(20260330)

    results = {}
    print('Part 1: truth-level facts ...', flush=True)
    results['part1_truth'] = part1(rng)
    print('Part 2: estimator behaviour ...', flush=True)
    results['part2_estimators'] = part2(rng)
    if args.n_reco > 0:
        for s in (240.0, 250.0):
            print(f'Part 3: repository chain, sample at {s:.0f} GeV ...',
                  flush=True)
            results[f'part3_reco_sample_{int(s)}'] = part3(rng, args.n_reco, s)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()

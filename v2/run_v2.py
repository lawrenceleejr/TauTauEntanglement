#!/usr/bin/env python3
"""
Driver for the version-2 analysis.

    python -m v2.run_v2 process --mode mumu --sample A     # reco of both versions
    python -m v2.run_v2 closure                             # WP2 closure gate
    python -m v2.run_v2 analyse                             # WP2-4 tables, WP3 limits
    python -m v2.run_v2 whizard                             # WP1 generator cross-check

Large intermediate files go to results/processed_*.npz (not committed);
results/v2_results.json holds every number used in the paper.
"""
import argparse
import json
import os
import time

import numpy as np

import beam_parameters as bp
from generate_sample import load_sample
from v2 import analysis as an
from v2 import kin

RES = "results"
SAMPLES = {
    ("mumu", "A"): ("samples/zh240_mumu_A", 1101), ("mumu", "B"): ("samples/zh240_mumu_B", 1202),
    ("ee", "A"): ("samples/zh240_ee_A", 2101), ("ee", "B"): ("samples/zh240_ee_B", 2202),
    ("qq", "A"): ("samples/zh240_qq_A", 3101), ("qq", "B"): ("samples/zh240_qq_B", 3202),
}
SCENARIOS = {
    "mumu": ["mumu"],
    "ll": ["ee", "mumu"],
    "visible": ["ee", "mumu", "qq"],
}
SCENARIO_LABEL = {
    "mumu": "Z→μμ, ππ",
    "ll": "Z→ee, μμ, ππ",
    "visible": "Z→ee, μμ, qq̄ (projection), ππ",
}
V_PSI_GRID = np.exp(np.linspace(np.log(1.0), np.log(300.0), 33))
LAMBDAS = {"no_correlation": 1.0, "local_bound": 1.0 - 1.0 / np.sqrt(2.0)}


def n_expected_generated(mode):
    return (bp.SIGMA_ZH_FB * bp.INTEGRATED_LUMI_AB * 1000.0 * bp.BR_Z[mode]
            * bp.BR_H_TAUTAU * bp.BR_TAU_PINU**2)


# ---------------------------------------------------------------------------
def cmd_process(mode, sample):
    path, seed = SAMPLES[(mode, sample)]
    a, meta = load_sample(path)
    t0 = time.time()
    out = an.process(a, meta, seed)
    eg = np.zeros(len(a["pi_m"]))
    np.add.at(eg, a["photon_event"], a["photon_p4"][:, 0])
    flat = {"v_true": out["v_true"], "photon_e_sum": eg}
    for ver in ("sm", "u"):
        for k, v in out[ver].items():
            if k == "steps":
                for sk, sv in v.items():
                    flat[f"{ver}__step__{sk}"] = sv
            else:
                flat[f"{ver}__{k}"] = v
    os.makedirs(RES, exist_ok=True)
    np.savez(f"{RES}/processed_{mode}_{sample}.npz", **flat)
    print(f"processed {mode} {sample}: {len(a['pi_m'])} events in {time.time() - t0:.0f} s")


def load_processed(mode, sample):
    with np.load(f"{RES}/processed_{mode}_{sample}.npz") as f:
        flat = {k: f[k] for k in f.files}
    pool = {"v_true": flat["v_true"], "photon_e_sum": flat["photon_e_sum"]}
    for ver in ("sm", "u"):
        d = {"steps": {}}
        for k, v in flat.items():
            if k.startswith(f"{ver}__step__"):
                d["steps"][k.split("__")[-1]] = v
            elif k.startswith(f"{ver}__"):
                d[k.split("__", 1)[1]] = v
        pool[ver] = d
    return pool


def apply_selection(pool, ip_cut):
    for ver in ("sm", "u"):
        d = pool[ver]
        pre = np.ones(len(pool["v_true"]), bool)
        for v in d["steps"].values():
            pre &= v
        d["presel"] = pre
        d["sel"] = pre & (d["ip_min"] > ip_cut)
    return pool


# ---------------------------------------------------------------------------
def cmd_closure():
    """WP2 gate: no smearing, events without radiated photons above 1 GeV."""
    from v2 import detector, reco
    path, seed = SAMPLES[("mumu", "A")]
    a, meta = load_sample(path)
    n = 50000
    sub = {k: (v[:n] if v.shape[:1] == (len(a["pi_m"]),) else v) for k, v in a.items()}
    keep = a["photon_event"] < n
    for k in ("photon_p4", "photon_tag", "photon_event"):
        sub[k] = a[k][keep]
    eg = np.zeros(n)
    np.add.at(eg, sub["photon_event"], sub["photon_p4"][:, 0])
    clean = eg < 1.0
    meas = detector.measure(sub, np.random.default_rng(seed), meta["sqrts"], "mumu", smear=False)
    r = an.chunked_reco(meas)
    ang = np.arccos(np.clip(kin.dot3(kin.unit(kin.p3(r["tau_m"])),
                                     kin.unit(kin.p3(sub["tau_m"]))), -1, 1))
    res = {
        "n_events": n, "fraction_clean": float(clean.mean()),
        "tau_dir_err_median_mrad_clean": float(np.median(ang[clean]) * 1e3),
        "tau_dir_err_median_mrad_all": float(np.median(ang) * 1e3),
        "nu_residual_median_gev_clean": float(np.median(r["nu_residual"][clean])),
        "note": ("nu_residual = |Σp_ν(fit) - (p_H - p_π- - p_π+)|; it vanishes by "
                 "construction when τ- + τ+ = p_H, which the Higgs-frame "
                 "parametrisation enforces."),
    }
    res["gate_direction"] = bool(res["tau_dir_err_median_mrad_clean"] < 0.2)
    res["gate_nu_residual"] = bool(res["nu_residual_median_gev_clean"] < 0.5)
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/closure.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------
def pct(x, ps=(2.5, 16, 50, 84, 97.5)):
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, ps)] if len(x) else [float("nan")] * len(ps)


def asymptotic_z(pool, n_exp, v_psi, lam, w_row, kappa, D):
    """Proxy significance used only to choose the IP cut: compound-Poisson
    mean and variance of the summed statistic under SM and H."""
    m = {}
    for hyp in ("sm", "h"):
        ts = []
        for ver in ("sm", "u"):
            p = pool[ver]
            lnr = an.log_ratio_u_over_sm(p["q_m"], p["q_p"], kappa, D)
            w = an.response_weight(w_row, p["v_reco"])
            ts.append(np.where(p["sel"], an.per_event_t(lnr, w, lam), 0.0))
        if hyp == "sm":
            t = ts[0]
            e1, e2 = t.mean(), (t**2).mean()
        else:
            above = pool["v_true"] > v_psi
            pu = np.where(above, lam, 0.0)
            e1 = ((1 - pu) * ts[0] + pu * ts[1]).mean()
            e2 = ((1 - pu) * ts[0]**2 + pu * ts[1]**2).mean()
        m[hyp] = (n_exp * e1, np.sqrt(n_exp * e2))
    return (m["sm"][0] - m["h"][0]) / m["h"][1]


def build_model(pool, v_grid):
    """Density ratio and response from a calibration pool (sample A)."""
    q_sm = (pool["sm"]["q_m"][pool["sm"]["sel"]], pool["sm"]["q_p"][pool["sm"]["sel"]])
    q_u = (pool["u"]["q_m"][pool["u"]["sel"]], pool["u"]["q_p"][pool["u"]["sel"]])
    kappa, D = an.fit_density_ratio(q_sm, q_u)
    sel = pool["sm"]["sel"]
    w = an.response(pool["sm"]["v_reco"][sel], pool["v_true"][sel], v_grid)
    return kappa, D, w


def subset_pool(pool, n):
    out = {"v_true": pool["v_true"][:n]}
    for ver in ("sm", "u"):
        out[ver] = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in pool[ver].items()
                    if k != "steps"}
    return out


def cmd_analyse(n_toys):
    t_start = time.time()
    modes = ["mumu", "ee", "qq"]
    A = {m: load_processed(m, "A") for m in modes}
    B = {m: load_processed(m, "B") for m in modes}
    results = {"parameters": {
        "sqrts": bp.SQRTS, "lumi_ab": bp.INTEGRATED_LUMI_AB, "sigma_zh_fb": bp.SIGMA_ZH_FB,
        "br_h_tautau": bp.BR_H_TAUTAU, "br_z": bp.BR_Z, "br_tau_pinu": bp.BR_TAU_PINU,
        "beam_energy_spread": bp.BEAM_ENERGY_SPREAD, "crossing_angle": bp.CROSSING_ANGLE,
        "luminous_region_mm": bp.LUMINOUS_REGION_MM, "detector": {
            k: v for k, v in bp.DETECTOR.items()}, "selection": dict(an.SELECTION),
        "v_psi_grid": V_PSI_GRID.tolist(), "lambdas": LAMBDAS, "n_toys": n_toys}}

    # ---- vertex-quality cut: optimised once on the SM μμ calibration sample
    cuts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    v_opt, lam_opt = 5.0, LAMBDAS["no_correlation"]
    scan = []
    for c in cuts:
        apply_selection(A["mumu"], c)
        kappa, D, w = build_model(A["mumu"], [v_opt])
        z = asymptotic_z(A["mumu"], n_expected_generated("mumu"), v_opt, lam_opt, w[0], kappa, D)
        scan.append({"cut": c, "z_proxy": float(z),
                     "efficiency": float(A["mumu"]["sm"]["sel"].mean())})
    best = max(scan, key=lambda s: s["z_proxy"])
    ip_cut = best["cut"]
    an.SELECTION["ip_sig_min"] = ip_cut
    results["ip_cut_scan"] = {"criterion": ("asymptotic Z for H(v_ψ = 5c, no correlation), "
                                            "Z→μμ calibration sample A"),
                              "scan": scan, "chosen": ip_cut}
    results["parameters"]["selection"]["ip_sig_min"] = ip_cut
    for m in modes:
        apply_selection(A[m], ip_cut)
        apply_selection(B[m], ip_cut)

    # ---- yields and cut flow (SM version, calibration samples)
    yields = {}
    for m in modes:
        p = A[m]["sm"]
        n = len(A[m]["v_true"])
        flow = {}
        mask = np.ones(n, bool)
        for step in ("acceptance", "pi_nu_id", "recoil_mass", "reco_ok"):
            mask &= p["steps"][step]
            flow[step] = float(mask.mean())
        mask &= p["ip_min"] > ip_cut
        flow["vertex_quality"] = float(mask.mean())
        ng = n_expected_generated(m)
        yields[m] = {"br_z": bp.BR_Z[m], "events_per_ab_generated": ng / bp.INTEGRATED_LUMI_AB,
                     "events_generated_at_lumi": ng, "cumulative_efficiency": flow,
                     "events_selected_at_lumi": ng * flow["vertex_quality"]}
    lumi_fb, lumi_ab = __import__("config").lumi_from_events(
        1.0, bp.SIGMA_ZH_FB * bp.BR_Z["mumu"] * bp.BR_H_TAUTAU * bp.BR_TAU_PINU**2)
    results["yields"] = yields
    results["yields_check_lumi_per_mumu_event_ab"] = lumi_ab

    # ---- resolution and migration (SM μμ, selected)
    p = A["mumu"]["sm"]
    s = p["sel"]
    lr = np.log(p["v_reco"][s] / A["mumu"]["v_true"][s])
    results["resolution_mumu"] = {
        "percentiles": [2.5, 16, 50, 84, 97.5],
        "tau_dir_err_mrad": pct(p["tau_dir_err"][s] * 1e3),
        "L_reco_over_true": pct(p["L_ratio"][s]),
        "ln_vreco_over_vtrue": pct(lr),
    }
    mig = []
    vt, vr = A["mumu"]["v_true"][s], p["v_reco"][s]
    for V in (2, 5, 9, 16, 30, 50):
        sel = vr > V
        mig.append({"v": V, "n_selected_fraction": float(sel.mean()),
                    "frac_truly_below": float(np.mean(vt[sel] < V)) if sel.any() else None,
                    "efficiency_for_true_above": float(np.sum(sel & (vt > V)) / max(np.sum(vt > V), 1))})
    results["migration_mumu"] = mig
    # same for the hadronic projection, whose recoil is worse
    pq = A["qq"]["sm"]
    sq = pq["sel"]
    results["resolution_qq"] = {
        "tau_dir_err_mrad": pct(pq["tau_dir_err"][sq] * 1e3),
        "ln_vreco_over_vtrue": pct(np.log(pq["v_reco"][sq] / A["qq"]["v_true"][sq]))}

    # ---- reco-level spin observables after selection (illustration)
    spin = {}
    for m in modes:
        p = A[m]["sm"]
        s = p["sel"]
        C, Ce = kin.c_matrix(p["q_m"][s], p["q_p"][s])
        B_, Be = kin.acoplanarity_b(p["dphi"][s])
        Bt, Bte = kin.acoplanarity_b(p["dphi_true"][s])
        Ball, Balle = kin.acoplanarity_b(p["dphi_true"])
        Cu, _ = kin.c_matrix(A[m]["u"]["q_m"][A[m]["u"]["sel"]], A[m]["u"]["q_p"][A[m]["u"]["sel"]])
        spin[m] = {"C_reco_selected": C.tolist(), "C_err": Ce.tolist(),
                   "B_reco_selected": [B_, Be], "B_truth_selected": [Bt, Bte],
                   "B_truth_all": [Ball, Balle], "C_reco_selected_uncorrelated": Cu.tolist()}
    results["spin_after_selection"] = spin

    # ---- statistical model per mode (calibration sample A)
    models = {}
    for m in modes:
        kappa, D, w = build_model(A[m], V_PSI_GRID)
        models[m] = (kappa, D, w)
        results.setdefault("density_ratio", {})[m] = {"kappa": kappa, "D": D.reshape(3, 3).tolist()}

    # ---- gate: Z = √N/2 at truth level, no detector, true v_min, λ = 1
    pz = A["mumu"]
    R = np.diag([1.0, 1.0, -1.0]).ravel()
    q_sm_t = (pz["sm"]["q_m_true"], pz["sm"]["q_p_true"])
    q_u_t = (pz["u"]["q_m_true"], pz["u"]["q_p_true"])
    k_t, D_t = an.fit_density_ratio(q_sm_t, q_u_t)
    gate_z = []
    rng = np.random.default_rng(5)
    for N in (100, 1000):
        for label, (kap, DD) in (("exact_R", (1.0, R)), ("fitted", (k_t, D_t))):
            t_sm = -2 * an.log_ratio_u_over_sm(*q_sm_t, kap, DD, floor=1e-12)
            t_u = -2 * an.log_ratio_u_over_sm(*q_u_t, kap, DD, floor=1e-12)
            ts = np.array([t_sm[rng.integers(0, len(t_sm), N)].sum() for _ in range(4000)])
            tu = np.array([t_u[rng.integers(0, len(t_u), N)].sum() for _ in range(4000)])
            z = (np.median(ts) - np.median(tu)) / tu.std()
            gate_z.append({"N": N, "model": label, "Z": float(z), "sqrtN_over_2": np.sqrt(N) / 2})
    results["gate_truth_Z"] = {"rows": gate_z, "D_fitted_truth": D_t.reshape(3, 3).tolist(),
                               "pass": bool(all(abs(r["Z"] / r["sqrtN_over_2"] - 1) < 0.05
                                                for r in gate_z))}

    # ---- limits
    limits = {}
    for scen, smodes in SCENARIOS.items():
        limits[scen] = {}
        for alt, lam in LAMBDAS.items():
            cls_rows = []
            for k, v in enumerate(V_PSI_GRID):
                t_sm = np.zeros(n_toys)
                t_h = np.zeros(n_toys)
                for m in smodes:
                    kappa, D, w = models[m]
                    ne = n_expected_generated(m)
                    seed = 1000 * k + 17 * len(m)
                    t_sm += an.toys(A[m], ne, v, lam, w[k], kappa, D, n_toys,
                                    np.random.default_rng(seed), "sm")
                    t_h += an.toys(A[m], ne, v, lam, w[k], kappa, D, n_toys,
                                   np.random.default_rng(seed + 1), "h")
                cls_rows.append(an.cls_expected(t_sm, t_h))
            reach = {q: an.crossing(V_PSI_GRID, [r[q] for r in cls_rows])
                     for q in an.QUANTILES}
            limits[scen][alt] = {"cls": cls_rows, "v95": reach}
            print(f"{scen:8s} {alt:15s} v95 median {reach['p50']:.2f}c "
                  f"[{reach['p15.9']:.2f}, {reach['p84.1']:.2f}] "
                  f"({time.time() - t_start:.0f} s)", flush=True)
    results["limits"] = limits

    # ---- gate: p-values uniform under each null, with toys from the
    # independent pseudo-data samples B and distributions from samples A
    uni = []
    for scen in ("ll",):
        smodes = SCENARIOS[scen]
        for alt, lam in LAMBDAS.items():
            for v in (2.0, 10.0):
                k = int(np.argmin(np.abs(V_PSI_GRID - v)))
                vv = V_PSI_GRID[k]
                dist = {"sm": np.zeros(n_toys), "h": np.zeros(n_toys)}
                test = {"sm": np.zeros(n_toys), "h": np.zeros(n_toys)}
                for m in smodes:
                    kappa, D, w = models[m]
                    ne = n_expected_generated(m)
                    for hyp in ("sm", "h"):
                        dist[hyp] += an.toys(A[m], ne, vv, lam, w[k], kappa, D, n_toys,
                                             np.random.default_rng(7 + k), hyp)
                        test[hyp] += an.toys(B[m], ne, vv, lam, w[k], kappa, D, n_toys,
                                             np.random.default_rng(99 + k), hyp)
                # test of H: reject when t is SM-like, p_H = P_H(t >= t_obs)
                th = np.sort(dist["h"])
                p_h = 1 - np.searchsorted(th, test["h"], side="left") / len(th)
                # test of SM: reject when t is H-like, p_SM = P_SM(t <= t_obs)
                tsm = np.sort(dist["sm"])
                p_sm = np.searchsorted(tsm, test["sm"], side="right") / len(tsm)
                for null, pv in (("H", p_h), ("SM", p_sm)):
                    rate = float(np.mean(pv < 0.05))
                    err = np.sqrt(0.05 * 0.95 / n_toys)
                    uni.append({"scenario": scen, "alternative": alt, "v_psi": float(vv),
                                "null": null, "rejection_rate_at_5pct": rate,
                                "binomial_err": float(err), "pull": float((rate - 0.05) / err)})
    results["gate_null_uniform"] = {"rows": uni,
                                    "pass": bool(all(abs(r["pull"]) < 3 for r in uni))}

    results["wall_time_s"] = round(time.time() - t_start)
    with open(f"{RES}/v2_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(json.dumps({"gate_truth_Z": results["gate_truth_Z"]["pass"],
                      "gate_null_uniform": results["gate_null_uniform"]["pass"],
                      "ip_cut": ip_cut}, indent=1))


# ---------------------------------------------------------------------------
def cmd_whizard():
    """WP1: Pythia at 250 GeV vs the bundled WHIZARD sample (generator level)."""
    from scipy.stats import ks_2samp
    import io
    import contextlib
    import config
    from parse_hepmc import parse_events
    config.set_sqrts(250.0)
    with contextlib.redirect_stdout(io.StringIO()):
        evs = parse_events("EventSample1002.hepmc", allowed_modes=["pi_nu"])
    e_pi_w = np.concatenate([[e.tau_minus.charged_pion_p4[0], e.tau_plus.charged_pion_p4[0]]
                             for e in evs])
    rec_w = np.array([kin.mass(np.array([250.0, 0, 0, 0]) - e.mu_plus_p4 - e.mu_minus_p4)
                      for e in evs])
    a, meta = load_sample("samples/zh250_mumu_check")
    e_pi_p = np.concatenate([a["pi_m"][:, 0], a["pi_p"][:, 0]])
    rec_p = kin.mass(np.array([250.0, 0, 0, 0]) - a["lep_m"] - a["lep_p"])
    res = {"n_whizard_events": len(evs), "n_pythia_events": len(a["pi_m"])}
    for name, w, p in (("pion_energy", e_pi_w, e_pi_p), ("recoil_mass", rec_w, rec_p)):
        ks = ks_2samp(w, p)
        res[name] = {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue),
                     "whizard_percentiles_10_50_90": pct(w, (10, 50, 90)),
                     "pythia_percentiles_10_50_90": pct(p, (10, 50, 90))}
    res["note"] = ("pion energies: 2 per event, so the KS p-value is approximate; "
                   "recoil mass from the generator-level muons")
    res["gate"] = bool(min(res["pion_energy"]["p_value"], res["recoil_mass"]["p_value"]) > 0.01)

    # Diagnosis of a recoil-mass disagreement: radiated energy fraction
    # ε = (√s - E_ll - E_H)/√s compared with the leading-log ISR radiator
    # β ε^(β-1) convolved with the Born σ(ZH)(ŝ) (which vanishes at threshold).
    import pyhepmc
    sq = 250.0
    eps_p = (sq - a["lep_m"][:, 0] - a["lep_p"][:, 0] - a["higgs"][:, 0]) / sq
    eps_w = []
    with pyhepmc.open("EventSample1002.hepmc") as f:
        for ev in f:
            ps = list(ev.particles)
            mu = [p for p in ps if abs(p.pid) == 13 and p.status == 1][:2]
            ta = [p for p in ps if abs(p.pid) == 15][:2]
            eps_w.append((sq - sum(p.momentum.e for p in mu + ta)) / sq)
    eps_w = np.array(eps_w)
    beta = 2 / 137.036 / np.pi * (np.log(sq**2 / 0.000511**2) - 1)
    mz, mh = 91.1876, 125.0

    def born(shat):
        lam = (1 - (mh + mz)**2 / shat) * (1 - (mh - mz)**2 / shat)
        lam = np.where(lam > 0, lam, 0.0)
        return np.sqrt(lam) * (lam + 12 * mz**2 / shat) / (shat * (1 - mz**2 / shat)**2)
    e = np.logspace(-9, np.log10(1 - (mh + mz)**2 / sq**2), 200000)
    wgt = beta * e**(beta - 1) * born(sq**2 * (1 - e))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (wgt[1:] + wgt[:-1]) * np.diff(e))])
    cdf /= cdf[-1]
    rows = []
    for e0 in (0.004, 0.02, 0.05):
        rows.append({"eps": e0, "LL_x_Born": float(1 - np.interp(e0, e, cdf)),
                     "pythia_incl_FSR": float(np.mean(eps_p > e0)),
                     "whizard_incl_FSR": float(np.mean(eps_w > e0))})
    res["radiation_diagnosis"] = {
        "beta_LL": beta, "rows": rows,
        "note": ("P(ε > ε0) for the radiated energy fraction.  'LL_x_Born' is ISR only; "
                 "the generator columns also contain final-state radiation off the "
                 "leptons (and, for Pythia, the τ leptons).  The WHIZARD file carries "
                 "no generator metadata, so whether it includes an ILC beam spectrum "
                 "(beamstrahlung) cannot be checked from the file."),
    }
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/whizard_check.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["process", "closure", "analyse", "whizard"])
    ap.add_argument("--mode")
    ap.add_argument("--sample")
    ap.add_argument("--toys", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "process":
        cmd_process(a.mode, a.sample)
    elif a.cmd == "closure":
        cmd_closure()
    elif a.cmd == "analyse":
        cmd_analyse(a.toys)
    else:
        cmd_whizard()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Work package 1 gates: truth-level checks of a generated sample.

Usage: python -m v2.truth_gates samples/zh240_mumu_A [more samples ...]
Writes results/truth_gates.json.
"""
import json
import os
import sys

import numpy as np

import beam_parameters as bp
from generate_sample import load_sample
from v2 import kin

B_SM = -np.pi**2 / 16


def nominal_beam_minus(sqrts):
    e = sqrts / 2
    th = bp.CROSSING_ANGLE / 2
    return np.array([e, e * np.sin(th), 0.0, e * np.cos(th)])


def truth_observables(a, sqrts, uncorrelated_perm=None):
    """Truth spin angles in the true Higgs frame and the true v_min."""
    ph = a["higgs"]
    bh = kin.beta_of(ph)
    tm_h = kin.boost_to_rest(a["tau_m"], bh)
    tp_h = kin.boost_to_rest(a["tau_p"], bh)
    beam = np.broadcast_to(nominal_beam_minus(sqrts), ph.shape)
    if a["beams"][:, 1].std() == 0 and abs(a["beams"][0, 1]) < 1e-9:
        e = sqrts / 2
        beam = np.broadcast_to(np.array([e, 0, 0, e]), ph.shape)
    k = kin.unit(kin.p3(tm_h))
    qm, qp, dphi = kin.spin_angles(a["pi_m"], a["pi_p"], ph, k, beam,
                                   tau_p_hat_h=kin.unit(kin.p3(tp_h)),
                                   tau_h=(tm_h, tp_h))
    return qm, qp, dphi


def gates(path):
    a, meta = load_sample(path)
    sqrts = meta["sqrts"]
    n = len(a["pi_m"])
    qm, qp, dphi = truth_observables(a, sqrts)
    C, Ce = kin.c_matrix(qm, qp)
    B, Be = kin.acoplanarity_b(dphi)
    target = np.diag([1.0, 1.0, -1.0])
    pull_C = (C - target) / Ce

    # space-time: lab frame and Higgs frame
    v_lab = kin.vmin(a["dv_m"], a["dv_p"])
    bh = kin.beta_of(a["higgs"])
    c = kin.C_LIGHT
    def to_h(dv):
        x = np.concatenate([(c * (dv[:, 0] - a["pv"][:, 0]))[:, None],
                            dv[:, 1:] - a["pv"][:, 1:]], axis=1)
        return kin.boost_to_rest(x, bh)
    xm, xp = to_h(a["dv_m"]), to_h(a["dv_p"])
    v_h = kin.mag(xp[:, 1:] - xm[:, 1:]) / np.abs(xp[:, 0] - xm[:, 0])
    tm_h = kin.boost_to_rest(a["tau_m"], bh)
    beta_tau = np.median(kin.mag(kin.p3(tm_h)) / tm_h[:, 0])
    tl = float(np.mean(v_lab < 1))
    eps = (1 - beta_tau) / (1 + beta_tau)
    tl_exp = 2 * eps / (1 + eps)

    tail = {}
    for V in (2, 5, 10, 20, 50, 100):
        f = float(np.mean(v_h > V))
        exp = beta_tau / V
        tail[str(V)] = {"measured": f, "expected": float(exp),
                        "pull": float(f - exp) / np.sqrt(exp * (1 - exp) / n)}

    ct = np.concatenate([a["ptime_m"], a["ptime_p"]]) * c * 1e6     # μm
    ct_mean, ct_err = ct.mean(), ct.std() / np.sqrt(len(ct))

    sq = float(np.median(a["beams"][:, 0]))

    res = {
        "sample": os.path.basename(path), "n_events": int(n), "sqrts": sqrts,
        "B": [float(B), float(Be)], "B_SM": B_SM, "B_pull": float((B - B_SM) / Be),
        "C": C.tolist(), "C_err": Ce.tolist(), "C_max_abs_pull": float(np.abs(pull_C).max()),
        "timelike_fraction": tl, "timelike_expected": float(tl_exp),
        "timelike_pull": float(tl - tl_exp) / np.sqrt(tl_exp / n),
        "P_vmin_higgs_frame": tail, "beta_tau": float(beta_tau),
        "ctau_um": [float(ct_mean), float(ct_err)], "ctau_pdg_um": 87.03,
        "ctau_pull": float((ct_mean - 87.03) / ct_err),
        "sqrts_from_beams": sq,
    }
    tail_pulls = [abs(t["pull"]) for t in tail.values()]
    # The P(v > V) checks are correlated (same events); require each within 3σ.
    res["gates"] = {
        "B": bool(abs(res["B_pull"]) < 3),
        "C_diag": bool(res["C_max_abs_pull"] < 3.5),    # 9 elements: allow the max-of-9
        "timelike": bool(abs(res["timelike_pull"]) < 3),
        "P_vmin_beta_over_V": bool(max(tail_pulls) < 3),
        "ctau": bool(abs(res["ctau_pull"]) < 3),
        "check_sqrts": bool(abs(sq - sqrts) < 0.5),
    }
    return res


def main():
    out = [gates(p) for p in sys.argv[1:]]
    os.makedirs("results", exist_ok=True)
    with open("results/truth_gates.json", "w") as f:
        json.dump(out, f, indent=1)
    for r in out:
        print(r["sample"], r["n_events"], json.dumps(r["gates"]))
        print("  B = %.4f ± %.4f (SM %.4f)  C diag = %s  max|pull| %.2f" % (
            r["B"][0], r["B"][1], r["B_SM"], np.round(np.diag(r["C"]), 4),
            r["C_max_abs_pull"]))
        print("  timelike %.2e (exp %.2e)  cτ = %.2f ± %.2f μm" % (
            r["timelike_fraction"], r["timelike_expected"], *r["ctau_um"]))
        print("  P(v>V) H frame:", {k: round(v["pull"], 2) for k, v in r["P_vmin_higgs_frame"].items()})


if __name__ == "__main__":
    main()

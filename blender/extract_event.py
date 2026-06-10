"""
Extract a single, clean real Monte-Carlo event from the HepMC sample and dump
everything the Blender event-display needs into a JSON file.

We re-use the analysis's own parser (parse_hepmc) and Jeans reconstruction
(tau_reconstruction.reconstruct_event) so that the diagram shows the *actual*
numbers produced by the analysis pipeline, not a hand-made cartoon.

The chosen event is the cleanest pi-nu x pi-nu topology (tau -> pi nu for both
taus), which is the channel the reconstruction section of the paper is built
around.  For that event we record, in the lab frame and in the Higgs rest
frame:

  * the e+e- beam axis
  * the Z -> mu+ mu- decay (the "tag" that fixes the Higgs 4-momentum)
  * the reconstructed Higgs 4-momentum  p_H = p_beam - p_Z
  * both taus: truth momentum, charged-pion momentum, neutrino momentum,
    production vertex (the Higgs decay point) and decay vertex
  * the Jeans reconstruction geometry: impact-parameter vector, track plane,
    opening angle alpha, decay length L = |d| / sin(alpha), reconstructed
    decay vertex, and the missing-momentum constraint.

All distances are converted to micrometres (um) for the spatial part because
the tau flight lengths are O(10-1000 um) -- that is the natural scale of the
"where did each tau decay" story.
"""
import sys
import os
import json
import numpy as np

# Make the analysis package importable when run from blender/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parse_hepmc import parse_events
from tau_reconstruction import (
    reconstruct_event, boost, beta_vec, p3vec, p3mag, mass,
    compute_impact_parameter, p3hat,
)
from config import P_BEAM_TOTAL, M_HIGGS, C_LIGHT, M_TAU


def v(a):
    """numpy -> plain python list (JSON serialisable)."""
    return [float(x) for x in np.asarray(a).ravel()]


def pick_event(hepmc_path, max_events=400):
    """Return the (event, reco) pair that best illustrates the method.

    We want both tau decay vertices to be well separated from the production
    vertex (large impact parameter / decay length) so the geometry is visible,
    and a successful, low-chi2 reconstruction.
    """
    events = parse_events(hepmc_path, max_events=max_events,
                          allowed_modes=["pi_nu"])
    best = None
    best_score = -1.0
    for ev in events:
        reco = reconstruct_event(ev)
        if reco is None:
            continue
        tm, tp = reco["tau_minus"], reco["tau_plus"]
        Lm, Lp = tm["decay_length_m"], tp["decay_length_m"]
        dm, dp = tm["d_mag"], tp["d_mag"]
        if Lm <= 0 or Lp <= 0:
            continue

        # Both impact parameters must be resolvable so the track-plane geometry
        # is genuinely visible (not a degenerate near-collinear case).
        if dm < 20e-6 or dp < 20e-6:        # 20 um floor
            continue

        # Reconstructed decay length should track the truth for both taus so
        # the "the constraints pin down where it decayed" message is honest.
        def agree(reco_L, truth_L):
            return min(reco_L, truth_L) / max(reco_L, truth_L)
        am = agree(Lm, tm["truth_decay_length_m"])
        ap = agree(Lp, tp["truth_decay_length_m"])
        if am < 0.6 or ap < 0.6:
            continue

        chi2 = reco["chi2"]
        # Reward: both fly a visible distance, both |d| sizeable, good
        # reco<->truth agreement, low fit chi2.
        score = ((Lm * Lp) ** 0.5) * ((dm * dp) ** 0.5) * am * ap / (1.0 + chi2)
        if score > best_score:
            best_score = score
            best = (ev, reco)
    return best


def build_payload(ev, reco):
    UM = 1e6  # metres -> micrometres

    p_Z = reco["p_Z"]
    p_H = reco["p_H"]
    p_miss = reco["p_miss"]

    # Boost velocity to go INTO the Higgs rest frame.
    beta_H = beta_vec(p_H)

    pv_m = ev.tau_minus.production_vertex[1:4]      # Higgs decay point (m)

    payload = {
        "event_number": ev.event_number,
        "constants": {
            "m_higgs_gev": float(M_HIGGS),
            "m_tau_gev": float(M_TAU),
            "c_light": float(C_LIGHT),
            "len_unit": "micrometre",
            "mom_unit": "GeV",
        },
        "lab": {
            "beam_total_p4": v(P_BEAM_TOTAL),
            "beam_axis": v([0, 0, 1]),
            "production_vertex_um": v(pv_m * UM),
            "p_Z": v(p_Z),
            "p_H": v(p_H),
            "p_miss": v(p_miss),
            "beta_higgs": v(beta_H),
            "mu_plus_p4": v(ev.mu_plus_p4),
            "mu_minus_p4": v(ev.mu_minus_p4),
            "higgs_mass_check": float(mass(p_H)),
            "z_mass_check": float(mass(p_Z)),
            "chi2": float(reco["chi2"]),
        },
        "taus": {},
    }

    for label, sign in [("tau_minus", -1), ("tau_plus", +1)]:
        tinfo = getattr(ev, label)
        r = reco[label]

        pv = tinfo.production_vertex[1:4]
        dv_truth = tinfo.decay_vertex[1:4]
        dv_reco = r["decay_vertex_xyz"]

        p_pi = tinfo.charged_pion_p4
        pi_hat = p3hat(p_pi)
        d_vec, d_mag, pca = compute_impact_parameter(pv, dv_truth, pi_hat)

        # Rest-frame quantities: boost the pion and tau into the Higgs frame.
        p_pi_rest = boost(p_pi, beta_H)
        p_tau_truth_rest = boost(tinfo.tau_p4, beta_H)
        p_tau_reco_rest = boost(r["p_tau_reco"], beta_H)

        payload["taus"][label] = {
            "pdgid": int(tinfo.tau_pdgid),
            "charge_sign": sign,
            # --- lab-frame truth ---
            "tau_p4_truth": v(tinfo.tau_p4),
            "pion_p4": v(p_pi),
            "neutrino_p4_truth": v(tinfo.neutrino_p4),
            "production_vertex_um": v(pv * UM),
            "decay_vertex_truth_um": v(dv_truth * UM),
            "decay_length_truth_um": float(np.linalg.norm(dv_truth - pv) * UM),
            # --- Jeans reconstruction ---
            "tau_p4_reco": v(r["p_tau_reco"]),
            "neutrino_p4_reco": v(r["p_nu_reco"]),
            "tau_dir": v(r["tau_dir"]),
            "alpha_rad": float(r["alpha"]),
            "decay_length_reco_um": float(r["decay_length_m"] * UM),
            "decay_vertex_reco_um": v(dv_reco * UM),
            "beta": float(r["beta"]),
            "gamma": float(r["gamma"]),
            # --- impact-parameter geometry (the "track plane") ---
            "pion_dir": v(pi_hat),
            "impact_param_vec_um": v(d_vec * UM),
            "impact_param_mag_um": float(d_mag * UM),
            "pca_point_um": v(pca * UM),
            "ip_significance": float(r["ip_significance"]),
            # --- Higgs rest frame ---
            "rest_frame": {
                "tau_p4_truth": v(p_tau_truth_rest),
                "tau_p4_reco": v(p_tau_reco_rest),
                "pion_p4": v(p_pi_rest),
            },
        }

    return payload


def main():
    here = os.path.dirname(__file__)
    # Prefer the larger sample (more pi-nu events to choose from).
    candidates = [
        os.path.join(here, "..", "EventSample1002.hepmc"),
        os.path.join(here, "..", "EventSample100.hepmc"),
    ]
    chosen = None
    for c in candidates:
        if os.path.exists(c):
            print(f"Scanning {c} ...")
            res = pick_event(c, max_events=600)
            if res is not None:
                chosen = res
                break
    if chosen is None:
        raise SystemExit("No suitable pi-nu x pi-nu event found.")

    ev, reco = chosen
    payload = build_payload(ev, reco)

    out = os.path.join(here, "data", "event.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSelected event #{payload['event_number']}")
    print(f"  Higgs mass check : {payload['lab']['higgs_mass_check']:.3f} GeV")
    print(f"  Z mass check     : {payload['lab']['z_mass_check']:.3f} GeV")
    print(f"  fit chi2         : {payload['lab']['chi2']:.3e}")
    for lbl in ("tau_minus", "tau_plus"):
        t = payload["taus"][lbl]
        print(f"  {lbl}: L_truth={t['decay_length_truth_um']:.1f} um  "
              f"L_reco={t['decay_length_reco_um']:.1f} um  "
              f"|d|={t['impact_param_mag_um']:.2f} um  "
              f"alpha={t['alpha_rad']*1e3:.2f} mrad")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

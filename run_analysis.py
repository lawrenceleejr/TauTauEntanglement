#!/usr/bin/env python3
"""
Main driver for the tau-tau entanglement analysis.

Usage:
    python run_analysis.py <path_to_hepmc_file> [--max-events N]

Analyses e+e- -> ZH -> mu+mu- tau+tau- events at sqrt(s) = 240 GeV.
Measures quantum entanglement between the two tau leptons as a function
of the spacetime separation between their decay points.

The primary measurement uses only experimentally observable quantities:
  - Pion momenta (measured tracks)
  - Higgs 4-momentum from beam - Z (measured muons)
  - Reconstructed tau direction (Jeans track geometry)
  - Tau momentum magnitude from Higgs-mass kinematic constraint
  - Reconstructed decay vertices (track geometry)

Truth-level information is used only for validation plots.
"""
import argparse
import sys
import os
import json
import numpy as np

from config import (
    N_BINS_SPACETIME, N_BINS_SIGNAL_SPEED, N_BOOTSTRAP,
    V_PSI_SCAN, OUTPUT_DIR, ALLOWED_DECAY_MODES,
    SIGMA_ZH_FB, BR_H_TAUTAU, BR_Z_MUMU, BR_TAU_PI_NU, BR_TAU_RHO_NU,
)
from parse_hepmc import parse_events
from tau_reconstruction import (
    reconstruct_event,
    boost, beta_vec,
)
from spacetime import compute_truth_intervals, compute_reco_intervals
from spin_analysis import compute_spin_observables, extract_correlation_matrix
from entanglement import (
    bootstrap_entanglement, locality_rejection_sigma,
    entanglement_rejection_sigma, scan_vpsi,
)
from plotting import (
    plot_spacetime_distributions,
    plot_entanglement_vs_spacetime,
    plot_vpsi_overlay,
    plot_vpsi_exclusion,
    plot_correlation_matrix,
    plot_acoplanarity,
    plot_acoplanarity_vs_vsignal,
    plot_vertex_comparison,
    print_summary,
)


def _build_lightlike_binning(signed_ds_arr, n_spacelike_bins):
    """Build bin edges for signed ds with a lightlike boundary at 0.

    Returns edges with the boundary at ds=0 (lightlike), one bin for
    timelike events (ds > 0), and n_spacelike_bins bins for spacelike.
    """
    ds_finite = signed_ds_arr[np.isfinite(signed_ds_arr)]
    spacelike = ds_finite[ds_finite < 0]
    timelike = ds_finite[ds_finite >= 0]

    if len(spacelike) == 0:
        # All timelike — uniform binning
        return np.linspace(0, np.percentile(timelike, 98), n_spacelike_bins + 2)

    # Spacelike bins: equal-width from the 2nd percentile to 0
    sl_lo = np.percentile(spacelike, 2)
    sl_edges = np.linspace(sl_lo, 0, n_spacelike_bins + 1)

    # One timelike bin: 0 to 98th percentile (or reasonable max)
    if len(timelike) > 0:
        tl_hi = np.percentile(timelike, 98) if len(timelike) > 5 else np.max(timelike)
        tl_hi = max(tl_hi, 0.1)  # at least a small window
    else:
        tl_hi = 1.0

    edges = np.concatenate([sl_edges, [tl_hi]])
    return edges


def _bin_entanglement(cos_plus, cos_minus, variable, bin_edges, n_bootstrap):
    """Bin events by a variable and compute entanglement in each bin."""
    n_bins = len(bin_edges) - 1
    results = []
    for b in range(n_bins):
        mask = (variable >= bin_edges[b]) & (variable < bin_edges[b+1])
        n_in_bin = int(np.sum(mask))
        if n_in_bin < 10:
            results.append({'m12': np.nan, 'm12_err': np.nan,
                            'concurrence': np.nan, 'concurrence_err': np.nan,
                            'n_events': n_in_bin})
            continue
        res = bootstrap_entanglement(cos_plus[mask], cos_minus[mask],
                                      n_bootstrap=n_bootstrap)
        res['n_events'] = n_in_bin
        results.append(res)
    return results


def process_events(filepath, max_events=None, smear=False):
    """Run the full analysis pipeline."""

    # ------------------------------------------------------------------
    # Phase 1: Parse HepMC
    # ------------------------------------------------------------------
    print("\n[Phase 1] Parsing HepMC3 file...")
    events = parse_events(filepath, max_events=max_events,
                          allowed_modes=ALLOWED_DECAY_MODES)

    if len(events) == 0:
        print("ERROR: No pi x pi events found. Exiting.")
        sys.exit(1)

    print(f"\n  Selected {len(events)} pi x pi events for analysis.")

    # ------------------------------------------------------------------
    # Phase 1b: Apply detector smearing (if enabled)
    # ------------------------------------------------------------------
    if smear:
        from smearing import smear_event, print_resolution_summary
        print("\n[Phase 1b] Applying ILC detector smearing...")
        print_resolution_summary()
        smear_rng = np.random.default_rng(123)
        events_meas = [smear_event(evt, smear_rng) for evt in events]
        print(f"  Smeared {len(events_meas)} events.")
    else:
        events_meas = events

    # ------------------------------------------------------------------
    # Phase 2: Reconstruct tau kinematics (Jeans method)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Reconstructing tau kinematics (Jeans method)...")
    reco_results = []
    n_reco_fail = 0
    for i, evt_meas in enumerate(events_meas):
        reco = reconstruct_event(evt_meas)
        if reco is None:
            n_reco_fail += 1
        reco_results.append(reco)
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(events)} events...")

    n_good = sum(1 for r in reco_results if r is not None)
    print(f"  Successfully reconstructed: {n_good}/{len(events)} "
          f"({100*n_good/len(events):.1f}%)")
    if n_reco_fail > 0:
        print(f"  Failed: {n_reco_fail}")

    # Filter to successfully reconstructed events
    good_indices = [i for i, r in enumerate(reco_results) if r is not None]
    events_good = [events[i] for i in good_indices]          # truth
    events_meas_good = [events_meas[i] for i in good_indices]  # measured (= truth if no smearing)
    reco_good = [reco_results[i] for i in good_indices]
    N = len(events_good)
    print(f"  Proceeding with {N} events.")

    # ------------------------------------------------------------------
    # Phase 3: Compute spacetime intervals
    # ------------------------------------------------------------------
    print("\n[Phase 3] Computing spacetime intervals...")

    # PRIMARY: Jeans-reconstructed vertices (the experimental observable)
    reco_intervals = [compute_reco_intervals(r) for r in reco_good]
    v_signal_reco = np.array([iv['v_signal_c'] for iv in reco_intervals])
    n_spacelike_reco = sum(1 for iv in reco_intervals if iv['is_spacelike'])
    n_timelike_reco = N - n_spacelike_reco
    print(f"  Reco:  {n_spacelike_reco} spacelike, {n_timelike_reco} timelike")
    print(f"  Reco  v_signal/c: median={np.median(v_signal_reco):.1f}, "
          f"mean={np.mean(np.clip(v_signal_reco, 0, 1e6)):.1f}")

    # VALIDATION ONLY: truth vertices
    truth_intervals = [compute_truth_intervals(evt) for evt in events_good]
    v_signal_truth = np.array([iv['v_signal_c'] for iv in truth_intervals])
    n_spacelike_truth = sum(1 for iv in truth_intervals if iv['is_spacelike'])
    n_timelike_truth = N - n_spacelike_truth
    print(f"  Truth: {n_spacelike_truth} spacelike, {n_timelike_truth} timelike "
          "(validation)")

    # Empirical v_signal resolution (used to smear hypothesis curves)
    sigma_v_frac = 0.0
    if smear:
        _ok = np.isfinite(v_signal_truth) & np.isfinite(v_signal_reco) & (v_signal_truth > 0.1)
        if np.sum(_ok) > 20:
            _frac_resid = (v_signal_reco[_ok] - v_signal_truth[_ok]) / v_signal_truth[_ok]
            # Use IQR-based robust width to ignore outliers
            q75, q25 = np.percentile(_frac_resid, [75, 25])
            sigma_v_frac = (q75 - q25) / 1.349  # IQR / 1.349 ≈ Gaussian sigma
            print(f"  v_signal fractional resolution: {sigma_v_frac:.3f} "
                  f"(IQR-based, {np.sum(_ok)} events)")

    # ------------------------------------------------------------------
    # Phase 4: Compute spin observables
    # ------------------------------------------------------------------
    print("\n[Phase 4] Extracting spin observables...")

    cos_theta_plus_reco = []
    cos_theta_minus_reco = []
    acoplanarity_reco = []

    cos_theta_plus_truth = []
    cos_theta_minus_truth = []
    acoplanarity_truth = []

    for evt_truth, evt_meas, reco in zip(events_good, events_meas_good, reco_good):
        # Primary measurement: reco tau direction + kinematic constraints
        # Uses measured (possibly smeared) pion momenta for spin analysis
        spin_reco = compute_spin_observables(evt_meas, reco, use_reco_tau=True)
        cos_theta_plus_reco.append(spin_reco['cos_theta_plus'])
        cos_theta_minus_reco.append(spin_reco['cos_theta_minus'])
        acoplanarity_reco.append(spin_reco['acoplanarity'])

        # Validation: truth tau momenta and truth pion momenta
        spin_truth = compute_spin_observables(evt_truth, reco, use_reco_tau=False)
        cos_theta_plus_truth.append(spin_truth['cos_theta_plus'])
        cos_theta_minus_truth.append(spin_truth['cos_theta_minus'])
        acoplanarity_truth.append(spin_truth['acoplanarity'])

    cos_theta_plus_reco = np.array(cos_theta_plus_reco)
    cos_theta_minus_reco = np.array(cos_theta_minus_reco)
    cos_theta_plus_truth = np.array(cos_theta_plus_truth)
    cos_theta_minus_truth = np.array(cos_theta_minus_truth)
    acoplanarity_reco = np.array(acoplanarity_reco)
    acoplanarity_truth = np.array(acoplanarity_truth)

    # ------------------------------------------------------------------
    # Phase 5: Global entanglement measurement
    # ------------------------------------------------------------------
    print("\n[Phase 5] Computing entanglement observables...")

    # PRIMARY: reco-based measurement
    print("  Measurement (reco tau momenta + kinematic constraints):")
    global_reco = bootstrap_entanglement(
        cos_theta_plus_reco, cos_theta_minus_reco, n_bootstrap=N_BOOTSTRAP)
    print(f"    m12 = {global_reco['m12']:.4f} +/- {global_reco['m12_err']:.4f}")
    print(f"    Concurrence = {global_reco['concurrence']:.4f} "
          f"+/- {global_reco['concurrence_err']:.4f}")
    print(f"    Bell score = {global_reco['bell_score']:.4f} "
          f"+/- {global_reco['bell_score_err']:.4f}")

    # VALIDATION: truth-based
    print("  Validation (truth tau momenta):")
    global_truth = bootstrap_entanglement(
        cos_theta_plus_truth, cos_theta_minus_truth, n_bootstrap=N_BOOTSTRAP)
    print(f"    m12 = {global_truth['m12']:.4f} +/- {global_truth['m12_err']:.4f}")
    print(f"    Concurrence = {global_truth['concurrence']:.4f} "
          f"+/- {global_truth['concurrence_err']:.4f}")

    # Significance from the RECO measurement
    loc_sigma = locality_rejection_sigma(
        global_reco['m12'], global_reco['m12_err'])
    ent_sigma = entanglement_rejection_sigma(
        global_reco['concurrence'], global_reco['concurrence_err'])

    print(f"\n  Reject locality (m12 <= 1):    {loc_sigma:.1f} sigma")
    print(f"  Reject separability (C <= 0):  {ent_sigma:.1f} sigma")

    # C_ij comparison: reco vs truth
    print("\n  C_ij comparison (reco vs truth):")
    _nrk = ['n', 'r', 'k']
    print(f"    {'':>5s} {'reco':>10s} {'truth':>10s} {'diff':>10s}")
    for i in range(3):
        for j in range(3):
            c_r = global_reco['C'][i, j]
            c_t = global_truth['C'][i, j]
            print(f"    C_{_nrk[i]}{_nrk[j]}: {c_r:+8.4f}  {c_t:+8.4f}  {c_r-c_t:+8.4f}")

    # ------------------------------------------------------------------
    # Phase 6: Entanglement vs spacetime interval (all reco-based)
    # ------------------------------------------------------------------
    print("\n[Phase 6] Binning entanglement vs spacetime variables...")

    signed_ds_reco = np.array([iv['signed_ds_mm'] for iv in reco_intervals])
    v_arr = v_signal_reco

    # --- Binned m12 vs signed ds (with lightlike boundary) ---
    ds_edges = _build_lightlike_binning(signed_ds_reco,
                                         n_spacelike_bins=N_BINS_SPACETIME - 1)
    n_ds_bins = len(ds_edges) - 1
    print(f"  Spacetime binning: {n_ds_bins} bins, "
          f"lightlike boundary at ds=0")

    binned_ds = _bin_entanglement(cos_theta_plus_reco, cos_theta_minus_reco,
                                   signed_ds_reco, ds_edges, N_BOOTSTRAP)

    # --- Binned m12 vs v_signal ---
    v_finite = v_arr[np.isfinite(v_arr)]
    if len(v_finite) > 0 and np.min(v_finite) > 0:
        v_lo = max(np.percentile(v_finite, 2), 0.5)
        v_hi = np.percentile(v_finite, 98)
        v_edges = np.logspace(np.log10(v_lo), np.log10(v_hi),
                               N_BINS_SIGNAL_SPEED + 1)
    else:
        v_edges = np.logspace(0, 3, N_BINS_SIGNAL_SPEED + 1)

    binned_v = _bin_entanglement(cos_theta_plus_reco, cos_theta_minus_reco,
                                  v_arr, v_edges, N_BOOTSTRAP)

    # ------------------------------------------------------------------
    # Phase 7: v_psi hypothesis scan (all reco-based)
    # ------------------------------------------------------------------
    print("\n[Phase 7] Scanning v_psi hypotheses...")
    vpsi_results = scan_vpsi(
        cos_theta_plus_reco, cos_theta_minus_reco, v_arr,
        V_PSI_SCAN, n_bootstrap=N_BOOTSTRAP)

    for r in vpsi_results:
        if r['n_events'] >= 10:
            print(f"  v_psi = {r['v_psi']:6.1f}c : N={r['n_events']:5d}, "
                  f"m12={r['m12']:.3f}+/-{r['m12_err']:.3f}, "
                  f"reject m12=0 at {r['sigma_vs_0']:.1f}sig, "
                  f"reject m12<=1 at {r['sigma_vs_1']:.1f}sig")

    # ------------------------------------------------------------------
    # Luminosity estimation
    # ------------------------------------------------------------------
    # Combined BR for the analysed final state:
    #   sigma(ZH) * BR(H->tautau) * BR(Z->mumu) * BR(tau->X)^2
    # where X is each allowed decay mode.
    _br_tau_mode = {
        "pi_nu":  BR_TAU_PI_NU,
        "rho_nu": BR_TAU_RHO_NU,
    }
    br_tau_sq = sum(_br_tau_mode.get(m, 0) for m in ALLOWED_DECAY_MODES) ** 2
    sigma_eff_fb = SIGMA_ZH_FB * BR_H_TAUTAU * BR_Z_MUMU * br_tau_sq
    # N is the number of fully-reconstructed events analysed
    if sigma_eff_fb > 0:
        int_lumi_fb = N / sigma_eff_fb           # fb^-1
        int_lumi_ab = int_lumi_fb * 1e-6          # ab^-1
    else:
        int_lumi_fb = 0.0
        int_lumi_ab = 0.0

    T_COLLECT = 1e7   # assumed data-taking time [s]
    inst_lumi = int_lumi_fb / T_COLLECT if T_COLLECT > 0 else 0.0  # fb^-1 s^-1
    # Convert to conventional units: cm^-2 s^-1  (1 fb = 1e-39 cm^2)
    inst_lumi_cgs = inst_lumi * 1e-39  # cm^-2 s^-1

    print(f"\n  Luminosity estimate (modes: {ALLOWED_DECAY_MODES}):")
    print(f"    sigma_eff = {sigma_eff_fb:.4f} fb")
    print(f"    N_analysed = {N}")
    print(f"    int. lumi  = {int_lumi_fb:.0f} fb^-1  = {int_lumi_ab:.2f} ab^-1")
    print(f"    inst. lumi = {inst_lumi_cgs:.2e} cm^-2 s^-1  (T = {T_COLLECT:.0e} s)")

    # Write luminosity info to markdown file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lumi_md_path = os.path.join(OUTPUT_DIR, "luminosity.md")
    with open(lumi_md_path, 'w') as f:
        f.write("# Luminosity Estimate\n\n")
        f.write(f"| Quantity | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Events analysed | {N} |\n")
        f.write(f"| Allowed decay modes | {', '.join(ALLOWED_DECAY_MODES)} |\n")
        f.write(f"| Effective cross section | {sigma_eff_fb:.4f} fb |\n")
        f.write(f"| Integrated luminosity | {int_lumi_fb:.0f} fb⁻¹ ({int_lumi_ab:.2f} ab⁻¹) |\n")
        f.write(f"| Data-taking time assumed | {T_COLLECT:.0e} s |\n")
        f.write(f"| Instantaneous luminosity | {inst_lumi_cgs:.2e} cm⁻²s⁻¹ |\n")
        f.write(f"\n## Cross Sections and Branching Ratios\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| σ(e⁺e⁻ → ZH) | {SIGMA_ZH_FB} fb |\n")
        f.write(f"| BR(H → ττ) | {BR_H_TAUTAU} |\n")
        f.write(f"| BR(Z → μμ) | {BR_Z_MUMU} |\n")
        f.write(f"| BR(τ → πν) | {BR_TAU_PI_NU} |\n")
        f.write(f"| BR(τ → ρν) | {BR_TAU_RHO_NU} |\n")
        f.write(f"| Combined BR(τ → X)² | {br_tau_sq:.6f} |\n")
    print(f"  Wrote {lumi_md_path}")

    # ------------------------------------------------------------------
    # Phase 8: Generate plots
    # ------------------------------------------------------------------
    print(f"\n[Phase 8] Generating plots in {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Spacetime distributions (truth + reco overlaid for comparison)
    plot_spacetime_distributions(truth_intervals, reco_intervals)

    # 2. Entanglement vs spacetime interval (lightlike boundary)
    plot_entanglement_vs_spacetime(
        binned_ds, ds_edges,
        xlabel=r"Signed $\sqrt{|\Delta s^2|}$ [mm]",
        suffix="spacetime_interval",
        lightlike_boundary=True)

    # 3. Entanglement vs signal speed
    plot_entanglement_vs_spacetime(
        binned_v, v_edges,
        xlabel=r"$v_{\rm signal} / c$",
        suffix="signal_speed")

    # 4. v_psi overlay plot (the money plot)
    v_psi_overlay = [v for v in V_PSI_SCAN if v <= v_edges[-1] * 1.5]
    if len(v_psi_overlay) > 6:
        v_psi_overlay = v_psi_overlay[:6]
    plot_vpsi_overlay(binned_v, v_edges, v_psi_overlay,
                      sigma_v_frac=sigma_v_frac)

    # 5. v_psi exclusion curve (with 95% CL line)
    plot_vpsi_exclusion(vpsi_results)

    # 6. Correlation matrix heatmaps
    plot_correlation_matrix(global_reco['C'], global_reco['C_err'],
                            suffix="")
    plot_correlation_matrix(global_truth['C'], global_truth['C_err'],
                            suffix="_truth_validation")

    # 7. Acoplanarity with cosine fit
    plot_acoplanarity(acoplanarity_reco, suffix="")
    plot_acoplanarity(acoplanarity_truth, suffix="_truth_validation")

    # 8. Acoplanarity vs signal speed
    plot_acoplanarity_vs_vsignal(acoplanarity_reco, v_arr, v_edges)

    # 9. Vertex comparison (with ratio diagnostic)
    plot_vertex_comparison(reco_good)

    # ------------------------------------------------------------------
    # Save numerical results to JSON
    # ------------------------------------------------------------------
    def _to_json(v):
        """Convert numpy types for JSON serialization."""
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    results_dict = {
        'settings': {
            'smear': smear,
            'hepmc_file': filepath,
            'n_events': N,
            'n_spacelike': n_spacelike_reco,
            'n_timelike': n_timelike_reco,
        },
        'reco': {
            'm12': float(global_reco['m12']),
            'm12_err': float(global_reco['m12_err']),
            'concurrence': float(global_reco['concurrence']),
            'concurrence_err': float(global_reco['concurrence_err']),
            'bell_score': float(global_reco['bell_score']),
            'bell_score_err': float(global_reco['bell_score_err']),
            'C': global_reco['C'].tolist(),
            'C_err': global_reco['C_err'].tolist(),
        },
        'truth': {
            'm12': float(global_truth['m12']),
            'm12_err': float(global_truth['m12_err']),
            'concurrence': float(global_truth['concurrence']),
            'concurrence_err': float(global_truth['concurrence_err']),
            'C': global_truth['C'].tolist(),
            'C_err': global_truth['C_err'].tolist(),
        },
        'vpsi_scan': [
            {k: _to_json(v) for k, v in r.items()}
            for r in vpsi_results
        ],
    }

    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved {json_path}")

    # ------------------------------------------------------------------
    # Summary (all numbers from reco measurement)
    # ------------------------------------------------------------------
    print_summary(global_reco, loc_sigma, ent_sigma,
                  n_spacelike_reco, n_timelike_reco, vpsi_results)

    if smear:
        print("\n  NOTE: Detector smearing was ENABLED for this run.")
        print("  Compare with --no smear to see the unsmeared baseline.")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Tau-tau entanglement analysis at ILC ZH(240)")
    parser.add_argument("hepmc_file", help="Path to HepMC3 file")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of events to process")
    parser.add_argument("--smear", action="store_true",
                        help="Apply ILC/ILD-like detector smearing")
    args = parser.parse_args()

    if not os.path.isfile(args.hepmc_file):
        print(f"ERROR: File not found: {args.hepmc_file}")
        sys.exit(1)

    process_events(args.hepmc_file, max_events=args.max_events,
                   smear=args.smear)


if __name__ == "__main__":
    main()

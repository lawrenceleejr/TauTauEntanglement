#!/usr/bin/env python3
"""
Main driver for the tau-tau entanglement analysis.

Usage:
    python run_analysis.py <path_to_hepmc_file> [--max-events N]

Analyses e+e- -> ZH -> mu+mu- tau+tau- events at sqrt(s) = 240 GeV.
Measures quantum entanglement between the two tau leptons as a function
of the spacetime separation between their decay points.
"""
import argparse
import sys
import os
import numpy as np

from config import (
    N_BINS_SPACETIME, N_BINS_SIGNAL_SPEED, N_BOOTSTRAP,
    V_PSI_SCAN, OUTPUT_DIR,
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
    plot_vertex_comparison,
    print_summary,
)


def process_events(filepath, max_events=None):
    """Run the full analysis pipeline."""

    # ------------------------------------------------------------------
    # Phase 1: Parse HepMC
    # ------------------------------------------------------------------
    print("\n[Phase 1] Parsing HepMC3 file...")
    events = parse_events(filepath, max_events=max_events, require_pi_pi=True)

    if len(events) == 0:
        print("ERROR: No pi x pi events found. Exiting.")
        sys.exit(1)

    print(f"\n  Selected {len(events)} pi x pi events for analysis.")

    # ------------------------------------------------------------------
    # Phase 2: Reconstruct tau kinematics (Jeans method)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Reconstructing tau kinematics (Jeans method)...")
    reco_results = []
    n_reco_fail = 0
    for i, evt in enumerate(events):
        reco = reconstruct_event(evt)
        if reco is None:
            n_reco_fail += 1
        reco_results.append(reco)
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(events)} events...")

    n_good = sum(1 for r in reco_results if r is not None)
    print(f"  Successfully reconstructed: {n_good}/{len(events)} ({100*n_good/len(events):.1f}%)")
    if n_reco_fail > 0:
        print(f"  Failed: {n_reco_fail}")

    # Filter to successfully reconstructed events
    good_indices = [i for i, r in enumerate(reco_results) if r is not None]
    events_good = [events[i] for i in good_indices]
    reco_good = [reco_results[i] for i in good_indices]
    N = len(events_good)
    print(f"  Proceeding with {N} events.")

    # ------------------------------------------------------------------
    # Phase 3: Compute spacetime intervals
    # ------------------------------------------------------------------
    print("\n[Phase 3] Computing spacetime intervals...")
    truth_intervals = [compute_truth_intervals(evt) for evt in events_good]
    reco_intervals = [compute_reco_intervals(r) for r in reco_good]

    n_spacelike = sum(1 for iv in truth_intervals if iv['is_spacelike'])
    n_timelike = N - n_spacelike
    print(f"  Truth: {n_spacelike} spacelike, {n_timelike} timelike")

    v_signal_truth = np.array([iv['v_signal_c'] for iv in truth_intervals])
    v_signal_reco = np.array([iv['v_signal_c'] for iv in reco_intervals])
    print(f"  Truth v_signal/c: median={np.median(v_signal_truth):.1f}, "
          f"mean={np.mean(np.clip(v_signal_truth, 0, 1e6)):.1f}")

    # ------------------------------------------------------------------
    # Phase 4: Compute spin observables
    # ------------------------------------------------------------------
    print("\n[Phase 4] Extracting spin observables...")
    cos_theta_plus_reco = []
    cos_theta_minus_reco = []
    cos_theta_plus_truth = []
    cos_theta_minus_truth = []
    acoplanarity_reco = []
    acoplanarity_truth = []

    for evt, reco in zip(events_good, reco_good):
        # Using reconstructed tau momenta
        spin_reco = compute_spin_observables(evt, reco, use_reco_tau=True)
        cos_theta_plus_reco.append(spin_reco['cos_theta_plus'])
        cos_theta_minus_reco.append(spin_reco['cos_theta_minus'])
        acoplanarity_reco.append(spin_reco['acoplanarity'])

        # Using truth tau momenta
        spin_truth = compute_spin_observables(evt, reco, use_reco_tau=False)
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

    # Use truth tau momenta for the spin analysis (cleanest)
    print("  Using truth tau momenta for spin projections...")
    global_truth = bootstrap_entanglement(
        cos_theta_plus_truth, cos_theta_minus_truth, n_bootstrap=N_BOOTSTRAP)

    print(f"  [Truth] m12 = {global_truth['m12']:.4f} ± {global_truth['m12_err']:.4f}")
    print(f"  [Truth] Concurrence = {global_truth['concurrence']:.4f} ± {global_truth['concurrence_err']:.4f}")
    print(f"  [Truth] Bell score = {global_truth['bell_score']:.4f} ± {global_truth['bell_score_err']:.4f}")

    print("  Using reco tau momenta for spin projections...")
    global_reco = bootstrap_entanglement(
        cos_theta_plus_reco, cos_theta_minus_reco, n_bootstrap=N_BOOTSTRAP)
    print(f"  [Reco]  m12 = {global_reco['m12']:.4f} ± {global_reco['m12_err']:.4f}")
    print(f"  [Reco]  Concurrence = {global_reco['concurrence']:.4f} ± {global_reco['concurrence_err']:.4f}")

    # Significance
    loc_sigma = locality_rejection_sigma(global_truth['m12'], global_truth['m12_err'])
    ent_sigma = entanglement_rejection_sigma(global_truth['concurrence'], global_truth['concurrence_err'])

    print(f"\n  Reject locality:      {loc_sigma:.1f} sigma")
    print(f"  Reject separability:  {ent_sigma:.1f} sigma")

    # ------------------------------------------------------------------
    # Phase 6: Entanglement vs spacetime interval
    # ------------------------------------------------------------------
    print("\n[Phase 6] Binning entanglement vs spacetime variables...")

    # Use truth v_signal for the physics measurement
    # (this is the actual causal signal speed)
    v_arr = v_signal_truth

    # --- Binned m12 vs signed ds ---
    signed_ds_truth = np.array([iv['signed_ds_mm'] for iv in truth_intervals])
    ds_finite = signed_ds_truth[np.isfinite(signed_ds_truth)]
    ds_lo = np.percentile(ds_finite, 2)
    ds_hi = np.percentile(ds_finite, 98)
    ds_edges = np.linspace(ds_lo, ds_hi, N_BINS_SPACETIME + 1)

    binned_ds = []
    for b in range(N_BINS_SPACETIME):
        mask = (signed_ds_truth >= ds_edges[b]) & (signed_ds_truth < ds_edges[b+1])
        n_in_bin = np.sum(mask)
        if n_in_bin < 10:
            binned_ds.append({'m12': np.nan, 'm12_err': np.nan,
                              'concurrence': np.nan, 'concurrence_err': np.nan,
                              'n_events': n_in_bin})
            continue
        res = bootstrap_entanglement(
            cos_theta_plus_truth[mask], cos_theta_minus_truth[mask],
            n_bootstrap=N_BOOTSTRAP)
        res['n_events'] = n_in_bin
        binned_ds.append(res)

    # --- Binned m12 vs v_signal ---
    v_finite = v_arr[np.isfinite(v_arr)]
    # Use log spacing for v_signal
    if len(v_finite) > 0 and np.min(v_finite) > 0:
        v_lo = max(np.percentile(v_finite, 2), 0.5)
        v_hi = np.percentile(v_finite, 98)
        v_edges = np.logspace(np.log10(v_lo), np.log10(v_hi), N_BINS_SIGNAL_SPEED + 1)
    else:
        v_edges = np.logspace(0, 3, N_BINS_SIGNAL_SPEED + 1)

    binned_v = []
    for b in range(N_BINS_SIGNAL_SPEED):
        mask = (v_arr >= v_edges[b]) & (v_arr < v_edges[b+1])
        n_in_bin = np.sum(mask)
        if n_in_bin < 10:
            binned_v.append({'m12': np.nan, 'm12_err': np.nan,
                             'concurrence': np.nan, 'concurrence_err': np.nan,
                             'n_events': n_in_bin})
            continue
        res = bootstrap_entanglement(
            cos_theta_plus_truth[mask], cos_theta_minus_truth[mask],
            n_bootstrap=N_BOOTSTRAP)
        res['n_events'] = n_in_bin
        binned_v.append(res)

    # ------------------------------------------------------------------
    # Phase 7: v_psi hypothesis scan
    # ------------------------------------------------------------------
    print("\n[Phase 7] Scanning v_psi hypotheses...")
    vpsi_results = scan_vpsi(
        cos_theta_plus_truth, cos_theta_minus_truth, v_arr,
        V_PSI_SCAN, n_bootstrap=N_BOOTSTRAP)

    for r in vpsi_results:
        if r['n_events'] >= 10:
            print(f"  v_psi = {r['v_psi']:6.1f}c : N={r['n_events']:5d}, "
                  f"m12={r['m12']:.3f}±{r['m12_err']:.3f}, "
                  f"reject m12=0 at {r['sigma_vs_0']:.1f}σ, "
                  f"reject m12≤1 at {r['sigma_vs_1']:.1f}σ")

    # ------------------------------------------------------------------
    # Phase 8: Generate plots
    # ------------------------------------------------------------------
    print(f"\n[Phase 8] Generating plots in {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Spacetime distributions
    plot_spacetime_distributions(truth_intervals, reco_intervals)

    # 2. Entanglement vs spacetime interval
    plot_entanglement_vs_spacetime(
        binned_ds, ds_edges,
        xlabel=r"signed $\sqrt{|\Delta s^2|}$ [mm]",
        suffix="spacetime_interval")

    # 3. Entanglement vs signal speed
    plot_entanglement_vs_spacetime(
        binned_v, v_edges,
        xlabel=r"$v_{\rm signal} / c$",
        suffix="signal_speed")

    # 4. v_psi overlay plot (the money plot)
    # Select a few representative v_psi values for the overlay
    v_psi_overlay = [v for v in V_PSI_SCAN if v <= v_edges[-1] * 1.5]
    if len(v_psi_overlay) > 6:
        v_psi_overlay = v_psi_overlay[:6]
    plot_vpsi_overlay(binned_v, v_edges, v_psi_overlay)

    # 5. v_psi exclusion curve
    plot_vpsi_exclusion(vpsi_results)

    # 6. Correlation matrix heatmap
    plot_correlation_matrix(global_truth['C'], global_truth['C_err'], suffix="_truth")
    plot_correlation_matrix(global_reco['C'], global_reco['C_err'], suffix="_reco")

    # 7. Acoplanarity
    plot_acoplanarity(acoplanarity_truth, suffix="_truth")
    plot_acoplanarity(acoplanarity_reco, suffix="_reco")

    # 8. Vertex comparison
    plot_vertex_comparison(reco_good)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(global_truth, loc_sigma, ent_sigma,
                  n_spacelike, n_timelike, vpsi_results)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Tau-tau entanglement analysis at ILC ZH(240)")
    parser.add_argument("hepmc_file", help="Path to HepMC3 file")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of events to process")
    args = parser.parse_args()

    if not os.path.isfile(args.hepmc_file):
        print(f"ERROR: File not found: {args.hepmc_file}")
        sys.exit(1)

    process_events(args.hepmc_file, max_events=args.max_events)


if __name__ == "__main__":
    main()

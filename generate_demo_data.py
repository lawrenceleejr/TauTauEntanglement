#!/usr/bin/env python3
"""Generate synthetic plot data for demonstration of plot aesthetics.

Creates a plot_data.pkl file that can be used with run_analysis.py --replot.
The synthetic data mimics SM H->tau+tau- entanglement at the ILC with
physically motivated distributions.
"""
import numpy as np
import pickle
import os
from config import OUTPUT_DIR, V_PSI_SCAN, C_LIGHT, CTAU_TAU

N_EVENTS = 2000
RNG = np.random.default_rng(42)


def _random_unit_vector(n):
    """Generate n random unit vectors uniformly on the sphere."""
    phi = RNG.uniform(0, 2 * np.pi, n)
    cos_theta = RNG.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.column_stack([sin_theta * np.cos(phi),
                            sin_theta * np.sin(phi),
                            cos_theta])


def make_spacetime_intervals(n, spacelike_frac=0.65):
    """Generate realistic spacetime interval distributions."""
    intervals = []
    for _ in range(n):
        # Tau decay length ~ exponential with mean ctau * gamma
        gamma = RNG.uniform(30, 40)  # Lorentz factor for 62 GeV tau
        L1 = RNG.exponential(CTAU_TAU * gamma)
        L2 = RNG.exponential(CTAU_TAU * gamma)

        # Spatial separation and time separation
        dr = abs(L1 - L2) + RNG.exponential(0.5e-3)  # metres, ~mm scale
        dt = abs(L1 - L2) / C_LIGHT * RNG.lognormal(0, 0.8)

        ds2 = (C_LIGHT * dt)**2 - dr**2
        is_spacelike = ds2 < 0
        signed_ds = np.sign(ds2) * np.sqrt(abs(ds2))
        v_signal = dr / (C_LIGHT * dt) if dt > 0 else np.inf

        intervals.append({
            'dr_m': dr,
            'dt_s': dt,
            'ds2_m2': ds2,
            'v_signal_c': v_signal,
            'is_spacelike': bool(is_spacelike),
            'signed_ds_m': signed_ds,
            'signed_ds_mm': signed_ds * 1e3,
            'dr_mm': dr * 1e3,
            'dt_ps': dt * 1e12,
        })
    return intervals


def make_entangled_cos_theta(n):
    """Generate cos_theta arrays with SM-like entanglement.

    SM prediction for H->tautau: C = diag(+1, +1, -1).
    """
    # Generate tau- direction cosines uniformly
    cos_minus = _random_unit_vector(n)

    # For entangled state: cos_plus correlations
    # C_nn = +1, C_rr = +1, C_kk = -1
    # cos_i+ ~ C_ii * cos_i- + noise
    noise = RNG.normal(0, 0.25, (n, 3))
    cos_plus = cos_minus * np.array([1.0, 1.0, -1.0]) + noise
    # Normalise each row to unit vector
    norms = np.linalg.norm(cos_plus, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    cos_plus = cos_plus / norms

    return cos_plus, cos_minus


def make_acoplanarity(n):
    """Generate acoplanarity with SM-like cos(phi) modulation."""
    # SM: 1 - 0.5*cos(phi)
    # Use rejection sampling
    phi = np.empty(n)
    count = 0
    while count < n:
        trial = RNG.uniform(-np.pi, np.pi, n * 2)
        prob = (1 - 0.5 * np.cos(trial)) / 1.5  # normalised
        accept = RNG.random(n * 2) < prob
        accepted = trial[accept]
        take = min(len(accepted), n - count)
        phi[count:count + take] = accepted[:take]
        count += take
    return phi


def make_reco_results(n, truth_intervals, reco_intervals):
    """Generate reco result dicts with tau decay length info."""
    results = []
    for i in range(n):
        # Decay lengths in metres
        gamma = RNG.uniform(30, 40)
        L_truth_m = RNG.exponential(CTAU_TAU * gamma)
        L_truth_p = RNG.exponential(CTAU_TAU * gamma)
        L_reco_m = L_truth_m * RNG.lognormal(0, 0.3)
        L_reco_p = L_truth_p * RNG.lognormal(0, 0.3)

        results.append({
            'tau_minus': {
                'truth_decay_length_m': L_truth_m,
                'decay_length_m': L_reco_m,
                'decay_vertex_xyz': RNG.normal(0, 1e-3, 3),
                'decay_vertex_t': L_reco_m / C_LIGHT,
            },
            'tau_plus': {
                'truth_decay_length_m': L_truth_p,
                'decay_length_m': L_reco_p,
                'decay_vertex_xyz': RNG.normal(0, 1e-3, 3),
                'decay_vertex_t': L_reco_p / C_LIGHT,
            },
        })
    return results


def bin_entanglement_synthetic(cos_plus, cos_minus, variable, bin_edges):
    """Bin and compute entanglement in each bin."""
    from entanglement import bootstrap_entanglement
    n_bins = len(bin_edges) - 1
    results = []
    for b in range(n_bins):
        mask = (variable >= bin_edges[b]) & (variable < bin_edges[b + 1])
        n_in_bin = int(np.sum(mask))
        if n_in_bin < 10:
            results.append({'m12': np.nan, 'm12_err': np.nan,
                            'concurrence': np.nan, 'concurrence_err': np.nan,
                            'n_events': n_in_bin})
            continue
        res = bootstrap_entanglement(cos_plus[mask], cos_minus[mask],
                                      n_bootstrap=200)
        res['n_events'] = n_in_bin
        results.append(res)
    return results


def main():
    print("Generating synthetic demo data...")

    # Spacetime intervals
    truth_intervals = make_spacetime_intervals(N_EVENTS)
    reco_intervals = make_spacetime_intervals(N_EVENTS)

    # Spin observables
    cos_plus, cos_minus = make_entangled_cos_theta(N_EVENTS)
    acoplanarity_reco = make_acoplanarity(N_EVENTS)
    acoplanarity_truth = make_acoplanarity(N_EVENTS)

    # Signal speeds
    v_arr = np.array([iv['v_signal_c'] for iv in reco_intervals])

    # Global entanglement
    from entanglement import (
        bootstrap_entanglement, locality_rejection_sigma,
        entanglement_rejection_sigma, scan_vpsi,
    )

    print("  Computing global entanglement (reco)...")
    global_reco = bootstrap_entanglement(cos_plus, cos_minus, n_bootstrap=500)
    print("  Computing global entanglement (truth)...")
    global_truth = bootstrap_entanglement(cos_plus, cos_minus, n_bootstrap=500)

    loc_sigma = locality_rejection_sigma(global_reco['m12'], global_reco['m12_err'])
    ent_sigma = entanglement_rejection_sigma(
        global_reco['concurrence'], global_reco['concurrence_err'])

    # Binned entanglement vs signed ds
    signed_ds = np.array([iv['signed_ds_mm'] for iv in reco_intervals])
    ds_finite = signed_ds[np.isfinite(signed_ds)]
    sl = ds_finite[ds_finite < 0]
    sl_lo = np.percentile(sl, 2) if len(sl) > 0 else -5
    ds_edges = np.concatenate([np.linspace(sl_lo, 0, 8),
                                [np.percentile(ds_finite[ds_finite >= 0], 98)]])

    print("  Binning entanglement vs spacetime interval...")
    binned_ds = bin_entanglement_synthetic(cos_plus, cos_minus, signed_ds, ds_edges)

    # Binned entanglement vs v_signal
    v_finite = v_arr[np.isfinite(v_arr)]
    v_lo = max(np.percentile(v_finite, 2), 0.5)
    v_hi = np.percentile(v_finite, 98)
    v_edges = np.logspace(np.log10(v_lo), np.log10(v_hi), 9)

    print("  Binning entanglement vs signal speed...")
    binned_v = bin_entanglement_synthetic(cos_plus, cos_minus, v_arr, v_edges)

    # v_psi scan
    print("  Scanning v_psi hypotheses...")
    vpsi_results = scan_vpsi(cos_plus, cos_minus, v_arr,
                              V_PSI_SCAN, n_bootstrap=200)

    # Reco results for vertex comparison
    reco_good = make_reco_results(N_EVENTS, truth_intervals, reco_intervals)

    n_spacelike = sum(1 for iv in reco_intervals if iv['is_spacelike'])
    n_timelike = N_EVENTS - n_spacelike

    # Build plot data dict
    plot_data = {
        'truth_intervals': truth_intervals,
        'reco_intervals': reco_intervals,
        'binned_ds': binned_ds,
        'ds_edges': ds_edges,
        'binned_v': binned_v,
        'v_edges': v_edges,
        'vpsi_results': vpsi_results,
        'global_reco': global_reco,
        'global_truth': global_truth,
        'acoplanarity_reco': acoplanarity_reco,
        'acoplanarity_truth': acoplanarity_truth,
        'v_arr': v_arr,
        'sigma_v_frac': 0.15,
        'reco_good': reco_good,
        'loc_sigma': loc_sigma,
        'ent_sigma': ent_sigma,
        'n_spacelike_reco': n_spacelike,
        'n_timelike_reco': n_timelike,
        'smear': False,
        'filepath': 'synthetic_demo',
        'N': N_EVENTS,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DIR, "plot_data.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(plot_data, f, protocol=4)
    print(f"  Saved {cache_path}")
    print("Done. Now run: python run_analysis.py --replot")


if __name__ == "__main__":
    main()

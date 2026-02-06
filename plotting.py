"""
Plotting module for the tau-tau entanglement analysis.

Uses the `hist` package for histogramming and matplotlib for rendering.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import hist
import os
from config import OUTPUT_DIR

hep.style.use("CMS")  # clean HEP style

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Spacetime interval distributions
# ---------------------------------------------------------------------------

def plot_spacetime_distributions(truth_intervals, reco_intervals, suffix=""):
    """Plot distributions of spacetime intervals, dr, dt, v_signal."""
    _ensure_output_dir()

    # --- Signed sqrt(|ds2|) distribution ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) signed ds in mm
    ax = axes[0, 0]
    signed_ds_truth = np.array([iv['signed_ds_mm'] for iv in truth_intervals])
    signed_ds_reco = np.array([iv['signed_ds_mm'] for iv in reco_intervals])

    h_truth = hist.Hist(hist.axis.Regular(50, -15, 15, label=r"signed $\sqrt{|\Delta s^2|}$ [mm]"))
    h_truth.fill(signed_ds_truth)
    h_reco = hist.Hist(hist.axis.Regular(50, -15, 15, label=r"signed $\sqrt{|\Delta s^2|}$ [mm]"))
    h_reco.fill(signed_ds_reco)

    hep.histplot(h_truth, ax=ax, label="Truth", histtype="step", color="blue")
    hep.histplot(h_reco, ax=ax, label="Reco (Jeans)", histtype="step", color="red", linestyle="--")
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title("Spacetime interval")
    ax.text(0.03, 0.92, "spacelike ←  → timelike", transform=ax.transAxes,
            fontsize=9, color='gray')

    # (b) dr in mm
    ax = axes[0, 1]
    dr_truth = np.array([iv['dr_mm'] for iv in truth_intervals])
    dr_reco = np.array([iv['dr_mm'] for iv in reco_intervals])

    h = hist.Hist(hist.axis.Regular(50, 0, 15, label=r"$\Delta r$ [mm]"))
    h.fill(dr_truth)
    h2 = hist.Hist(hist.axis.Regular(50, 0, 15, label=r"$\Delta r$ [mm]"))
    h2.fill(dr_reco)
    hep.histplot(h, ax=ax, label="Truth", histtype="step", color="blue")
    hep.histplot(h2, ax=ax, label="Reco", histtype="step", color="red", linestyle="--")
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title("Spatial separation")

    # (c) v_signal / c
    ax = axes[1, 0]
    v_truth = np.array([iv['v_signal_c'] for iv in truth_intervals])
    v_reco = np.array([iv['v_signal_c'] for iv in reco_intervals])
    v_truth_clip = np.clip(v_truth, 0, 200)
    v_reco_clip = np.clip(v_reco, 0, 200)

    h = hist.Hist(hist.axis.Regular(60, 0, 200, label=r"$v_{\rm signal} / c$"))
    h.fill(v_truth_clip)
    h2 = hist.Hist(hist.axis.Regular(60, 0, 200, label=r"$v_{\rm signal} / c$"))
    h2.fill(v_reco_clip)
    hep.histplot(h, ax=ax, label="Truth", histtype="step", color="blue")
    hep.histplot(h2, ax=ax, label="Reco", histtype="step", color="red", linestyle="--")
    ax.axvline(1.0, color='green', linestyle='-', linewidth=2, label='$v = c$')
    ax.set_ylabel("Events")
    ax.set_yscale('log')
    ax.legend()
    ax.set_title("Required causal signal speed")

    # (d) spacelike fraction vs timelike
    ax = axes[1, 1]
    n_sl_truth = sum(1 for iv in truth_intervals if iv['is_spacelike'])
    n_tl_truth = len(truth_intervals) - n_sl_truth
    n_sl_reco = sum(1 for iv in reco_intervals if iv['is_spacelike'])
    n_tl_reco = len(reco_intervals) - n_sl_reco
    labels = ['Spacelike', 'Timelike']
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, [n_sl_truth, n_tl_truth], width, label='Truth', color='steelblue')
    ax.bar(x + width/2, [n_sl_reco, n_tl_reco], width, label='Reco', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title("Causal classification")
    for i, (nt, nr) in enumerate(zip([n_sl_truth, n_tl_truth], [n_sl_reco, n_tl_reco])):
        ax.text(i - width/2, nt + 5, str(nt), ha='center', fontsize=9, color='steelblue')
        ax.text(i + width/2, nr + 5, str(nr), ha='center', fontsize=9, color='indianred')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved spacetime_distributions{suffix}.pdf")


# ---------------------------------------------------------------------------
# 2. Entanglement vs spacetime interval
# ---------------------------------------------------------------------------

def plot_entanglement_vs_spacetime(binned_results, bin_edges, xlabel, suffix=""):
    """Plot m12 and concurrence vs a spacetime variable (ds or v_signal)."""
    _ensure_output_dir()

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    m12_vals = np.array([r['m12'] for r in binned_results])
    m12_errs = np.array([r['m12_err'] for r in binned_results])
    conc_vals = np.array([r['concurrence'] for r in binned_results])
    conc_errs = np.array([r['concurrence_err'] for r in binned_results])
    n_events = np.array([r.get('n_events', 0) for r in binned_results])

    # Mask out empty bins
    valid = ~np.isnan(m12_vals)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                              gridspec_kw={'height_ratios': [3, 3, 1]})

    # (a) m12
    ax = axes[0]
    ax.errorbar(bin_centers[valid], m12_vals[valid], yerr=m12_errs[valid],
                fmt='o', color='navy', markersize=6, capsize=3, label='Measured $m_{12}$')
    ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='SM prediction ($m_{12}=2$)')
    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5, label='Bell nonlocality threshold')
    ax.set_ylabel(r'$m_{12}$', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 3.0)
    ax.set_title(r"Horodecki parameter $m_{12}$ vs " + xlabel, fontsize=13)

    # (b) Concurrence
    ax = axes[1]
    ax.errorbar(bin_centers[valid], conc_vals[valid], yerr=conc_errs[valid],
                fmt='s', color='darkgreen', markersize=6, capsize=3, label='Measured concurrence')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='SM prediction ($C=1$)')
    ax.axhline(0.0, color='orange', linestyle=':', linewidth=1.5, label='Separability threshold')
    ax.set_ylabel('Concurrence', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.3, 1.5)

    # (c) Event count per bin
    ax = axes[2]
    ax.bar(bin_centers, n_events, width=np.diff(bin_edges), color='lightgray',
           edgecolor='gray', alpha=0.7)
    ax.set_ylabel('Events')
    ax.set_xlabel(xlabel, fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved entanglement_vs_{suffix}.pdf")


# ---------------------------------------------------------------------------
# 3. v_psi hypothesis overlay plot
# ---------------------------------------------------------------------------

def plot_vpsi_overlay(binned_results_vs_v, bin_edges_v, v_psi_values):
    """The money plot: m12 vs v_signal with v_psi model step-functions overlaid.

    Shows data as points with error bars, and for several v_psi hypotheses,
    the expected m12 as step functions (m12=2 for v < v_psi, m12=0 for v > v_psi).
    """
    _ensure_output_dir()

    bin_centers = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    m12_vals = np.array([r['m12'] for r in binned_results_vs_v])
    m12_errs = np.array([r['m12_err'] for r in binned_results_vs_v])
    valid = ~np.isnan(m12_vals)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot data
    ax.errorbar(bin_centers[valid], m12_vals[valid], yerr=m12_errs[valid],
                fmt='o', color='black', markersize=7, capsize=4, zorder=10,
                label=r'Measured $m_{12}$')

    # Plot v_psi step functions
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(v_psi_values)))
    v_fine = np.linspace(bin_edges_v[0], bin_edges_v[-1], 500)

    for v_psi, color in zip(v_psi_values, colors):
        m12_model = np.where(v_fine <= v_psi, 2.0, 0.0)
        ax.plot(v_fine, m12_model, color=color, linewidth=1.5, alpha=0.7,
                label=rf'$v_{{\psi}} = {v_psi:.0f}\,c$')

    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Bell nonlocality threshold')
    ax.axhline(2.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel(r'$v_{\rm signal} / c$', fontsize=14)
    ax.set_ylabel(r'$m_{12}$', fontsize=14)
    ax.set_title(r'Entanglement vs required signal speed: $v_\psi$ hypothesis test', fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc='center right')
    ax.set_ylim(-0.5, 3.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_overlay.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_overlay.png"), dpi=150)
    plt.close(fig)
    print("  Saved vpsi_overlay.pdf")


# ---------------------------------------------------------------------------
# 4. v_psi exclusion curve (significance vs v_psi)
# ---------------------------------------------------------------------------

def plot_vpsi_exclusion(vpsi_scan_results):
    """Plot the rejection significance as a function of v_psi.

    Two curves:
      - sigma_vs_0: significance of m12 > 0 (correlations exist)
      - sigma_vs_1: significance of m12 > 1 (Bell nonlocality)
    """
    _ensure_output_dir()

    v_psi_arr = np.array([r['v_psi'] for r in vpsi_scan_results])
    sigma_0 = np.array([r['sigma_vs_0'] for r in vpsi_scan_results])
    sigma_1 = np.array([r['sigma_vs_1'] for r in vpsi_scan_results])
    n_events = np.array([r['n_events'] for r in vpsi_scan_results])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    valid = ~np.isnan(sigma_0) & (n_events >= 10)
    ax.plot(v_psi_arr[valid], sigma_0[valid], 'o-', color='navy',
            markersize=6, label=r'Reject $m_{12}=0$ (no correlation)')
    ax.plot(v_psi_arr[valid], sigma_1[valid], 's-', color='darkred',
            markersize=6, label=r'Reject $m_{12}\leq 1$ (locality)')

    ax.axhline(2.0, color='gray', linestyle=':', alpha=0.5, label=r'$2\sigma$')
    ax.axhline(3.0, color='gray', linestyle='--', alpha=0.5, label=r'$3\sigma$')
    ax.axhline(5.0, color='gray', linestyle='-', alpha=0.3, label=r'$5\sigma$')

    ax.set_ylabel(r'Rejection significance [$\sigma$]', fontsize=14)
    ax.set_title(r'Exclusion of finite-speed signal hypothesis $v_\psi$', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)

    # Event count
    ax2 = axes[1]
    ax2.bar(v_psi_arr, n_events, width=np.diff(np.concatenate([[0], v_psi_arr])),
            color='lightgray', edgecolor='gray', alpha=0.7, align='center')
    ax2.set_xlabel(r'$v_\psi / c$', fontsize=14)
    ax2.set_ylabel('Events')
    ax2.set_xscale('log')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_exclusion.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_exclusion.png"), dpi=150)
    plt.close(fig)
    print("  Saved vpsi_exclusion.pdf")


# ---------------------------------------------------------------------------
# 5. Correlation matrix heatmap
# ---------------------------------------------------------------------------

def plot_correlation_matrix(C, C_err, suffix=""):
    """Plot the 3x3 spin correlation matrix as a heatmap."""
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ['n', 'r', 'k']

    im = ax.imshow(C, cmap='RdBu_r', vmin=-2, vmax=2, aspect='equal')
    plt.colorbar(im, ax=ax, label=r'$C_{ij}$')

    for i in range(3):
        for j in range(3):
            text = f"{C[i,j]:.3f}\n±{C_err[i,j]:.3f}"
            color = 'white' if abs(C[i,j]) > 1.0 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11, color=color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel(r'$j$ (from $\tau^+$ decay)', fontsize=13)
    ax.set_ylabel(r'$i$ (from $\tau^-$ decay)', fontsize=13)
    ax.set_title(r'Spin correlation matrix $C_{ij}$' + (f' ({suffix})' if suffix else ''),
                 fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved correlation_matrix{suffix}.pdf")


# ---------------------------------------------------------------------------
# 6. Acoplanarity distribution
# ---------------------------------------------------------------------------

def plot_acoplanarity(delta_phi_arr, suffix=""):
    """Plot the acoplanarity angle distribution."""
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(8, 6))

    h = hist.Hist(hist.axis.Regular(30, -np.pi, np.pi, label=r"$\Delta\phi$ [rad]"))
    h.fill(delta_phi_arr)
    hep.histplot(h, ax=ax, histtype="fill", color="steelblue", alpha=0.7, edgecolor="navy")

    ax.set_ylabel("Events", fontsize=13)
    ax.set_xlabel(r"Acoplanarity $\Delta\phi$ [rad]", fontsize=13)
    ax.set_title("Acoplanarity angle (CP-sensitive)", fontsize=13)

    # Expected: cos(dphi) for CP-even, cos(dphi - pi) for CP-odd
    phi_fine = np.linspace(-np.pi, np.pi, 200)
    norm = len(delta_phi_arr) * 2 * np.pi / 30  # bin width normalisation
    cp_even = norm / (2 * np.pi) * (1 - 0.5 * np.cos(phi_fine))
    ax.plot(phi_fine, cp_even, 'r--', linewidth=2, label='CP-even expectation (schematic)')
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved acoplanarity{suffix}.pdf")


# ---------------------------------------------------------------------------
# 7. Truth vs reco vertex comparison
# ---------------------------------------------------------------------------

def plot_vertex_comparison(reco_results):
    """Compare truth and reconstructed decay vertices."""
    _ensure_output_dir()

    truth_L_m = []
    reco_L_m = []
    truth_L_p = []
    reco_L_p = []

    for r in reco_results:
        if r is None:
            continue
        truth_L_m.append(r['tau_minus']['truth_decay_length_m'] * 1e3)  # mm
        reco_L_m.append(r['tau_minus']['decay_length_m'] * 1e3)
        truth_L_p.append(r['tau_plus']['truth_decay_length_m'] * 1e3)
        reco_L_p.append(r['tau_plus']['decay_length_m'] * 1e3)

    truth_L = np.array(truth_L_m + truth_L_p)
    reco_L = np.array(reco_L_m + reco_L_p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Scatter plot
    ax = axes[0]
    ax.scatter(truth_L, reco_L, s=2, alpha=0.3, color='steelblue')
    lim = max(truth_L.max(), reco_L.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='Perfect reco')
    ax.set_xlabel("Truth decay length [mm]", fontsize=12)
    ax.set_ylabel("Reco decay length [mm]", fontsize=12)
    ax.set_title("Decay length: truth vs reco", fontsize=13)
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # (b) Residual distribution
    ax = axes[1]
    residual = reco_L - truth_L
    h = hist.Hist(hist.axis.Regular(50, -5, 5, label="Reco - Truth [mm]"))
    h.fill(np.clip(residual, -5, 5))
    hep.histplot(h, ax=ax, histtype="fill", color="steelblue", alpha=0.7, edgecolor="navy")
    ax.set_ylabel("Entries")
    ax.set_title(f"Decay length residual (mean={np.mean(residual):.3f} mm)", fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "vertex_comparison.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "vertex_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved vertex_comparison.pdf")


# ---------------------------------------------------------------------------
# 8. Summary statistics printout
# ---------------------------------------------------------------------------

def print_summary(global_result, locality_sigma, entanglement_sigma,
                  n_spacelike, n_timelike, vpsi_scan_results=None):
    """Print a summary of the entanglement analysis."""
    print("\n" + "="*70)
    print("  TAU-TAU ENTANGLEMENT ANALYSIS SUMMARY")
    print("="*70)

    C = global_result['C']
    print(f"\n  Spin correlation matrix C_ij:")
    labels = ['n', 'r', 'k']
    print(f"       {'n':>10s} {'r':>10s} {'k':>10s}")
    for i in range(3):
        row = "  " + labels[i] + "  "
        for j in range(3):
            row += f"  {C[i,j]:+.4f}±{global_result['C_err'][i,j]:.4f}"
        print(row)

    print(f"\n  SM prediction: C = diag(+1, +1, -1)")

    print(f"\n  Horodecki parameter:   m12 = {global_result['m12']:.4f} ± {global_result['m12_err']:.4f}")
    print(f"  SM prediction:         m12 = 2.0")
    print(f"  Bell score:            2*sqrt(m12) = {global_result['bell_score']:.4f} ± {global_result['bell_score_err']:.4f}")
    print(f"  Concurrence:           C = {global_result['concurrence']:.4f} ± {global_result['concurrence_err']:.4f}")
    print(f"  SM prediction:         C = 1.0")

    print(f"\n  Reject locality (m12 <= 1):    {locality_sigma:.1f} sigma")
    print(f"  Reject separability (C <= 0):  {entanglement_sigma:.1f} sigma")

    print(f"\n  Spacetime classification:")
    n_total = n_spacelike + n_timelike
    print(f"    Spacelike: {n_spacelike} ({100*n_spacelike/n_total:.1f}%)")
    print(f"    Timelike:  {n_timelike} ({100*n_timelike/n_total:.1f}%)")

    if vpsi_scan_results is not None:
        print(f"\n  v_psi exclusion scan:")
        print(f"    {'v_psi/c':>10s} {'N_events':>10s} {'m12':>10s} {'sigma(m12>0)':>14s} {'sigma(m12>1)':>14s}")
        for r in vpsi_scan_results:
            if r['n_events'] >= 10:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d} {r['m12']:10.3f} "
                      f"{r['sigma_vs_0']:14.1f} {r['sigma_vs_1']:14.1f}")
            else:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d}    (too few events)")

    print("\n" + "="*70)

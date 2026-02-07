"""
Plotting module for the tau-tau entanglement analysis.

Uses the `hist` package for histogramming and matplotlib for rendering.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) signed ds in mm
    ax = axes[0, 0]
    signed_ds_truth = np.array([iv['signed_ds_mm'] for iv in truth_intervals])
    signed_ds_reco = np.array([iv['signed_ds_mm'] for iv in reco_intervals])

    ds_all = np.concatenate([signed_ds_truth, signed_ds_reco])
    ds_lo, ds_hi = np.percentile(ds_all[np.isfinite(ds_all)], [1, 99])
    ds_lo = min(ds_lo, -1)
    ds_hi = max(ds_hi, 1)

    h_truth = hist.Hist(hist.axis.Regular(50, ds_lo, ds_hi, label=r"signed $\sqrt{|\Delta s^2|}$ [mm]"))
    h_truth.fill(np.clip(signed_ds_truth, ds_lo, ds_hi))
    h_reco = hist.Hist(hist.axis.Regular(50, ds_lo, ds_hi, label=r"signed $\sqrt{|\Delta s^2|}$ [mm]"))
    h_reco.fill(np.clip(signed_ds_reco, ds_lo, ds_hi))

    hep.histplot(h_truth, ax=ax, label="Truth", histtype="step", color="blue")
    hep.histplot(h_reco, ax=ax, label="Reco (Jeans)", histtype="step", color="red", linestyle="--")
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title("Spacetime interval")
    ax.text(0.03, 0.92, r"spacelike $\leftarrow$  $\rightarrow$ timelike",
            transform=ax.transAxes, fontsize=9, color='gray')

    # (b) dr in mm
    ax = axes[0, 1]
    dr_truth = np.array([iv['dr_mm'] for iv in truth_intervals])
    dr_reco = np.array([iv['dr_mm'] for iv in reco_intervals])
    dr_max = np.percentile(np.concatenate([dr_truth, dr_reco]), 99) * 1.1

    h = hist.Hist(hist.axis.Regular(50, 0, dr_max, label=r"$\Delta r$ [mm]"))
    h.fill(np.clip(dr_truth, 0, dr_max))
    h2 = hist.Hist(hist.axis.Regular(50, 0, dr_max, label=r"$\Delta r$ [mm]"))
    h2.fill(np.clip(dr_reco, 0, dr_max))
    hep.histplot(h, ax=ax, label="Truth", histtype="step", color="blue")
    hep.histplot(h2, ax=ax, label="Reco", histtype="step", color="red", linestyle="--")
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title("Spatial separation")

    # (c) v_signal / c
    ax = axes[1, 0]
    v_truth = np.array([iv['v_signal_c'] for iv in truth_intervals])
    v_reco = np.array([iv['v_signal_c'] for iv in reco_intervals])
    v_max = np.percentile(np.concatenate([
        np.clip(v_truth, 0, 1e6), np.clip(v_reco, 0, 1e6)]), 99)
    v_max = max(v_max, 10) * 1.1

    h = hist.Hist(hist.axis.Regular(60, 0, v_max, label=r"$v_{\rm signal} / c$"))
    h.fill(np.clip(v_truth, 0, v_max))
    h2 = hist.Hist(hist.axis.Regular(60, 0, v_max, label=r"$v_{\rm signal} / c$"))
    h2.fill(np.clip(v_reco, 0, v_max))
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
    ymax_bar = max(n_sl_truth, n_tl_truth, n_sl_reco, n_tl_reco)
    ax.set_ylim(0, ymax_bar * 1.15)
    for i, (nt, nr) in enumerate(zip([n_sl_truth, n_tl_truth], [n_sl_reco, n_tl_reco])):
        ax.text(i - width/2, nt + ymax_bar*0.02, str(nt), ha='center', fontsize=9, color='steelblue')
        ax.text(i + width/2, nr + ymax_bar*0.02, str(nr), ha='center', fontsize=9, color='indianred')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved spacetime_distributions{suffix}.pdf")


# ---------------------------------------------------------------------------
# 2. Entanglement vs spacetime interval
# ---------------------------------------------------------------------------

def plot_entanglement_vs_spacetime(binned_results, bin_edges, xlabel, suffix="",
                                    lightlike_boundary=False):
    """Plot m12 and concurrence vs a spacetime variable (ds or v_signal)."""
    _ensure_output_dir()

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    m12_vals = np.array([r['m12'] for r in binned_results])
    m12_errs = np.array([r['m12_err'] for r in binned_results])
    conc_vals = np.array([r['concurrence'] for r in binned_results])
    conc_errs = np.array([r['concurrence_err'] for r in binned_results])
    n_events = np.array([r.get('n_events', 0) for r in binned_results])

    valid = ~np.isnan(m12_vals)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                              gridspec_kw={'height_ratios': [3, 3, 1]})

    # (a) m12
    ax = axes[0]
    ax.errorbar(bin_centers[valid], m12_vals[valid], yerr=m12_errs[valid],
                fmt='o', color='navy', markersize=6, capsize=3, label=r'Measured $m_{12}$')
    ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label=r'SM prediction ($m_{12}=2$)')
    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5, label='Bell nonlocality threshold')
    if lightlike_boundary:
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Lightlike boundary')
    ax.set_ylabel(r'$m_{12}$', fontsize=14)
    ax.legend(fontsize=10)
    # Auto y-range with padding
    valid_m12 = m12_vals[valid]
    valid_m12_err = m12_errs[valid]
    if len(valid_m12) > 0:
        ylo = max(0, np.min(valid_m12 - valid_m12_err) - 0.5)
        yhi = max(3.0, np.max(valid_m12 + valid_m12_err) + 0.5)
        ax.set_ylim(ylo, yhi)
    ax.set_title(r"Horodecki parameter $m_{12}$ vs " + xlabel, fontsize=13)

    # (b) Concurrence
    ax = axes[1]
    valid_c = ~np.isnan(conc_vals)
    ax.errorbar(bin_centers[valid_c], conc_vals[valid_c], yerr=conc_errs[valid_c],
                fmt='s', color='darkgreen', markersize=6, capsize=3, label='Measured concurrence')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label=r'SM prediction ($\mathcal{C}=1$)')
    ax.axhline(0.0, color='orange', linestyle=':', linewidth=1.5, label='Separability threshold')
    if lightlike_boundary:
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_ylabel('Concurrence', fontsize=14)
    ax.legend(fontsize=10)
    # Auto y-range
    vc = conc_vals[valid_c]
    vc_err = conc_errs[valid_c]
    if len(vc) > 0:
        ylo_c = min(-0.2, np.min(vc - vc_err) - 0.15)
        yhi_c = max(1.3, np.max(vc + vc_err) + 0.15)
        ax.set_ylim(ylo_c, yhi_c)

    # (c) Event count per bin
    ax = axes[2]
    ax.bar(bin_centers, n_events, width=np.diff(bin_edges), color='lightgray',
           edgecolor='gray', alpha=0.7)
    ax.set_ylabel('Events')
    ax.set_xlabel(xlabel, fontsize=14)
    if lightlike_boundary:
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved entanglement_vs_{suffix}.pdf")


# ---------------------------------------------------------------------------
# 3. v_psi hypothesis overlay plot
# ---------------------------------------------------------------------------

def plot_vpsi_overlay(binned_results_vs_v, bin_edges_v, v_psi_values):
    """The money plot: m12 vs v_signal with v_psi model step-functions overlaid."""
    _ensure_output_dir()

    bin_centers = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    m12_vals = np.array([r['m12'] for r in binned_results_vs_v])
    m12_errs = np.array([r['m12_err'] for r in binned_results_vs_v])
    valid = ~np.isnan(m12_vals)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.errorbar(bin_centers[valid], m12_vals[valid], yerr=m12_errs[valid],
                fmt='o', color='black', markersize=7, capsize=4, zorder=10,
                label=r'Measured $m_{12}$')

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
    # Auto y-range
    if np.any(valid):
        yhi = max(3.5, np.nanmax(m12_vals[valid] + m12_errs[valid]) + 0.5)
        ax.set_ylim(-0.5, yhi)

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

    Includes 95% CL (1.96 sigma) exclusion line alongside 2/3/5 sigma markers.
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

    # 95% CL exclusion line
    ax.axhline(1.96, color='forestgreen', linestyle='-', linewidth=1.5,
               alpha=0.7, label=r'95% CL ($1.96\sigma$)')
    ax.axhline(3.0, color='gray', linestyle='--', alpha=0.5, label=r'$3\sigma$')
    ax.axhline(5.0, color='gray', linestyle='-', alpha=0.3, label=r'$5\sigma$')

    ax.set_ylabel(r'Rejection significance [$\sigma$]', fontsize=14)
    ax.set_title(r'Exclusion of finite-speed signal hypothesis $v_\psi$', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    # Auto y-range
    all_sigma = np.concatenate([sigma_0[valid], sigma_1[valid]])
    if len(all_sigma) > 0:
        yhi = max(6, np.nanmax(all_sigma) * 1.15)
        ax.set_ylim(0, yhi)

    # Event count
    ax2 = axes[1]
    # Use log-spaced bar widths
    for i, (v, n) in enumerate(zip(v_psi_arr, n_events)):
        if i == 0:
            w = v_psi_arr[1] - v_psi_arr[0] if len(v_psi_arr) > 1 else v
        else:
            w = v_psi_arr[i] - v_psi_arr[i-1]
        ax2.bar(v, n, width=w*0.8, color='lightgray', edgecolor='gray', alpha=0.7)
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
            text = f"{C[i,j]:.3f}\n" + r"$\pm$" + f"{C_err[i,j]:.3f}"
            color = 'white' if abs(C[i,j]) > 1.0 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11, color=color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel(r'$j$ (from $\tau^+$ decay)', fontsize=13)
    ax.set_ylabel(r'$i$ (from $\tau^-$ decay)', fontsize=13)
    title_extra = f' ({suffix})' if suffix else ''
    ax.set_title(r'Spin correlation matrix $C_{ij}$' + title_extra, fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved correlation_matrix{suffix}.pdf")


# ---------------------------------------------------------------------------
# 6. Acoplanarity distribution with cosine fit
# ---------------------------------------------------------------------------

def _cosine_model(phi, A, B):
    """Model: A * (1 + B * cos(phi))."""
    return A * (1.0 + B * np.cos(phi))


def plot_acoplanarity(delta_phi_arr, suffix=""):
    """Plot the acoplanarity angle distribution with a cosine fit."""
    _ensure_output_dir()

    n_bins = 30
    fig, ax = plt.subplots(figsize=(8, 6))

    h = hist.Hist(hist.axis.Regular(n_bins, -np.pi, np.pi,
                                     label=r"$\Delta\phi$ [rad]"))
    h.fill(delta_phi_arr)
    hep.histplot(h, ax=ax, histtype="fill", color="steelblue", alpha=0.7,
                 edgecolor="navy", label="Data")

    # Cosine fit: N(phi) = A * (1 + B * cos(phi))
    bin_centers = h.axes[0].centers
    bin_counts = h.values()
    bin_width = 2 * np.pi / n_bins
    bin_errors = np.sqrt(np.maximum(bin_counts, 1))

    try:
        popt, pcov = curve_fit(_cosine_model, bin_centers, bin_counts,
                               p0=[np.mean(bin_counts), -0.5],
                               sigma=bin_errors, absolute_sigma=True)
        A_fit, B_fit = popt
        A_err, B_err = np.sqrt(np.diag(pcov))

        phi_fine = np.linspace(-np.pi, np.pi, 200)
        ax.plot(phi_fine, _cosine_model(phi_fine, A_fit, B_fit), 'r-',
                linewidth=2,
                label=f'Fit: $A(1 + B\\cos\\phi)$\n  $B = {B_fit:.3f} \\pm {B_err:.3f}$')
    except RuntimeError:
        A_fit, B_fit, A_err, B_err = 0, 0, 0, 0

    # SM expectation (schematic)
    phi_fine = np.linspace(-np.pi, np.pi, 200)
    norm = len(delta_phi_arr) * bin_width / (2 * np.pi)
    cp_even = norm * (1 - 0.5 * np.cos(phi_fine))
    ax.plot(phi_fine, cp_even, 'g--', linewidth=1.5,
            label='CP-even expectation ($B=-0.5$)')

    ax.set_ylabel("Events", fontsize=13)
    ax.set_xlabel(r"Acoplanarity $\Delta\phi$ [rad]", fontsize=13)
    ax.set_title("Acoplanarity angle (CP-sensitive)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved acoplanarity{suffix}.pdf")


# ---------------------------------------------------------------------------
# 7. Acoplanarity vs signal speed
# ---------------------------------------------------------------------------

def plot_acoplanarity_vs_vsignal(acoplanarity_arr, v_signal_arr, v_edges):
    """Plot the acoplanarity cosine fit parameter B vs signal speed."""
    _ensure_output_dir()

    n_bins_v = len(v_edges) - 1
    bin_centers_v = 0.5 * (v_edges[:-1] + v_edges[1:])

    B_vals = []
    B_errs = []
    n_events = []

    for b in range(n_bins_v):
        mask = (v_signal_arr >= v_edges[b]) & (v_signal_arr < v_edges[b+1])
        n = np.sum(mask)
        n_events.append(n)

        if n < 20:
            B_vals.append(np.nan)
            B_errs.append(np.nan)
            continue

        dphi = acoplanarity_arr[mask]
        h = hist.Hist(hist.axis.Regular(15, -np.pi, np.pi))
        h.fill(dphi)
        bc = h.axes[0].centers
        counts = h.values()
        errors = np.sqrt(np.maximum(counts, 1))

        try:
            popt, pcov = curve_fit(_cosine_model, bc, counts,
                                   p0=[np.mean(counts), -0.5],
                                   sigma=errors, absolute_sigma=True)
            B_vals.append(popt[1])
            B_errs.append(np.sqrt(pcov[1, 1]))
        except RuntimeError:
            B_vals.append(np.nan)
            B_errs.append(np.nan)

    B_vals = np.array(B_vals)
    B_errs = np.array(B_errs)
    n_events = np.array(n_events)
    valid = ~np.isnan(B_vals)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    if np.any(valid):
        ax.errorbar(bin_centers_v[valid], B_vals[valid], yerr=B_errs[valid],
                    fmt='o', color='navy', markersize=6, capsize=3,
                    label=r'Measured $B$ from $A(1+B\cos\phi)$ fit')
    ax.axhline(-0.5, color='red', linestyle='--', linewidth=1.5,
               label=r'SM CP-even ($B=-0.5$)')
    ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel(r'Cosine coefficient $B$', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_title(r'Acoplanarity modulation vs signal speed', fontsize=13)
    # Auto y-range
    vb = B_vals[valid]
    if len(vb) > 0:
        ylo = min(-1.0, np.nanmin(vb - B_errs[valid]) - 0.2)
        yhi = max(0.5, np.nanmax(vb + B_errs[valid]) + 0.2)
        ax.set_ylim(ylo, yhi)

    ax2 = axes[1]
    ax2.bar(bin_centers_v, n_events, width=np.diff(v_edges),
            color='lightgray', edgecolor='gray', alpha=0.7)
    ax2.set_ylabel('Events')
    ax2.set_xlabel(r'$v_{\rm signal} / c$', fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "acoplanarity_vs_vsignal.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "acoplanarity_vs_vsignal.png"), dpi=150)
    plt.close(fig)
    print("  Saved acoplanarity_vs_vsignal.pdf")


# ---------------------------------------------------------------------------
# 8. Truth vs reco vertex comparison (with ratio diagnostic)
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
        truth_L_m.append(r['tau_minus']['truth_decay_length_m'] * 1e3)
        reco_L_m.append(r['tau_minus']['decay_length_m'] * 1e3)
        truth_L_p.append(r['tau_plus']['truth_decay_length_m'] * 1e3)
        reco_L_p.append(r['tau_plus']['decay_length_m'] * 1e3)

    truth_L = np.array(truth_L_m + truth_L_p)
    reco_L = np.array(reco_L_m + reco_L_p)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Scatter plot
    ax = axes[0, 0]
    ax.scatter(truth_L, reco_L, s=2, alpha=0.3, color='steelblue')
    p99 = np.percentile(np.concatenate([truth_L, reco_L]), 99)
    lim = p99 * 1.2
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='Perfect reco')
    ax.set_xlabel("Truth decay length [mm]", fontsize=12)
    ax.set_ylabel("Reco decay length [mm]", fontsize=12)
    ax.set_title("Decay length: truth vs reco", fontsize=13)
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # (b) Residual distribution
    ax = axes[0, 1]
    residual = reco_L - truth_L
    res_lo, res_hi = np.percentile(residual, [2, 98])
    res_range = max(abs(res_lo), abs(res_hi)) * 1.2
    h = hist.Hist(hist.axis.Regular(50, -res_range, res_range,
                                     label="Reco - Truth [mm]"))
    h.fill(np.clip(residual, -res_range, res_range))
    hep.histplot(h, ax=ax, histtype="fill", color="steelblue", alpha=0.7,
                 edgecolor="navy")
    ax.set_ylabel("Entries")
    ax.set_title(f"Decay length residual\n"
                 f"mean={np.mean(residual):.3f} mm, "
                 f"RMS={np.std(residual):.3f} mm", fontsize=12)

    # (c) Ratio distribution (reco/truth) — diagnostic for the tail
    ax = axes[1, 0]
    safe = truth_L > 0.01  # avoid division by zero
    ratio = reco_L[safe] / truth_L[safe]
    h_ratio = hist.Hist(hist.axis.Regular(60, 0, 5, label="Reco / Truth"))
    h_ratio.fill(np.clip(ratio, 0, 5))
    hep.histplot(h_ratio, ax=ax, histtype="fill", color="steelblue", alpha=0.7,
                 edgecolor="navy")
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect')
    ax.set_ylabel("Entries")
    ax.set_title(f"Decay length ratio (median={np.median(ratio):.3f})", fontsize=12)
    ax.legend()

    # (d) Reco/truth ratio vs truth decay length — shows where the tail comes from
    ax = axes[1, 1]
    ax.scatter(truth_L[safe], ratio, s=2, alpha=0.3, color='steelblue')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("Truth decay length [mm]", fontsize=12)
    ax.set_ylabel("Reco / Truth", fontsize=12)
    ax.set_title("Ratio vs truth decay length", fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, p99 * 1.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "vertex_comparison.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "vertex_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved vertex_comparison.pdf")


# ---------------------------------------------------------------------------
# 9. Summary statistics printout
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
            row += f"  {C[i,j]:+.4f}+/-{global_result['C_err'][i,j]:.4f}"
        print(row)

    print(f"\n  SM prediction: C = diag(+1, +1, -1)")

    print(f"\n  Horodecki parameter:   m12 = {global_result['m12']:.4f} +/- {global_result['m12_err']:.4f}")
    print(f"  SM prediction:         m12 = 2.0")
    print(f"  Bell score:            2*sqrt(m12) = {global_result['bell_score']:.4f} +/- {global_result['bell_score_err']:.4f}")
    print(f"  Concurrence:           C = {global_result['concurrence']:.4f} +/- {global_result['concurrence_err']:.4f}")
    print(f"  SM prediction:         C = 1.0")

    print(f"\n  Reject locality (m12 <= 1):    {locality_sigma:.1f} sigma")
    print(f"  Reject separability (C <= 0):  {entanglement_sigma:.1f} sigma")

    print(f"\n  Spacetime classification:")
    n_total = n_spacelike + n_timelike
    print(f"    Spacelike: {n_spacelike} ({100*n_spacelike/n_total:.1f}%)")
    print(f"    Timelike:  {n_timelike} ({100*n_timelike/n_total:.1f}%)")

    if vpsi_scan_results is not None:
        print(f"\n  v_psi exclusion scan:")
        print(f"    {'v_psi/c':>10s} {'N_events':>10s} {'m12':>10s} "
              f"{'sigma(m12>0)':>14s} {'sigma(m12>1)':>14s}")
        for r in vpsi_scan_results:
            if r['n_events'] >= 10:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d} {r['m12']:10.3f} "
                      f"{r['sigma_vs_0']:14.1f} {r['sigma_vs_1']:14.1f}")
            else:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d}    (too few events)")

    print("\n" + "="*70)

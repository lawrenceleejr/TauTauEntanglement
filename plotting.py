"""
Plotting module for the tau-tau entanglement analysis.

Style: Tufte-inspired (maximize data-ink, remove chartjunk, direct labels).
Layout: sized for PRL two-column format.
  - Single column: 3.375 in wide
  - Double column: 7.0 in wide
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.optimize import curve_fit
import os
from config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# PRL dimensions (inches)
# ---------------------------------------------------------------------------
COL1 = 3.375          # single-column width
COL2 = 7.0            # double-column width
GOLDEN = (1 + np.sqrt(5)) / 2  # ~1.618

# ---------------------------------------------------------------------------
# Tufte-inspired global style
# ---------------------------------------------------------------------------
_TUFTE_RC = {
    # Font
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern', 'DejaVu Serif'],
    'font.size': 8,
    'mathtext.fontset': 'cm',
    # Axes
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.titlepad': 4,
    # Ticks
    'xtick.major.size': 3,
    'xtick.minor.size': 1.5,
    'xtick.major.width': 0.4,
    'xtick.minor.width': 0.3,
    'xtick.direction': 'in',
    'xtick.labelsize': 7,
    'ytick.major.size': 3,
    'ytick.minor.size': 1.5,
    'ytick.major.width': 0.4,
    'ytick.minor.width': 0.3,
    'ytick.direction': 'in',
    'ytick.labelsize': 7,
    # Legend
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.4,
    'legend.labelspacing': 0.3,
    'legend.columnspacing': 1.0,
    # Figure
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    # Lines / markers
    'lines.linewidth': 0.8,
    'lines.markersize': 3,
}


def _apply_style():
    """Apply the Tufte RC params (call at top of every plot function)."""
    plt.rcParams.update(_TUFTE_RC)


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# Muted colour palette
_C = {
    'data':     '#1a1a1a',   # near-black for measured data
    'truth':    '#4878A8',   # steel blue
    'reco':     '#C0392B',   # muted red
    'sm':       '#888888',   # grey for SM reference
    'bell':     '#D4A017',   # dark gold for thresholds
    'accent':   '#2E7D32',   # dark green
    'light':    '#B0B0B0',   # light grey for secondary
}


def _step_hist(ax, edges, values, **kwargs):
    """Draw a step histogram (no fill) from pre-computed bin edges and values."""
    x = np.repeat(edges, 2)
    y = np.concatenate([[0], np.repeat(values, 2), [0]])
    ax.plot(x, y, **kwargs)


def _annotate_inline(ax, x, y, text, color='k', fontsize=6, offset=(4, 2)):
    """Place a small inline annotation near a data point."""
    ax.annotate(text, (x, y), textcoords='offset points', xytext=offset,
                fontsize=fontsize, color=color, va='center')


# ---------------------------------------------------------------------------
# 1. Spacetime interval distributions  (double-wide)
# ---------------------------------------------------------------------------

def plot_spacetime_distributions(truth_intervals, reco_intervals, suffix=""):
    _apply_style()
    _ensure_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 / GOLDEN))
    fig.subplots_adjust(hspace=0.35, wspace=0.32)

    # --- helpers ---
    def _hist_vals(data, lo, hi, nbins=50):
        counts, edges = np.histogram(np.clip(data, lo, hi), bins=nbins, range=(lo, hi))
        return edges, counts

    # (a) signed ds
    ax = axes[0, 0]
    sd_t = np.array([iv['signed_ds_mm'] for iv in truth_intervals])
    sd_r = np.array([iv['signed_ds_mm'] for iv in reco_intervals])
    ds_all = np.concatenate([sd_t, sd_r])
    ds_lo, ds_hi = np.percentile(ds_all[np.isfinite(ds_all)], [1, 99])
    ds_lo, ds_hi = min(ds_lo, -1), max(ds_hi, 1)

    e, v = _hist_vals(sd_t, ds_lo, ds_hi)
    _step_hist(ax, e, v, color=_C['truth'], label='Truth')
    e, v = _hist_vals(sd_r, ds_lo, ds_hi)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--', label='Reco')
    ax.axvline(0, color=_C['light'], linewidth=0.4, zorder=0)
    ax.set_xlabel(r'Signed $\sqrt{|\Delta s^2|}$ [mm]')
    ax.set_ylabel('Events')
    ax.legend()
    ax.text(0.03, 0.93, r'spacelike $\leftarrow|\rightarrow$ timelike',
            transform=ax.transAxes, fontsize=5.5, color=_C['light'])
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (b) dr
    ax = axes[0, 1]
    dr_t = np.array([iv['dr_mm'] for iv in truth_intervals])
    dr_r = np.array([iv['dr_mm'] for iv in reco_intervals])
    dr_max = np.percentile(np.concatenate([dr_t, dr_r]), 99) * 1.1

    e, v = _hist_vals(dr_t, 0, dr_max)
    _step_hist(ax, e, v, color=_C['truth'], label='Truth')
    e, v = _hist_vals(dr_r, 0, dr_max)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--', label='Reco')
    ax.set_xlabel(r'$\Delta r$ [mm]')
    ax.set_ylabel('Events')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (c) v_signal
    ax = axes[1, 0]
    v_t = np.array([iv['v_signal_c'] for iv in truth_intervals])
    v_r = np.array([iv['v_signal_c'] for iv in reco_intervals])
    v_max = max(np.percentile(np.concatenate([
        np.clip(v_t, 0, 1e6), np.clip(v_r, 0, 1e6)]), 99), 10) * 1.1

    e, v = _hist_vals(v_t, 0, v_max, nbins=60)
    _step_hist(ax, e, v, color=_C['truth'], label='Truth')
    e, v = _hist_vals(v_r, 0, v_max, nbins=60)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--', label='Reco')
    ax.axvline(1.0, color=_C['accent'], linewidth=0.6, label='$v = c$')
    ax.set_xlabel(r'$v_{\mathrm{signal}} / c$')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    ax.legend()

    # (d) spacelike / timelike counts
    ax = axes[1, 1]
    n_sl_t = sum(1 for iv in truth_intervals if iv['is_spacelike'])
    n_tl_t = len(truth_intervals) - n_sl_t
    n_sl_r = sum(1 for iv in reco_intervals if iv['is_spacelike'])
    n_tl_r = len(reco_intervals) - n_sl_r

    x = np.array([0, 1])
    w = 0.28
    ax.bar(x - w/2, [n_sl_t, n_tl_t], w, color=_C['truth'], alpha=0.75,
           edgecolor='none', label='Truth')
    ax.bar(x + w/2, [n_sl_r, n_tl_r], w, color=_C['reco'], alpha=0.75,
           edgecolor='none', label='Reco')
    ax.set_xticks(x)
    ax.set_xticklabels(['Spacelike', 'Timelike'])
    ax.set_ylabel('Events')
    ax.legend()
    ymax = max(n_sl_t, n_tl_t, n_sl_r, n_tl_r)
    ax.set_ylim(0, ymax * 1.18)
    for i, (nt, nr) in enumerate(zip([n_sl_t, n_tl_t], [n_sl_r, n_tl_r])):
        ax.text(i - w/2, nt + ymax * 0.02, str(nt), ha='center', fontsize=6,
                color=_C['truth'])
        ax.text(i + w/2, nr + ymax * 0.02, str(nr), ha='center', fontsize=6,
                color=_C['reco'])

    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.png"))
    plt.close(fig)
    print(f"  Saved spacetime_distributions{suffix}.pdf")


# ---------------------------------------------------------------------------
# 2. Entanglement vs spacetime interval  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_entanglement_vs_spacetime(binned_results, bin_edges, xlabel, suffix="",
                                    lightlike_boundary=False):
    _apply_style()
    _ensure_output_dir()

    bc = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hw = np.diff(bin_edges) / 2

    m12  = np.array([r['m12'] for r in binned_results])
    m12e = np.array([r['m12_err'] for r in binned_results])
    conc  = np.array([r['concurrence'] for r in binned_results])
    conce = np.array([r['concurrence_err'] for r in binned_results])
    nev   = np.array([r.get('n_events', 0) for r in binned_results])
    ok    = ~np.isnan(m12)
    ok_c  = ~np.isnan(conc)

    fig, axes = plt.subplots(3, 1, figsize=(COL1, COL1 * 1.45),
                              gridspec_kw={'height_ratios': [3, 3, 1]})
    fig.subplots_adjust(hspace=0.08)

    # (a) m12
    ax = axes[0]
    ax.errorbar(bc[ok], m12[ok], yerr=m12e[ok], xerr=hw[ok],
                fmt='o', color=_C['data'], markersize=3, capsize=0,
                elinewidth=0.5, zorder=5)
    ax.axhline(2.0, color=_C['sm'], linewidth=0.5, linestyle='--', zorder=1)
    ax.axhline(1.0, color=_C['bell'], linewidth=0.5, linestyle=':', zorder=1)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.4, zorder=0)
    # Direct labels instead of legend
    ax.text(0.97, 0.92, r'$m_{12}$', transform=ax.transAxes, ha='right',
            fontsize=8, fontweight='bold')
    ax.text(0.97, 0.78, r'SM ($m_{12}=2$)', transform=ax.transAxes, ha='right',
            fontsize=5.5, color=_C['sm'])
    ax.text(0.97, 0.15, 'Bell threshold', transform=ax.transAxes, ha='right',
            fontsize=5.5, color=_C['bell'])
    ax.set_ylabel(r'$m_{12}$')
    ax.tick_params(labelbottom=False)
    if len(m12[ok]) > 0:
        ylo = max(0, np.min(m12[ok] - m12e[ok]) - 0.5)
        yhi = max(3.0, np.max(m12[ok] + m12e[ok]) + 0.5)
        ax.set_ylim(ylo, yhi)

    # (b) Concurrence
    ax = axes[1]
    ax.errorbar(bc[ok_c], conc[ok_c], yerr=conce[ok_c], xerr=hw[ok_c],
                fmt='s', color=_C['accent'], markersize=2.5, capsize=0,
                elinewidth=0.5, zorder=5)
    ax.axhline(1.0, color=_C['sm'], linewidth=0.5, linestyle='--', zorder=1)
    ax.axhline(0.0, color=_C['bell'], linewidth=0.5, linestyle=':', zorder=1)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.4, zorder=0)
    ax.text(0.97, 0.92, 'Concurrence', transform=ax.transAxes, ha='right',
            fontsize=8, fontweight='bold')
    ax.text(0.97, 0.78, r'SM ($\mathcal{C}=1$)', transform=ax.transAxes,
            ha='right', fontsize=5.5, color=_C['sm'])
    ax.set_ylabel(r'$\mathcal{C}$')
    ax.tick_params(labelbottom=False)
    if len(conc[ok_c]) > 0:
        ylo = min(-0.2, np.min(conc[ok_c] - conce[ok_c]) - 0.15)
        yhi = max(1.3, np.max(conc[ok_c] + conce[ok_c]) + 0.15)
        ax.set_ylim(ylo, yhi)

    # (c) Event count
    ax = axes[2]
    ax.bar(bc, nev, width=np.diff(bin_edges), color=_C['light'], edgecolor='none',
           alpha=0.6)
    ax.set_ylabel('Events')
    ax.set_xlabel(xlabel)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.4, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.png"))
    plt.close(fig)
    print(f"  Saved entanglement_vs_{suffix}.pdf")


# ---------------------------------------------------------------------------
# 3. v_psi hypothesis overlay  (double-wide)
# ---------------------------------------------------------------------------

def plot_vpsi_overlay(binned_results_vs_v, bin_edges_v, v_psi_values):
    _apply_style()
    _ensure_output_dir()

    bc = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    hw = np.diff(bin_edges_v) / 2
    m12  = np.array([r['m12'] for r in binned_results_vs_v])
    m12e = np.array([r['m12_err'] for r in binned_results_vs_v])
    ok   = ~np.isnan(m12)

    fig, ax = plt.subplots(figsize=(COL2, COL2 / GOLDEN / 1.3))

    # Data points
    ax.errorbar(bc[ok], m12[ok], yerr=m12e[ok], xerr=hw[ok],
                fmt='o', color=_C['data'], markersize=3.5, capsize=0,
                elinewidth=0.6, zorder=10, label=r'Measured $m_{12}$')

    # v_psi step models — thin grey lines with direct end labels
    v_fine = np.linspace(bin_edges_v[0], bin_edges_v[-1], 500)
    greys = np.linspace(0.45, 0.80, len(v_psi_values))
    for v_psi, g in zip(v_psi_values, greys):
        c = str(g)
        m12_model = np.where(v_fine <= v_psi, 2.0, 0.0)
        ax.plot(v_fine, m12_model, color=c, linewidth=0.6, zorder=2)
        ax.text(v_psi, 2.12, rf'${v_psi:.0f}c$', fontsize=5, color=c,
                ha='center', va='bottom', rotation=0)

    ax.axhline(1.0, color=_C['bell'], linewidth=0.5, linestyle=':', zorder=1)
    ax.text(bin_edges_v[-1], 1.05, 'Bell threshold', fontsize=5.5,
            color=_C['bell'], ha='right', va='bottom')
    ax.axhline(2.0, color=_C['sm'], linewidth=0.4, linestyle='--', zorder=1)

    ax.set_xlabel(r'$v_{\mathrm{signal}} / c$')
    ax.set_ylabel(r'$m_{12}$')
    ax.legend(loc='upper right', fontsize=7)
    if np.any(ok):
        yhi = max(3.5, np.nanmax(m12[ok] + m12e[ok]) + 0.5)
        ax.set_ylim(-0.3, yhi)

    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_overlay.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_overlay.png"))
    plt.close(fig)
    print("  Saved vpsi_overlay.pdf")


# ---------------------------------------------------------------------------
# 4. v_psi exclusion curve  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_vpsi_exclusion(vpsi_scan_results):
    _apply_style()
    _ensure_output_dir()

    v_psi = np.array([r['v_psi'] for r in vpsi_scan_results])
    sig0  = np.array([r['sigma_vs_0'] for r in vpsi_scan_results])
    sig1  = np.array([r['sigma_vs_1'] for r in vpsi_scan_results])
    nev   = np.array([r['n_events'] for r in vpsi_scan_results])
    ok    = ~np.isnan(sig0) & (nev >= 10)

    fig, axes = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.15),
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.08)

    ax = axes[0]
    ax.plot(v_psi[ok], sig0[ok], 'o-', color=_C['truth'], markersize=2.5,
            linewidth=0.7, label=r'Reject $m_{12}=0$')
    ax.plot(v_psi[ok], sig1[ok], 's-', color=_C['reco'], markersize=2.5,
            linewidth=0.7, label=r'Reject $m_{12}\leq 1$')
    ax.axhline(1.96, color=_C['accent'], linewidth=0.5,
               label=r'95\% CL')
    ax.axhline(3.0, color=_C['light'], linewidth=0.4, linestyle='--')
    ax.axhline(5.0, color=_C['light'], linewidth=0.4, linestyle='-')
    # Direct labels for sigma lines
    ax.text(v_psi[ok][-1] * 1.1, 3.15, r'$3\sigma$', fontsize=5,
            color=_C['light'], va='bottom')
    ax.text(v_psi[ok][-1] * 1.1, 5.15, r'$5\sigma$', fontsize=5,
            color=_C['light'], va='bottom')
    ax.set_ylabel(r'Rejection significance [$\sigma$]')
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.tick_params(labelbottom=False)
    all_sig = np.concatenate([sig0[ok], sig1[ok]])
    if len(all_sig) > 0:
        ax.set_ylim(0, max(6, np.nanmax(all_sig) * 1.15))

    ax2 = axes[1]
    ax2.plot(v_psi[ok], nev[ok], 'o', color=_C['light'], markersize=2.5)
    ax2.set_xlabel(r'$v_\psi / c$')
    ax2.set_ylabel('Events')
    ax2.set_xscale('log')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_exclusion.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_exclusion.png"))
    plt.close(fig)
    print("  Saved vpsi_exclusion.pdf")


# ---------------------------------------------------------------------------
# 5. Correlation matrix heatmap  (single-column, square)
# ---------------------------------------------------------------------------

def plot_correlation_matrix(C, C_err, suffix=""):
    _apply_style()
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.92))
    labels = ['$n$', '$r$', '$k$']

    # Use a diverging colourmap centered on zero
    vlim = max(abs(C).max(), 1.0)
    im = ax.imshow(C, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='equal',
                   interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=15, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$C_{ij}$', fontsize=7)

    for i in range(3):
        for j in range(3):
            txt = f'{C[i,j]:+.2f}\n$\\pm${C_err[i,j]:.2f}'
            clr = 'white' if abs(C[i, j]) > 0.6 * vlim else _C['data']
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                    color=clr)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel(r'$j$ ($\tau^+$ decay)')
    ax.set_ylabel(r'$i$ ($\tau^-$ decay)')
    # Restore all spines for the matrix box
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
    ax.tick_params(top=True, right=True, direction='out', length=0)

    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.png"))
    plt.close(fig)
    print(f"  Saved correlation_matrix{suffix}.pdf")


# ---------------------------------------------------------------------------
# 6. Acoplanarity distribution  (single-column)
# ---------------------------------------------------------------------------

def _cosine_model(phi, A, B):
    return A * (1.0 + B * np.cos(phi))


def plot_acoplanarity(delta_phi_arr, suffix=""):
    _apply_style()
    _ensure_output_dir()

    n_bins = 25
    fig, ax = plt.subplots(figsize=(COL1, COL1 / GOLDEN))

    counts, edges = np.histogram(delta_phi_arr, bins=n_bins, range=(-np.pi, np.pi))
    bc = 0.5 * (edges[:-1] + edges[1:])
    bw = 2 * np.pi / n_bins
    errs = np.sqrt(np.maximum(counts, 1))

    # Data as points with error bars (Tufte: dots, not filled bars)
    ax.errorbar(bc, counts, yerr=errs, fmt='o', color=_C['data'],
                markersize=2.5, capsize=0, elinewidth=0.4, zorder=5)

    # Cosine fit
    try:
        popt, pcov = curve_fit(_cosine_model, bc, counts,
                               p0=[np.mean(counts), -0.5],
                               sigma=errs, absolute_sigma=True)
        A_fit, B_fit = popt
        B_err = np.sqrt(pcov[1, 1])

        phi_fine = np.linspace(-np.pi, np.pi, 200)
        ax.plot(phi_fine, _cosine_model(phi_fine, *popt), color=_C['reco'],
                linewidth=0.7, zorder=3)
        ax.text(0.03, 0.92,
                rf'$B = {B_fit:.3f} \pm {B_err:.3f}$',
                transform=ax.transAxes, fontsize=6.5, color=_C['reco'])
    except RuntimeError:
        pass

    # SM expectation
    phi_fine = np.linspace(-np.pi, np.pi, 200)
    norm = len(delta_phi_arr) * bw / (2 * np.pi)
    ax.plot(phi_fine, norm * (1 - 0.5 * np.cos(phi_fine)),
            color=_C['sm'], linewidth=0.5, linestyle='--', zorder=2)
    ax.text(0.03, 0.82, r'SM ($B=-0.5$)', transform=ax.transAxes,
            fontsize=5.5, color=_C['sm'])

    ax.set_xlabel(r'Acoplanarity $\Delta\phi$ [rad]')
    ax.set_ylabel('Events')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.png"))
    plt.close(fig)
    print(f"  Saved acoplanarity{suffix}.pdf")


# ---------------------------------------------------------------------------
# 7. Acoplanarity vs signal speed  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_acoplanarity_vs_vsignal(acoplanarity_arr, v_signal_arr, v_edges):
    _apply_style()
    _ensure_output_dir()

    n_bins_v = len(v_edges) - 1
    bc_v = 0.5 * (v_edges[:-1] + v_edges[1:])
    hw_v = np.diff(v_edges) / 2

    B_vals, B_errs, n_events = [], [], []
    for b in range(n_bins_v):
        mask = (v_signal_arr >= v_edges[b]) & (v_signal_arr < v_edges[b + 1])
        n = np.sum(mask)
        n_events.append(n)
        if n < 20:
            B_vals.append(np.nan); B_errs.append(np.nan)
            continue
        dphi = acoplanarity_arr[mask]
        c, e = np.histogram(dphi, bins=15, range=(-np.pi, np.pi))
        cbc = 0.5 * (e[:-1] + e[1:])
        cerr = np.sqrt(np.maximum(c, 1))
        try:
            popt, pcov = curve_fit(_cosine_model, cbc, c,
                                   p0=[np.mean(c), -0.5],
                                   sigma=cerr, absolute_sigma=True)
            B_vals.append(popt[1])
            B_errs.append(np.sqrt(pcov[1, 1]))
        except RuntimeError:
            B_vals.append(np.nan); B_errs.append(np.nan)

    B_vals = np.array(B_vals)
    B_errs = np.array(B_errs)
    n_events = np.array(n_events)
    ok = ~np.isnan(B_vals)

    fig, axes = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.15),
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.08)

    ax = axes[0]
    if np.any(ok):
        ax.errorbar(bc_v[ok], B_vals[ok], yerr=B_errs[ok], xerr=hw_v[ok],
                    fmt='o', color=_C['data'], markersize=2.5, capsize=0,
                    elinewidth=0.5)
    ax.axhline(-0.5, color=_C['sm'], linewidth=0.5, linestyle='--')
    ax.axhline(0.0, color=_C['light'], linewidth=0.3)
    ax.text(0.97, 0.08, r'SM ($B=-0.5$)', transform=ax.transAxes,
            ha='right', fontsize=5.5, color=_C['sm'])
    ax.set_ylabel(r'Cosine coefficient $B$')
    ax.tick_params(labelbottom=False)
    vb = B_vals[ok]
    if len(vb) > 0:
        ylo = min(-1.0, np.nanmin(vb - B_errs[ok]) - 0.2)
        yhi = max(0.5, np.nanmax(vb + B_errs[ok]) + 0.2)
        ax.set_ylim(ylo, yhi)

    ax2 = axes[1]
    ax2.bar(bc_v, n_events, width=np.diff(v_edges), color=_C['light'],
            edgecolor='none', alpha=0.6)
    ax2.set_xlabel(r'$v_{\mathrm{signal}} / c$')
    ax2.set_ylabel('Events')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    fig.savefig(os.path.join(OUTPUT_DIR, "acoplanarity_vs_vsignal.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "acoplanarity_vs_vsignal.png"))
    plt.close(fig)
    print("  Saved acoplanarity_vs_vsignal.pdf")


# ---------------------------------------------------------------------------
# 8. Vertex comparison  (double-wide)
# ---------------------------------------------------------------------------

def plot_vertex_comparison(reco_results):
    _apply_style()
    _ensure_output_dir()

    truth_L_m, reco_L_m, truth_L_p, reco_L_p = [], [], [], []
    for r in reco_results:
        if r is None:
            continue
        truth_L_m.append(r['tau_minus']['truth_decay_length_m'] * 1e3)
        reco_L_m.append(r['tau_minus']['decay_length_m'] * 1e3)
        truth_L_p.append(r['tau_plus']['truth_decay_length_m'] * 1e3)
        reco_L_p.append(r['tau_plus']['decay_length_m'] * 1e3)

    truth_L = np.array(truth_L_m + truth_L_p)
    reco_L = np.array(reco_L_m + reco_L_p)

    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 / GOLDEN))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    p99 = np.percentile(np.concatenate([truth_L, reco_L]), 99)
    lim = p99 * 1.2

    # (a) Scatter
    ax = axes[0, 0]
    ax.scatter(truth_L, reco_L, s=0.5, alpha=0.2, color=_C['truth'],
               edgecolors='none', rasterized=True)
    ax.plot([0, lim], [0, lim], color=_C['sm'], linewidth=0.5, linestyle='--')
    ax.set_xlabel('Truth decay length [mm]')
    ax.set_ylabel('Reco decay length [mm]')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')

    # (b) Residual
    ax = axes[0, 1]
    residual = reco_L - truth_L
    res_range = max(abs(np.percentile(residual, 2)),
                    abs(np.percentile(residual, 98))) * 1.2
    c, e = np.histogram(np.clip(residual, -res_range, res_range),
                        bins=50, range=(-res_range, res_range))
    _step_hist(ax, e, c, color=_C['truth'])
    ax.set_xlabel('Reco $-$ Truth [mm]')
    ax.set_ylabel('Entries')
    ax.text(0.97, 0.92,
            f'mean {np.mean(residual):.3f}\nRMS {np.std(residual):.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=5.5,
            color=_C['data'])

    # (c) Ratio distribution
    ax = axes[1, 0]
    safe = truth_L > 0.01
    ratio = reco_L[safe] / truth_L[safe]
    c, e = np.histogram(np.clip(ratio, 0, 5), bins=60, range=(0, 5))
    _step_hist(ax, e, c, color=_C['truth'])
    ax.axvline(1.0, color=_C['sm'], linewidth=0.5, linestyle='--')
    ax.set_xlabel('Reco / Truth')
    ax.set_ylabel('Entries')
    ax.text(0.97, 0.92, f'median {np.median(ratio):.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=5.5,
            color=_C['data'])

    # (d) Ratio vs truth
    ax = axes[1, 1]
    ax.scatter(truth_L[safe], ratio, s=0.5, alpha=0.2, color=_C['truth'],
               edgecolors='none', rasterized=True)
    ax.axhline(1.0, color=_C['sm'], linewidth=0.5, linestyle='--')
    ax.set_xlabel('Truth decay length [mm]')
    ax.set_ylabel('Reco / Truth')
    ax.set_ylim(0, 5)
    ax.set_xlim(0, lim)

    fig.savefig(os.path.join(OUTPUT_DIR, "vertex_comparison.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "vertex_comparison.png"))
    plt.close(fig)
    print("  Saved vertex_comparison.pdf")


# ---------------------------------------------------------------------------
# 9. Summary statistics printout
# ---------------------------------------------------------------------------

def print_summary(global_result, locality_sigma, entanglement_sigma,
                  n_spacelike, n_timelike, vpsi_scan_results=None):
    print("\n" + "=" * 70)
    print("  TAU-TAU ENTANGLEMENT ANALYSIS SUMMARY")
    print("=" * 70)

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

    m12 = global_result['m12']
    print(f"\n  Horodecki parameter:   m12 = {m12:.4f} +/- {global_result['m12_err']:.4f}")
    print(f"  SM prediction:         m12 = 2.0")
    print(f"  Bell score:            2*sqrt(m12) = {global_result['bell_score']:.4f} "
          f"+/- {global_result['bell_score_err']:.4f}")
    print(f"  Concurrence:           C = {global_result['concurrence']:.4f} "
          f"+/- {global_result['concurrence_err']:.4f}")
    print(f"  SM prediction:         C = 1.0")

    print(f"\n  Reject locality (m12 <= 1):    {locality_sigma:.1f} sigma")
    print(f"  Reject separability (C <= 0):  {entanglement_sigma:.1f} sigma")

    n_total = n_spacelike + n_timelike
    print(f"\n  Spacetime classification:")
    print(f"    Spacelike: {n_spacelike} ({100*n_spacelike/n_total:.1f}%)")
    print(f"    Timelike:  {n_timelike} ({100*n_timelike/n_total:.1f}%)")

    if vpsi_scan_results is not None:
        print(f"\n  v_psi exclusion scan:")
        print(f"    {'v_psi/c':>10s} {'N_events':>10s} {'m12':>10s} "
              f"{'sigma(m12>0)':>14s} {'sigma(m12>1)':>14s}")
        for r in vpsi_scan_results:
            if r['n_events'] >= 10:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d} "
                      f"{r['m12']:10.3f} "
                      f"{r['sigma_vs_0']:14.1f} {r['sigma_vs_1']:14.1f}")
            else:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d}"
                      f"    (too few events)")

    print("\n" + "=" * 70)

"""
Plotting module for the tau-tau entanglement analysis.

Style: Tufte-inspired art — maximise data-ink, remove chartjunk, direct labels,
drop shadows on data marks, subtle hatched texture in fills, range-frame axes,
letterpress-thin ruling lines.  White background for clean PDF embedding.

Layout: sized for PRL two-column format.
  - Single column: 3.375 in wide
  - Double column: 7.0 in wide
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
# Palette — muted, ink-on-paper tones
# ---------------------------------------------------------------------------
_C = {
    'data':     '#2b2b2b',   # warm near-black
    'truth':    '#3e6b8a',   # dusted slate blue
    'reco':     '#a0413a',   # brick red
    'sm':       '#7a7a7a',   # warm grey
    'bell':     '#b8942e',   # muted gold
    'accent':   '#3b7a3e',   # forest green
    'light':    '#b5b0a8',   # parchment grey
    'paper':    '#ffffff',   # white — sits cleanly on PDF page
    'shadow':   '#c8c8c8',   # neutral light-grey shadow
}

# ---------------------------------------------------------------------------
# Tufte-inspired global style — letterpress-thin ruling
# ---------------------------------------------------------------------------
_TUFTE_RC = {
    # Font
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern', 'DejaVu Serif'],
    'font.size': 8,
    'mathtext.fontset': 'cm',
    # Axes
    'axes.linewidth': 0.35,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.titlepad': 4,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#555555',
    # Ticks
    'xtick.major.size': 3,
    'xtick.minor.size': 1.5,
    'xtick.major.width': 0.3,
    'xtick.minor.width': 0.2,
    'xtick.direction': 'in',
    'xtick.labelsize': 7,
    'xtick.color': '#555555',
    'ytick.major.size': 3,
    'ytick.minor.size': 1.5,
    'ytick.major.width': 0.3,
    'ytick.minor.width': 0.2,
    'ytick.direction': 'in',
    'ytick.labelsize': 7,
    'ytick.color': '#555555',
    # Legend
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.4,
    'legend.labelspacing': 0.3,
    'legend.columnspacing': 1.0,
    # Figure
    'figure.dpi': 600,
    'figure.facecolor': 'white',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'savefig.facecolor': 'white',
    # Lines / markers — whisper-thin
    'lines.linewidth': 0.6,
    'lines.markersize': 3,
    # Text
    'text.color': '#2b2b2b',
    'axes.labelcolor': '#2b2b2b',
}


def _apply_style():
    """Apply the Tufte RC params (call at top of every plot function)."""
    plt.rcParams.update(_TUFTE_RC)


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# Drop-shadow helpers
# ===================================================================

# Reusable path-effect lists
_MARKER_SHADOW = [
    pe.SimpleLineShadow(offset=(0.6, -0.6), shadow_color=_C['shadow'],
                        alpha=0.35, linewidth=0),
    pe.Normal(),
]

_LINE_SHADOW = [
    pe.SimpleLineShadow(offset=(0.5, -0.5), shadow_color=_C['shadow'],
                        alpha=0.25),
    pe.Normal(),
]


def _shadow_errorbar(ax, x, y, yerr=None, xerr=None, color=_C['data'],
                     marker='o', ms=3, elinewidth=0.4, label=None,
                     zorder=5, **kw):
    """Plot an errorbar with a soft drop shadow underneath.

    Draws a slightly-offset shadow replica first, then the real data on top.
    This gives markers and error bars a letterpress "lifted" look.
    """
    dx, dy = 0.0015, -0.0015          # shadow offset in axes fraction
    shadow_color = _C['shadow']
    shadow_alpha = 0.30

    # --- shadow layer (slightly offset, no label) ---
    # We transform offset to data coords via axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    sx = (xlim[1] - xlim[0]) * dx
    sy = (ylim[1] - ylim[0]) * dy

    ax.errorbar(x + sx, y + sy, yerr=yerr, xerr=xerr,
                fmt=marker, color=shadow_color, markersize=ms,
                ecolor=shadow_color, elinewidth=elinewidth,
                capsize=0, alpha=shadow_alpha, zorder=zorder - 1,
                markeredgewidth=0, **kw)

    # --- real data layer ---
    container = ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                            fmt=marker, color=color, markersize=ms,
                            ecolor=color, elinewidth=elinewidth,
                            capsize=0, zorder=zorder, label=label,
                            markeredgewidth=0, **kw)
    return container


def _shadow_markers(ax, x, y, color=_C['data'], marker='o', ms=3,
                    label=None, zorder=5, **kw):
    """Scatter-style markers with a subtle drop shadow."""
    dx, dy = 0.0015, -0.0015
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    sx = (xlim[1] - xlim[0]) * dx
    sy = (ylim[1] - ylim[0]) * dy

    ax.plot(x + sx, y + sy, marker=marker, linestyle='none',
            color=_C['shadow'], markersize=ms, alpha=0.30,
            zorder=zorder - 1, markeredgewidth=0, **kw)
    ax.plot(x, y, marker=marker, linestyle='none', color=color,
            markersize=ms, zorder=zorder, label=label,
            markeredgewidth=0, **kw)


# ===================================================================
# Paper ground + background hatch texture
# ===================================================================

def _paper_bg(fig, ax_or_axes):
    """Set figure and axes to clean white for seamless PDF embedding."""
    fig.patch.set_facecolor(_C['paper'])
    axes = np.atleast_1d(ax_or_axes).ravel()
    for ax in axes:
        ax.set_facecolor(_C['paper'])


# ===================================================================
# Textured / hatched fills
# ===================================================================

def _step_hist(ax, edges, values, hatch=None, fill_alpha=0.0,
               fill_color=None, **kwargs):
    """Draw a step histogram with optional subtle diagonal-line fill.

    Parameters
    ----------
    hatch : str or None
        Matplotlib hatch pattern, e.g. '////' for fine diagonals.
    fill_alpha : float
        Opacity of the filled region (0 = outline only).
    fill_color : str or None
        Colour of the fill; falls back to the line colour.
    """
    x = np.repeat(edges, 2)
    y = np.concatenate([[0], np.repeat(values, 2), [0]])
    color = kwargs.get('color', _C['data'])
    lw = kwargs.get('linewidth', kwargs.get('lw', 0.6))
    ls = kwargs.get('linestyle', kwargs.get('ls', '-'))
    label = kwargs.get('label', None)

    # filled region (hatched / translucent)
    if fill_alpha > 0 or hatch:
        fc = fill_color or color
        ax.fill(x, y, facecolor=fc, alpha=fill_alpha,
                hatch=hatch, edgecolor=color, linewidth=0.0, zorder=1)
        # hatching lines need their own edge colour
        if hatch:
            ax.fill(x, y, facecolor='none',
                    hatch=hatch, edgecolor=color, linewidth=0.0,
                    alpha=0.20, zorder=2)

    # outline
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls,
            label=label, zorder=3,
            path_effects=_LINE_SHADOW if ls == '-' else [])


def _textured_bar(ax, x, heights, width, color, hatch='....', alpha=0.65,
                  label=None, zorder=3):
    """Bar chart with stipple-dot texture and a soft drop shadow."""
    # shadow bars (slightly offset)
    dx = width * 0.025
    dy_frac = -0.008
    ylim = ax.get_ylim()
    dy = (ylim[1] - ylim[0]) * dy_frac if ylim[1] != ylim[0] else 0
    ax.bar(x + dx, heights, width, color=_C['shadow'], alpha=0.20,
           edgecolor='none', zorder=zorder - 1)
    # real bars
    bars = ax.bar(x, heights, width, color=color, alpha=alpha,
                  edgecolor='none', label=label, zorder=zorder)
    # overlay hatch
    ax.bar(x, heights, width, facecolor='none', edgecolor=color,
           hatch=hatch, linewidth=0, alpha=0.18, zorder=zorder + 1)
    return bars


# ===================================================================
# Tufte range-frame axes
# ===================================================================

def _range_frame(ax, x_data=None, y_data=None, pad_frac=0.02):
    """Trim spines so they span only the data range (Tufte range frame).

    Only acts on bottom and left spines (top/right already hidden).
    Gracefully degrades if data is empty.
    """
    for spine_name, data in [('bottom', x_data), ('left', y_data)]:
        if data is None or len(data) == 0:
            continue
        lo, hi = np.nanmin(data), np.nanmax(data)
        pad = (hi - lo) * pad_frac
        ax.spines[spine_name].set_bounds(lo - pad, hi + pad)


# ===================================================================
# Small helpers
# ===================================================================

def _annotate_inline(ax, x, y, text, color='k', fontsize=6, offset=(4, 2)):
    """Place a small inline annotation near a data point."""
    ax.annotate(text, (x, y), textcoords='offset points', xytext=offset,
                fontsize=fontsize, color=color, va='center')


def _label_shadow(ax, x_frac, y_frac, text, fontsize=8, color=_C['data'],
                  ha='right', va='top', **kw):
    """Annotate inside axes with a very faint text shadow for depth."""
    shadow_fx = [pe.withStroke(linewidth=1.5, foreground='white', alpha=0.9),
                 pe.Normal()]
    ax.text(x_frac, y_frac, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, ha=ha, va=va,
            path_effects=shadow_fx, **kw)


def _stamp_lumi(fig, lumi_label):
    """Place the luminosity / event-count annotation at the top of the figure."""
    if not lumi_label:
        return
    fig.text(0.5, 0.995, lumi_label, ha='center', va='top',
             fontsize=5, color=_C['sm'])


# ---------------------------------------------------------------------------
# 1. Spacetime interval distributions  (double-wide)
# ---------------------------------------------------------------------------

def plot_spacetime_distributions(truth_intervals, reco_intervals, suffix="",
                                 lumi_label=None):
    _apply_style()
    _ensure_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 / GOLDEN))
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    # --- helpers ---
    def _hist_vals(data, lo, hi, nbins=50):
        counts, edges = np.histogram(np.clip(data, lo, hi),
                                      bins=nbins, range=(lo, hi))
        return edges, counts

    # (a) signed ds
    ax = axes[0, 0]
    sd_t = np.array([iv['signed_ds_mm'] for iv in truth_intervals])
    sd_r = np.array([iv['signed_ds_mm'] for iv in reco_intervals])
    ds_all = np.concatenate([sd_t, sd_r])
    ds_lo, ds_hi = np.percentile(ds_all[np.isfinite(ds_all)], [1, 99])
    ds_lo, ds_hi = min(ds_lo, -1), max(ds_hi, 1)

    e, v = _hist_vals(sd_t, ds_lo, ds_hi)
    _step_hist(ax, e, v, color=_C['truth'], label='Truth',
               hatch='////', fill_alpha=0.06, fill_color=_C['truth'])
    e, v = _hist_vals(sd_r, ds_lo, ds_hi)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--', label='Reco')
    ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    ax.set_xlabel(r'Signed $\sqrt{|\Delta s^2|}$ [mm]')
    ax.set_ylabel('Events / Bin')
    ax.legend()
    _label_shadow(ax, 0.03, 0.93,
                  r'Spacelike $\leftarrow|\rightarrow$ Timelike',
                  fontsize=5.5, color=_C['light'], ha='left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (b) dr
    ax = axes[0, 1]
    dr_t = np.array([iv['dr_mm'] for iv in truth_intervals])
    dr_r = np.array([iv['dr_mm'] for iv in reco_intervals])
    dr_max = np.percentile(np.concatenate([dr_t, dr_r]), 99) * 1.1

    e, v = _hist_vals(dr_t, 0, dr_max)
    _step_hist(ax, e, v, color=_C['truth'], label='Truth',
               hatch='////', fill_alpha=0.06, fill_color=_C['truth'])
    e, v = _hist_vals(dr_r, 0, dr_max)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--', label='Reco')
    ax.set_xlabel(r'Spatial Separation $\Delta r$ [mm]')
    ax.set_ylabel('Events / Bin')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (c) v_signal
    ax = axes[1, 0]
    v_t = np.array([iv['v_signal_c'] for iv in truth_intervals])
    v_r = np.array([iv['v_signal_c'] for iv in reco_intervals])
    v_max = max(np.percentile(np.concatenate([
        np.clip(v_t, 0, 1e6), np.clip(v_r, 0, 1e6)]), 99), 10) * 1.1

    e, v = _hist_vals(v_t, 0, v_max, nbins=60)
    _step_hist(ax, e, v, color=_C['truth'], label='Truth',
               hatch='////', fill_alpha=0.06, fill_color=_C['truth'])
    e, v = _hist_vals(v_r, 0, v_max, nbins=60)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--', label='Reco')
    ax.axvline(1.0, color=_C['accent'], linewidth=0.5, label='$v = c$')
    ax.set_xlabel(r'Signal Speed $v_{\mathrm{signal}} / c$')
    ax.set_ylabel('Events / Bin')
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
    ymax = max(n_sl_t, n_tl_t, n_sl_r, n_tl_r)
    ax.set_ylim(0, ymax * 1.18)
    _textured_bar(ax, x - w / 2, [n_sl_t, n_tl_t], w, color=_C['truth'],
                  hatch='....', label='Truth')
    _textured_bar(ax, x + w / 2, [n_sl_r, n_tl_r], w, color=_C['reco'],
                  hatch='....', label='Reco')
    ax.set_xticks(x)
    ax.set_xticklabels(['Spacelike', 'Timelike'])
    ax.set_ylabel('Events')
    ax.legend()
    for i, (nt, nr) in enumerate(zip([n_sl_t, n_tl_t], [n_sl_r, n_tl_r])):
        ax.text(i - w / 2, nt + ymax * 0.02, str(nt), ha='center',
                fontsize=6, color=_C['truth'])
        ax.text(i + w / 2, nr + ymax * 0.02, str(nr), ha='center',
                fontsize=6, color=_C['reco'])

    _paper_bg(fig, axes)
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"spacetime_distributions{suffix}.png"))
    plt.close(fig)
    print(f"  Saved spacetime_distributions{suffix}.pdf")


# ---------------------------------------------------------------------------
# 2. Entanglement vs spacetime interval  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_entanglement_vs_spacetime(binned_results, bin_edges, xlabel, suffix="",
                                    lightlike_boundary=False, lumi_label=None):
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

    fig, axes = plt.subplots(3, 1, figsize=(COL1, COL1 * 1.55),
                              gridspec_kw={'height_ratios': [3, 3, 1]})
    fig.subplots_adjust(hspace=0.12, left=0.18, right=0.95, bottom=0.10, top=0.97)

    # (a) m12
    ax = axes[0]
    if len(m12[ok]) > 0:
        ylo = max(0, np.min(m12[ok] - m12e[ok]) - 0.5)
        yhi = max(3.0, np.max(m12[ok] + m12e[ok]) + 0.5)
        ax.set_ylim(ylo, yhi)
    ax.axhline(2.0, color=_C['sm'], linewidth=0.35, linestyle='--', zorder=1)
    ax.axhline(1.0, color=_C['bell'], linewidth=0.35, linestyle=':', zorder=1)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    _shadow_errorbar(ax, bc[ok], m12[ok], yerr=m12e[ok], xerr=hw[ok],
                     color=_C['data'], marker='o', ms=3.5, elinewidth=0.4)
    _label_shadow(ax, 0.97, 0.92, r'$m_{12}$', fontsize=8,
                  color=_C['data'], fontweight='bold')
    _label_shadow(ax, 0.97, 0.78, r'SM ($m_{12}=2$)', fontsize=5.5,
                  color=_C['sm'])
    _label_shadow(ax, 0.97, 0.15, 'Bell threshold', fontsize=5.5,
                  color=_C['bell'])
    ax.set_ylabel(r'$m_{12}$')
    ax.tick_params(labelbottom=False)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    # (b) Concurrence
    ax = axes[1]
    if len(conc[ok_c]) > 0:
        ylo = min(-0.2, np.min(conc[ok_c] - conce[ok_c]) - 0.15)
        yhi = max(1.3, np.max(conc[ok_c] + conce[ok_c]) + 0.15)
        ax.set_ylim(ylo, yhi)
    ax.axhline(1.0, color=_C['sm'], linewidth=0.35, linestyle='--', zorder=1)
    ax.axhline(0.0, color=_C['bell'], linewidth=0.35, linestyle=':', zorder=1)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    _shadow_errorbar(ax, bc[ok_c], conc[ok_c], yerr=conce[ok_c], xerr=hw[ok_c],
                     color=_C['accent'], marker='s', ms=2.8, elinewidth=0.4)
    _label_shadow(ax, 0.97, 0.92, 'Concurrence', fontsize=8,
                  color=_C['data'], fontweight='bold')
    _label_shadow(ax, 0.97, 0.78, r'SM ($\mathcal{C}=1$)', fontsize=5.5,
                  color=_C['sm'])
    ax.set_ylabel(r'$\mathcal{C}$')
    ax.tick_params(labelbottom=False)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    # (c) Event count — textured bars
    ax = axes[2]
    ax.set_ylim(0, max(nev) * 1.2 if max(nev) > 0 else 1)
    _textured_bar(ax, bc, nev, width=np.diff(bin_edges), color=_C['light'],
                  hatch='....', alpha=0.50, zorder=3)
    ax.set_ylabel('Events')
    ax.set_xlabel(xlabel)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    _paper_bg(fig, axes)
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"entanglement_vs_{suffix}.png"))
    plt.close(fig)
    print(f"  Saved entanglement_vs_{suffix}.pdf")


# ---------------------------------------------------------------------------
# 3. v_psi hypothesis overlay  (double-wide)
# ---------------------------------------------------------------------------

def plot_vpsi_overlay(binned_results_vs_v, bin_edges_v, v_psi_values,
                      lumi_label=None):
    _apply_style()
    _ensure_output_dir()

    bc = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    hw = np.diff(bin_edges_v) / 2
    m12  = np.array([r['m12'] for r in binned_results_vs_v])
    m12e = np.array([r['m12_err'] for r in binned_results_vs_v])
    ok   = ~np.isnan(m12)

    fig, ax = plt.subplots(figsize=(COL2, COL2 / GOLDEN / 1.3))

    if np.any(ok):
        yhi = max(3.5, np.nanmax(m12[ok] + m12e[ok]) + 0.5)
        ax.set_ylim(-0.3, yhi)

    # v_psi step models — thin grey lines with direct end labels
    v_fine = np.linspace(bin_edges_v[0], bin_edges_v[-1], 500)
    greys = np.linspace(0.45, 0.78, len(v_psi_values))
    for v_psi, g in zip(v_psi_values, greys):
        c = str(g)
        m12_model = np.where(v_fine <= v_psi, 2.0, 0.0)
        ax.plot(v_fine, m12_model, color=c, linewidth=0.45, zorder=2)
        ax.text(v_psi, 2.12, rf'${v_psi:.0f}c$', fontsize=5, color=c,
                ha='center', va='bottom')

    ax.axhline(1.0, color=_C['bell'], linewidth=0.35, linestyle=':', zorder=1)
    _label_shadow(ax, 0.99, 0.32, 'Bell threshold', fontsize=5.5,
                  color=_C['bell'], va='bottom')
    ax.axhline(2.0, color=_C['sm'], linewidth=0.3, linestyle='--', zorder=1)

    # Data points — with drop shadow
    _shadow_errorbar(ax, bc[ok], m12[ok], yerr=m12e[ok], xerr=hw[ok],
                     color=_C['data'], marker='o', ms=3.5, elinewidth=0.5,
                     label=r'Measured $m_{12}$')

    ax.set_xlabel(r'Signal Speed $v_{\mathrm{signal}} / c$')
    ax.set_ylabel(r'Horodecki Parameter $m_{12}$')
    ax.legend(loc='upper right', fontsize=7)
    if np.any(ok):
        _range_frame(ax, x_data=bc[ok], y_data=m12[ok])

    _paper_bg(fig, ax)
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_overlay.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_overlay.png"))
    plt.close(fig)
    print("  Saved vpsi_overlay.pdf")


# ---------------------------------------------------------------------------
# 4. v_psi exclusion curve  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_vpsi_exclusion(vpsi_scan_results, lumi_label=None):
    _apply_style()
    _ensure_output_dir()

    v_psi = np.array([r['v_psi'] for r in vpsi_scan_results])
    sig0  = np.array([r['sigma_vs_0'] for r in vpsi_scan_results])
    sig1  = np.array([r['sigma_vs_1'] for r in vpsi_scan_results])
    nev   = np.array([r['n_events'] for r in vpsi_scan_results])
    ok    = ~np.isnan(sig0) & (nev >= 10)

    fig, axes = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.15),
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.12, left=0.18, right=0.95, bottom=0.12, top=0.97)

    ax = axes[0]
    all_sig = np.concatenate([sig0[ok], sig1[ok]])
    if len(all_sig) > 0:
        ax.set_ylim(0, max(6, np.nanmax(all_sig) * 1.15))

    # Lines with shadow path effects
    ax.plot(v_psi[ok], sig0[ok], 'o-', color=_C['truth'], markersize=2.5,
            linewidth=0.5, label=r'Reject $m_{12}=0$',
            path_effects=_LINE_SHADOW, markeredgewidth=0)
    ax.plot(v_psi[ok], sig1[ok], 's-', color=_C['reco'], markersize=2.5,
            linewidth=0.5, label=r'Reject $m_{12}\leq 1$',
            path_effects=_LINE_SHADOW, markeredgewidth=0)
    ax.axhline(1.96, color=_C['accent'], linewidth=0.35,
               label='95% CL')
    ax.axhline(3.0, color=_C['light'], linewidth=0.3, linestyle='--')
    ax.axhline(5.0, color=_C['light'], linewidth=0.3, linestyle='-')
    # Sigma labels
    _label_shadow(ax, 0.98, 0, r'$3\sigma$', fontsize=5, color=_C['light'])
    ax.annotate(r'$3\sigma$', xy=(v_psi[ok][-1], 3.0),
                xytext=(4, 3), textcoords='offset points',
                fontsize=5, color=_C['light'], va='bottom')
    ax.annotate(r'$5\sigma$', xy=(v_psi[ok][-1], 5.0),
                xytext=(4, 3), textcoords='offset points',
                fontsize=5, color=_C['light'], va='bottom')
    ax.set_ylabel(r'Rejection Significance [$\sigma$]')
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.tick_params(labelbottom=False)

    # (b) event count
    ax2 = axes[1]
    _shadow_markers(ax2, v_psi[ok], nev[ok], color=_C['light'],
                    marker='o', ms=2.5)
    ax2.set_xlabel(r'Signal Speed $v_\psi / c$')
    ax2.set_ylabel('Events')
    ax2.set_xscale('log')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    _paper_bg(fig, axes)
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_exclusion.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "vpsi_exclusion.png"))
    plt.close(fig)
    print("  Saved vpsi_exclusion.pdf")


# ---------------------------------------------------------------------------
# 5. Correlation matrix heatmap  (single-column, square)
# ---------------------------------------------------------------------------

def plot_correlation_matrix(C, C_err, suffix="", lumi_label=None):
    _apply_style()
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.92))
    labels = ['$n$', '$r$', '$k$']

    vlim = max(abs(C).max(), 1.0)
    im = ax.imshow(C, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='equal',
                   interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=15, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$C_{ij}$', fontsize=7)
    cbar.outline.set_linewidth(0.3)

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
    ax.set_xlabel(r'$j$ ($\tau^+$ Decay)')
    ax.set_ylabel(r'$i$ ($\tau^-$ Decay)')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.3)
    ax.tick_params(top=True, right=True, direction='out', length=0)

    fig.patch.set_facecolor(_C['paper'])
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix{suffix}.png"))
    plt.close(fig)
    print(f"  Saved correlation_matrix{suffix}.pdf")


# ---------------------------------------------------------------------------
# 6. Acoplanarity distribution  (single-column)
# ---------------------------------------------------------------------------

def _cosine_model(phi, A, B):
    return A * (1.0 + B * np.cos(phi))


def plot_acoplanarity(delta_phi_arr, suffix="", lumi_label=None):
    _apply_style()
    _ensure_output_dir()

    n_bins = 25
    fig, ax = plt.subplots(figsize=(COL1, COL1 / GOLDEN))

    counts, edges = np.histogram(delta_phi_arr, bins=n_bins,
                                  range=(-np.pi, np.pi))
    bc = 0.5 * (edges[:-1] + edges[1:])
    bw = 2 * np.pi / n_bins
    errs = np.sqrt(np.maximum(counts, 1))

    # Set ylim before shadow_errorbar so offset computation works
    ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 1)
    ax.set_xlim(-np.pi, np.pi)

    # Data as shadow-lifted dots with error bars
    _shadow_errorbar(ax, bc, counts, yerr=errs,
                     color=_C['data'], marker='o', ms=2.5, elinewidth=0.35)

    # Cosine fit
    try:
        popt, pcov = curve_fit(_cosine_model, bc, counts,
                               p0=[np.mean(counts), -0.5],
                               sigma=errs, absolute_sigma=True)
        A_fit, B_fit = popt
        B_err = np.sqrt(pcov[1, 1])

        phi_fine = np.linspace(-np.pi, np.pi, 200)
        ax.plot(phi_fine, _cosine_model(phi_fine, *popt), color=_C['reco'],
                linewidth=0.55, zorder=3, path_effects=_LINE_SHADOW)
        _label_shadow(ax, 0.03, 0.95,
                      rf'$B = {B_fit:.3f} \pm {B_err:.3f}$',
                      fontsize=6.5, color=_C['reco'], ha='left')
    except RuntimeError:
        pass

    # SM expectation
    phi_fine = np.linspace(-np.pi, np.pi, 200)
    norm = len(delta_phi_arr) * bw / (2 * np.pi)
    ax.plot(phi_fine, norm * (1 - 0.5 * np.cos(phi_fine)),
            color=_C['sm'], linewidth=0.4, linestyle='--', zorder=2)
    _label_shadow(ax, 0.03, 0.85, r'SM ($B=-0.5$)', fontsize=5.5,
                  color=_C['sm'], ha='left')

    ax.set_xlabel(r'Acoplanarity $\Delta\phi$ [rad]')
    ax.set_ylabel('Events / Bin')
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    _paper_bg(fig, ax)
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"acoplanarity{suffix}.png"))
    plt.close(fig)
    print(f"  Saved acoplanarity{suffix}.pdf")


# ---------------------------------------------------------------------------
# 7. Acoplanarity vs signal speed  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_acoplanarity_vs_vsignal(acoplanarity_arr, v_signal_arr, v_edges,
                                 lumi_label=None):
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
        if n < 10:
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
    fig.subplots_adjust(hspace=0.12, left=0.18, right=0.95, bottom=0.12, top=0.97)

    # (a) B coefficient — plot all bins that have a fit result
    ax = axes[0]
    vb = B_vals[ok]
    if len(vb) > 0:
        ylo = min(-1.0, np.nanmin(vb - B_errs[ok]) - 0.2)
        yhi = max(0.5, np.nanmax(vb + B_errs[ok]) + 0.2)
        ax.set_ylim(ylo, yhi)
    if np.any(ok):
        _shadow_errorbar(ax, bc_v[ok], B_vals[ok], yerr=B_errs[ok],
                         xerr=hw_v[ok], color=_C['data'], marker='o',
                         ms=2.5, elinewidth=0.4)
    ax.axhline(-0.5, color=_C['sm'], linewidth=0.35, linestyle='--')
    ax.axhline(0.0, color=_C['light'], linewidth=0.25)
    _label_shadow(ax, 0.97, 0.08, r'SM ($B=-0.5$)', fontsize=5.5,
                  color=_C['sm'])
    ax.set_ylabel(r'Cosine Coefficient $B$')
    ax.tick_params(labelbottom=False)
    # Shared x-limits with event count panel
    ax.set_xlim(v_edges[0], v_edges[-1])

    # (b) event count — same bins as panel (a)
    ax2 = axes[1]
    ax2.set_ylim(0, max(n_events) * 1.2 if max(n_events) > 0 else 1)
    _textured_bar(ax2, bc_v, n_events, width=np.diff(v_edges),
                  color=_C['light'], hatch='....', alpha=0.50)
    ax2.set_xlabel(r'Signal Speed $v_{\mathrm{signal}} / c$')
    ax2.set_ylabel('Events')
    ax2.set_xlim(v_edges[0], v_edges[-1])
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    _paper_bg(fig, axes)
    _stamp_lumi(fig, lumi_label)
    fig.savefig(os.path.join(OUTPUT_DIR, "acoplanarity_vs_vsignal.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "acoplanarity_vs_vsignal.png"))
    plt.close(fig)
    print("  Saved acoplanarity_vs_vsignal.pdf")


# ---------------------------------------------------------------------------
# 8. Vertex comparison  (double-wide)
# ---------------------------------------------------------------------------

def plot_vertex_comparison(reco_results, lumi_label=None):
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
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    p99 = np.percentile(np.concatenate([truth_L, reco_L]), 99)
    lim = p99 * 1.2

    # (a) Scatter — symlog to reveal small-scale structure
    ax = axes[0, 0]
    ax.scatter(truth_L, reco_L, s=0.8, alpha=0.35, color=_C['data'],
               edgecolors='none', rasterized=True, zorder=3)
    ax.plot([1e-4, lim], [1e-4, lim], color=_C['sm'], linewidth=0.35,
            linestyle='--')
    ax.set_xlabel('Truth Decay Length [mm]')
    ax.set_ylabel('Reco Decay Length [mm]')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # (b) Residual with hatched fill
    ax = axes[0, 1]
    residual = reco_L - truth_L
    res_range = max(abs(np.percentile(residual, 2)),
                    abs(np.percentile(residual, 98))) * 1.2
    c, e = np.histogram(np.clip(residual, -res_range, res_range),
                        bins=50, range=(-res_range, res_range))
    _step_hist(ax, e, c, color=_C['truth'],
               hatch='////', fill_alpha=0.06, fill_color=_C['truth'])
    ax.set_xlabel('Reco $-$ Truth [mm]')
    ax.set_ylabel('Entries / Bin')
    _label_shadow(ax, 0.97, 0.92,
                  f'mean {np.mean(residual):.3f}\nRMS {np.std(residual):.3f}',
                  fontsize=5.5, color=_C['data'])

    # (c) Ratio distribution with hatched fill
    ax = axes[1, 0]
    safe = truth_L > 0.01
    ratio = reco_L[safe] / truth_L[safe]
    c, e = np.histogram(np.clip(ratio, 0, 5), bins=60, range=(0, 5))
    _step_hist(ax, e, c, color=_C['truth'],
               hatch='////', fill_alpha=0.06, fill_color=_C['truth'])
    ax.axvline(1.0, color=_C['sm'], linewidth=0.35, linestyle='--')
    ax.set_xlabel('Reco / Truth')
    ax.set_ylabel('Entries / Bin')
    _label_shadow(ax, 0.97, 0.92, f'median {np.median(ratio):.3f}',
                  fontsize=5.5, color=_C['data'])

    # (d) Ratio vs truth — symlog to match panel (a)
    ax = axes[1, 1]
    ax.scatter(truth_L[safe], ratio, s=0.8, alpha=0.35, color=_C['data'],
               edgecolors='none', rasterized=True, zorder=3)
    ax.axhline(1.0, color=_C['sm'], linewidth=0.35, linestyle='--')
    ax.set_xlabel('Truth Decay Length [mm]')
    ax.set_ylabel('Reco / Truth')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, lim)

    _paper_bg(fig, axes)
    _stamp_lumi(fig, lumi_label)
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

"""
Plotting module for the tau-tau entanglement analysis.

Style: Tufte-inspired art — maximise data-ink, remove chartjunk, direct labels,
drop shadows on data marks, light translucent fills, range-frame axes,
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
import matplotlib.colors as mcolors
import matplotlib.text as mtext
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from matplotlib.transforms import ScaledTranslation
from scipy.optimize import curve_fit
from scipy.special import erfc
import os
from config import OUTPUT_DIR

# Exact SM acoplanarity coefficient for the pi-pi channel:
# dGamma/dphi ~ 1 + B cos(phi) with B = -pi^2/16 ~= -0.617
# (C_nn = C_rr = 1 enter weighted by <sin theta>^2 = (pi/4)^2 per side).
B_SM = -np.pi**2 / 16.0
B_SM_LABEL = r'SM ($B=-\pi^2/16$)'

__all__ = [
    'plot_spacetime_distributions',
    'plot_entanglement_vs_spacetime',
    'plot_vpsi_overlay',
    'plot_vpsi_combined',
    'plot_vpsi_exclusion',
    'plot_correlation_matrix',
    'plot_acoplanarity',
    'plot_acoplanarity_vs_vsignal',
    'plot_acoplanarity_2d',
    'plot_vertex_comparison',
    'print_summary',
]

# ---------------------------------------------------------------------------
# PRL dimensions (inches)
# ---------------------------------------------------------------------------
COL1 = 3.375          # single-column width
COL2 = 7.0            # double-column width
GOLDEN = (1 + np.sqrt(5)) / 2  # ~1.618

# ---------------------------------------------------------------------------
# Palette — jewel tones on white: striking, print-safe, colorblind-aware
# ---------------------------------------------------------------------------
_C = {
    'data':     '#1a1a2e',   # midnight navy (near-black primary)
    'truth':    '#0d7377',   # deep teal
    'reco':     '#c23b48',   # ruby red
    'sm':       '#7c8594',   # blue-grey (reference lines)
    'bell':     '#c49030',   # burnished amber
    'accent':   '#2a7f62',   # sea-green
    'light':    '#aeb5c0',   # cool silver
    'paper':    '#ffffff',   # white — sits cleanly on PDF page
    'shadow':   '#c8c8c8',   # neutral light-grey shadow
}

# ---------------------------------------------------------------------------
# Centralised style constants — single source of truth for visual tuning
# ---------------------------------------------------------------------------
_S = {
    # Markers
    'data_ms': 3.5,              # primary data marker size
    'data_ms_large': 4.5,        # money-plot data marker size
    'data_elinewidth': 0.4,      # errorbar line width
    # Reference lines
    'ref_lw': 0.35,              # reference/threshold line width
    'ref_ls_sm': '--',           # SM prediction line style
    'ref_ls_bell': ':',          # Bell threshold line style
    # Fills & transparency
    'fill_alpha': 0.12,          # histogram fill alpha
    'shadow_alpha': 0.30,        # drop-shadow alpha
    'shadow_offset_pt': 0.5,     # shadow offset in points
    # Annotations
    'annot_fs': 7,               # small annotation font size
    'label_fs': 9,               # in-plot label font size
    # Hypothesis curves
    'hypo_lw': 0.45,             # v_psi hypothesis line width
    'hypo_label_fs': 7,          # v_psi label font size
}

# ---------------------------------------------------------------------------
# Tufte-inspired global style — letterpress-thin ruling
# ---------------------------------------------------------------------------
_TUFTE_RC = {
    # Font — STIX harmonises math and body text in a CM-like style
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'STIXGeneral', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 9,
    'mathtext.fontset': 'stix',
    # Axes
    'axes.linewidth': 0.35,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titlepad': 4,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#555555',
    # Ticks
    'xtick.major.size': 3,
    'xtick.minor.size': 1.5,
    'xtick.major.width': 0.3,
    'xtick.minor.width': 0.2,
    'xtick.direction': 'in',
    'xtick.labelsize': 8,
    'xtick.color': '#555555',
    'ytick.major.size': 3,
    'ytick.minor.size': 1.5,
    'ytick.major.width': 0.3,
    'ytick.minor.width': 0.2,
    'ytick.direction': 'in',
    'ytick.labelsize': 8,
    'ytick.color': '#555555',
    # Legend
    'legend.fontsize': 8,
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
    'savefig.pad_inches': 0.08,
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


def _invert_color(c):
    """Invert an RGB(A) colour, preserving alpha.

    Fully transparent colours are returned unchanged so invisible artists
    stay invisible.  This is a literal colour inversion (rgb -> 1-rgb), the
    same transform as toggling "invert colours" in a viewer.
    """
    r, g, b, a = mcolors.to_rgba(c)
    if a == 0:
        return (r, g, b, a)
    return (1.0 - r, 1.0 - g, 1.0 - b, a)


def _invert_color_array(arr):
    """Invert an (N, 4) RGBA colour array, leaving transparent rows alone."""
    arr = np.atleast_2d(np.array(arr, dtype=float))
    out = arr.copy()
    if out.shape[1] == 4:
        visible = out[:, 3] > 0
        out[visible, :3] = 1.0 - out[visible, :3]
    else:
        out[:, :3] = 1.0 - out[:, :3]
    return out


def _invert_path_effect(effect):
    """Invert the colours baked into a path effect (halo / drop shadow)."""
    gc = getattr(effect, '_gc', None)
    if isinstance(gc, dict) and gc.get('foreground') is not None:
        gc['foreground'] = _invert_color(gc['foreground'])
    if getattr(effect, '_shadow_color', None) is not None:
        effect._shadow_color = _invert_color(effect._shadow_color)


def _invert_figure_colors(fig):
    """Invert every artist colour in place for a dark-background variant.

    Figure and axes background patches are left untouched: they are made
    transparent at save time (``transparent=True``) so the plot drops onto
    any dark slide.  Everything else (text, lines, markers, spines, ticks,
    fills, path-effect halos and shadows) is flipped light-on-dark.
    """
    background = {id(fig.patch)}
    for ax in fig.get_axes():
        background.add(id(ax.patch))

    for obj in fig.findobj():
        pes = getattr(obj, 'get_path_effects', None)
        if callable(pes):
            for effect in (pes() or []):
                _invert_path_effect(effect)

        if isinstance(obj, mtext.Text):
            obj.set_color(_invert_color(obj.get_color()))
        elif isinstance(obj, mlines.Line2D):
            # Read every colour *before* mutating any of them: the marker
            # face/edge default to 'auto' (they track the line colour), so
            # once the line colour is flipped a later read would return the
            # already-inverted value and invert it a second time.
            line_c = obj.get_color()
            mfc = obj.get_markerfacecolor()
            mec = obj.get_markeredgecolor()
            obj.set_color(_invert_color(line_c))
            if mfc not in (None, 'none'):
                obj.set_markerfacecolor(_invert_color(mfc))
            if mec not in (None, 'none'):
                obj.set_markeredgecolor(_invert_color(mec))
        elif isinstance(obj, mpatches.Patch):
            if id(obj) in background:
                continue
            fc = obj.get_facecolor()
            if fc is not None and mcolors.to_rgba(fc)[3] > 0:
                obj.set_facecolor(_invert_color(fc))
            ec = obj.get_edgecolor()
            if ec is not None and mcolors.to_rgba(ec)[3] > 0:
                obj.set_edgecolor(_invert_color(ec))
        elif isinstance(obj, mcoll.Collection):
            fcs = obj.get_facecolor()
            if len(fcs):
                obj.set_facecolor(_invert_color_array(fcs))
            ecs = obj.get_edgecolor()
            if len(ecs):
                obj.set_edgecolor(_invert_color_array(ecs))


def _save(fig, name):
    """Save the figure and close it.

    Writes three files per plot:
      * ``<name>.pdf`` / ``<name>.png`` — white-background versions for the
        paper and quick viewing.
      * ``<name>_dark.pdf`` — colour-inverted, transparent-background version
        for talks on a dark slide background.
    """
    _ensure_output_dir()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"))

    # Dark-background talk variant: invert every colour, drop the background.
    _invert_figure_colors(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_dark.pdf"), transparent=True)

    plt.close(fig)
    print(f"  Saved {name}.pdf (+ {name}_dark.pdf)")


# ===================================================================
# Drop-shadow helpers — transform-based for consistent offsets
# ===================================================================

# Reusable path-effect lists
_MARKER_SHADOW = [
    pe.SimpleLineShadow(offset=(0.6, -0.6), shadow_color=_C['shadow'],
                        alpha=0.35, linewidth=0),
    pe.Normal(),
]

_LINE_SHADOW = [
    pe.SimpleLineShadow(offset=(_S['shadow_offset_pt'], -_S['shadow_offset_pt']),
                        shadow_color=_C['shadow'], alpha=0.25),
    pe.Normal(),
]


def _shadow_errorbar(ax, x, y, yerr=None, xerr=None, color=_C['data'],
                     marker='o', ms=None, elinewidth=None, label=None,
                     zorder=5, **kw):
    """Plot an errorbar with a soft drop shadow underneath.

    Uses a resolution-independent transform offset (in points) so the shadow
    is consistent regardless of data range or axis scale.
    """
    if ms is None:
        ms = _S['data_ms']
    if elinewidth is None:
        elinewidth = _S['data_elinewidth']

    fig = ax.get_figure()
    offset = ScaledTranslation(
        _S['shadow_offset_pt'] / 72, -_S['shadow_offset_pt'] / 72,
        fig.dpi_scale_trans)
    shadow_trans = ax.transData + offset

    # --- shadow layer (slightly offset, no label) ---
    ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                fmt=marker, color=_C['shadow'], markersize=ms,
                ecolor=_C['shadow'], elinewidth=elinewidth,
                capsize=0, alpha=_S['shadow_alpha'], zorder=zorder - 1,
                markeredgewidth=0, transform=shadow_trans, **kw)

    # --- real data layer ---
    container = ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                            fmt=marker, color=color, markersize=ms,
                            ecolor=color, elinewidth=elinewidth,
                            capsize=0, zorder=zorder, label=label,
                            markeredgewidth=0, **kw)
    return container


def _shadow_markers(ax, x, y, color=_C['data'], marker='o', ms=None,
                    label=None, zorder=5, **kw):
    """Scatter-style markers with a subtle drop shadow."""
    if ms is None:
        ms = _S['data_ms']

    fig = ax.get_figure()
    offset = ScaledTranslation(
        _S['shadow_offset_pt'] / 72, -_S['shadow_offset_pt'] / 72,
        fig.dpi_scale_trans)
    shadow_trans = ax.transData + offset

    ax.plot(x, y, marker=marker, linestyle='none',
            color=_C['shadow'], markersize=ms, alpha=_S['shadow_alpha'],
            zorder=zorder - 1, markeredgewidth=0, transform=shadow_trans, **kw)
    ax.plot(x, y, marker=marker, linestyle='none', color=color,
            markersize=ms, zorder=zorder, label=label,
            markeredgewidth=0, **kw)


# ===================================================================
# Paper ground
# ===================================================================

def _paper_bg(fig, ax_or_axes):
    """Set figure and axes to clean white for seamless PDF embedding."""
    fig.patch.set_facecolor(_C['paper'])
    axes = np.atleast_1d(ax_or_axes).ravel()
    for ax in axes:
        ax.set_facecolor(_C['paper'])


# ===================================================================
# Histogram and bar helpers
# ===================================================================

def _step_hist(ax, edges, values, fill_alpha=0.0, fill_color=None, **kwargs):
    """Draw a step histogram with optional light translucent fill.

    Parameters
    ----------
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

    # light solid fill
    if fill_alpha > 0:
        fc = fill_color or color
        ax.fill(x, y, facecolor=fc, alpha=fill_alpha,
                edgecolor='none', linewidth=0.0, zorder=1)

    # outline
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls,
            label=label, zorder=3,
            path_effects=_LINE_SHADOW if ls == '-' else [])


def _textured_bar(ax, x, heights, width, color, alpha=0.55,
                  label=None, zorder=3):
    """Bar chart with light solid fill and a soft drop shadow."""
    fig = ax.get_figure()
    offset = ScaledTranslation(
        0.4 / 72, -0.4 / 72, fig.dpi_scale_trans)

    # shadow bars (transform-offset)
    ax.bar(x, heights, width, color=_C['shadow'], alpha=0.20,
           edgecolor='none', zorder=zorder - 1,
           transform=ax.transData + offset)
    # real bars
    bars = ax.bar(x, heights, width, color=color, alpha=alpha,
                  edgecolor='none', label=label, zorder=zorder)
    # step outline around the perimeter (no internal bin dividers)
    x = np.asarray(x)
    heights = np.asarray(heights)
    w = np.broadcast_to(np.asarray(width), x.shape)
    edges = np.concatenate([x - w / 2, [x[-1] + w[-1] / 2]])
    sx = np.repeat(edges, 2)
    sy = np.concatenate([[0], np.repeat(heights, 2), [0]])
    ax.plot(sx, sy, color=color, linewidth=0.6, zorder=zorder + 1)
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


def _label_shadow(ax, x_frac, y_frac, text, fontsize=None, color=_C['data'],
                  ha='right', va='top', **kw):
    """Annotate inside axes with a very faint text shadow for depth."""
    if fontsize is None:
        fontsize = _S['label_fs']
    shadow_fx = [pe.withStroke(linewidth=1.5, foreground='white', alpha=0.9),
                 pe.Normal()]
    ax.text(x_frac, y_frac, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, ha=ha, va=va,
            path_effects=shadow_fx, **kw)


def _jewel_colormap():
    """Create a bespoke two-tone colormap from the jewel palette.

    Teal (negative) → white (zero) → ruby (positive).
    """
    return mcolors.LinearSegmentedColormap.from_list(
        'jewel', [_C['truth'], '#f5f5f5', _C['reco']], N=256)


# ---------------------------------------------------------------------------
# 1. Spacetime interval distributions  (double-wide)
# ---------------------------------------------------------------------------

def plot_spacetime_distributions(truth_intervals, reco_intervals, suffix=""):
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 / GOLDEN))
    fig.subplots_adjust(hspace=0.50, wspace=0.35)

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
    _step_hist(ax, e, v, color=_C['truth'],
               fill_alpha=_S['fill_alpha'], fill_color=_C['truth'])
    e, v = _hist_vals(sd_r, ds_lo, ds_hi)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--')
    ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    ax.set_xlabel(r'Signed $\sqrt{|\Delta s^2|}$ [mm]')
    ax.set_ylabel('Events / Bin')
    _label_shadow(ax, 0.97, 0.92, 'Truth', color=_C['truth'])
    _label_shadow(ax, 0.97, 0.80, 'Reco', color=_C['reco'])
    _label_shadow(ax, 0.03, 0.93,
                  r'Spacelike $\leftarrow|\rightarrow$ Timelike',
                  fontsize=_S['annot_fs'], color=_C['light'], ha='left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (b) dr
    ax = axes[0, 1]
    dr_t = np.array([iv['dr_mm'] for iv in truth_intervals])
    dr_r = np.array([iv['dr_mm'] for iv in reco_intervals])
    dr_max = np.percentile(np.concatenate([dr_t, dr_r]), 99) * 1.1

    e, v = _hist_vals(dr_t, 0, dr_max)
    _step_hist(ax, e, v, color=_C['truth'],
               fill_alpha=_S['fill_alpha'], fill_color=_C['truth'])
    e, v = _hist_vals(dr_r, 0, dr_max)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--')
    ax.set_xlabel(r'Spatial Separation $\Delta r$ [mm]')
    ax.set_ylabel('Events / Bin')
    _label_shadow(ax, 0.97, 0.92, 'Truth', color=_C['truth'])
    _label_shadow(ax, 0.97, 0.80, 'Reco', color=_C['reco'])
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (c) v_signal
    ax = axes[1, 0]
    v_t = np.array([iv['v_signal_c'] for iv in truth_intervals])
    v_r = np.array([iv['v_signal_c'] for iv in reco_intervals])
    v_max = max(np.percentile(np.concatenate([
        np.clip(v_t, 0, 1e6), np.clip(v_r, 0, 1e6)]), 99), 10) * 1.1

    e, v = _hist_vals(v_t, 0, v_max, nbins=60)
    _step_hist(ax, e, v, color=_C['truth'],
               fill_alpha=_S['fill_alpha'], fill_color=_C['truth'])
    e, v = _hist_vals(v_r, 0, v_max, nbins=60)
    _step_hist(ax, e, v, color=_C['reco'], linestyle='--')
    ax.axvline(1.0, color=_C['accent'], linewidth=0.5, zorder=0)
    ax.set_xlabel(r'$v_{\mathrm{sig}} / c$')
    ax.set_ylabel('Events / Bin')
    ax.set_yscale('log')
    _label_shadow(ax, 0.97, 0.92, 'Truth', color=_C['truth'])
    _label_shadow(ax, 0.97, 0.80, 'Reco', color=_C['reco'])
    ax.annotate('$v = c$', xy=(1.0, 0.75), xycoords=('data', 'axes fraction'),
                fontsize=_S['annot_fs'], color=_C['accent'],
                ha='right', xytext=(-3, 0), textcoords='offset points')

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
    _textured_bar(ax, x - w / 2, [n_sl_t, n_tl_t], w, color=_C['truth'])
    _textured_bar(ax, x + w / 2, [n_sl_r, n_tl_r], w, color=_C['reco'])
    ax.set_xticks(x)
    ax.set_xticklabels(['Spacelike', 'Timelike'])
    ax.set_ylabel('Events')
    for i, (nt, nr) in enumerate(zip([n_sl_t, n_tl_t], [n_sl_r, n_tl_r])):
        ax.text(i - w / 2, nt + ymax * 0.02, str(nt), ha='center',
                fontsize=_S['annot_fs'], color=_C['truth'])
        ax.text(i + w / 2, nr + ymax * 0.02, str(nr), ha='center',
                fontsize=_S['annot_fs'], color=_C['reco'])
    # Direct labels instead of legend
    ax.text(0 - w / 2, ymax * 0.85, 'Truth', ha='center',
            fontsize=_S['annot_fs'], color=_C['truth'])
    ax.text(0 + w / 2, ymax * 0.85, 'Reco', ha='center',
            fontsize=_S['annot_fs'], color=_C['reco'])

    _paper_bg(fig, axes)
    fig.align_ylabels()

    _save(fig, f"spacetime_distributions{suffix}")


# ---------------------------------------------------------------------------
# 2. Entanglement vs spacetime interval  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_entanglement_vs_spacetime(binned_results, bin_edges, xlabel, suffix="",
                                    lightlike_boundary=False, binned_truth=None):
    _apply_style()

    bc = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hw = np.diff(bin_edges) / 2

    # Display the bootstrap-bias-corrected m12 when available: the raw
    # eigenvalue-sum estimator carries a positive noise bias ~78.5/N.
    m12  = np.array([r.get('m12_bc', r['m12']) for r in binned_results])
    m12e = np.array([r['m12_err'] for r in binned_results])
    # Concurrence witness W (linear, unbiased; concurrence >= W) preferred
    # over the physicalised-rho concurrence for binned displays.
    conc  = np.array([r.get('witness_W', r['concurrence'])
                      if not np.isnan(r.get('witness_W', np.nan))
                      else r['concurrence'] for r in binned_results])
    conce = np.array([r.get('witness_W_err', r['concurrence_err'])
                      if not np.isnan(r.get('witness_W_err', np.nan))
                      else r['concurrence_err'] for r in binned_results])
    nev   = np.array([r.get('n_events', 0) for r in binned_results])
    ok    = ~np.isnan(m12)
    ok_c  = ~np.isnan(conc)

    fig, axes = plt.subplots(3, 1, figsize=(COL1, COL1 * 1.55),
                              gridspec_kw={'height_ratios': [3, 3, 1],
                                           'hspace': 0.08})
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.10, top=0.97)

    # m12 — adaptive top so no point is silently clipped
    ax = axes[0]
    ytop = 3.5
    if np.any(ok):
        ytop = min(max(3.5, np.nanmax(m12[ok]) + 0.4), 8.0)
    ax.set_ylim(0, ytop)
    ax.axhline(2.0, color=_C['sm'], linewidth=_S['ref_lw'],
               linestyle=_S['ref_ls_sm'], zorder=1)
    ax.axhline(1.0, color=_C['bell'], linewidth=_S['ref_lw'],
               linestyle=_S['ref_ls_bell'], zorder=1)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    _shadow_errorbar(ax, bc[ok], m12[ok], yerr=m12e[ok], xerr=hw[ok],
                     color=_C['data'], marker='o', ms=_S['data_ms'])
    _label_shadow(ax, 0.97, 0.92, r'$m_{12}$',
                  color=_C['data'], fontweight='bold')
    _label_shadow(ax, 0.97, 0.55, r'SM ($m_{12}=2$)',
                  fontsize=_S['annot_fs'], color=_C['sm'])
    _label_shadow(ax, 0.97, 0.26, 'Bell threshold',
                  fontsize=_S['annot_fs'], color=_C['bell'])
    ax.set_ylabel(r'$m_{12}$')
    ax.tick_params(labelbottom=False)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    # Concurrence
    ax = axes[1]
    if len(conc[ok_c]) > 0:
        ylo = min(-0.2, np.min(conc[ok_c] - conce[ok_c]) - 0.15)
        yhi = max(1.3, np.max(conc[ok_c] + conce[ok_c]) + 0.15)
        ax.set_ylim(ylo, yhi)
    ax.axhline(1.0, color=_C['sm'], linewidth=_S['ref_lw'],
               linestyle=_S['ref_ls_sm'], zorder=1)
    ax.axhline(0.0, color=_C['bell'], linewidth=_S['ref_lw'],
               linestyle=_S['ref_ls_bell'], zorder=1)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    # Truth-level validation: open markers (flat at W = 1; any trend in
    # the reconstructed witness is dilution, not decoherence)
    if binned_truth is not None:
        w_t = np.array([r.get('witness_W', np.nan) for r in binned_truth])
        we_t = np.array([r.get('witness_W_err', np.nan) for r in binned_truth])
        ok_t = ~np.isnan(w_t)
        ax.errorbar(bc[ok_t], w_t[ok_t], yerr=we_t[ok_t],
                    fmt='o', markersize=2.6, markerfacecolor='white',
                    markeredgecolor=_C['truth'], markeredgewidth=0.5,
                    ecolor=_C['truth'], elinewidth=0.3, capsize=0,
                    zorder=4, alpha=0.85)
        _label_shadow(ax, 0.97, 0.64, 'truth (open)',
                      fontsize=_S['annot_fs'], color=_C['truth'])
    _shadow_errorbar(ax, bc[ok_c], conc[ok_c], yerr=conce[ok_c], xerr=hw[ok_c],
                     color=_C['accent'], marker='s', ms=2.8)
    _label_shadow(ax, 0.97, 0.92, r'Witness $\mathcal{W}\leq\mathcal{C}$',
                  color=_C['data'], fontweight='bold')
    _label_shadow(ax, 0.97, 0.78, r'SM ($\mathcal{W}=1$)',
                  fontsize=_S['annot_fs'], color=_C['sm'])
    ax.set_ylabel(r'$\mathcal{W}$')
    ax.tick_params(labelbottom=False)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    # Event count — textured bars
    ax = axes[2]
    ax.set_ylim(0, max(nev) * 1.2 if max(nev) > 0 else 1)
    _textured_bar(ax, bc, nev, width=np.diff(bin_edges), color=_C['light'],
                  alpha=0.50, zorder=3)
    ax.set_ylabel('Events')
    ax.set_xlabel(xlabel)
    if lightlike_boundary:
        ax.axvline(0, color=_C['light'], linewidth=0.3, zorder=0)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    # Symlog x-axis when quantile bins concentrate near the lightlike
    # boundary (most events have |ds| << the range): keeps all bins legible.
    if lightlike_boundary and len(bin_edges) > 2:
        widths = np.diff(bin_edges)
        if np.max(widths) > 20 * np.min(widths):
            linthresh = max(np.min(widths[widths > 0]), 1e-3)
            for a in axes:
                a.set_xscale('symlog', linthresh=linthresh, linscale=0.8)

    _paper_bg(fig, axes)
    fig.align_ylabels(axes)

    _save(fig, f"entanglement_vs_{suffix}")


# ---------------------------------------------------------------------------
# 3. v_psi hypothesis overlay  (double-wide) — the money plot
# ---------------------------------------------------------------------------

def _hypothesis_colors(n):
    """Generate a cool-blue → warm-amber gradient for hypothesis curves."""
    cool = np.array(mcolors.to_rgb(_C['truth']))   # teal
    warm = np.array(mcolors.to_rgb(_C['bell']))     # amber
    return [mcolors.to_hex(cool + t * (warm - cool))
            for t in np.linspace(0, 1, n)]


def plot_vpsi_overlay(binned_results_vs_v, bin_edges_v, v_psi_values,
                      sigma_v_frac=0.0):
    """Plot m12 vs signal speed with v_psi hypothesis curves.

    Parameters
    ----------
    sigma_v_frac : float
        Fractional resolution on v_signal (sigma_v / v).  When > 0 the
        sharp step-function hypothesis curves are convolved with the
        detector resolution, producing smooth error-function turn-offs.
    """
    _apply_style()

    bc = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    hw = np.diff(bin_edges_v) / 2
    m12  = np.array([r.get('m12_bc', r['m12']) for r in binned_results_vs_v])
    m12e = np.array([r['m12_err'] for r in binned_results_vs_v])
    ok   = ~np.isnan(m12)

    fig, ax = plt.subplots(figsize=(COL2, COL2 / GOLDEN / 1.3))

    # Dynamic y-range: extend to show data error bars while keeping
    # hypothesis curves readable.  Cap at 10 so curves aren't squished.
    ylim_top = 3.5
    if np.any(ok):
        # Show at least the lower 1-sigma bound of every point
        need = np.max(m12[ok] - m12e[ok]) + 0.5
        ylim_top = min(max(ylim_top, need), 10.0)
    ax.set_ylim(-0.3, ylim_top)

    # x-range: extend to cover all hypothesis curves with some padding
    x_lo = bin_edges_v[0]
    x_hi = max(bin_edges_v[-1], max(v_psi_values) * 1.15) if len(v_psi_values) > 0 else bin_edges_v[-1]
    ax.set_xlim(x_lo, x_hi)

    # Shaded Bell-local band: m12 < 1 region
    ax.axhspan(-0.3, 1.0, facecolor=_C['bell'], alpha=0.05, zorder=0)

    # v_psi hypothesis models — teal→amber gradient with direct end labels
    v_fine = np.linspace(x_lo, x_hi, 500)
    hypo_colors = _hypothesis_colors(len(v_psi_values))
    for v_psi, hc in zip(v_psi_values, hypo_colors):
        if sigma_v_frac > 0:
            sigma_v = sigma_v_frac * v_fine.clip(1e-6)
            m12_model = 2.0 * 0.5 * erfc(
                (v_fine - v_psi) / (np.sqrt(2) * sigma_v))
        else:
            m12_model = np.where(v_fine <= v_psi, 2.0, 0.0)
        ax.plot(v_fine, m12_model, color=hc,
                linewidth=_S['hypo_lw'], zorder=2)
        # Label at the midpoint of the transition, offset right
        ax.text(v_psi * 1.08, 1.05, rf'${v_psi:g}c$',
                fontsize=_S['hypo_label_fs'], color=hc,
                ha='left', va='bottom')

    ax.axhline(1.0, color=_C['bell'], linewidth=_S['ref_lw'],
               linestyle=_S['ref_ls_bell'], zorder=1)
    ax.annotate('Bell threshold', xy=(x_hi, 1.0),
                xytext=(-3, 2), textcoords='offset points',
                fontsize=_S['annot_fs'], color=_C['bell'],
                ha='right', va='bottom')
    ax.axhline(2.0, color=_C['sm'], linewidth=0.3,
               linestyle=_S['ref_ls_sm'], zorder=1)
    ax.annotate(r'SM ($m_{12}=2$)', xy=(x_hi, 2.0),
                xytext=(-3, 2), textcoords='offset points',
                fontsize=_S['annot_fs'], color=_C['sm'],
                ha='right', va='bottom')

    # Data points — prominent markers on the money plot
    _shadow_errorbar(ax, bc[ok], m12[ok], yerr=m12e[ok], xerr=hw[ok],
                     color=_C['data'], marker='o', ms=_S['data_ms_large'],
                     elinewidth=0.5)

    # For points above the chart, add small upward-pointing triangles
    # at the top edge to indicate "continues above".
    if np.any(ok):
        above = m12[ok] > ylim_top
        if np.any(above):
            ax.plot(bc[ok][above],
                    np.full(np.sum(above), ylim_top - 0.15),
                    marker='^', linestyle='none', color=_C['data'],
                    markersize=_S['data_ms_large'], markeredgewidth=0,
                    zorder=6, clip_on=False)

    ax.set_xlabel(r'$v_{\mathrm{sig}} / c$')
    ax.set_ylabel(r'$m_{12}$')

    _paper_bg(fig, ax)

    _save(fig, "vpsi_overlay")


# ---------------------------------------------------------------------------
# 3b. Combined B + m12 vs v_psi  (single-column, two panels)
# ---------------------------------------------------------------------------

def plot_vpsi_combined(binned_results_vs_v, bin_edges_v,
                       acoplanarity_arr, v_signal_arr, v_edges,
                       v_psi_values, sigma_v_frac=0.0):
    """Two-panel figure: B coefficient (top) and m12 (bottom) vs v_psi/c.

    Single-column width with a shared x-axis.
    """
    _apply_style()

    # --- Compute B(v) from acoplanarity data ---
    n_bins_v = len(v_edges) - 1
    bc_v = 0.5 * (v_edges[:-1] + v_edges[1:])
    hw_v = np.diff(v_edges) / 2
    B_vals, B_errs = [], []
    for b in range(n_bins_v):
        mask = (v_signal_arr >= v_edges[b]) & (v_signal_arr < v_edges[b + 1])
        n = np.sum(mask)
        if n < 10:
            B_vals.append(np.nan); B_errs.append(np.nan)
            continue
        dphi = acoplanarity_arr[mask]
        c, e = np.histogram(dphi, bins=15, range=(-np.pi, np.pi))
        cbc = 0.5 * (e[:-1] + e[1:])
        cerr = np.sqrt(np.maximum(c, 1))
        try:
            popt, pcov = curve_fit(_cosine_model, cbc, c,
                                   p0=[np.mean(c), B_SM],
                                   sigma=cerr, absolute_sigma=True)
            B_vals.append(popt[1])
            B_errs.append(np.sqrt(pcov[1, 1]))
        except RuntimeError:
            B_vals.append(np.nan); B_errs.append(np.nan)
    B_vals = np.array(B_vals)
    B_errs = np.array(B_errs)
    ok_B = ~np.isnan(B_vals)

    # --- m12 data (bias-corrected) ---
    bc_m = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    hw_m = np.diff(bin_edges_v) / 2
    m12  = np.array([r.get('m12_bc', r['m12']) for r in binned_results_vs_v])
    m12e = np.array([r['m12_err'] for r in binned_results_vs_v])
    ok_m = ~np.isnan(m12)

    # --- Shared x-range ---
    x_lo = min(bin_edges_v[0], v_edges[0])
    x_hi = max(bin_edges_v[-1], v_edges[-1])
    if len(v_psi_values) > 0:
        x_hi = max(x_hi, max(v_psi_values) * 1.15)

    # --- Figure: two rows, shared x ---
    fig, (ax_B, ax_m) = plt.subplots(
        2, 1, figsize=(COL1, COL1 * 1.4),
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.06},
        sharex=True)

    # ---- Hypothesis curves (shared colours, drawn on both panels) ----
    v_fine = np.linspace(x_lo, x_hi, 500)
    hypo_colors = _hypothesis_colors(len(v_psi_values))

    for v_psi, hc in zip(v_psi_values, hypo_colors):
        if sigma_v_frac > 0:
            sigma_v = sigma_v_frac * v_fine.clip(1e-6)
            frac = 0.5 * erfc(
                (v_fine - v_psi) / (np.sqrt(2) * sigma_v))
        else:
            frac = np.where(v_fine <= v_psi, 1.0, 0.0)

        # B panel
        ax_B.plot(v_fine, B_SM * frac, color=hc,
                  linewidth=_S['hypo_lw'], zorder=2)
        if v_psi <= x_hi:
            # Nudge the rightmost label a bit more to avoid clipping
            nudge = 1.15 if v_psi == max(v_psi_values) else 1.08
            ax_B.text(v_psi * nudge, -0.18, rf'${v_psi:g}c$',
                      fontsize=_S['hypo_label_fs'], color=hc,
                      ha='left', va='top', clip_on=True)

        # m12 panel
        ax_m.plot(v_fine, 2.0 * frac, color=hc,
                  linewidth=_S['hypo_lw'], zorder=2)

    # ---- Top panel: B coefficient ----
    vb = B_vals[ok_B]
    if len(vb) > 0:
        ylo_B = min(-1.0, np.nanmin(vb - B_errs[ok_B]) - 0.2)
    else:
        ylo_B = -1.0
    ax_B.set_ylim(ylo_B, 0.2)
    ax_B.set_xlim(x_lo, x_hi)

    if np.any(ok_B):
        _shadow_errorbar(ax_B, bc_v[ok_B], B_vals[ok_B], yerr=B_errs[ok_B],
                         xerr=hw_v[ok_B], color=_C['data'], marker='o',
                         ms=_S['data_ms_large'])
    ax_B.axhline(B_SM, color=_C['sm'], linewidth=_S['ref_lw'],
                 linestyle=_S['ref_ls_sm'], zorder=1)
    ax_B.axhline(0.0, color=_C['light'], linewidth=0.25, zorder=1)
    ax_B.annotate(B_SM_LABEL, xy=(x_hi, B_SM),
                  xytext=(-3, 2), textcoords='offset points',
                  fontsize=_S['annot_fs'], color=_C['sm'],
                  ha='right', va='bottom')
    ax_B.set_ylabel(r'Cosine Coefficient $B$')

    # ---- Bottom panel: m12 ----
    ylim_top = 4.0
    if np.any(ok_m):
        need = np.max(m12[ok_m] - m12e[ok_m]) + 0.5
        ylim_top = min(max(ylim_top, need), 10.0)
    ax_m.set_ylim(-0.3, ylim_top)

    ax_m.axhspan(-0.3, 1.0, facecolor=_C['bell'], alpha=0.05, zorder=0)
    ax_m.axhline(1.0, color=_C['bell'], linewidth=_S['ref_lw'],
                 linestyle=_S['ref_ls_bell'], zorder=1)
    ax_m.annotate('Bell threshold', xy=(x_hi, 1.0),
                  xytext=(-3, 2), textcoords='offset points',
                  fontsize=_S['annot_fs'], color=_C['bell'],
                  ha='right', va='bottom')
    ax_m.axhline(2.0, color=_C['sm'], linewidth=0.3,
                 linestyle=_S['ref_ls_sm'], zorder=1)
    ax_m.annotate(r'SM ($m_{12}=2$)', xy=(x_hi, 2.0),
                  xytext=(-3, 2), textcoords='offset points',
                  fontsize=_S['annot_fs'], color=_C['sm'],
                  ha='right', va='bottom')

    _shadow_errorbar(ax_m, bc_m[ok_m], m12[ok_m], yerr=m12e[ok_m],
                     xerr=hw_m[ok_m], color=_C['data'], marker='o',
                     ms=_S['data_ms_large'], elinewidth=0.5)
    if np.any(ok_m):
        above = m12[ok_m] > ylim_top
        if np.any(above):
            ax_m.plot(bc_m[ok_m][above],
                      np.full(np.sum(above), ylim_top - 0.15),
                      marker='^', linestyle='none', color=_C['data'],
                      markersize=_S['data_ms_large'], markeredgewidth=0,
                      zorder=6, clip_on=False)

    ax_m.set_xlabel(r'$v_{\mathrm{sig}} / c$')
    ax_m.set_ylabel(r'$m_{12}$')

    # ILD resolution label — upper-right of top panel
    ax_B.text(0.97, 0.93, 'ILD resolutions',
              transform=ax_B.transAxes, fontsize=_S['annot_fs'],
              color=_C['sm'], ha='right', va='top', style='italic',
              path_effects=[pe.withStroke(linewidth=1.5, foreground='white',
                                          alpha=0.9),
                            pe.Normal()])

    _paper_bg(fig, [ax_B, ax_m])
    fig.align_ylabels([ax_B, ax_m])

    _save(fig, "vpsi_combined")


# ---------------------------------------------------------------------------
# 4. v_psi exclusion curve  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_vpsi_exclusion(vpsi_scan_results):
    _apply_style()

    v_psi = np.array([r['v_psi'] for r in vpsi_scan_results])
    sig0  = np.array([r['sigma_vs_0'] for r in vpsi_scan_results])
    sig1  = np.array([r['sigma_vs_1'] for r in vpsi_scan_results])
    nev   = np.array([r['n_events'] for r in vpsi_scan_results])
    ok    = ~np.isnan(sig0) & (nev >= 6)

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(COL1, COL1 * 1.05),
        gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08},
        sharex=True)

    all_sig = np.concatenate([sig0[ok], sig1[ok]])
    ymax_sig = max(6, np.nanmax(all_sig) * 1.15) if len(all_sig) > 0 else 6
    ax.set_ylim(0, ymax_sig)

    # Subtle exclusion shading above 95% CL
    ax.axhspan(1.96, ymax_sig, facecolor=_C['accent'], alpha=0.04, zorder=0)

    # Extend curves to zero beyond the last measured point to show
    # that we can't make any statement at very high speeds.
    v_plot0, s_plot0 = v_psi[ok], sig0[ok]
    v_plot1, s_plot1 = v_psi[ok], sig1[ok]
    if np.any(ok):
        # Find the first v_psi beyond the data with too few events
        all_beyond = v_psi[~ok & (v_psi > v_psi[ok][-1])]
        v_zero = all_beyond[0] if len(all_beyond) > 0 else v_psi[ok][-1] * 2
        v_plot0 = np.append(v_plot0, v_zero)
        s_plot0 = np.append(s_plot0, 0.0)
        v_plot1 = np.append(v_plot1, v_zero)
        s_plot1 = np.append(s_plot1, 0.0)

    # Data lines
    ax.plot(v_plot0, s_plot0, 'o-', color=_C['truth'], markersize=2.5,
            linewidth=0.5,
            path_effects=_LINE_SHADOW, markeredgewidth=0, zorder=5)
    ax.plot(v_plot1, s_plot1, 's-', color=_C['reco'], markersize=2.5,
            linewidth=0.5,
            path_effects=_LINE_SHADOW, markeredgewidth=0, zorder=5)
    ax.axhline(1.96, color=_C['accent'], linewidth=_S['ref_lw'], zorder=1)
    ax.axhline(3.0, color=_C['light'], linewidth=0.3, linestyle='--',
               zorder=1)
    ax.axhline(5.0, color=_C['light'], linewidth=0.3, linestyle='-',
               zorder=1)

    # Direct line labels — anchored to each curve, offset clear of it.
    if np.any(ok):
        n_ok = int(np.sum(ok))
        # "Reject no correlation" (teal, upper curve): anchor to the curve
        # past its steepest drop, where it has started to flatten, and set
        # the label above-right of that point so it reads next to the curve
        # without sitting on top of it.
        mid0 = max(0, min(n_ok - 1, n_ok // 2))
        ax.annotate('Reject no correlation',
                    xy=(v_plot0[mid0], s_plot0[mid0]),
                    xytext=(8, 10), textcoords='offset points',
                    fontsize=_S['annot_fs'], color=_C['truth'],
                    ha='left', va='bottom')
        # "Reject CHSH S<=2" (ruby, lower curve): anchor to the ruby curve
        # and lift the label into the open wedge between the two curves.
        mid1 = max(0, min(n_ok - 1, n_ok // 2 - 1))
        ax.annotate(r'Reject CHSH $S\leq 2$',
                    xy=(v_plot1[mid1], s_plot1[mid1]),
                    xytext=(0, 13), textcoords='offset points',
                    fontsize=_S['annot_fs'], color=_C['reco'],
                    ha='center', va='bottom')
    # Threshold labels — small, tucked at the right edge against their lines.
    thr_fs = _S['annot_fs'] - 2
    ax.annotate('95% CL', xy=(0.995, 1.96),
                xycoords=('axes fraction', 'data'),
                xytext=(0, -2), textcoords='offset points',
                fontsize=thr_fs, color=_C['accent'],
                ha='right', va='top')
    ax.annotate(r'$3\sigma$', xy=(0.995, 3.0),
                xycoords=('axes fraction', 'data'),
                xytext=(0, 1), textcoords='offset points',
                fontsize=thr_fs, color=_C['light'], ha='right', va='bottom')
    ax.annotate(r'$5\sigma$', xy=(0.995, 5.0),
                xycoords=('axes fraction', 'data'),
                xytext=(0, 1), textcoords='offset points',
                fontsize=thr_fs, color=_C['light'], ha='right', va='bottom')
    ax.set_ylabel(r'Rejection Significance [$\sigma$]')
    ax.set_xscale('log')
    ax.tick_params(labelbottom=False)

    # --- Bottom panel: surviving event counts + exact Pareto reference ---
    pos = nev > 0
    ax_n.plot(v_psi[pos], nev[pos], 'o', color=_C['data'],
              markersize=2.2, markeredgewidth=0, zorder=5)
    if np.any(pos):
        # Pareto prediction P(v_sig > v) = beta/v, anchored at the
        # first scan point: N(v) = N(v1) * v1 / v.
        v1, n1 = v_psi[pos][0], nev[pos][0]
        v_fine = np.geomspace(v_psi[pos][0], max(v_psi[pos][-1], v1 * 2), 200)
        ax_n.plot(v_fine, n1 * v1 / v_fine, color=_C['sm'],
                  linewidth=_S['ref_lw'], linestyle='--', zorder=2)
    ax_n.set_yscale('log')
    ax_n.set_xscale('log')
    ax_n.set_ylabel(r'$N(v_{\mathrm{sig}}>v_\psi)$')
    ax_n.set_xlabel(r'$v_\psi / c$')

    _paper_bg(fig, [ax, ax_n])
    fig.align_ylabels([ax, ax_n])

    _save(fig, "vpsi_exclusion")


# ---------------------------------------------------------------------------
# 5. Correlation matrix heatmap  (single-column, square)
# ---------------------------------------------------------------------------

def plot_correlation_matrix(C, C_err, suffix=""):
    _apply_style()

    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.92))
    labels = ['$n$', '$r$', '$k$']

    vlim = max(abs(C).max(), 1.0)
    cmap = _jewel_colormap()
    im = ax.imshow(C, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='equal',
                   interpolation='nearest')

    # White grid lines between cells for a modern design feel
    for i in range(4):
        ax.axhline(i - 0.5, color='white', linewidth=1.2, zorder=2)
        ax.axvline(i - 0.5, color='white', linewidth=1.2, zorder=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=15, pad=0.04)
    cbar.ax.tick_params(labelsize=_S['annot_fs'])
    cbar.set_label(r'$C_{ij}$', fontsize=_S['annot_fs'])
    cbar.outline.set_linewidth(0.3)

    for i in range(3):
        for j in range(3):
            txt = f'{C[i,j]:+.2f}\n$\\pm${C_err[i,j]:.2f}'
            clr = 'white' if abs(C[i, j]) > 0.5 * vlim else _C['data']
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color=clr,
                    path_effects=[pe.withStroke(linewidth=1.0,
                                               foreground='white',
                                               alpha=0.5 if clr != 'white' else 0),
                                  pe.Normal()])

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

    _save(fig, f"correlation_matrix{suffix}")


# ---------------------------------------------------------------------------
# 6. Acoplanarity distribution  (single-column)
# ---------------------------------------------------------------------------

def _cosine_model(phi, A, B):
    return A * (1.0 + B * np.cos(phi))


def plot_acoplanarity(delta_phi_arr, suffix=""):
    _apply_style()

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
                               p0=[np.mean(counts), B_SM],
                               sigma=errs, absolute_sigma=True)
        A_fit, B_fit = popt
        B_err = np.sqrt(pcov[1, 1])

        phi_fine = np.linspace(-np.pi, np.pi, 200)
        ax.plot(phi_fine, _cosine_model(phi_fine, *popt), color=_C['reco'],
                linewidth=0.55, zorder=2, path_effects=_LINE_SHADOW)
        _label_shadow(ax, 0.03, 0.95,
                      rf'$B = {B_fit:.3f} \pm {B_err:.3f}$',
                      fontsize=_S['annot_fs'], color=_C['reco'], ha='left')
    except RuntimeError:
        pass

    # SM expectation
    phi_fine = np.linspace(-np.pi, np.pi, 200)
    norm = len(delta_phi_arr) * bw / (2 * np.pi)
    ax.plot(phi_fine, norm * (1 + B_SM * np.cos(phi_fine)),
            color=_C['sm'], linewidth=0.4, linestyle='--', zorder=1)
    _label_shadow(ax, 0.03, 0.85, B_SM_LABEL,
                  fontsize=_S['annot_fs'], color=_C['sm'], ha='left')

    ax.set_xlabel(r'Acoplanarity $\Delta\phi$')
    ax.set_ylabel('Events / Bin')

    # Named pi-ticks instead of decimal numbers
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    _paper_bg(fig, ax)

    _save(fig, f"acoplanarity{suffix}")


# ---------------------------------------------------------------------------
# 7. Acoplanarity vs signal speed  (single-column, tall)
# ---------------------------------------------------------------------------

def plot_acoplanarity_vs_vsignal(acoplanarity_arr, v_signal_arr, v_edges,
                                 v_psi_values=None, sigma_v_frac=0.0):
    _apply_style()

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
                                   p0=[np.mean(c), B_SM],
                                   sigma=cerr, absolute_sigma=True)
            B_vals.append(popt[1])
            B_errs.append(np.sqrt(pcov[1, 1]))
        except RuntimeError:
            B_vals.append(np.nan); B_errs.append(np.nan)

    B_vals = np.array(B_vals)
    B_errs = np.array(B_errs)
    n_events = np.array(n_events)
    ok = ~np.isnan(B_vals)

    fig, ax = plt.subplots(figsize=(COL1, COL1 / GOLDEN))

    vb = B_vals[ok]
    if len(vb) > 0:
        ylo = min(-1.0, np.nanmin(vb - B_errs[ok]) - 0.2)
        yhi = max(0.5, np.nanmax(vb + B_errs[ok]) + 0.2)
        ax.set_ylim(ylo, yhi)

    # v_psi hypothesis models for B: entangled B=-0.5, separable B=0
    x_lo_v, x_hi_v = v_edges[0], v_edges[-1]
    ax.set_xlim(x_lo_v, x_hi_v)
    if v_psi_values is not None and len(v_psi_values) > 0:
        v_fine = np.linspace(x_lo_v, x_hi_v, 500)
        hypo_colors = _hypothesis_colors(len(v_psi_values))
        for v_psi, hc in zip(v_psi_values, hypo_colors):
            if sigma_v_frac > 0:
                sigma_v = sigma_v_frac * v_fine.clip(1e-6)
                frac_ent = 0.5 * erfc(
                    (v_fine - v_psi) / (np.sqrt(2) * sigma_v))
                B_model = B_SM * frac_ent
            else:
                B_model = np.where(v_fine <= v_psi, B_SM, 0.0)
            ax.plot(v_fine, B_model, color=hc,
                    linewidth=_S['hypo_lw'], zorder=2)
            # Label at the midpoint of the transition, offset right
            if v_psi <= x_hi_v:
                ax.text(v_psi * 1.08, -0.18, rf'${v_psi:g}c$',
                        fontsize=_S['hypo_label_fs'], color=hc,
                        ha='left', va='top', clip_on=True)

    if np.any(ok):
        _shadow_errorbar(ax, bc_v[ok], B_vals[ok], yerr=B_errs[ok],
                         xerr=hw_v[ok], color=_C['data'], marker='o',
                         ms=2.5)
    ax.axhline(B_SM, color=_C['sm'], linewidth=_S['ref_lw'],
               linestyle=_S['ref_ls_sm'], zorder=1)
    ax.axhline(0.0, color=_C['light'], linewidth=0.25, zorder=1)
    _label_shadow(ax, 0.97, 0.08, B_SM_LABEL,
                  fontsize=_S['annot_fs'], color=_C['sm'])
    ax.set_xlabel(r'$v_{\mathrm{sig}} / c$')
    ax.set_ylabel(r'Cosine Coefficient $B$')

    _paper_bg(fig, ax)

    _save(fig, "acoplanarity_vs_vsignal")


# ---------------------------------------------------------------------------
# 8b. Acoplanarity vs v_psi 2D histogram  (single-column)
# ---------------------------------------------------------------------------

def plot_acoplanarity_2d(acoplanarity_arr, v_signal_arr):
    """2D histogram of acoplanarity vs v_psi/c, column-normalised."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(COL1, COL1 / GOLDEN))

    v_clip = np.clip(v_signal_arr, 0, np.percentile(v_signal_arr, 98) * 1.1)

    # Use few wide v_psi bins so every column has decent statistics.
    n_vbins = max(3, min(6, len(v_signal_arr) // 15))
    n_phibins = 6

    h, xedges, yedges = np.histogram2d(
        v_clip, acoplanarity_arr,
        bins=[n_vbins, n_phibins],
        range=[[0, v_clip.max()], [-np.pi, np.pi]])

    # Normalise each v_psi column so all slices have equal visual weight.
    col_sums = h.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1  # avoid division by zero
    h_norm = h / col_sums

    cmap = _jewel_colormap()
    im = ax.pcolormesh(xedges, yedges, h_norm.T, cmap=cmap, rasterized=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=15, pad=0.04)
    cbar.ax.tick_params(labelsize=_S['annot_fs'])
    cbar.set_label('Fraction per slice', fontsize=_S['annot_fs'])
    cbar.outline.set_linewidth(0.3)

    ax.set_xlabel(r'$v_{\mathrm{sig}} / c$')
    ax.set_ylabel(r'Acoplanarity $\Delta\phi$')
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])

    _paper_bg(fig, ax)

    _save(fig, "acoplanarity_2d")


# ---------------------------------------------------------------------------
# 9. Vertex comparison  (double-wide)
# ---------------------------------------------------------------------------

def plot_vertex_comparison(reco_results):
    _apply_style()

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
    fig.subplots_adjust(hspace=0.50, wspace=0.35)

    p99 = np.percentile(np.concatenate([truth_L, reco_L]), 99)
    lim = p99 * 1.2

    # (a) Scatter: reco vs truth decay length
    ax = axes[0, 0]
    ax.scatter(truth_L, reco_L, s=1.5, color=_C['data'], alpha=0.25,
               edgecolors='none', rasterized=True, zorder=3)
    ax.plot([1e-4, lim], [1e-4, lim], color=_C['sm'], linewidth=_S['ref_lw'],
            linestyle='--', zorder=1)
    ax.set_xlabel('Truth Decay Length [mm]')
    ax.set_ylabel('Reco Decay Length [mm]')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # (b) Residual distribution
    ax = axes[0, 1]
    residual = reco_L - truth_L
    res_range = max(abs(np.percentile(residual, 2)),
                    abs(np.percentile(residual, 98))) * 1.2
    c, e = np.histogram(np.clip(residual, -res_range, res_range),
                        bins=50, range=(-res_range, res_range))
    _step_hist(ax, e, c, color=_C['truth'],
               fill_alpha=_S['fill_alpha'], fill_color=_C['truth'])
    ax.set_xlabel('Reco $-$ Truth [mm]')
    ax.set_ylabel('Entries / Bin')
    # Robust location/width (median and IQR-based sigma): a handful of
    # far outliers would otherwise dominate mean/RMS and mislead.
    q75, q25 = np.percentile(residual, [75, 25])
    _label_shadow(ax, 0.97, 0.92,
                  f'median {np.median(residual):.3f} mm\n'
                  rf'$\sigma_{{\mathrm{{IQR}}}}$ {(q75 - q25) / 1.349:.3f} mm',
                  fontsize=_S['annot_fs'], color=_C['data'])

    # (c) Ratio distribution
    ax = axes[1, 0]
    safe = truth_L > 0.01
    ratio = reco_L[safe] / truth_L[safe]
    c, e = np.histogram(np.clip(ratio, 0, 5), bins=60, range=(0, 5))
    _step_hist(ax, e, c, color=_C['truth'],
               fill_alpha=_S['fill_alpha'], fill_color=_C['truth'])
    ax.axvline(1.0, color=_C['sm'], linewidth=_S['ref_lw'], linestyle='--',
              zorder=1)
    ax.set_xlabel('Reco / Truth')
    ax.set_ylabel('Entries / Bin')
    _label_shadow(ax, 0.97, 0.92, f'median {np.median(ratio):.3f}',
                  fontsize=_S['annot_fs'], color=_C['data'])

    # (d) Scatter: ratio vs truth decay length
    ax = axes[1, 1]
    ax.scatter(truth_L[safe], ratio, s=1.5, color=_C['data'], alpha=0.25,
               edgecolors='none', rasterized=True, zorder=3)
    ax.axhline(1.0, color=_C['sm'], linewidth=_S['ref_lw'], linestyle='--',
               zorder=1)
    ax.set_xlabel('Truth Decay Length [mm]')
    ax.set_ylabel('Reco / Truth')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, lim)

    _paper_bg(fig, axes)
    fig.align_ylabels()

    _save(fig, "vertex_comparison")


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
    if 'm12_bc' in global_result:
        print(f"  (bias-corrected):      m12 = {global_result['m12_bc']:.4f}")
    print(f"  SM prediction:         m12 = 2.0")
    if 'chsh_S' in global_result:
        print(f"  CHSH (fixed axes):     S = {global_result['chsh_S']:.4f} "
              f"+/- {global_result['chsh_S_err']:.4f}")
        print(f"  SM prediction:         S = 2*sqrt(2) = 2.8284 (Tsirelson)")
    if 'witness_W' in global_result:
        print(f"  Concurrence witness:   W = {global_result['witness_W']:.4f} "
              f"+/- {global_result['witness_W_err']:.4f}  (concurrence >= W)")
    print(f"  Concurrence (rho rec.): C = {global_result['concurrence']:.4f} "
          f"+/- {global_result['concurrence_err']:.4f}")
    print(f"  SM prediction:         C = 1.0")

    print(f"\n  Reject local realism (CHSH S <= 2):    {locality_sigma:.1f} sigma")
    print(f"  Reject separability (witness W <= 0):  {entanglement_sigma:.1f} sigma")

    n_total = n_spacelike + n_timelike
    print(f"\n  Spacetime classification:")
    print(f"    Spacelike: {n_spacelike} ({100*n_spacelike/n_total:.1f}%)")
    print(f"    Timelike:  {n_timelike} ({100*n_timelike/n_total:.1f}%)")

    if vpsi_scan_results is not None:
        print(f"\n  v_psi exclusion scan "
              f"(LR test = optimal no-correlation test, permutation-calibrated):")
        print(f"    {'v_psi/c':>10s} {'N_events':>10s} {'m12_bc':>10s} "
              f"{'z(LR test)':>12s} {'z(CHSH S>2)':>12s}")
        for r in vpsi_scan_results:
            if r['n_events'] >= 6:
                m12_show = r.get('m12_bc', r['m12'])
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d} "
                      f"{m12_show:10.3f} "
                      f"{r['sigma_vs_0']:12.1f} {r['sigma_vs_1']:12.1f}")
            else:
                print(f"    {r['v_psi']:10.1f} {r['n_events']:10d}"
                      f"    (too few events)")

    print("\n" + "=" * 70)

# Enhancement Plan: Plot Aesthetics & Code Quality for PRL Submission

## Overview

The codebase is well-structured and the physics is solid. The Tufte-inspired design
philosophy is a strong foundation. The changes below will elevate the plots from
"good physics paper" to "striking graphic design" while also addressing coding
practices for a robust PRL submission.

---

## A. Plot Aesthetics — Typography & Fonts

### A1. Fix font fallback chain (high impact)

CMU Serif is not installed on most systems (including this one). The current fallback
chain `['CMU Serif', 'Computer Modern', 'DejaVu Serif']` silently falls back to
DejaVu Serif, which has a very different character from Computer Modern. This means
the plots are probably rendering in DejaVu Serif right now.

**Change:** Add `'STIXGeneral'` to the font stack (available on this system and
most others). STIX was designed as a near-match for Computer Modern and is the best
available stand-in. Also add `'Liberation Serif'` as a fallback before DejaVu.

```python
'font.serif': ['CMU Serif', 'STIXGeneral', 'Liberation Serif', 'DejaVu Serif'],
```

### A2. Use `text.usetex: False` with `mathtext.fontset: 'stix'`

The current setting `'mathtext.fontset': 'cm'` uses CM-style math glyphs, but the
surrounding text is in DejaVu (see A1). Switching to `'stix'` harmonizes math and
body text when STIX is the active font:

```python
'mathtext.fontset': 'stix',
```

This gives the closest approximation to the LaTeX look in PRL without requiring a
full TeX installation.

---

## B. Plot Aesthetics — Spacing, Layout & Panel Labels

### B1. Add panel labels (a), (b), (c), (d) — PRL requirement

PRL requires panel labels for multi-panel figures. Currently none of the 4 multi-panel
figures have them. Add a helper function:

```python
def _panel_label(ax, label, x=-0.12, y=1.06, fontsize=9, fontweight='bold'):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight, va='top', ha='right')
```

Apply to every subplot in: `plot_spacetime_distributions`, `plot_entanglement_vs_spacetime`,
`plot_vpsi_exclusion`, `plot_acoplanarity_vs_vsignal`, `plot_vertex_comparison`.

### B2. Replace `subplots_adjust` with `constrained_layout` or tuned `gridspec`

Several figures use `fig.subplots_adjust(hspace=..., wspace=...)` with values that
don't account for axis labels and tick labels, leading to potential overlap or wasted
space. For stacked-panel figures that share an x-axis (`entanglement_vs_spacetime`,
`vpsi_exclusion`, `acoplanarity_vs_vsignal`), the approach should be:

- Use `gridspec_kw` with explicit `hspace=0.08` (very tight, since shared x-axis)
- Remove `subplots_adjust` calls
- Use `fig.align_ylabels()` to ensure y-axis labels line up across panels

For grid figures (`spacetime_distributions`, `vertex_comparison`):
- Increase `hspace` slightly (0.42 → 0.50) to give breathing room for x-axis labels on top row
- Add `fig.align_ylabels()` and `fig.align_xlabels()`

### B3. Increase savefig padding

The current `savefig.pad_inches: 0.04` is very tight and occasionally clips axis
labels in PDF viewers. Increase to `0.08`.

---

## C. Plot Aesthetics — Color & Visual Design

### C1. Refine the shadow effect implementation

The current `_shadow_errorbar` computes shadow offsets from axis limits (`ax.get_xlim()`),
but this is called *before* the axes autoscale to the data, meaning the offsets are
computed from default (0,1) limits rather than the actual data range. This makes the
shadow offset inconsistent.

**Fix:** Use `matplotlib.transforms` for a resolution-independent offset:

```python
from matplotlib.transforms import ScaledTranslation
offset = ScaledTranslation(0.5/72, -0.5/72, fig.dpi_scale_trans)
```

This gives a consistent 0.5pt offset regardless of data range or axis scale.

### C2. Add a subtle warm paper-tone option

For a graphic-design look, pure white backgrounds can feel stark. Add an option for
a very faint warm tone (`#FDFBF7`) that evokes high-quality paper. Keep `#ffffff` as
default for PRL (journals require white), but this could be striking for talks or
a preprint cover figure.

### C3. Enhance the correlation matrix heatmap

The current `RdBu_r` colormap is standard but not distinctive. Create a custom
two-tone colormap derived from the jewel palette (`_C['truth']` teal → white →
`_C['reco']` ruby) for a more bespoke look. Also:
- Round the cell corners using `FancyBboxPatch` for a modern design feel
- Increase cell text size slightly (6.5 → 7pt) for readability
- Add a thin border between cells (linewidth 0.5, white)

### C4. Use filled markers with thin white edge

Instead of `markeredgewidth=0` (fully filled dots), use a thin white edge
(`markeredgewidth=0.3, markeredgecolor='white'`). This gives markers a cleaner,
"die-cut" appearance that separates overlapping points — a classic graphic-design
technique.

---

## D. Individual Plot Improvements

### D1. vpsi_overlay — "the money plot"

This is the central result figure. Make it exceptional:

- **Increase marker size** from 3.5 → 4.5 for the data points (these need to
  dominate visually)
- **Use a color gradient** for the hypothesis curves instead of grey scale: step
  from a cool blue (low v_psi) to a warm amber (high v_psi), using the existing
  palette endpoints. This adds visual narrative — "the hypothesis speed warms up."
- **Add a shaded "Bell-nonlocal" band** (amber, alpha=0.06) below m12=1 to visually
  highlight the region where locality holds, making the violation immediately apparent
- **Place the v_psi labels along a subtle curved arc** instead of all at y=2.12,
  reducing clutter
- **Add a concise figure caption placeholder** as a text annotation at the bottom
  (common in design books, can be removed for PRL)

### D2. vpsi_exclusion — tighten and polish

- The duplicate `_label_shadow` + `ax.annotate` for the 3σ label (lines 577-580) is
  a bug — it places two overlapping labels. Remove the `_label_shadow` call.
- Align the 3σ and 5σ annotations at a consistent x position (right edge)
- Use subtle dashed grey shading above 95% CL line (alpha=0.04) to visually mark the
  exclusion region

### D3. Acoplanarity — refine

- Add a vertical π and −π tick label using `ax.set_xticks([-π, -π/2, 0, π/2, π])`
  with LaTeX labels `[r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']`
- This is a standard graphic-design improvement for angular distributions — named
  ticks instead of decimal numbers

### D4. Spacetime distributions — balance the grid

- Panel (d) (bar chart) looks visually different from the other three histogram panels.
  Give the bars a subtle gradient fill or use the same step-histogram style with
  side-by-side hatching to maintain visual consistency across the grid.

### D5. Vertex comparison — declutter

- The scatter panels (a) and (d) use 0.8pt markers at alpha=0.35. For a design-forward
  look, consider a 2D histogram (hexbin or hist2d) with the custom teal colormap to
  replace the scatter. This eliminates overplotting and gives a smoother, more polished
  visual.

---

## E. Coding Design Practices

### E1. Add unit tests (critical for PRL submission confidence)

There are currently zero tests. Add a `tests/` directory with:

- `test_lorentz.py`: Test `mass()`, `boost()`, `beta_vec()` with known inputs
  (e.g., boost a particle to its rest frame, check mass is preserved)
- `test_entanglement.py`: Test `compute_m12()` with known correlation matrices:
  - C = diag(1, 1, -1) → m12 = 2 (SM)
  - C = 0 → m12 = 0 (no correlation)
  - C = diag(0.5, 0.5, 0) → m12 = 0.5 (classical)
- `test_concurrence.py`: Test concurrence for maximally entangled state (should be 1)
  and separable state (should be 0)
- `test_spacetime.py`: Test `compute_spacetime_interval()` with known lightlike,
  spacelike, and timelike separations
- `test_spin_analysis.py`: Test basis definition `_define_basis()` returns orthonormal
  vectors; test C_ij extraction with synthetic uniform distributions

This is a modest test suite (~150 lines total) that verifies the core physics
calculations are correct. Essential before publication.

### E2. Extract plotting constants into a style configuration

The plotting code has many hardcoded values scattered across functions (marker sizes,
font sizes, linewidths, alpha values). Extract these into a `_STYLE` dict at module
level:

```python
_STYLE = {
    'data_marker_size': 3.5,
    'data_elinewidth': 0.4,
    'ref_linewidth': 0.35,
    'ref_linestyle_sm': '--',
    'ref_linestyle_bell': ':',
    'fill_alpha': 0.12,
    'annotation_fontsize': 5.5,
    'label_fontsize': 8,
    'panel_label_fontsize': 9,
}
```

This makes the style trivially adjustable and ensures consistency across all plots.

### E3. DRY out the save-and-close pattern

Every plot function ends with the same 4-line pattern:

```python
fig.savefig(os.path.join(OUTPUT_DIR, f"name.pdf"))
fig.savefig(os.path.join(OUTPUT_DIR, f"name.png"))
plt.close(fig)
print(f"  Saved name.pdf")
```

Extract into a helper:

```python
def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}.pdf")
```

### E4. Add `__all__` to plotting.py

The module exposes many private helpers (`_shadow_errorbar`, `_C`, etc.) alongside
the public API. Add an `__all__` list to make the public interface explicit.

### E5. Add a `pyproject.toml` or modernize `requirements.txt`

The current `requirements.txt` has no version pins. For reproducibility (critical for
PRL claims), at minimum pin major versions:

```
numpy>=1.24,<3
scipy>=1.10,<2
matplotlib>=3.7,<4
```

---

## F. PRL Submission Checklist

### F1. Figure format

PRL prefers EPS or PDF (already producing PDF — good). Ensure all fonts are embedded.
The current `savefig.dpi: 600` is fine for rasterized elements; vector elements in
PDF are resolution-independent.

### F2. Figure width compliance

The figures already use `COL1 = 3.375"` and `COL2 = 7.0"` which are correct PRL
column widths.

### F3. Color accessibility

Run the palette through a colorblind simulator. The current palette is described as
"colorblind-aware" but hasn't been verified against deuteranopia/protanopia. The
teal-ruby pair is generally safe, but the amber-green pair (`_C['bell']` vs
`_C['accent']`) could be problematic. Consider using distinct line styles (already
partially done) as a redundant encoding.

### F4. Verify physics with tests before submission

See E1 — run the test suite and confirm all entanglement measures match known
analytical results.

---

## Implementation Order

1. **B1** — Panel labels (quick, high impact, PRL-required)
2. **A1 + A2** — Font stack fix (quick, visual consistency)
3. **C1** — Shadow effect fix (correctness)
4. **E3** — DRY save pattern (cleanup before other changes)
5. **E2** — Style constants (makes subsequent changes easier)
6. **D1** — Money plot enhancements
7. **D3** — Acoplanarity π-labels
8. **C3** — Correlation matrix redesign
9. **D2** — vpsi_exclusion polish
10. **C4** — Marker edge styling
11. **B2** — Spacing improvements
12. **E1** — Unit tests
13. **D4, D5** — Optional: spacetime/vertex polish
14. **E5** — Requirements pinning
15. **F3** — Colorblind verification

#!/usr/bin/env python3
"""
Figures for the version-2 paper (Tufte style, one panel per PDF).

    python -m v2.figures export   # processed samples -> results/figure_data.json
    python -m v2.figures plot     # results/*.json -> paper/v2/figures/*.pdf

`export` needs the (uncommitted) results/processed_*.npz; `plot` needs only
committed JSON files.
"""
import json
import os
import sys

import numpy as np

RES = "results"
OUT = "paper/v2/figures"
B_SM = -np.pi**2 / 16
INK = "#1f1f1f"
GREY = "#8a8a8a"
BLUE = "#4c72b0"
RED = "#c44e52"


# ---------------------------------------------------------------------------
def export():
    from v2.run_v2 import load_processed, apply_selection, n_expected_generated
    with open(f"{RES}/v2_results.json") as f:
        res = json.load(f)
    cut = res["ip_cut_scan"]["chosen"]
    data = {}
    modes = ["ee", "mumu", "qq"]
    A = {m: apply_selection(load_processed(m, "A"), cut) for m in modes}
    B = {m: apply_selection(load_processed(m, "B"), cut) for m in modes}

    # Figure 1: acoplanarity.  One pseudo-experiment at the expected yield of
    # the visible-Z scenario, drawn from the independent samples B; expected
    # shape after selection from samples A; truth-level curve for reference.
    edges = np.linspace(-np.pi, np.pi, 17)
    rng = np.random.default_rng(2026)
    counts = np.zeros(16)
    shape = np.zeros(16)
    n_sel_exp = 0.0
    for m in modes:
        ne = n_expected_generated(m)
        b = B[m]
        idx = rng.integers(0, len(b["v_true"]), rng.poisson(ne))
        idx = idx[b["sm"]["sel"][idx]]
        counts += np.histogram(b["sm"]["dphi"][idx], edges)[0]
        a = A[m]["sm"]
        h = np.histogram(a["dphi"][a["sel"]], edges)[0].astype(float)
        eff = a["sel"].mean()
        shape += h / h.sum() * ne * eff
        n_sel_exp += ne * eff
    data["acoplanarity"] = {"edges": edges.tolist(), "pseudo_data": counts.tolist(),
                            "expected_after_selection": shape.tolist(),
                            "n_selected_expected": n_sel_exp, "scenario": "visible"}

    # Figure 2: v_min response (μμ, selected, SM)
    a = A["mumu"]
    s = a["sm"]["sel"]
    vt, vr = a["v_true"][s], a["sm"]["v_reco"][s]
    le = np.linspace(0, np.log10(300), 41)
    H, _, _ = np.histogram2d(np.log10(np.clip(vt, 1, 1e6)), np.log10(np.clip(vr, 1e-3, 1e6)),
                             bins=[le, le])
    data["response"] = {"log10_edges": le.tolist(), "counts_true_x_reco_y": H.tolist()}
    pur = {}
    V = np.exp(np.linspace(0, np.log(100), 30))
    for m in ("mumu", "qq"):
        a = A[m]
        s = a["sm"]["sel"]
        vt, vr = a["v_true"][s], a["sm"]["v_reco"][s]
        pur[m] = [float(np.mean(vt[vr > v] > v)) if np.sum(vr > v) > 50 else None for v in V]
    data["purity"] = {"V": V.tolist(), **pur}

    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/figure_data.json", "w") as f:
        json.dump(data, f, indent=1)
    print("wrote results/figure_data.json")


# ---------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(os.path.dirname(__file__), "tufte.mplstyle"))
    return plt


def _range_frame(ax, xr, yr):
    ax.spines["bottom"].set_bounds(*xr)
    ax.spines["left"].set_bounds(*yr)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    assert len(fig.axes) == 1, "one panel per file"
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    print(f"wrote {OUT}/{name}.pdf")


def plot():
    plt = _style()
    with open(f"{RES}/figure_data.json") as f:
        fd = json.load(f)
    with open(f"{RES}/v2_results.json") as f:
        res = json.load(f)

    # ---- Figure 1: acoplanarity
    d = fd["acoplanarity"]
    e = np.array(d["edges"])
    c = 0.5 * (e[1:] + e[:-1])
    y = np.array(d["pseudo_data"])
    expd = np.array(d["expected_after_selection"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    phi = np.linspace(-np.pi, np.pi, 300)
    norm = d["n_selected_expected"] / len(c)
    ax.plot(phi, norm * (1 + B_SM * np.cos(phi)), color=GREY, lw=0.8, ls=(0, (4, 2)))
    ax.step(e, np.r_[expd, expd[-1]], where="post", color=RED, lw=0.9)
    ax.errorbar(c, y, yerr=np.sqrt(np.maximum(y, 1)), fmt="o", ms=2.6, color=INK,
                elinewidth=0.6, capsize=0)
    ax.set_xlim(-np.pi, np.pi)
    ymax = max(y.max() + np.sqrt(y.max()), expd.max()) * 1.08
    ax.set_ylim(0, ymax)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"Acoplanarity $\Delta\phi$")
    ax.set_ylabel("Events / bin")
    yt = [t for t in ax.get_yticks() if t <= ymax]
    _range_frame(ax, (-np.pi, np.pi), (0, yt[-1]))
    ax.text(-np.pi * 0.97, norm * (1 - B_SM) * 1.02, r"SM, truth ($B=-\pi^2/16$)",
            color=GREY, fontsize=7, va="bottom")
    ax.text(0, expd.min() * 0.72, "SM after selection", color=RED, fontsize=7, ha="center",
            va="top")
    _save(fig, "fig1_acoplanarity")
    plt.close(fig)

    # ---- Figure 2a: response
    d = fd["response"]
    le = np.array(d["log10_edges"])
    H = np.array(d["counts_true_x_reco_y"])
    Hn = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.pcolormesh(10**le, 10**le, Hn.T, cmap="Greys", vmin=0, vmax=np.percentile(Hn[Hn > 0], 99),
                  rasterized=True)
    ax.plot([1, 300], [1, 300], color=RED, lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 300)
    ax.set_ylim(1, 300)
    ax.set_xlabel(r"True $v_{\min}/c$")
    ax.set_ylabel(r"Reconstructed $v_{\min}/c$")
    ax.text(150, 210, "true = reco", color=RED, fontsize=7, ha="right", rotation=45)
    _range_frame(ax, (1, 300), (1, 300))
    _save(fig, "fig2a_vmin_response")
    plt.close(fig)

    # ---- Figure 2b: purity
    d = fd["purity"]
    V = np.array(d["V"])
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    for m, col, lab in (("mumu", INK, r"$Z\to\mu\mu$"), ("qq", BLUE, r"$Z\to q\bar q$")):
        p = np.array([np.nan if x is None else x for x in d[m]])
        ax.plot(V, p, color=col, lw=1.0)
        k = np.where(np.isfinite(p))[0][-1]
        ax.text(V[k] * 1.08, p[k], lab, color=col, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 100)
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$P(v_{\min}^{\rm true}>V \mid v_{\min}^{\rm reco}>V)$")
    _range_frame(ax, (1, 100), (0, 1))
    _save(fig, "fig2b_purity")
    plt.close(fig)

    # ---- Figure 3: expected exclusion, dot-and-band chart
    lim = res["limits"]
    rows = []
    for scen, lab in (("mumu", r"$Z\to\mu\mu$"), ("ll", r"$Z\to ee,\mu\mu$"),
                      ("visible", r"all visible $Z$ (proj.)")):
        for alt, alab, col in (("no_correlation", "no correlation", INK),
                               ("local_bound", "local bound", RED)):
            r = lim[scen][alt]["v95"]
            rows.append((lab, alab, col, r))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ypos = []
    for i, (lab, alab, col, r) in enumerate(rows):
        y = len(rows) - i + (0.15 if i % 2 == 0 else -0.15) - (i // 2) * 0.6
        ypos.append(y)
        lo2, lo1, med, hi1, hi2 = (max(r[q], 1.0) for q in ("p02.3", "p15.9", "p50", "p84.1", "p97.7"))
        ax.plot([lo2, hi2], [y, y], color=col, lw=0.6, solid_capstyle="butt")
        ax.plot([lo1, hi1], [y, y], color=col, lw=2.4, solid_capstyle="butt", alpha=0.55)
        ax.plot([med], [y], "o", color=col, ms=3.2)
        if i % 2 == 0:
            ax.text(0.85, y - 0.15, lab, ha="right", va="center", fontsize=7.5,
                    transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    xs = [max(r[q], 1.0) for *_, r in rows for q in ("p02.3", "p97.7")]
    ax.set_xlim(1, max(xs) * 1.3)
    ax.set_xlabel(r"Expected 95% CL exclusion: $v_\psi/c$ below")
    ax.text(max(xs) * 1.25, ypos[-2] + 0.15, "no correlation", color=INK, fontsize=7, ha="right",
            va="bottom")
    ax.text(max(xs) * 1.25, ypos[-1] - 0.15, "local bound", color=RED, fontsize=7, ha="right",
            va="top")
    _save(fig, "fig3_exclusion")
    plt.close(fig)

    # ---- Figure 3b: CL_s vs v_ψ, visible scenario
    vg = np.array(res["parameters"]["v_psi_grid"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for alt, col, lab in (("no_correlation", INK, "no correlation"),
                          ("local_bound", RED, "local bound")):
        cls = lim["visible"][alt]["cls"]
        med = np.array([c["p50"] for c in cls])
        lo = np.array([c["p15.9"] for c in cls])
        hi = np.array([c["p84.1"] for c in cls])
        ax.fill_between(vg, np.clip(hi, 1e-4, 1), np.clip(lo, 1e-4, 1), color=col, alpha=0.15, lw=0)
        ax.plot(vg, np.clip(med, 1e-4, 1), color=col, lw=1.0)
        k = int(np.argmin(np.abs(med - 0.3)))
        ax.text(vg[k] * 1.1, med[k], lab, color=col, fontsize=7, va="bottom")
    ax.axhline(0.05, color=GREY, lw=0.5, ls=(0, (2, 2)))
    ax.text(vg[-1], 0.055, "95% CL", color=GREY, fontsize=7, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlim(vg[0], vg[-1])
    ax.set_xlabel(r"$v_\psi/c$")
    ax.set_ylabel(r"Expected CL$_s$")
    _range_frame(ax, (vg[0], vg[-1]), (1e-4, 1))
    _save(fig, "fig3b_cls_visible")
    plt.close(fig)


if __name__ == "__main__":
    {"export": export, "plot": plot}[sys.argv[1]]()

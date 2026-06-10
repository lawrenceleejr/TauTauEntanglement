# Tau-Tau Entanglement Analysis

Measurement of quantum entanglement between tau leptons produced in Higgs
boson decay at a 240 GeV e+e- collider (ILC / CEPC). The analysis uses
the process e+e- -> ZH -> mu+mu- tau+tau- and reconstructs tau spin
correlations as a function of the spacetime separation between their
decay vertices.

## Physics overview

A CP-even Higgs decays to tau pairs through the unique 3P0 partial wave,
so the tau spins are produced in the maximally entangled triplet
(|ud> + |du>)/sqrt(2) along the decay axis. The spin correlation matrix
is **exactly** C = diag(+1, +1, -1) in the {n, r, k} basis, for any tau
velocity. The pi-pi acoplanarity follows 1 + B cos(phi) with
B = -pi^2/16 ~ -0.617 (exact). This analysis measures the spin
correlations as a function of the spacetime interval between the two tau
decay points, testing whether entanglement persists across spacelike
separations and scanning hypothetical signal-speed thresholds for
finite-speed causal models.

Key observables (see `entanglement.py` for derivations):

- **CHSH score S** (fixed a-priori axes) -- linear in C, hence an
  unbiased estimator with exact per-event variance (28 under the SM);
  S > 2 excludes local hidden-variable models, SM gives S = 2 sqrt(2).
- **Concurrence witness W = (C_nn + C_rr - C_kk - 1)/2** -- lower bound
  on the concurrence via the fully-entangled fraction (Bennett et al.);
  W > 0 certifies entanglement, SM gives W = 1.
- **m12 (Horodecki parameter)** -- largest two eigenvalues of C^T C
  summed; m12 > 1 implies Bell nonlocality. NOTE: the moment estimator
  is noise-biased, E[m12_hat | C=0] = 78.5/N; only bias-corrected and
  calibrated values are meaningful at small N.
- **Optimal LR test** -- per-event log-likelihood ratio
  t = ln(1 - h+.C_SM.h-) with exact moments (E_SM = ln2 - 1/2,
  Var_SM = 1/4; E_0 = ln2 - 1, Var_0 = 1), calibrated by permuting the
  tau+/tau- pairing (an exact realisation of the no-correlation null).
- **Signal speed v_sig** -- evaluated in the Higgs rest frame, where it
  obeys the exact Pareto law P(v_sig > v) = beta/v, giving the analytic
  exclusion reach v95 = N beta / (4 z95^2).

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: numpy, scipy, matplotlib, pyhepmc, hist, mplhep, particle.

## Usage

```bash
python run_analysis.py <path_to_hepmc_file> [--max-events N] [--smear]
```

| Argument | Description |
|---|---|
| `hepmc_file` | Path to a HepMC3 file with e+e- -> ZH -> mu+mu- tau+tau- events |
| `--max-events N` | Process at most N events (default: all) |
| `--smear` | Apply ILC detector resolution smearing to tracks |

Output is written to the `plots/` directory (PDFs, PNGs, and a
`results.json` with numerical results).

## Enabling rho decay mode

By default, only the cleanest tau decay channel is used:

```python
# config.py
ALLOWED_DECAY_MODES = ["pi_nu"]   # tau -> pi nu  (BR ~ 10.8%)
```

To also accept the higher-statistics rho channel (tau -> rho nu -> pi pi0
nu, BR ~ 25.5%), edit `config.py`:

```python
ALLOWED_DECAY_MODES = ["pi_nu", "rho_nu"]
```

Or to use only the rho channel:

```python
ALLOWED_DECAY_MODES = ["rho_nu"]
```

The supported decay modes are:

| Mode | Decay | Branching ratio |
|---|---|---|
| `"pi_nu"` | tau -> pi nu | 10.8% |
| `"rho_nu"` | tau -> rho nu -> pi pi0 nu | 25.5% |

The luminosity estimate printed on every plot adjusts automatically for
the combined branching ratio of the selected modes.

## Analysis pipeline

1. **Parse HepMC3** -- extract tau decay products, classify decay modes,
   filter by `ALLOWED_DECAY_MODES`.
2. **Detector smearing** (optional) -- apply ILC-like track resolution.
3. **Tau reconstruction** -- Jeans impact-parameter method using pion
   tracks and the Higgs mass constraint.
4. **Spacetime intervals** -- compute truth and reconstructed separations
   between tau decay vertices.
5. **Spin observables** -- project pion momenta onto tau rest-frame spin
   basis to build the 3x3 correlation matrix C_ij.
6. **Entanglement measures** -- extract m12, concurrence, and Bell score
   from C_ij with bootstrap uncertainties.
7. **Binned analysis** -- bin entanglement vs spacetime interval and
   signal speed.
8. **Signal-speed scan** -- test v_psi hypotheses for exclusion.
9. **Plots** -- generate publication-quality figures.

## Output plots

| File | Description |
|---|---|
| `spacetime_distributions.pdf` | Truth vs reconstructed spacetime intervals |
| `entanglement_vs_spacetime_interval.pdf` | m12 and concurrence vs spacetime interval |
| `entanglement_vs_signal_speed.pdf` | m12 and concurrence vs signal speed |
| `vpsi_overlay.pdf` | m12 vs signal speed with v_psi hypothesis curves |
| `vpsi_exclusion.pdf` | Signal-speed exclusion at 95% CL |
| `correlation_matrix.pdf` | Reconstructed C_ij heatmap |
| `acoplanarity.pdf` | Pion acoplanarity distribution with cosine fit |
| `acoplanarity_vs_vsignal.pdf` | Acoplanarity binned by signal speed |
| `vertex_comparison.pdf` | Reconstructed vs truth decay vertices |

Truth-level validation plots (`*_truth_validation.pdf`) are also produced.

## Luminosity estimation

Given the number of analysed events, the code estimates the corresponding
ILC integrated luminosity:

```
L_int = N_events / sigma_eff
```

where

```
sigma_eff = sigma(ZH) * BR(H->tautau) * BR(Z->mumu) * [sum of BR(tau->X)]^2
```

Cross sections and branching ratios are taken from the ILC TDR and PDG
2024 and are set in `config.py`. The instantaneous luminosity is computed
assuming 10^7 seconds of data collection.

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `ALLOWED_DECAY_MODES` | `["pi_nu"]` | Accepted tau decay channels |
| `SQRT_S` | 240.0 GeV | Centre-of-mass energy |
| `N_BINS_SPACETIME` | 8 | Bins for spacetime interval plots |
| `N_BINS_SIGNAL_SPEED` | 8 | Bins for signal speed plots |
| `N_BOOTSTRAP` | 1000 | Bootstrap resamples for uncertainties |
| `V_PSI_SCAN` | 1 -- 1000 c | Signal-speed hypotheses to test |
| `OUTPUT_DIR` | `"plots"` | Output directory |

## Project structure

| File | Purpose |
|---|---|
| `run_analysis.py` | Main driver |
| `parse_hepmc.py` | HepMC3 event parsing and decay classification |
| `config.py` | Constants and analysis settings |
| `tau_reconstruction.py` | Jeans method tau reconstruction |
| `spacetime.py` | Spacetime interval calculations |
| `spin_analysis.py` | Spin observable extraction |
| `entanglement.py` | Entanglement witnesses and bootstrap |
| `plotting.py` | Publication-quality Tufte-styled plots |
| `smearing.py` | ILC detector resolution simulation |

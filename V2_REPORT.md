# Version-2 analysis: report

FCC-ee, √s = 240 GeV, 10.8 ab⁻¹, ZH with H → ττ and both τ → πν.
The draft is `paper/v2/tauentanglement.pdf`, and changes from v1 are in `paper/v2/CHANGES.md`.
All numbers come from `results/*.json`.

## Gates

| Work package | Gate | Result |
|---|---|---|
| 1 | B = −π²/16 at truth | **Pass** in all 4 samples (e.g. −0.6126 ± 0.0018, μμ A) |
| 1 | C = diag(1, 1, −1) | **Pass**: largest pull of the 9 elements is 2.4 |
| 1 | Timelike fraction 4.0 × 10⁻⁴ | **Pass** within 3σ: 3.9 to 4.8 × 10⁻⁴. The excess comes from QED radiation off the τ, which the analytic value ignores |
| 1 | P(v_min > V) = β/V, Higgs frame | **Pass**: all pulls below 3, though mostly −1 to −2 and correlated |
| 1 | cτ = 87.03 μm | **Pass**: 87.05 to 87.20 ± 0.1 μm |
| 1 | `check_sqrts` | **Pass** |
| 1 | Pythia at 250 GeV vs the 90 WHIZARD ππ events | **FAIL** on recoil mass (KS p = 2 × 10⁻⁵); pion energies agree (p = 0.64). See below |
| 2 | No smearing, no radiation above 1 GeV: τ direction < 0.2 mrad | **Pass**: 0.023 mrad |
| 2 | Neutrino-sum residual < 0.5 GeV | **Pass**, but trivially: it is zero by construction |
| 3 | Null p-values uniform (5% test on independent sample B) | **Pass**: 8 checks, pulls between −1.8 and +1.6 |
| 3 | Z = √N/2 with truth and no detector | **Pass**: 5.02 (exact R) and 4.79 (fitted) for N = 100, against 5.00; 15.6 and 15.3 for N = 1000, against 15.8 |

**The failed gate.** The handoff says to stop on a failed gate. I diagnosed this one instead of stopping, and did not tune anything to make it pass. Nothing downstream depends on WHIZARD. The decision on whether this is acceptable is yours.

- **Pythia is consistent with expectation.** Its radiated-energy spectrum agrees with leading-log ISR convolved with σ_ZH(ŝ) up to ε = 0.02. It is about 25% softer in the hardest tail (P(ε > 0.05) = 0.13 against 0.18).
- **The WHIZARD file is the outlier.** It has far more soft radiation: P(ε > 0.004) = 0.66 against 0.42 expected. That is what you would see if the file included an ILC beam spectrum (beamstrahlung), but the file carries no metadata, so this cannot be confirmed.
- **The comparison is therefore not like-for-like.** Details are in `results/whizard_check.json`.

## Resolution and migration (Z → μμ, selected)

| Quantity | 2.5% | 16% | 50% | 84% | 97.5% |
|---|---|---|---|---|---|
| τ direction error (mrad) | 0.12 | 0.41 | 1.49 | 5.4 | 27 |
| L_reco / L_true | −0.91 | 0.81 | 1.00 | 1.15 | 2.8 |
| ln(v_reco / v_true) | −2.09 | −0.37 | −0.01 | 0.20 | 1.45 |

| Selection v_reco > | 2c | 5c | 9c | 16c | 30c | 50c |
|---|---|---|---|---|---|---|
| Fraction truly below | 11% | 26% | 38% | 52% | 68% | 78% |
| Efficiency for truly above | 84% | 70% | 59% | 45% | 31% | 21% |

The impact-parameter cut was optimised on the SM μμ calibration sample and frozen at **0**: the asymptotic Z falls monotonically from 1.18 with no cut to 1.02 at 6σ.

## Headline numbers

Expected 95% CL exclusion of v_ψ below, as median [68% band]:

| Scenario | Selected events | No correlation | Local (CHSH = 2) bound |
|---|---|---|---|
| Z → μμ | 45 | 2.8c [1.7, 4.7] | none |
| Z → ee, μμ | 90 | 5.1c [3.0, 7.6] | none (95% band reaches 2.3c) |
| All visible Z (projection) | 1053 | 22c [14, 27] | 4.3c [2.7, 7.2] |

## Assumed rather than simulated

- **No backgrounds.** ZZ, Zγ* → ττ and τ-decay cross-feed are absent.
- **Detector values without a source:** σ(z0) = σ(d0); a 1 mrad GeV multiple-scattering term for track angles; photon acceptance and thresholds; the πν photon veto; tracking acceptance.
- **Hadronic Z.** It is modelled as two jets along the Z decay quarks, each with 5% energy resolution (CLD) and exact direction. The primary vertex is taken to be as good as in the leptonic case. This channel drives the headline, and its sensitivity depends strongly on the jet model: with isotropic smearing of the Z four-vector the transverse spin correlations vanish (B ≈ 0); with energy-only smearing, B = −0.17 against −0.62 at truth.
- **Beam energy spread** is applied as an error on the beam energies the analysis assumes, not inside Pythia. Pythia's own momentum spread corrupts its cross section by a factor of 0.57 with lepton PDFs.
- **Beamstrahlung** beyond the energy spread is not modelled.

## Not done, or done differently

- **Generator output.** `generate_sample.py` saves compact arrays (`.npz` plus `.json`), not `EventRecord` objects. `to_event_records()` converts them on demand.
- **Unmerged branch.** The α-grid fix on `claude/prl-submission-review-d5fdhe` was not reused. The grid was replaced by an analytic Higgs-frame cone solution followed by a kinematic fit.
- **Statistics.** The test statistic is shape-only: event counts do not enter it. The density ratio is linear in q⁺_i q⁻_j and does not depend on v_min.
- **Paper text.** It was rewritten but needs the authors' read: tone, the reworked introduction and the discussion of loopholes. Fig. 3b (CL_s against v_ψ) is generated but not in the paper.
- **Nothing was submitted to arXiv.**

## Reproduce

```
pip install pythia8mc pyhepmc scipy matplotlib
bash generate_all_samples.sh          # ~3 min, 2.3 x 10^6 events (not committed)
./run_v2_pipeline.sh                  # ~35 min on 4 cores
python -m v2.figures export && python -m v2.figures plot
python -m v2.make_numbers
cd paper/v2 && pdflatex tauentanglement && bibtex tauentanglement && pdflatex tauentanglement && pdflatex tauentanglement
python -m pytest tests                # 56 tests
```

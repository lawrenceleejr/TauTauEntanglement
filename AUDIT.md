# Audit of arXiv:2603.28868v1

"Higgs Boson Spookiness: Probing Quantum Nonlocality with Spacetime-Resolved
H→τ⁺τ⁻ Decays" (Lee, Lawless, Riggall), audited against the arXiv source and
the code in this repository.

## Verdict

The idea survives: measuring the separation of the two τ decays event by event
and using it to bound a finite-speed influence is new at a collider, and the
reconstruction makes it possible. The v1 numbers do not survive. The headline
luminosity is wrong by a factor of 1000, the statistical method does not give
valid significances, and two reference values the results are compared to are
wrong. A v2 needs new numbers, a different statistical treatment and a changed
motivation. It does not need a new idea.

## What was checked, and what was not

- The arXiv source and figures were read in full.
- The code on `main`, and the branch `claude/update-significance-calculation-eAQj2`
  whose significance calculation matches the arXiv figure, were read.
- `audit/toy_validation.py` generates ZH→(μμ)(ττ→πνπν) from first principles,
  independent of the analysis code. It gives truth-level facts from 2×10⁶
  events, estimator behaviour from 2000 pseudo-experiments per sample size,
  and pushes 8000 events per scenario through this repository's smearing and
  reconstruction. All numbers below marked "toy" are in `audit/toy_results.json`.
- The pipeline was run on the 90 ππ events in `EventSample1002.hepmc`.
- **Not checked:** the 4462-event sample behind the paper is not in the
  repository, so the paper's figures could not be regenerated. Findings 2 and
  6 are established on the bundled samples and the toy, and need one command
  on the full sample to confirm (see finding 2).

## Findings

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Luminosity is 752 ab⁻¹, not 0.75 ab⁻¹ | Invalidates headline | Certain. Code fixed |
| 2 | Sample is at 250 GeV, analysis assumes 240 GeV | Biases every observable | Certain for bundled samples. Guard added |
| 3 | m₁₂ estimator is biased and the quoted significances are not calibrated | Invalidates Fig. 3 | Certain |
| 4 | Events selected at high v_min are mostly migrations from low v_min | Overstates reach above ~10c | Toy |
| 5 | SM acoplanarity coefficient is −π²/16, not −0.5 | "Agreement with SM" is a 6σ dilution | Certain. Code fixed |
| 6 | True timelike fraction is 0.04%, not 5%; the motivation overstates the loophole | Changes the framing | Certain |
| 7 | Impact parameter is ~55 μm, not 2.4 μm; an unreported cut removes 36% of events | Text wrong, selection undocumented | Certain |
| 8 | Missing realism: exact primary vertex, no backgrounds, no ISR treatment, 100% efficiency | Referee will ask | By inspection |
| 9 | Text errors and overclaims | Minor to moderate | By inspection |

### 1. The luminosity is off by a factor of 1000

The paper states that 4462 ππ events correspond to 0.75 ab⁻¹. With the
paper's own inputs, σ(ZH) × BR(H→ττ) × BR(Z→μμ) × BR(τ→πν)² =
240 fb × 0.0627 × 0.03366 × 0.1082² = 0.00593 fb, which is **5.9 events per
ab⁻¹**. So 4462 events need **752 ab⁻¹**.

The cause was `int_lumi_ab = int_lumi_fb * 1e-6` in `run_analysis.py`
(1 ab⁻¹ is 1000 fb⁻¹, and the code then mislabelled the result). The wrong
conversion reproduces the paper's 0.75 exactly.

Events in the channel the paper analyses, at 100% efficiency:

| Programme | ∫L | ππ events, Z→μμ |
|---|---|---|
| Paper's stated scenario | 0.75 ab⁻¹ | 4 |
| ILC 250 | 2 ab⁻¹ | 12 |
| FCC-ee 240 | 10.8 ab⁻¹ | 64 (53 with σ = 200 fb) |
| CEPC 240 | 20 ab⁻¹ | 119 |
| Paper's sample | 752 ab⁻¹ | 4462 |

As analysed (ππ only, Z→μμ only), no planned collider reaches the paper's
sample. The "1.5 ab⁻¹" curves are the same sample with errors divided by √2,
so they are 1500 ab⁻¹.

### 2. The sample and the analysis disagree on √s

Both bundled HepMC files sum to exactly 250.0 GeV per event. The analysis
builds the Higgs from p_H = (240, 0, 0, 0) − p_Z. On the 90 real ππ events
this gives a median recoil mass of 116 GeV instead of 125–127 GeV, and so the
wrong Higgs rest frame and the wrong τ energy constraint.

| 90 real ππ events, with smearing | assumed 240 | assumed 250 |
|---|---|---|
| Median recoil mass | 115.8 GeV | 126.9 GeV |
| Median τ direction error | 3.6 mrad | 3.1 mrad |
| L_reco/L_true, 84th percentile | 2.27 | 1.64 |

The 8000-event toy shows the same: with the mismatch the global m₁₂ estimate
drops from 2.03 to 1.67 and the transverse correlations C_nn, C_rr from
0.88, 0.90 to 0.78, 0.78.

The paper says 240 GeV. To check the full sample:

```
python -c "from parse_hepmc import sample_sqrts; print(sample_sqrts('FULL.hepmc'))"
```

If it prints 250, either the paper is an ILC-250 study and should say so, or
a 240 GeV sample is needed. `parse_events()` now stops on a mismatch, and
`--sqrts` sets the value.

### 3. The m₁₂ estimator and the quoted significances

m₁₂ is the sum of the two largest eigenvalues of ĈᵀĈ. It is quadratic in a
noisy matrix, so it is biased upward, and the bias grows as the sample
shrinks, which is exactly what happens at high v_min. Toy, exact SM density,
no detector:

| N events | ⟨m̂₁₂⟩, no correlation (true 0) | ⟨m̂₁₂⟩, SM (true 2) | Pseudo-experiments with m̂₁₂ > 2 |
|---|---|---|---|
| 30 | 2.66 | 5.39 | 99% |
| 100 | 0.80 | 3.35 | 97% |
| 300 | 0.26 | 2.66 | 94% |
| 1000 | 0.08 | 2.33 | 93% |
| 4462 | 0.02 | 2.14 | 92% |

Consequences for the paper:

- Fig. 2 shows m₁₂ = 2.5 to 3.9, above the physical maximum of 2. This is the
  bias, not physics. The toy reproduces the pattern: 2.4 in the lowest v_min
  bin rising to 3.9 in the highest. About half of the excess is finite-N
  bias and half is the impact-parameter cut sculpting the pion angles
  (finding 7), which the estimator C = −9⟨cos cos⟩ assumes are unsculpted.
- With 100 uncorrelated events, 26% of experiments give m̂₁₂ > 1. A point
  above the "Bell threshold" line is not evidence of anything at these
  sample sizes.
- Fig. 3 plots 2/σ_boot and 1/σ_boot, where σ_boot is the bootstrap width of
  the biased estimator. That is not a significance: the null distribution of
  m̂₁₂ is neither centred on 0 nor of width σ_boot, and m₁₂ = 0 is on the
  boundary of the parameter space.
- The estimator is also inefficient. The likelihood ratio between the SM
  density W = (1 − q⁺·R q⁻)/2 and W = 1/4 gives Z = √N/2 against "no
  correlation" (toy: 8.5 at N = 300, against 4.7 for 2/σ). That is a factor
  3.4 in luminosity.

For the Bell threshold, an unbiased alternative is the CHSH combination with
axes fixed in advance, S = √2 (C_nn + C_rr) ≤ 2. It is linear in the data,
Gaussian, with per-event variance 28 (toy), so Z = 0.157 √N.

### 4. The high-v_min region is dominated by migration

v_min = Δr/(cΔt) is large when the two decay times are nearly equal, so it
is the ratio of a well-measured sum to a poorly measured difference. Toy,
ILD-like smearing, √s consistent:

| Selection v_min(reco) > | Selected events truly below the cut | Efficiency for events truly above |
|---|---|---|
| 2 | 7% | 93% |
| 5 | 21% | 79% |
| 9 | 37% | 63% |
| 16 | 50% | 46% |
| 30 | 72% | 27% |

In the reconstructed bin above 32c the median true v_min is 18c. The paper's
statement that v_min is probed "to roughly 50c" is not supported. With the
√s mismatch the numbers are worse (42% at 9c, 62% at 16c).

This matters for the exclusion. Under a hypothesis with speed v_ψ, the
events selected with v_min(reco) > v_ψ are not uncorrelated: the migrated
ones carry the full SM correlation. Testing that selection against m₁₂ = 0
tests something the hypothesis does not predict, in the direction that
overstates the exclusion. The hypothesis curves in Fig. 2 use a Gaussian with
width proportional to v_min; the toy response has a 68% width of ±25% but 5%
of events are off by more than a factor of four.

### 5. The SM acoplanarity coefficient is −π²/16

For both taus decaying to πν, dN/dΔφ ∝ 1 − (π²/16) cos Δφ, so B = −0.617, not
−0.5. The factor is ⟨sin θ⁺⟩⟨sin θ⁻⟩ = (π/4)². Toy at truth level:
B = −0.6172 ± 0.0009.

Fig. 1 reports B = −0.496 ± 0.019 "in good agreement with the SM
expectation". It is 6σ from the SM value. It is a 20% dilution from the
τ-direction resolution (toy: −0.547 ± 0.015 with √s consistent,
−0.521 ± 0.015 with the mismatch), which happened to land on the wrong
reference. Fig. 2 has the same wrong reference line. The points there at
−0.8 and −0.93 are beyond the physical value; after the impact-parameter cut
the toy gives −0.66 overall because the cut prefers pions emitted at large
angle in the τ frame.

### 6. Almost every H→ττ event is spacelike, with or without this measurement

In the Higgs frame the taus are back to back with speed β, so the decays are
timelike only if the proper-time ratio is below (1−β)/(1+β) = 2×10⁻⁴. For
exponential lifetimes the timelike fraction is 2ε/(1+ε) = **4.0×10⁻⁴**
(toy: 4.0×10⁻⁴). The same argument gives P(v_min > V) = 1/V in the Higgs
frame (toy: 0.100 at V = 10; 0.090 in the lab frame at 240 GeV).

- The paper's "small timelike population (~5%)" that "arises when one tau
  decays promptly" does not exist at truth level. With 4462 events, 1.8
  timelike events are expected. The reconstruction toy gives 0.02%, or 0.7%
  with the √s mismatch; 5% could not be reproduced and can only be
  mis-reconstruction. It cannot serve as an "internal consistency check".
- The introduction argues that existing proposals leave a loophole because
  "even timelike signaling could reproduce the results". For relativistic
  pairs this is already answered by kinematics, and the paper's own
  Ref. [Ehataht:2023zzt] says so: Ehatäht et al. give the same criterion,
  |t₁−t₂|c / ((t₁+t₂)v) < 1, find more than 95% of Belle II τ pairs
  spacelike, and conclude the locality loophole is closed (arXiv:2311.17555,
  Sec. 3, Eq. 3.8). For H→ττ the same formula gives 99.96%.

What is new here is the event-by-event measurement of the separation, the
resulting v_min spectrum, and a bound on v_ψ. The paper should claim that
rather than the closing of a timelike loophole.

### 7. Impact parameter and the unreported cut

The paper says the typical τ impact parameter is 2.4 μm, giving
d/σ(d₀) ≈ 0.5. That number came from a hard-coded 0.8 mrad opening angle in
a printout in `smearing.py`. The median τ–π angle is 26 mrad and the median
impact parameter is **55 μm**, so d/σ ≈ 10 (toy). The measurement would not
work at d/σ = 0.5.

The analysis requires d/σ > 3 on both pions for every vertex-dependent
result. This removes 36% of events (toy; 38 of 90 on the real sample) and is
not mentioned in the paper. It also biases the angular distributions
(findings 3 and 5), so the SM expectation after the cut has to come from
simulation, not from the analytic values.

### 8. Missing realism

- The primary vertex is taken from truth with no uncertainty. The Z→μμ
  vertex resolution is of the same order as σ(d₀).
- No backgrounds: ZZ and Zγ* with ττ, H→WW, and other τ decays reconstructed
  as πν (ρν with a lost π⁰, Kν).
- The sample contains radiated photons: 39% of bundled events have more
  than 5 GeV outside the μμττ system (median 2.9 GeV), and the recoil mass
  has a tail to 144 GeV at the 90th percentile. p_H = p_beam − p_Z ignores
  this, so the Higgs frame is wrong in a large fraction of events. The paper
  does not discuss it. The crossing angle is absent.
- Efficiencies are 100%.
- Without any smearing, the selected solution misses the missing-momentum
  constraint by a median of 16 GeV and the τ direction by 1.2 mrad on the
  real events. The α scan in `tau_reconstruction.py` has 5 mrad spacing
  where the τ–π angle peaks; the unmerged branch
  `claude/prl-submission-review-d5fdhe` attributes the direction floor to
  that and reports a fix. The radiated photons above also contribute. This
  was not separated here.

### 9. Text

- H→ττ is not a spin singlet. A CP-even scalar decays to the ³P₀ state, the
  S = 1, m = 0 Bell state along the flight axis, with C = diag(+1, +1, −1).
  The singlet would be diag(−1, −1, −1) and is the CP-odd case. It is
  maximally entangled either way. Appears in the introduction twice.
- "The observed Bell violation in this sample therefore constitutes a test
  of nonlocality in the strict sense that Bell's theorem requires" and
  "model-independent test" contradict the paragraph two above them, which
  concedes there is no free choice of measurement axis. Reconstructing C
  from decay angles assumes the quantum description of the decay.
- A finite v_ψ needs a preferred frame. The optical bounds cited are quoted
  in candidate preferred frames, not "the lab frame". The paper can state
  that for v_ψ far below c/β_frame its lab-frame bound holds in any frame
  moving slowly relative to the lab, which includes the CMB frame
  (β ≈ 10⁻³).
- Bancal et al., Nature Phys. 8, 867 (2012), arXiv:1110.3795, show that any
  finite-speed influence model allows superluminal signalling. It should be
  cited where the v_ψ hypothesis is introduced. Yin et al., PRL 110, 260407
  (2013) is the standard recent optical bound.
- "Quantum states represent behavior established outside of spacetime or via
  superluminal connection" overstates what Bell's theorem shows.
- Z→qq̄ multiplies the leptonic sample by 21, not "an order of magnitude".
- σ(ZH) = 240 fb needs a reference and a stated polarisation; FCC-ee quotes
  200 fb at 240 GeV.

## What a publishable v2 needs

1. **Restate the reach in events, then map to programmes.** Use all visible
   Z decays (23 times Z→μμ), as Jeans and Wilson did, and state efficiencies.
   Ideal statistical reach, with 64% passing the vertex-quality cut:

   | Scenario | Events with vertices | v_ψ reach, no correlation | v_ψ reach, CHSH bound |
   |---|---|---|---|
   | FCC-ee 10.8 ab⁻¹, Z→μμ, ππ | 41 | 2.7c | none |
   | FCC-ee 10.8 ab⁻¹, visible Z, ππ | 930 | 60c statistically, ~10–15c with resolution | 6c |
   | Paper's sample | 2850 | resolution limited | 18c statistically |

   These use reach = N/15.4 (likelihood ratio, 95% CL) and N/156 (linear
   CHSH), before selection efficiency and dilution. The paper's 9c and 2c are
   therefore roughly what FCC-ee could do with hadronic Z decays and a proper
   test, which is a defensible headline.
2. **Fix √s** and regenerate everything.
3. **Replace the m₁₂ machinery for the exclusion.** Fold each v_ψ hypothesis
   through the simulated response in v_min, fit the unbinned angular
   likelihood, and obtain expected limits and bands from pseudo-experiments.
   Keep m₁₂ only as an illustration at large N, bias-corrected.
4. **Take SM expectations after selection from simulation**, including the
   dilution, and show them as the reference in Figs. 1 and 2.
5. **Change the motivation** from closing a timelike loophole to measuring
   the separation and bounding v_ψ, and remove the 5% timelike paragraph.
6. **Add the missing detector and physics effects** in finding 8, or quote
   efficiencies and resolutions from a full-simulation study.
7. **Correct the text** in finding 9 and describe the vertex-quality cut.

## Changes made on this branch

- `run_analysis.py`: luminosity conversion; `--sqrts` option.
- `config.py`: `set_sqrts()`, `lumi_from_events()`, `B_ACOPLANARITY_SM`.
- `parse_hepmc.py`: `sample_sqrts()`, `check_sqrts()`; parsing stops on a
  mismatch.
- `spin_analysis.py`: sign of the τ⁻ polarisation estimator.
- `plotting.py`: SM acoplanarity reference is −π²/16.
- `smearing.py`: typical impact parameter in the printout.
- `tests/test_audit_fixes.py`: regression tests for the above. One stale
  test that still expected C = +9⟨cos cos⟩ was updated. 50 tests pass.
- `audit/toy_validation.py`, `audit/toy_results.json`.

Not changed: the statistical method (finding 3), the migration treatment
(finding 4) and the paper text. The first two need the full sample and a
decision on the scenario in item 1.

## Reproduce

```
python audit/toy_validation.py --n-reco 8000      # about 15 minutes
python run_analysis.py EventSample1002.hepmc --smear --sqrts 250
python -m pytest tests
```

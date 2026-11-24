# Handoff: version 2 of arXiv:2603.28868

You are taking over the revision of a physics paper and its analysis code.
Read `AUDIT.md` first. It lists what is wrong with version 1 and the evidence.
This file says what to build.

## Goal

Produce a corrected analysis and a v2 draft of the paper for an **FCC-ee run
at √s = 250 GeV with 10.8 ab⁻¹**, using a signal sample we generate
ourselves. The paper's claim becomes: the separation of the two τ decays in
H→ττ can be measured event by event, and the resulting v_min spectrum bounds
the speed v_ψ of any finite-speed influence.

Decisions already made by the author. Do not reopen them:

- Collider scenario: FCC-ee, √s = 250 GeV. Keep √s as a single parameter
  (`config.set_sqrts`) so it can be changed in one place.
- We generate our own sample. The bundled `EventSample*.hepmc` files (WHIZARD,
  250 GeV, from D. Jeans) are for cross-checks only.
- The m₁₂-based significance of v1 is dropped as the basis of the exclusion.

## Working rules

- Branch: `claude/arxiv-analysis-audit-2xgidw`. Do not open a pull request.
- Commit and push before running tests or long jobs.
- All figures follow Tufte's principles: direct labels, no legend boxes, no
  chart junk, range frames, small multiples where they help.
- Do not commit generated samples. Commit the generator, the seeds and small
  result files (JSON). Add sample paths to `.gitignore`.
- Every number that goes into the paper must come from a committed script and
  a committed result file.
- Each work package has gates. If a gate fails, stop and report what failed
  and why. Do not tune anything to recover the v1 numbers (2c and 9c).
- Do not submit anything to arXiv.

## What is already verified in this environment

- `pip install pythia8mc` works and generates the signal at about 3000 events
  per second. These settings were tested:

  ```
  Beams:idA = 11
  Beams:idB = -11
  Beams:eCM = 250.
  PDF:lepton = on              # ISR
  HiggsSM:ffbar2HZ = on
  25:m0 = 125.
  23:onMode = off
  23:onIfAny = 13              # Z -> mu mu
  25:onMode = off
  25:onIfMatch = 15 -15
  15:onMode = off
  15:onIfMatch = 16 -211       # tau -> pi nu
  ```

  With them, Pythia's τ decays carry the H→ττ spin correlation (pion
  energies are anticorrelated, as expected), decay vertices are available
  from `xProd(), yProd(), zProd(), tProd()` of the pion in mm and mm/c, and
  1 event in 3000 was timelike (0.04% expected).
- `audit/toy_validation.py` is an independent generator with the exact SM
  spin density. Use it as the reference for closure tests.
- 50 tests pass with `python -m pytest tests`.

## Work packages, in order

### 1. Generator

Write `generate_sample.py` using `pythia8mc`. It builds `EventRecord` objects
directly (see `parse_hepmc.py`) and saves them in a compact file, with the
seed and all settings stored alongside.

Samples needed:

| Sample | Content | Size |
|---|---|---|
| Signal, SM | Z→μμ, both τ→πν, spin correlations on | 5×10⁵ |
| Signal, uncorrelated | same, with τ⁺ decays swapped between events | derived from the SM sample |
| Pseudo-data | statistically independent of the above | 5×10⁵ |

Also record the generator cross section and compare it with the literature
value for unpolarised e⁺e⁻→ZH at this energy. Cite the value you use.

Gates, at truth level, each within statistical error:

- Acoplanarity coefficient B = −π²/16 = −0.617.
- C = diag(+1, +1, −1) in the {n, r, k} basis with C_ij = −9⟨cos⁺ cos⁻⟩.
- Timelike fraction 4.0×10⁻⁴.
- P(v_min > V) = β/V in the Higgs frame.
- cτ = 87.03 μm from the proper decay times.
- `check_sqrts` passes.
- Agreement with the 90 ππ events of the bundled WHIZARD sample in the pion
  energy and recoil-mass distributions.

### 2. Reconstruction and detector model

- Replace the fixed α grid in `tau_reconstruction.py` with a solver that
  closes on truth. Gate: with no smearing and no radiated photon above 1 GeV,
  median τ direction error below 0.2 mrad and median missing-momentum
  residual below 0.5 GeV. The unmerged branch
  `claude/prl-submission-review-d5fdhe` has a grid refinement; review it
  before reusing it.
- Handle initial-state radiation. 39% of the bundled events radiate more than
  5 GeV. Either fit one collinear photon along the beam axis or cut on the
  recoil mass. Report the efficiency and show the recoil-mass distribution.
- Smear the primary vertex. Take the resolution from an FCC-ee detector
  reference and cite it.
- Replace the ILD resolution formulas with those of an FCC-ee detector
  concept (IDEA or CLD). Look the numbers up and cite them. Do not use
  values from memory.
- Keep the vertex-quality cut on impact-parameter significance, optimise its
  threshold once on the SM sample, then freeze it. Report its efficiency.

Gate: a table of resolution in τ direction, decay length and ln(v_reco/v_true)
(16th, 50th, 84th, 2.5th, 97.5th percentiles), and the migration table from
`AUDIT.md` finding 4 for the new detector model.

### 3. Statistical analysis

The hypothesis H(v_ψ): a pair is correlated as in the SM if its **true**
v_min is below v_ψ, and otherwise follows the alternative. Two alternatives:

- No correlation: the τ⁺ decay is replaced by one from another event.
- Local bound: correlations scaled so that the CHSH value is 2, that is
  C → C/√2.

State the frame in which v_min is defined (lab frame) and note in the paper
that for v_ψ ≪ c/β_frame the bound holds in any frame moving slowly relative
to the lab, including the CMB frame.

Method:

1. Per-event statistic: the log-likelihood ratio between the SM angular
   density W = (1 − q⁺·R q⁻)/4π², R = diag(1, 1, −1), and the alternative.
   Use densities **after** selection, taken from simulation, because the
   vertex-quality cut sculpts the pion angles.
2. For each v_ψ, weight each event by P(true v_min > v_ψ | reconstructed
   v_min) from the simulated response, and sum.
3. Build the distribution of the summed statistic under H(v_ψ) and under the
   SM from pseudo-experiments drawn at the expected yield. Quote the median
   expected exclusion with 68% and 95% bands, using CL_s.

Gates:

- Under each null the p-value is uniform: the 5% test rejects 5% of null
  pseudo-experiments within binomial error.
- With no detector effects and the true v_min, the result reproduces
  Z = √N/2 for the no-correlation alternative.

Keep m₁₂ and B only as illustrations. If shown, each point must be compared
with the simulated SM expectation after selection for that bin, not with 2
and −0.617.

### 4. Yields

One table: cross section, branching ratios, events per ab⁻¹, events at
10.8 ab⁻¹, then the efficiency of each selection step. Use
`config.lumi_from_events`.

Scenarios to report:

| Scenario | Status |
|---|---|
| Z→μμ, ππ | Full analysis |
| Z→ee and μμ, ππ | Full analysis; add the Z→ee sample |
| All visible Z, ππ | Projection. Generate Z→qq̄, smear the Z four-momentum with a cited jet resolution, and state every assumption |

Z→μμ alone gives about 55 to 65 events at 10.8 ab⁻¹ before efficiency, so
the paper's headline will come from the third scenario. Say so plainly in
the text. The ρν channel is future work, not a result.

### 5. Figures

Three figures, each generated by a committed script:

1. Reconstructed acoplanarity with the simulated SM expectation after
   selection, and the truth-level curve for reference.
2. The v_min response: reconstructed against true, and the purity of
   "v_min above V" as a function of V.
3. Expected 95% CL exclusion on v_ψ for both alternatives, with bands, for
   each scenario in work package 4.

### 6. Paper

- Download the v1 source from `https://arxiv.org/src/2603.28868` into
  `paper/v2/`. Extract it in an empty directory. Edit that source; keep the
  authors' voice and structure.
- Apply every text correction in `AUDIT.md` finding 9 and the reframing in
  finding 6.
- Replace all numbers and figures. Describe the selection, including the
  vertex-quality cut, the detector model with citations, the statistical
  method, and what is and is not simulated.
- Remove the claims "model-independent", "test of nonlocality in the strict
  sense that Bell's theorem requires", the 5% timelike paragraph and the
  2.4 μm impact-parameter sentence.
- Add citations: Bancal et al., Nature Phys. 8, 867 (2012); Yin et al., PRL
  110, 260407 (2013); the FCC-ee luminosity and detector references. Check
  each reference against its source before adding it.
- Write `paper/v2/CHANGES.md`: every changed claim and number, old and new.
- Delete or clearly mark as obsolete the old `paper/paper.tex`.

### 7. Final report

End with a short report: the gate results as a table, the headline numbers
with their bands, everything that was assumed rather than simulated, and
anything you could not do.

## Known open points for the author

- FCC-ee's baseline ZH run is at 240 GeV with 10.8 ab⁻¹. This handoff uses
  250 GeV as instructed. State the energy and luminosity assumption
  explicitly in the paper.
- Backgrounds are not in scope here. The paper must say so, or a later pass
  adds ZZ, Zγ* and τ-decay cross-feed.

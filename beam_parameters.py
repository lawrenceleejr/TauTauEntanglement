"""
FCC-ee beam, physics and detector parameters for the ZH run at 240 GeV.

Every value carries its source.  ASSUMPTION marks values not taken from a
document; each is listed in paper/v2/CHANGES.md and in the paper.

Sources
  [FSR1] FCC Feasibility Study Report Vol. 1, arXiv:2505.00272
  [FSR2] FCC Feasibility Study Report Vol. 2, arXiv:2505.00274
  [CLD]  N. Bacchetta et al., CLD - A Detector Concept for the FCC-ee,
         arXiv:1911.12230
  [ZH]   A. Li et al., Model-independent ZH production cross section at
         FCC-ee, arXiv:2512.21290
  [YR4]  LHC Higgs XS WG, CERN Yellow Report 4, arXiv:1610.07922
  [PDG]  S. Navas et al. (PDG), Phys. Rev. D 110, 030001 (2024)
"""
import math

SQRTS = 240.0                       # GeV
INTEGRATED_LUMI_AB = 10.8           # ab^-1, 4 IPs, 3 years [FSR2 Table 2.1]
CROSSING_ANGLE = 0.030              # rad, full angle [FSR2 Table 1.2]
BEAM_ENERGY_SPREAD = 0.00176        # σ_E/E incl. beamstrahlung [FSR2 Table 2]
SIGMA_X_STAR_M = 13e-6              # [FSR2 Table 2]
SIGMA_Y_STAR_M = 32e-9              # [FSR2 Table 2]
SIGMA_Z_BUNCH_M = 5.59e-3           # incl. beamstrahlung [FSR2 Table 2]

# Luminous region (derived): two Gaussian beams of equal size overlap with
# σ_lum = σ*/√2 transversely; longitudinally the crossing angle shortens it to
# (σz/√2)/sqrt(1 + Φ²) with Piwinski angle Φ = σz tan(θc/2)/σx*.
_PHI = SIGMA_Z_BUNCH_M * math.tan(CROSSING_ANGLE / 2) / SIGMA_X_STAR_M
LUMINOUS_REGION_MM = (SIGMA_X_STAR_M / math.sqrt(2) * 1e3,
                      SIGMA_Y_STAR_M / math.sqrt(2) * 1e3,
                      SIGMA_Z_BUNCH_M / math.sqrt(2) / math.sqrt(1 + _PHI**2) * 1e3)

SIGMA_ZH_FB = 200.0                 # unpolarised, 240 GeV [ZH]
BR_H_TAUTAU = 0.06272               # m_H = 125.0 GeV [YR4 Table 174]
BR_Z = {"ee": 0.033632, "mumu": 0.033662, "qq": 0.69911}   # [PDG]
BR_TAU_PINU = 0.1082                # [PDG]
M_TAU = 1.77693                     # GeV [PDG]
CTAU_TAU_MM = 0.08703               # mm [PDG]

# ---------------------------------------------------------------------------
# Detector model (CLD).  Lengths in metres, momenta in GeV.
#   σ(pT)/pT² = a(θ) ⊕ b(θ)/(p sin^{3/2}θ)      [CLD Eq. 3, Table 15]
#   σ(d0)     = d0_a ⊕ d0_b/(p sin^{3/2}θ)       [CLD Eq. 2: a = 5 μm, b = 15 μm GeV]
#   σ(z0)     = same as d0                        ASSUMPTION (CLD gives a figure only)
#   σ(θ), σ(φ) = ang_a ⊕ ang_b/(p sin^{1/2}θ)    ang_a = 0.023 mrad [CLD p. 39];
#                                                  ang_b = 1 mrad GeV ASSUMPTION
#   σ(E)/E    = 15%/√E ⊕ 1% (photons)            [FSR1 Table 12, CLD p. 43]
#   jets: σ(E)/E = 5% per jet (RMS90 4.5-5% for 45 GeV jets) [CLD p. 52];
#         jet directions exact                              ASSUMPTION (projection)
# ---------------------------------------------------------------------------
DETECTOR = {
    "pt_table_theta_deg": (10.0, 30.0, 50.0, 70.0, 89.0),   # [CLD Table 15]
    "pt_table_a": (7.4e-5, 1.8e-5, 9.2e-6, 8.3e-6, 8.6e-6),  # GeV^-1
    "pt_table_b": (0.010, 0.005, 0.004, 0.003, 0.003),
    "d0_a": 5e-6, "d0_b": 15e-6,
    "z0_a": 5e-6, "z0_b": 15e-6,
    "ang_a": 0.023e-3, "ang_b": 1e-3,
    "ecal_stoch": 0.15, "ecal_const": 0.01,
    "ecal_cos_max": 0.99,               # ASSUMPTION: photon acceptance |cos θ| < 0.99
    "photon_e_min": 0.5,                # ASSUMPTION: photon threshold, GeV
    "pi_photon_veto_angle": 0.10,       # ASSUMPTION: π ν ID fails if a photon
    "pi_photon_veto_e": 1.0,            #   above 1 GeV lies within 0.1 rad of the pion
    "lep_photon_cone": 0.10,            # ASSUMPTION: FSR recovery cone, rad
    "jet_rel_e": 0.05,
    "beam_energy_spread": BEAM_ENERGY_SPREAD,
    "luminous_region_m": tuple(x * 1e-3 for x in LUMINOUS_REGION_MM),
    "m_pi": 0.13957, "m_mu": 0.1056584, "m_e": 0.000511,
}

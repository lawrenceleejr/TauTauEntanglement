"""
Configuration and physical constants for the tau-tau entanglement analysis.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
M_TAU = 1.77686        # tau mass [GeV]
CTAU_TAU = 87.03e-6    # tau c*tau [m]  (87.03 um)
M_HIGGS = 125.0        # Higgs mass [GeV] (use PDG-like round value)
M_Z = 91.1876          # Z mass [GeV]
M_MU = 0.10566         # muon mass [GeV]
M_PI = 0.13957         # charged pion mass [GeV]
M_PI0 = 0.13498        # neutral pion mass [GeV]
M_RHO = 0.77526        # rho(770) mass [GeV]
SQRT_S = 240.0         # centre-of-mass energy [GeV]; see set_sqrts()
C_LIGHT = 2.99792458e8 # speed of light [m/s]

# Beam four-momenta (symmetric e+e- at 240 GeV)
E_BEAM = SQRT_S / 2.0  # 120 GeV per beam
P_BEAM_MINUS = np.array([E_BEAM, 0.0, 0.0,  E_BEAM])  # (E, px, py, pz) e-
P_BEAM_PLUS  = np.array([E_BEAM, 0.0, 0.0, -E_BEAM])  # (E, px, py, pz) e+
P_BEAM_TOTAL = P_BEAM_MINUS + P_BEAM_PLUS               # (sqrt(s), 0, 0, 0)


def set_sqrts(sqrts):
    """Set the centre-of-mass energy assumed by the analysis.

    The recoil Higgs four-momentum p_H = p_beam - p_Z, and with it the Higgs
    rest frame and every spin observable, depends on this value.  It must
    equal the energy at which the sample was generated: parse_events() checks
    this and refuses to continue on a mismatch.  Consumers must read
    config.P_BEAM_TOTAL / config.P_BEAM_MINUS at call time.
    """
    global SQRT_S, E_BEAM, P_BEAM_MINUS, P_BEAM_PLUS, P_BEAM_TOTAL
    SQRT_S = float(sqrts)
    E_BEAM = SQRT_S / 2.0
    P_BEAM_MINUS = np.array([E_BEAM, 0.0, 0.0,  E_BEAM])
    P_BEAM_PLUS = np.array([E_BEAM, 0.0, 0.0, -E_BEAM])
    P_BEAM_TOTAL = P_BEAM_MINUS + P_BEAM_PLUS


def lumi_from_events(n_events, sigma_eff_fb):
    """Integrated luminosity needed for n_events at 100% efficiency.

    Returns (fb^-1, ab^-1).  1 ab^-1 = 1000 fb^-1.
    """
    if sigma_eff_fb <= 0:
        return 0.0, 0.0
    lumi_fb = n_events / sigma_eff_fb
    return lumi_fb, lumi_fb * 1e-3


# Acoplanarity coefficient for a CP-even scalar with both taus -> pi nu:
# dN/dphi ~ 1 + B cos(phi), B = -(pi/4)^2 from averaging sin(theta+) sin(theta-)
# over the two polar angles.
B_ACOPLANARITY_SM = -np.pi**2 / 16.0

# ---------------------------------------------------------------------------
# Cross sections and branching ratios (for luminosity estimation)
# ---------------------------------------------------------------------------
SIGMA_ZH_FB = 240.0           # sigma(e+e- -> ZH) at 240 GeV [fb]  (ILC TDR)
BR_H_TAUTAU = 0.0627          # BR(H -> tau tau)  (PDG 2024)
BR_Z_MUMU   = 0.03366         # BR(Z -> mu mu)    (PDG 2024)
BR_TAU_PI_NU  = 0.1082        # BR(tau -> pi nu)  (PDG 2024)
BR_TAU_RHO_NU = 0.2549        # BR(tau -> rho nu) (PDG 2024)

# ---------------------------------------------------------------------------
# PDG IDs
# ---------------------------------------------------------------------------
PDGID_TAU_MINUS  = 15
PDGID_TAU_PLUS   = -15
PDGID_MU_MINUS   = 13
PDGID_MU_PLUS    = -13
PDGID_NU_TAU     = 16
PDGID_NU_TAU_BAR = -16
PDGID_NU_MU      = 14
PDGID_NU_MU_BAR  = -14
PDGID_PI_PLUS    = 211
PDGID_PI_MINUS   = -211
PDGID_PI0        = 111
PDGID_HIGGS      = 25
PDGID_Z          = 23

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------
# Accepted tau decay modes.  Options:
#   "pi_nu"   — single charged pion (tau -> pi nu), cleanest channel
#   "rho_nu"  — rho meson (tau -> rho nu -> pi pi0 nu), higher BR
# Set to a list to accept multiple modes, e.g. ["pi_nu", "rho_nu"].
ALLOWED_DECAY_MODES = ["pi_nu"]

# Binning for spacetime interval plots
N_BINS_SPACETIME = 8
N_BINS_SIGNAL_SPEED = 8

# Bootstrap resampling for uncertainties
N_BOOTSTRAP = 1000

# v_psi hypotheses to scan [in units of c]
V_PSI_SCAN = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0,
                        200.0, 500.0, 1000.0])

# Output directory for plots
OUTPUT_DIR = "plots"

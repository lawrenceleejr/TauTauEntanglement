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
SQRT_S = 240.0         # centre-of-mass energy [GeV]
C_LIGHT = 2.99792458e8 # speed of light [m/s]

# Beam four-momenta (symmetric e+e- at 240 GeV)
E_BEAM = SQRT_S / 2.0  # 120 GeV per beam
P_BEAM_MINUS = np.array([E_BEAM, 0.0, 0.0,  E_BEAM])  # (E, px, py, pz) e-
P_BEAM_PLUS  = np.array([E_BEAM, 0.0, 0.0, -E_BEAM])  # (E, px, py, pz) e+
P_BEAM_TOTAL = P_BEAM_MINUS + P_BEAM_PLUS               # (240, 0, 0, 0)

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
# Require both taus decay to pi nu (single charged pion + neutrino)
REQUIRED_DECAY_MODE = "pi_nu"

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

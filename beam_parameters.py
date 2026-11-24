"""
FCC-ee beam and luminous-region parameters for the ZH run.

Every value carries its source.  PROVISIONAL values are placeholders that must
be replaced by cited numbers before production samples are generated.
"""
SQRTS = 240.0                     # GeV
CROSSING_ANGLE = 0.030            # rad, full angle            PROVISIONAL
BEAM_ENERGY_SPREAD = 0.00185      # sigma_E/E per beam          PROVISIONAL
# Luminous-region r.m.s. (x, y, z) in mm, used for the primary vertex
LUMINOUS_REGION_MM = (0.0137 / 1.41421356, 0.000036 / 1.41421356, 0.30)  # PROVISIONAL
INTEGRATED_LUMI_AB = 10.8         # ab^-1, 240 GeV run, 4 IPs   PROVISIONAL
SIGMA_ZH_FB = 200.0               # fb, unpolarised, 240 GeV    PROVISIONAL

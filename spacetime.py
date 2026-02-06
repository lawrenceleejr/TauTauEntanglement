"""
Spacetime interval and causal signal speed calculations.

Given two tau decay 4-positions (t, x, y, z), compute:
  - Spatial separation  dr = |x1 - x2|
  - Time separation     dt = |t1 - t2|
  - Spacetime interval  ds2 = c^2 dt^2 - dr^2
  - Signal speed        v_signal = dr / dt  (in units of c)
  - Classification: spacelike (ds2 < 0) or timelike (ds2 > 0)
"""
import numpy as np
from config import C_LIGHT


def compute_spacetime_interval(decay_vtx_1, decay_vtx_2):
    """Compute spacetime interval between two decay events.

    Parameters
    ----------
    decay_vtx_1, decay_vtx_2 : dict or tuple
        Each must provide (t, x, y, z) in seconds and metres.
        If dict, expects keys 'decay_vertex_xyz' (m) and 'decay_vertex_t' (s).
        If ndarray of length 4, expects (t, x, y, z).

    Returns
    -------
    dict with keys:
        'dr_m'          : spatial separation [m]
        'dt_s'          : time separation [s]
        'ds2_m2'        : spacetime interval [m^2]  (c^2 dt^2 - dr^2)
        'v_signal_c'    : signal speed in units of c (dr / (c * dt))
        'is_spacelike'  : bool
        'signed_ds_m'   : sign(ds2) * sqrt(|ds2|) [m]  (for plotting)
    """
    if isinstance(decay_vtx_1, dict):
        xyz1 = decay_vtx_1['decay_vertex_xyz']
        t1 = decay_vtx_1['decay_vertex_t']
        xyz2 = decay_vtx_2['decay_vertex_xyz']
        t2 = decay_vtx_2['decay_vertex_t']
    else:
        t1, xyz1 = decay_vtx_1[0], decay_vtx_1[1:4]
        t2, xyz2 = decay_vtx_2[0], decay_vtx_2[1:4]

    dx = xyz2 - xyz1
    dr = np.linalg.norm(dx)   # metres
    dt = abs(t2 - t1)         # seconds

    # ds^2 = c^2 dt^2 - dr^2
    ds2 = (C_LIGHT * dt)**2 - dr**2

    # Signal speed: v = dr / dt in units of c
    if dt > 0:
        v_signal = dr / (C_LIGHT * dt)
    else:
        v_signal = np.inf if dr > 0 else 0.0

    is_spacelike = ds2 < 0

    # Signed sqrt for plotting: negative = spacelike, positive = timelike
    signed_ds = np.sign(ds2) * np.sqrt(abs(ds2))

    return {
        'dr_m': dr,
        'dt_s': dt,
        'ds2_m2': ds2,
        'v_signal_c': v_signal,
        'is_spacelike': is_spacelike,
        'signed_ds_m': signed_ds,
        'signed_ds_mm': signed_ds * 1e3,  # in mm for nicer plotting
        'dr_mm': dr * 1e3,
        'dt_ps': dt * 1e12,  # picoseconds
    }


def compute_truth_intervals(event):
    """Compute spacetime intervals using truth decay vertices."""
    tau_m = event.tau_minus
    tau_p = event.tau_plus

    vtx1 = {'decay_vertex_xyz': tau_m.decay_vertex[1:4],
             'decay_vertex_t': tau_m.decay_vertex[0]}
    vtx2 = {'decay_vertex_xyz': tau_p.decay_vertex[1:4],
             'decay_vertex_t': tau_p.decay_vertex[0]}

    return compute_spacetime_interval(vtx1, vtx2)


def compute_reco_intervals(reco):
    """Compute spacetime intervals using Jeans-reconstructed vertices."""
    vtx1 = {'decay_vertex_xyz': reco['tau_minus']['decay_vertex_xyz'],
             'decay_vertex_t': reco['tau_minus']['decay_vertex_t']}
    vtx2 = {'decay_vertex_xyz': reco['tau_plus']['decay_vertex_xyz'],
             'decay_vertex_t': reco['tau_plus']['decay_vertex_t']}

    return compute_spacetime_interval(vtx1, vtx2)

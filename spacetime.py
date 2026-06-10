"""
Spacetime interval and causal signal speed calculations.

Given two tau decay 4-positions (t, x, y, z), compute:
  - Spatial separation  dr = |x1 - x2|
  - Time separation     dt = |t1 - t2|
  - Spacetime interval  ds2 = c^2 dt^2 - dr^2
  - Signal speed        v_signal = dr / dt  (in units of c)
  - Classification: spacelike (ds2 < 0) or timelike (ds2 > 0)

The interval ds2 is Lorentz invariant; dr, dt and hence v_signal are not.
The analysis quotes v_signal in the HIGGS REST FRAME, where the taus are
exactly back-to-back: with i.i.d. exponential decay lengths L1, L2 the
ratio (L1-L2)/(L1+L2) is uniform on [-1, 1], so
    v_signal = beta (L1+L2)/|L1-L2|   has the exact Pareto tail
    P(v_signal > v) = beta / v        (v >= beta).
This closed form underpins the analytic sensitivity projections.
"""
import numpy as np
from config import C_LIGHT


def boost_position(t, xyz, beta):
    """Boost an event 4-position (t [s], xyz [m]) by velocity beta (3-vector).

    Transforms into the frame moving with velocity beta (e.g. pass
    beta = p_H/E_H to go to the Higgs rest frame).
    """
    b2 = float(np.dot(beta, beta))
    if b2 < 1e-30:
        return t, np.array(xyz, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    ct = C_LIGHT * t
    bx = float(np.dot(beta, xyz))
    ct_new = gamma * (ct - bx)
    xyz_new = xyz + (((gamma - 1.0) * bx / b2) - gamma * ct) * beta
    return ct_new / C_LIGHT, xyz_new


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


def _intervals_with_higgs_frame(t1, xyz1, t2, xyz2, beta_H):
    """Lab-frame intervals plus Higgs-rest-frame v_signal/dr/dt.

    ds2 is invariant (cross-checked); dr, dt and v_signal are evaluated in
    both frames.  Higgs-frame values carry the '_hf' suffix.
    """
    iv = compute_spacetime_interval(
        {'decay_vertex_xyz': xyz1, 'decay_vertex_t': t1},
        {'decay_vertex_xyz': xyz2, 'decay_vertex_t': t2})

    if beta_H is not None:
        t1h, x1h = boost_position(t1, xyz1, beta_H)
        t2h, x2h = boost_position(t2, xyz2, beta_H)
        ivh = compute_spacetime_interval(
            {'decay_vertex_xyz': x1h, 'decay_vertex_t': t1h},
            {'decay_vertex_xyz': x2h, 'decay_vertex_t': t2h})
        iv['v_signal_c_hf'] = ivh['v_signal_c']
        iv['dr_mm_hf'] = ivh['dr_mm']
        iv['dt_ps_hf'] = ivh['dt_ps']
        iv['signed_ds_mm_hf'] = ivh['signed_ds_mm']
    else:
        iv['v_signal_c_hf'] = iv['v_signal_c']
        iv['dr_mm_hf'] = iv['dr_mm']
        iv['dt_ps_hf'] = iv['dt_ps']
        iv['signed_ds_mm_hf'] = iv['signed_ds_mm']
    return iv


def compute_truth_intervals(event):
    """Compute spacetime intervals using truth decay vertices."""
    tau_m = event.tau_minus
    tau_p = event.tau_plus

    p_H = tau_m.tau_p4 + tau_p.tau_p4
    beta_H = p_H[1:4] / p_H[0]

    return _intervals_with_higgs_frame(
        tau_m.decay_vertex[0], tau_m.decay_vertex[1:4],
        tau_p.decay_vertex[0], tau_p.decay_vertex[1:4],
        beta_H)


def compute_reco_intervals(reco):
    """Compute spacetime intervals using Jeans-reconstructed vertices.

    The Higgs-frame quantities use the recoil-reconstructed p_H (from the
    measured muons and the known beam energy), not the noisy tau sum.
    """
    p_H = reco.get('p_H')
    beta_H = p_H[1:4] / p_H[0] if p_H is not None else None

    return _intervals_with_higgs_frame(
        reco['tau_minus']['decay_vertex_t'], reco['tau_minus']['decay_vertex_xyz'],
        reco['tau_plus']['decay_vertex_t'], reco['tau_plus']['decay_vertex_xyz'],
        beta_H)

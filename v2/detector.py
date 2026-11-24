"""
Parametrised FCC-ee detector response, vectorised over events.

All resolution parameters live in DETECTOR and must carry a citation (see
beam_parameters.py and paper/v2 for the sources).  `measure()` turns a
generated sample into the quantities a detector provides:

  * charged tracks (pions, leptons): momentum and direction smeared, and the
    track line displaced at its point of closest approach to the true primary
    vertex by the d0 / z0 resolution;
  * photons: detected inside the calorimeter acceptance above a threshold,
    energy smeared; photons close to a pion spoil the τ -> π ν identification,
    photons close to a lepton are added to it;
  * the hadronic Z (projection only): four-momentum smeared with a jet
    resolution;
  * the primary vertex: least-squares fit of the two lepton tracks with the
    luminous-region constraint;
  * the beam energies known to the analysis: nominal, while the true beam
    energies scatter with the beam energy spread.
"""
import numpy as np

import beam_parameters as bp
from v2 import kin

DETECTOR = dict(bp.DETECTOR)


def sigma_inv_pt(p, sin_t, det=DETECTOR):
    """σ(1/pT) = σ(pT)/pT² with θ-dependent a, b interpolated from CLD Table 15."""
    th = np.degrees(np.arcsin(np.clip(sin_t, 0, 1)))
    a = np.interp(th, det["pt_table_theta_deg"], det["pt_table_a"])
    b = np.interp(th, det["pt_table_theta_deg"], det["pt_table_b"])
    return np.hypot(a, b / (p * np.maximum(sin_t, 0.05)**1.5))


def sigma_d0(p, sin_t, det=DETECTOR):
    return np.hypot(det["d0_a"], det["d0_b"] / (p * np.maximum(sin_t, 0.05)**1.5))


def sigma_z0(p, sin_t, det=DETECTOR):
    return np.hypot(det["z0_a"], det["z0_b"] / (p * np.maximum(sin_t, 0.05)**1.5))


def sigma_angle(p, sin_t, det=DETECTOR):
    return np.hypot(det["ang_a"], det["ang_b"] / (p * np.sqrt(np.maximum(sin_t, 0.05))))


def _perp_axes(u):
    """Unit vectors (e_d0, e_z0) perpendicular to track direction u:
    e_d0 is transverse to the beam (z) axis, e_z0 completes the triad."""
    z = np.array([0.0, 0.0, 1.0])
    e1 = kin.unit(np.cross(z, u))
    e2 = np.cross(u, e1)
    return e1, e2


def smear_track(p4, point, pv_true, mass, rng, det=DETECTOR, smear=True):
    """Return measured (p4, pca_point, direction, sigma_1, sigma_2), where
    sigma_1 = σ(d0) and sigma_2 = σ(z0) sin θ are the position errors along
    the two directions perpendicular to the track (see _perp_axes).

    point : any point on the true track (e.g. the decay vertex), metres
    The measured line passes through the true PCA to the true primary vertex,
    displaced by (d0, z0) errors; its direction is smeared in θ and φ."""
    u = kin.unit(kin.p3(p4))
    p = kin.mag(kin.p3(p4))
    sin_t = np.sqrt(np.maximum(1 - u[:, 2]**2, 1e-12))
    s = -kin.dot3(point - pv_true, u)
    pca = point + s[:, None] * u
    sd0, sz0 = sigma_d0(p, sin_t, det), sigma_z0(p, sin_t, det) * sin_t
    if not smear:
        return p4.copy(), pca, u, sd0, sz0
    e1, e2 = _perp_axes(u)
    # z0 is measured along the beam axis; perpendicular to the track that is
    # a displacement of z0 * sin(theta) along e2
    pca_m = pca + (rng.normal(size=len(p)) * sd0)[:, None] * e1 \
        + (rng.normal(size=len(p)) * sz0)[:, None] * e2
    # direction: θ and φ errors
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    sa = sigma_angle(p, sin_t, det)
    th = th + rng.normal(size=len(p)) * sa
    ph = ph + rng.normal(size=len(p)) * sa / np.maximum(sin_t, 0.05)
    um = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    pt = p * sin_t
    inv = 1 / pt + rng.normal(size=len(p)) * sigma_inv_pt(p, sin_t, det)
    pt_m = 1 / np.maximum(inv, 1e-6)
    pm = pt_m / np.maximum(np.sin(th), 1e-6)
    e = np.sqrt(pm**2 + mass**2)
    return kin.four(e, pm[:, None] * um), pca_m, um, sd0, sz0


def fit_primary_vertex(points, dirs, sig_d0, sig_z0, det=DETECTOR):
    """Least-squares vertex from straight tracks plus the luminous region.

    points, dirs: lists of (N, 3) arrays; returns vertex (N, 3), cov (N, 3, 3)."""
    n = len(points[0])
    lr = np.array(det["luminous_region_m"])
    W = np.zeros((n, 3, 3)) + np.diag(1 / lr**2)
    b = np.zeros((n, 3))
    eye = np.eye(3)
    for a, u, s1, s2 in zip(points, dirs, sig_d0, sig_z0):
        e1, e2 = _perp_axes(u)
        Wi = (e1[:, :, None] * e1[:, None, :]) / s1[:, None, None]**2 \
            + (e2[:, :, None] * e2[:, None, :]) / s2[:, None, None]**2
        W += Wi
        b += np.einsum("nij,nj->ni", Wi, a)
    cov = np.linalg.inv(W)
    return np.einsum("nij,nj->ni", cov, b), cov


def nominal_beams(sqrts=None, crossing=None):
    sqrts = bp.SQRTS if sqrts is None else sqrts
    e = sqrts / 2
    th = (bp.CROSSING_ANGLE if crossing is None else crossing) / 2
    return (np.array([e, e * np.sin(th), 0, e * np.cos(th)]),
            np.array([e, e * np.sin(th), 0, -e * np.cos(th)]))


def measure(a, rng, sqrts, zmode, smear=True, det=DETECTOR, beam_effects=True):
    """Detector-level quantities for every event of sample `a`.

    Returns a dict of arrays.  `pass_id` flags events whose τ candidates pass
    the τ -> π ν identification (no photon above threshold near the pion)."""
    n = len(a["pi_m"])
    pv_true = a["pv"][:, 1:]
    m = {}
    for lab in ("m", "p"):
        p4, pca, u, sd0, sz0 = smear_track(a[f"pi_{lab}"], a[f"dv_{lab}"][:, 1:],
                                           pv_true, det["m_pi"], rng, det, smear)
        m[f"pi_{lab}"], m[f"pi_{lab}_pt"], m[f"pi_{lab}_dir"] = p4, pca, u
        m[f"pi_{lab}_sd0"], m[f"pi_{lab}_sz0"] = sd0, sz0

    # photons
    g = a["photon_p4"]
    ge = a["photon_event"]
    gdir = kin.unit(kin.p3(g))
    det_ok = (np.abs(gdir[:, 2]) < det["ecal_cos_max"]) & (g[:, 0] > det["photon_e_min"])
    e_meas = g[:, 0].copy()
    if smear:
        rel = np.hypot(det["ecal_stoch"] / np.sqrt(np.maximum(g[:, 0], 1e-3)), det["ecal_const"])
        e_meas = np.maximum(g[:, 0] * (1 + rng.normal(size=len(g)) * rel), 0)
    g_meas = kin.four(e_meas, e_meas[:, None] * gdir)
    near_pi = np.zeros(len(g), bool)
    for lab in ("m", "p"):
        c = kin.dot3(gdir, m[f"pi_{lab}_dir"][ge])
        near_pi |= c > np.cos(det["pi_photon_veto_angle"])
    veto = det_ok & near_pi & (g[:, 0] > det["pi_photon_veto_e"])
    pass_id = np.ones(n, bool)
    pass_id[ge[veto]] = False
    m["pass_id"] = pass_id

    # Z system
    if zmode in ("mumu", "ee"):
        mass = det["m_mu"] if zmode == "mumu" else det["m_e"]
        leps, lpts, ldirs, lsd0, lsz0 = [], [], [], [], []
        for lab in ("m", "p"):
            p4, pca, u, sd0, sz0 = smear_track(a[f"lep_{lab}"], pv_true + 0.0, pv_true,
                                               mass, rng, det, smear)
            leps.append(p4); lpts.append(pca); ldirs.append(u); lsd0.append(sd0); lsz0.append(sz0)
        near_l = np.zeros(len(g), bool)
        for u in ldirs:
            near_l |= kin.dot3(gdir, u[ge]) > np.cos(det["lep_photon_cone"])
        fsr = det_ok & near_l & ~near_pi
        pz = leps[0] + leps[1]
        np.add.at(pz, ge[fsr], g_meas[fsr])
        pv, pv_cov = fit_primary_vertex(lpts, ldirs, lsd0, lsz0, det)
        m["lep_m"], m["lep_p"] = leps
    else:
        fsr = np.zeros(len(g), bool)
        pz = a["z"].copy()
        if smear:
            rel = det["z_hadronic_rel_e"]
            pz[:, 0] *= 1 + rng.normal(size=n) * rel
            pz[:, 1:] += rng.normal(size=(n, 3)) * (rel * a["z"][:, 0] / np.sqrt(3))[:, None]
        # projection: primary vertex resolution taken equal to the leptonic case
        # (two high-momentum tracks + luminous region): conservative for Z -> qq.
        sd = sigma_d0(np.full(n, 45.0), np.ones(n), det)
        u1 = kin.unit(kin.p3(a["z"]) + 1e-9)
        u2 = kin.unit(np.cross(u1, [0, 0, 1.0]) + 1e-9)
        pts = [pv_true + (rng.normal(size=(n, 1)) * sd[:, None]) * _perp_axes(u)[0]
               + (rng.normal(size=(n, 1)) * sd[:, None]) * _perp_axes(u)[1]
               if smear else pv_true for u in (u1, u2)]
        pv, pv_cov = fit_primary_vertex(pts, [u1, u2], [sd, sd], [sd, sd], det)
    m["p_z"] = pz
    m["pv"], m["pv_cov"] = (pv, pv_cov) if smear else (pv_true.copy(), pv_cov)

    other = det_ok & ~fsr & ~(near_pi & (g[:, 0] > det["pi_photon_veto_e"]))
    pg = np.zeros((n, 4))
    np.add.at(pg, ge[other], g_meas[other])
    m["p_photons"] = pg

    # beam energies known to the analysis
    bm, bpl = nominal_beams(sqrts, None if beam_effects else 0.0)
    m["beam_minus"] = np.broadcast_to(bm, (n, 4)).copy()
    m["beam_plus"] = np.broadcast_to(bpl, (n, 4)).copy()
    if smear and beam_effects:
        # true beams are nominal; the analysis's knowledge is off by the spread
        for key, b in (("beam_minus", bm), ("beam_plus", bpl)):
            d = 1 + rng.normal(size=n) * det["beam_energy_spread"]
            m[key] = b[None, :] * d[:, None]
    return m

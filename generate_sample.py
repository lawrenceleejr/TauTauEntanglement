#!/usr/bin/env python3
"""
Generate e+e- -> ZH, H -> tau+ tau-, tau -> pi nu with Pythia 8 (pythia8mc).

The τ decays are done by Pythia's TauDecays, which applies the H -> ττ spin
correlations.  Initial-state radiation (lepton PDF and QED shower), final-state
QED radiation (including off the τ leptons), the beam energy spread, the
crossing angle and the luminous region are all on by default.

Output is a compressed .npz file of fixed-size per-event arrays (units: GeV,
metres, seconds) plus a ragged list of final-state photons, and a JSON file
with the settings, seeds and generator cross section.  `load_sample()` reads
it back and `to_event_records()` converts events to parse_hepmc.EventRecord
for use with the event-by-event code.

Usage:
    python generate_sample.py --zmode mumu --n 500000 --seed 1 --out samples/zh240_mumu_A
    python generate_sample.py --zmode qq --n 250000 --seed 7 --out samples/zh240_qq_A
    python generate_sample.py --sqrts 250 --no-beam-effects --n 20000 --seed 99 \
        --out samples/zh250_mumu_whizard_check
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import beam_parameters as bp

MM = 1e-3                      # mm -> m
MM_OVER_C = 1e-3 / 2.99792458e8   # mm/c -> s

Z_MODES = {
    "mumu": "23:onIfAny = 13",
    "ee": "23:onIfAny = 11",
    "qq": "23:onIfAny = 1 2 3 4 5",
}

FOUR = ("beams", "higgs", "z", "tau_m", "tau_p", "pi_m", "pi_p", "nu_m",
        "nu_p", "lep_m", "lep_p", "jet_1", "jet_2")
VTX = ("pv", "dv_m", "dv_p")


def pythia_settings(sqrts, zmode, seed, beam_effects=True):
    """Return the list of Pythia settings for one run."""
    e = sqrts / 2.0
    s = [
        "Beams:idA = 11", "Beams:idB = -11",
        "PDF:lepton = on",                 # ISR via electron structure function
        "HiggsSM:ffbar2HZ = on",
        "25:m0 = 125.0",
        "23:onMode = off", Z_MODES[zmode],
        "25:onMode = off", "25:onIfMatch = 15 -15",
        "15:onMode = off", "15:onIfMatch = 16 -211",
        # PDG 2024 τ mass and lifetime (Pythia defaults: 1.77682 GeV, 87.11 μm)
        f"15:m0 = {bp.M_TAU}", f"15:tau0 = {bp.CTAU_TAU_MM}",
        "Random:setSeed = on", f"Random:seed = {seed}",
        "Next:numberCount = 0", "Next:numberShowEvent = 0",
        "Next:numberShowInfo = 0", "Next:numberShowProcess = 0",
        "Init:showChangedSettings = off", "Init:showChangedParticleData = off",
        "Init:showProcesses = off",
    ]
    if zmode == "qq":
        # Only the Z four-momentum is used for the hadronic projection, so
        # the quarks are showered but not hadronised.  τ decays still run.
        s += ["HadronLevel:Hadronize = off"]
    if beam_effects:
        # The beam energy spread is NOT applied here: Pythia's
        # allowMomentumSpread corrupts its cross-section bookkeeping (factor
        # 0.57 with lepton PDFs).  It is applied in detector.py as an error
        # on the beam energies known to the analysis, which gives the same
        # error on the recoil four-momentum.
        th = bp.CROSSING_ANGLE / 2.0
        s += [
            "Beams:frameType = 3",
            f"Beams:pxA = {e * np.sin(th):.9f}", f"Beams:pzA = {e * np.cos(th):.9f}",
            f"Beams:pxB = {e * np.sin(th):.9f}", f"Beams:pzB = {-e * np.cos(th):.9f}",
            "Beams:allowVertexSpread = on",
            f"Beams:sigmaVertexX = {bp.LUMINOUS_REGION_MM[0]:.7f}",
            f"Beams:sigmaVertexY = {bp.LUMINOUS_REGION_MM[1]:.7f}",
            f"Beams:sigmaVertexZ = {bp.LUMINOUS_REGION_MM[2]:.7f}",
            "Beams:sigmaTime = 0.",
        ]
    else:
        s += [f"Beams:eCM = {sqrts}"]
    return s


def _p4(p):
    return (p.e(), p.px(), p.py(), p.pz())


def _vtx(p):
    return (p.tProd() * MM_OVER_C, p.xProd() * MM, p.yProd() * MM, p.zProd() * MM)


def _photon_tag(ev, i):
    """0: beam / ISR, 1: radiated by a τ, 2: radiated in the Z decay."""
    j = ev[i].mother1()
    for _ in range(40):
        if j <= 0:
            return 0
        aid = abs(ev[j].id())
        if aid == 15:
            return 1
        if aid == 23:
            return 2
        if aid == 25:
            return 1
        j = ev[j].mother1()
    return 0


def _worker(job):
    import pythia8mc as p8
    sqrts, zmode, seed, n, beam_effects = job
    py = p8.Pythia("", False)
    for s in pythia_settings(sqrts, zmode, seed, beam_effects):
        if not py.readString(s):
            raise RuntimeError(f"Pythia rejected setting: {s}")
    if not py.init():
        raise RuntimeError("Pythia init failed")

    out = {k: np.zeros((n, 4)) for k in FOUR + VTX}
    out["ptime_m"] = np.zeros(n)
    out["ptime_p"] = np.zeros(n)
    phot, phot_tag, phot_evt = [], [], []
    k = 0
    n_fail = 0
    while k < n:
        if not py.next():
            n_fail += 1
            continue
        ev = py.event
        taus = {}
        z_last = h_last = None
        leps = {}
        for i in range(ev.size()):
            p = ev[i]
            aid = abs(p.id())
            if aid == 15 and p.daughter1() > 0 and abs(ev[p.daughter1()].id()) != 15:
                dl = [ev[d] for d in p.daughterList()]
                pis = [d for d in dl if abs(d.id()) == 211]
                nus = [d for d in dl if abs(d.id()) == 16]
                if len(pis) == 1 and len(nus) == 1 and len(dl) == 2:
                    taus[p.id()] = (p, pis[0], nus[0])
            elif p.id() == 23 and p.daughter1() > 0 and ev[p.daughter1()].id() != 23:
                z_last = p
            elif p.id() == 25 and p.daughter1() > 0 and ev[p.daughter1()].id() != 25:
                h_last = p
            elif p.isFinal() and aid in (11, 13):
                leps[p.id()] = p
            elif p.isFinal() and p.id() == 22:
                phot.append(_p4(p))
                phot_tag.append(_photon_tag(ev, i))
                phot_evt.append(k)
        if 15 not in taus or -15 not in taus or z_last is None or h_last is None:
            # A τ decay with extra radiation not in the τ -> π ν topology.
            # Drop the event (photons appended for it are removed below).
            while phot_evt and phot_evt[-1] == k:
                phot.pop(); phot_tag.pop(); phot_evt.pop()
            n_fail += 1
            continue
        lid = 13 if zmode == "mumu" else 11
        out["beams"][k] = np.add(_p4(ev[1]), _p4(ev[2]))
        out["higgs"][k] = _p4(h_last)
        out["z"][k] = _p4(z_last)
        out["pv"][k] = _vtx(h_last)
        for sgn, lab in ((15, "m"), (-15, "p")):
            tau, pi, nu = taus[sgn]
            out[f"tau_{lab}"][k] = _p4(tau)
            out[f"pi_{lab}"][k] = _p4(pi)
            out[f"nu_{lab}"][k] = _p4(nu)
            out[f"dv_{lab}"][k] = _vtx(pi)
            out[f"ptime_{lab}"][k] = tau.tau() * MM_OVER_C
        if zmode != "qq":
            out["lep_m"][k] = _p4(leps[lid])
            out["lep_p"][k] = _p4(leps[-lid])
        else:
            # the Z decay quarks (before showering) stand in for the two jets
            out["jet_1"][k] = _p4(ev[z_last.daughter1()])
            out["jet_2"][k] = _p4(ev[z_last.daughter2()])
        k += 1
    out["photon_p4"] = np.array(phot, dtype=float).reshape(-1, 4)
    out["photon_tag"] = np.array(phot_tag, dtype=np.int8)
    out["photon_event"] = np.array(phot_evt, dtype=np.int64)
    info = py.infoPython()
    out["_sigma_fb"] = info.sigmaGen() * 1e12
    out["_sigma_err_fb"] = info.sigmaErr() * 1e12
    out["_n_tried"] = info.nTried()
    out["_n_fail"] = n_fail
    return out


def generate(sqrts, zmode, n, seed, beam_effects=True, n_proc=None):
    n_proc = n_proc or min(mp.cpu_count(), 8)
    sizes = [n // n_proc + (1 if i < n % n_proc else 0) for i in range(n_proc)]
    jobs = [(sqrts, zmode, seed * 1000 + i, m, beam_effects)
            for i, m in enumerate(sizes) if m > 0]
    with mp.Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    out = {}
    for key in FOUR + VTX + ("ptime_m", "ptime_p", "photon_p4", "photon_tag"):
        out[key] = np.concatenate([p[key] for p in parts])
    offs = np.cumsum([0] + sizes[:-1])
    out["photon_event"] = np.concatenate(
        [p["photon_event"] + o for p, o in zip(parts, offs)])
    sig = np.array([p["_sigma_fb"] for p in parts])
    err = np.array([p["_sigma_err_fb"] for p in parts])
    meta = {
        "sqrts": sqrts, "zmode": zmode, "n_events": int(n), "seed": seed,
        "worker_seeds": [j[2] for j in jobs], "beam_effects": beam_effects,
        "pythia_settings": pythia_settings(sqrts, zmode, seed, beam_effects),
        "sigma_gen_fb": float(np.average(sig, weights=1 / err**2)),
        "sigma_gen_err_fb": float(1 / np.sqrt(np.sum(1 / err**2))),
        "n_dropped": int(sum(p["_n_fail"] for p in parts)),
        "note": ("sigma_gen includes Pythia's BR(Z->mode) x BR(H->tau tau); "
                 "τ -> π ν branching fractions are NOT included."),
    }
    return out, meta


def save_sample(path, arrays, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path + ".npz", **arrays)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)


def load_sample(path):
    path = path[:-4] if path.endswith(".npz") else path
    with np.load(path + ".npz") as f:
        arrays = {k: f[k] for k in f.files}
    with open(path + ".json") as f:
        meta = json.load(f)
    return arrays, meta


def to_event_records(arrays, idx=None):
    """Convert (a subset of) a sample to parse_hepmc.EventRecord objects."""
    from parse_hepmc import EventRecord, TauDecayInfo
    idx = range(len(arrays["pi_m"])) if idx is None else idx
    recs = []
    for i in idx:
        def tau(lab, pid):
            return TauDecayInfo(
                tau_p4=arrays[f"tau_{lab}"][i].copy(), tau_pdgid=pid,
                decay_mode="pi_nu", charged_pion_p4=arrays[f"pi_{lab}"][i].copy(),
                neutral_pions_p4=[], neutrino_p4=arrays[f"nu_{lab}"][i].copy(),
                visible_p4=arrays[f"pi_{lab}"][i].copy(),
                production_vertex=arrays["pv"][i].copy(),
                decay_vertex=arrays[f"dv_{lab}"][i].copy())
        recs.append(EventRecord(
            event_number=int(i), mu_plus_p4=arrays["lep_p"][i].copy(),
            mu_minus_p4=arrays["lep_m"][i].copy(),
            tau_minus=tau("m", 15), tau_plus=tau("p", -15), is_pi_pi=True))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqrts", type=float, default=bp.SQRTS)
    ap.add_argument("--zmode", choices=sorted(Z_MODES), default="mumu")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--no-beam-effects", action="store_true",
                    help="head-on beams, no energy spread, no luminous region")
    ap.add_argument("--nproc", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    arrays, meta = generate(a.sqrts, a.zmode, a.n, a.seed,
                            beam_effects=not a.no_beam_effects, n_proc=a.nproc)
    meta["wall_time_s"] = round(time.time() - t0, 1)
    save_sample(a.out, arrays, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "pythia_settings"},
                     indent=1))


if __name__ == "__main__":
    main()

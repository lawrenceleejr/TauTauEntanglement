"""
HepMC3 parser for e+e- -> ZH -> mu+mu- tau+tau-(hadronic) events.

Extracts truth-level information and identifies tau decay modes.
"""
import numpy as np
import pyhepmc
from dataclasses import dataclass, field
from typing import Optional
from config import (
    PDGID_TAU_MINUS, PDGID_TAU_PLUS, PDGID_MU_MINUS, PDGID_MU_PLUS,
    PDGID_NU_TAU, PDGID_NU_TAU_BAR, PDGID_PI_PLUS, PDGID_PI_MINUS,
    PDGID_PI0, PDGID_HIGGS, PDGID_Z, C_LIGHT,
    ALLOWED_DECAY_MODES,
)


def _to_p4(particle):
    """Extract (E, px, py, pz) numpy array from a HepMC particle."""
    mom = particle.momentum
    return np.array([mom.e, mom.px, mom.py, mom.pz])


def _vertex_pos(vertex):
    """Extract (t, x, y, z) from a HepMC vertex position.

    HepMC3 stores position as (x, y, z, t) in mm and mm/c.
    We convert to metres for the spatial components and seconds for time.
    """
    pos = vertex.position
    # pos components are in mm (spatial) and mm/c (time)
    x_m = pos.x * 1e-3   # mm -> m
    y_m = pos.y * 1e-3
    z_m = pos.z * 1e-3
    t_s = pos.t * 1e-3 / C_LIGHT  # mm/c -> m/c -> s  (t_hepmc is in mm/c)
    return np.array([t_s, x_m, y_m, z_m])


@dataclass
class TauDecayInfo:
    """Information about a single tau decay."""
    tau_p4: np.ndarray                # truth tau (E, px, py, pz) [GeV]
    tau_pdgid: int                    # +15 or -15
    decay_mode: str                   # "pi_nu", "rho_nu", "a1_nu", "other"
    charged_pion_p4: np.ndarray       # leading charged pion (E, px, py, pz)
    neutral_pions_p4: list            # list of pi0 four-vectors
    neutrino_p4: np.ndarray           # truth neutrino (E, px, py, pz)
    visible_p4: np.ndarray            # sum of all visible decay products
    production_vertex: np.ndarray     # (t, x, y, z) of tau production [s, m]
    decay_vertex: np.ndarray          # (t, x, y, z) of tau decay [s, m]


@dataclass
class EventRecord:
    """Parsed event with all relevant physics objects."""
    event_number: int
    mu_plus_p4: np.ndarray
    mu_minus_p4: np.ndarray
    tau_minus: Optional[TauDecayInfo] = None
    tau_plus: Optional[TauDecayInfo] = None
    is_pi_pi: bool = False  # both taus decay to pi nu


def _classify_tau_decay(tau_particle):
    """Walk the tau decay tree and classify the decay mode.

    Returns TauDecayInfo or None if the decay products can't be found.
    """
    # Find the decay vertex of this tau
    end_vertex = tau_particle.end_vertex
    if end_vertex is None:
        return None

    # Collect final-state descendants (walk tree iteratively)
    charged_pions = []
    neutral_pions = []
    neutrinos = []
    other_charged = []
    other_neutral = []

    stack = list(end_vertex.particles_out)
    visited = set()

    while stack:
        p = stack.pop()
        pid = p.pid
        if p.id in visited:
            continue
        visited.add(p.id)

        # If this particle decays further, follow its children
        # (except for pi0 which we keep as-is since we want its 4-mom)
        if p.end_vertex is not None and abs(pid) != PDGID_PI0:
            stack.extend(p.end_vertex.particles_out)
            continue

        # Final-state particle (or pi0 which we treat as visible)
        apid = abs(pid)
        if apid == abs(PDGID_PI_PLUS):
            charged_pions.append(p)
        elif apid == PDGID_PI0:
            neutral_pions.append(p)
        elif apid in (abs(PDGID_NU_TAU), abs(PDGID_NU_TAU_BAR)):
            neutrinos.append(p)
        elif apid in (12, 14, 16):  # any neutrino
            neutrinos.append(p)
        else:
            # photons from pi0 decay, etc -- skip if we already got pi0
            # Also catches K, etc.
            if p.status == 1:
                if pid in (22,):  # photon -- likely from pi0
                    pass
                else:
                    other_charged.append(p) if pid != 0 else None

    n_charged_pi = len(charged_pions)
    n_neutral_pi = len(neutral_pions)

    # Classify
    if n_charged_pi == 1 and n_neutral_pi == 0 and len(neutrinos) >= 1:
        mode = "pi_nu"
    elif n_charged_pi == 1 and n_neutral_pi == 1 and len(neutrinos) >= 1:
        mode = "rho_nu"
    elif n_charged_pi == 1 and n_neutral_pi == 2 and len(neutrinos) >= 1:
        mode = "a1_nu"
    elif n_charged_pi == 3 and len(neutrinos) >= 1:
        mode = "a1_3pr_nu"
    else:
        mode = "other"

    # Sum neutrino 4-momenta
    nu_p4 = np.zeros(4)
    for nu in neutrinos:
        nu_p4 += _to_p4(nu)

    # Charged pion 4-momentum (use the one matching tau charge)
    if n_charged_pi >= 1:
        ch_pi_p4 = _to_p4(charged_pions[0])
    else:
        ch_pi_p4 = np.zeros(4)

    # Neutral pion 4-momenta
    pi0_p4_list = [_to_p4(p) for p in neutral_pions]

    # Total visible = all pions
    vis_p4 = ch_pi_p4.copy()
    for pi0 in pi0_p4_list:
        vis_p4 += pi0

    # Vertices
    prod_vtx = tau_particle.production_vertex
    decay_vtx = tau_particle.end_vertex

    prod_pos = _vertex_pos(prod_vtx) if prod_vtx is not None else np.zeros(4)
    decay_pos = _vertex_pos(decay_vtx) if decay_vtx is not None else np.zeros(4)

    return TauDecayInfo(
        tau_p4=_to_p4(tau_particle),
        tau_pdgid=tau_particle.pid,
        decay_mode=mode,
        charged_pion_p4=ch_pi_p4,
        neutral_pions_p4=pi0_p4_list,
        neutrino_p4=nu_p4,
        visible_p4=vis_p4,
        production_vertex=prod_pos,
        decay_vertex=decay_pos,
    )


def _find_particle(particles, pdgid, require_decays=False):
    """Find a particle by PDG ID in the event, preferring status 2."""
    candidates = [p for p in particles if p.pid == pdgid]
    if require_decays:
        candidates = [p for p in candidates if p.end_vertex is not None]
    if not candidates:
        return None
    # Prefer particles that actually decay (status 2 equivalent)
    decaying = [p for p in candidates if p.end_vertex is not None]
    if decaying:
        return decaying[0]
    return candidates[0]


def _find_stable_particle(particles, pdgid):
    """Find a final-state particle by PDG ID."""
    for p in particles:
        if p.pid == pdgid and p.status == 1:
            return p
    # Fallback: any particle with that PDGID
    for p in particles:
        if p.pid == pdgid:
            return p
    return None


def parse_events(filepath, max_events=None, require_pi_pi=True,
                  allowed_modes=None):
    """Parse a HepMC3 file and return a list of EventRecord objects.

    Parameters
    ----------
    filepath : str
        Path to the .hepmc file
    max_events : int or None
        Maximum number of events to process (None = all)
    require_pi_pi : bool
        Legacy flag. If True and *allowed_modes* is None, only keep events
        where both taus decay to pi+nu.  Ignored when *allowed_modes* is set.
    allowed_modes : list of str or None
        Accepted tau decay-mode names (e.g. ["pi_nu", "rho_nu"]).  Both taus
        must have a mode in this list for the event to be kept.  Defaults to
        ALLOWED_DECAY_MODES from config.py.

    Returns
    -------
    list of EventRecord
    """
    if allowed_modes is None:
        allowed_modes = ALLOWED_DECAY_MODES
    allowed_set = set(allowed_modes)

    events = []
    n_total = 0
    n_found_taus = 0
    n_pi_pi = 0
    n_selected_mode = 0
    decay_mode_counts = {}

    with pyhepmc.open(filepath) as f:
        for event in f:
            n_total += 1
            if max_events is not None and n_total > max_events:
                break

            particles = list(event.particles)

            # Find muons from Z decay (final state)
            mu_minus = _find_stable_particle(particles, PDGID_MU_MINUS)
            mu_plus = _find_stable_particle(particles, PDGID_MU_PLUS)

            if mu_minus is None or mu_plus is None:
                continue

            # Find tau leptons (should be intermediate, i.e. they decay)
            tau_minus = _find_particle(particles, PDGID_TAU_MINUS, require_decays=True)
            tau_plus = _find_particle(particles, PDGID_TAU_PLUS, require_decays=True)

            if tau_minus is None or tau_plus is None:
                continue

            n_found_taus += 1

            # Classify tau decays
            tau_m_info = _classify_tau_decay(tau_minus)
            tau_p_info = _classify_tau_decay(tau_plus)

            if tau_m_info is None or tau_p_info is None:
                continue

            # Count decay modes
            mode_key = f"{tau_m_info.decay_mode} x {tau_p_info.decay_mode}"
            decay_mode_counts[mode_key] = decay_mode_counts.get(mode_key, 0) + 1

            is_pi_pi = (tau_m_info.decay_mode == "pi_nu" and
                        tau_p_info.decay_mode == "pi_nu")

            if is_pi_pi:
                n_pi_pi += 1

            # Check whether both taus have an allowed decay mode
            passes_mode = (tau_m_info.decay_mode in allowed_set and
                           tau_p_info.decay_mode in allowed_set)
            if passes_mode:
                n_selected_mode += 1

            if not passes_mode:
                continue

            rec = EventRecord(
                event_number=n_total,
                mu_plus_p4=_to_p4(mu_plus),
                mu_minus_p4=_to_p4(mu_minus),
                tau_minus=tau_m_info,
                tau_plus=tau_p_info,
                is_pi_pi=is_pi_pi,
            )
            events.append(rec)

    print(f"Parsed {n_total} events")
    print(f"  Found both taus: {n_found_taus}")
    print(f"  pi x pi events: {n_pi_pi}")
    print(f"  Allowed modes: {sorted(allowed_set)}")
    print(f"  Selected: {len(events)}")
    print(f"  Decay mode breakdown:")
    for mode, count in sorted(decay_mode_counts.items(), key=lambda x: -x[1]):
        print(f"    {mode}: {count} ({100*count/max(n_found_taus,1):.1f}%)")

    return events

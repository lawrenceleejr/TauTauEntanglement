# Blender event display — H → ττ entanglement, the talk kit

A cinematic 3-D visualisation of the analysis, built from a **real
Monte-Carlo event** (not a cartoon) and rendered with Cycles in 4K.  It is
organised as a **talk kit**: an 11-frame storyboard (one slide per beat of
the argument) plus four drop-in animations, all named by the beat they
belong to, so `00 → 10` in filename order IS the talk.

### The storyboard (`--shot steps`) — one slide per beat

No step-number captions and no event-specific numbers — each frame carries
only its physics labels, so you narrate it however you like.  A suggested
line per slide:

| Frame | What it shows | Suggested narration |
|---|---|---|
| `00_lab_event` | The lab event, beam horizontal: μ⁺μ⁻ + τ⁺τ⁻ | "A Higgsstrahlung event: the Z gives two muons, the Higgs two taus — each τ → π ν." |
| `01_what_is_measured` | Measured tracks solid; τ flights, decay points, ν's ghosted | "The detector only sees four charged tracks and the vertex — everything else must be reconstructed." |
| `02_measure_the_muons` | Z→μ⁺μ⁻ bright, Higgs recoil `p_H = p_beam − p_Z` | "We measure the muons; recoil fixes the Higgs four-momentum…" |
| `03_boost_to_rest_frame` | Two τ's exactly back-to-back, `\|p\| ≈ M_H/2` | "…so we can boost into the Higgs rest frame, where the taus are exactly back-to-back." |
| `04_higgs_rest_frame` | The full decay in the Higgs frame: flights, π's, ν's | "In that frame the whole decay lives on one axis — this is where the spin physics is clean." |
| `05_decay_planes` | Each decay spans a translucent plane hinged on the τ axis | "Each τ decay defines a plane containing its pion." |
| `06_acoplanarity` | The single dihedral angle φ between the planes | "Between the two planes there is a single angle — the acoplanarity. Its distribution encodes the spin correlation." |
| `07_acoplanarity_axial` | Down-the-axis "clock face" | "Looking down the axis: one angle, unmistakably." |
| `08_impact_parameters` | Zoom: impact parameters d of the pion tracks | "The pion tracks miss the vertex by measurable impact parameters…" |
| `09_decay_locations` | `L = \|d\|/sin α` right triangles fix both decay points | "…and d with the opening angle α pins down *where* each τ decayed." |
| `10_spacetime_separation` | Dimension line Δx between the two decay vertices | "So we know the spacetime separation of the two decays — always **spacelike** — and can measure the spin correlation as a function of it. That's the entanglement test." |

### The four talk animations (`--shot anims`)

Each is the animated companion of a storyboard frame (same scene, same
palette), named to slot in right after it:

| Animation | Companion of | Move |
|---|---|---|
| `00_lab_event.mp4` | frame 00 | slow orbit of the lab event |
| `03_boost_to_rest_frame.mp4` | frame 03 | the lab-frame τ momenta **morph** into the back-to-back configuration |
| `06_acoplanarity.mp4` | frame 06 | slow orbit of the two planes + φ |
| `08_zoom_to_impact_parameters.mp4` | frame 08 | dolly from the wide rest-frame view down into the vertex region |

`--shot anims-extra` renders orbit companions for other beats
(`03_rest_frame_orbit`, `07_acoplanarity_axial`, `10_spacetime_separation`,
`A1_reco_triangle`).

### Appendix / backup-slide assets

| Output | Use |
|---|---|
| `A1_reco_triangle` (.png / .mp4) | Deep-dive on the Jeans method for ONE τ: track plane, d, α, `L = \|d\|/sin α` right triangle (for questions) |
| `A2_event_angular_only` (.png / .mp4) | The lab event *without* resolved displacement — the "angular analysis only" comparison |
| `tautau_event.blend` | The full Blender scene, for opening / tweaking by hand |

Standalone still shots (`event`, `measurable`, `measure-muons`, `rest`,
`planes`, `planes-axial`, `separation`) re-render single storyboard frames
under the same filenames.

Physics conventions match the analysis (`spin_analysis.py`): the common axis
k̂ is the τ⁻ direction in the Higgs frame, both pions' azimuths are measured
about it, and the displayed φ is exactly the analysis acoplanarity.  Labels are **generic** (no specific numbers) so the figures are a stand-in for any event, while still being this real event's geometry.  The
rest-frame opening angles α and decay lengths L are the reconstructed values
(Lorentz-boosted into the Higgs frame by `extract_event.py`).  The displayed
rest-frame space is rotated so the τ axis is world-X (horizontal in frame,
level horizon) and the decay planes open symmetrically upward.

In the Higgs rest frame the two τ's are **exactly back-to-back** with equal
momenta (|p| = M_H/2) — that is an exact consequence of momentum conservation,
so the diagram draws it that way rather than advertising the small residual
from reconstructing the two τ's independently.  Each τ's decay is then laid out
on the common axis at its reconstructed L and α, with the pion in the plane
spanned by that axis and its azimuth q̂; so each **decay plane has an edge along
the τ momentum and entirely contains that τ's pion vector**, and the right
triangle giving `L = |d|/sin α` holds by construction.  Labels render with real
sub/superscripts (e.g. M_H, p_H = p_beam − p_Z), not ASCII underscores.

## Look & feel

- **Modern type** — labels use Inter (bundled at `/opt/fonts/label.ttf`), with
  real Greek/symbol glyphs (μ, τ, π, ν, **α**, **φ**) and screen-aligned,
  carefully placed text.
- **Warm light, native temperature** — lights use Blender's *actual* colour
  temperature controls (`use_temperature` + `temperature` in Kelvin; requires
  Blender ≥ 4.5): a 3100 K warm key, a 3600 K top fill and a 4500 K rim.
- **Softly lit dome** — the whole scene sits inside a giant sphere whose warm
  vertical emission gradient is both a wraparound softbox and a seamless
  cyclorama backdrop; a low-albedo ground catches **soft shadows**.
- **Horizontal beam** — a physics→scene axis remap (`beam → +X`) keeps the beam
  axis horizontal in every frame.
- **Sophisticated decay planes** — translucent tinted glass panels with glowing
  rims and normal indicators; the angle between them (acoplanarity) is drawn
  with a proper **φ** symbol, and the per-τ opening angle with **α**.
- **Satin-matte materials**, **cinematic shallow DOF** (9-blade bokeh), slow
  ease-in-out camera moves, **motion blur** on all animations.
- Cycles 4K, rendered as a **single tile** (auto-tiling off) so Cycles never
  spills its accumulation buffer to a temp `.exr` — that disk round-trip fails
  on some systems (notably macOS) with "Error writing tile to file".
- One consistent palette (τ⁻ teal, τ⁺ orange, μ blue, Higgs/vertex gold,
  ν pale grey, reconstruction magenta, angles warm amber); subjects composed
  on the left/right third for slide backgrounds.

## Running it

By default `render.sh` fetches Blender on the host, builds a Docker image, and
renders inside it (the committed `data/event.json` is used as-is):

```bash
cd blender
./render.sh                       # the full talk kit: storyboard + 4 animations
./render.sh steps 160             # the 11-frame storyboard (00 ... 10)
./render.sh anims 96 GPU          # the 4 talk animations, on the GPU
./render.sh anims-extra 96 GPU    # orbit companions for the other beats
./render.sh planes 192            # a single still
```

Shots: `event`, `reco`, `rest`, `planes`, `planes-axial`, `event-angular`,
`measurable`, `measure-muons`, `separation` (single stills) · `steps` (the
11-frame storyboard) · `anims` (the 4 talk animations) · `anims-extra` ·
`event-anim`, `reco-anim`, `rest-anim`, `planes-anim`, `separation-anim`,
`boost`, `zoom` (single animations) · `stills`, `all`.

### Running without Docker (`--host`)

Pass `--host` to skip Docker and render with a Blender already installed on
this machine — handy on a Mac, where Docker can't reach the GPU and Blender's
own Metal backend can:

```bash
./render.sh steps 200 GPU --host
```

The flag may appear anywhere on the command line. The script looks for Blender
on `PATH` and at the standard macOS location
(`/Applications/Blender.app/Contents/MacOS/Blender`); override with
`BLENDER_BIN=/path/to/blender`. Blender ≥ 4.5 is required (native light
temperature). The bundled `fonts/label.ttf` is picked up automatically.

### GPU rendering (for the finals)

Pass `GPU` as the third argument. Inside Docker this needs the NVIDIA container
runtime (Linux only); with `--host` it uses whatever backend Blender finds
(OPTIX → CUDA → HIP → METAL → ONEAPI). You can also drive Blender directly:

```bash
docker run --rm --gpus all -v "$PWD/blender":/work tautau-blender \
    -b -P /work/build_scene.py -- --shot anims --samples 128 --device GPU
```

`--device GPU` auto-selects OPTIX → CUDA → HIP → METAL → ONEAPI, falling back
to CPU if none is found.

### `build_scene.py` arguments (after the `--`)

| Arg | Default | Meaning |
|---|---|---|
| `--shot` | `all` | shot name or group (see above) |
| `--samples` | `128` | Cycles samples (adaptive + OpenImageDenoise on top) |
| `--res-percent` | `100` | resolution scale (use ~30 for quick previews) |
| `--device` | `CPU` | `CPU` or `GPU` |
| `--frame-start` / `--frame-end` | `1` / `96` | animation frame range |
| `--fps` | `24` | animation frame rate |

## Notes

- Built and tested against **Blender 4.5.10 LTS** (needed for native light
  temperature). The Inter font is committed under `fonts/`.
- CPU previews work fine at reduced `--res-percent`; the 4K finals and the
  ~100-frame animations are GPU territory.
- `blender.tar.xz` and everything under `output/` are git-ignored; rerun
  `render.sh` to regenerate them.

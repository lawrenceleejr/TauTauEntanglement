# Blender event display — H → ττ reconstruction

A cinematic 3-D visualisation of the reconstruction method used in this
analysis, built from a **real Monte-Carlo event** (not a cartoon) and rendered
with Cycles in 4K inside a Blender Docker image.

The shots tell the story in several beats:

| Output | Beat |
|---|---|
| `01_event_display` (.png / .mp4) | The lab-frame event `e⁺e⁻ → ZH → μ⁺μ⁻ τ⁺τ⁻`, beam horizontal. The Z→μμ "tag" fixes the Higgs 4-momentum; each τ flies a short distance and decays to `π ν`. |
| `02_reconstruction_geometry` (.png / .mp4) | The Jeans impact-parameter method for one τ: the measured π track misses the primary vertex by **d**; the τ flight, the opening angle **α**, and **L = \|d\|/sin α** form a right triangle in a translucent track plane. |
| `03_higgs_rest_frame` (.png / .mp4) | Boosted into the Higgs rest frame the two τ's are **back-to-back**, each ≈ M_H/2. |
| `04_boost_to_rest_frame.mp4` | The boost itself: the lab-frame τ momenta morph into the back-to-back rest-frame configuration. |
| `05_decay_planes` (.png / .mp4) | Rest frame: the two **translucent τ decay planes** hinged on the common τ axis and the **single acoplanarity angle φ** between them. |
| `05_zoom_to_impact_parameters.mp4` | Animated dolly from the wide rest-frame view down into the vertex region where the impact parameters live. |
| `06_decay_planes_axial` (.png) | The same decay planes viewed **down the τ axis** — the classic "clock face" where the one angle φ is unmistakable. |
| `07_event_angular` (.png) | The lab event display without the resolved displacement (pions drawn straight from the PV). |
| storyboard `01 … 06` (.png) | The **pedagogical storyboard** — one rendered frame per step of the method (see below). |
| `tautau_event.blend` | The full Blender scene, for opening / tweaking by hand. |

### The step-by-step storyboard (`--shot steps`)

A clean diagram per step, in narrative order, **all in the Higgs rest frame**
after the boost.  No step-number captions — each frame carries only its
physics labels, so the sequence can be narrated however you like.

| Frame | Diagram |
|---|---|
| `00_lab_event` | The lab event: μ⁺μ⁻ from the Z and τ⁺τ⁻ from the Higgs (beam horizontal) |
| `01_what_is_measured` | What a detector **actually measures**: the four charged tracks, the primary vertex and the pion impact parameters (solid/bright); the τ flights, decay vertices and neutrinos are **inferred** (faint ghosts) |
| `02_measure_the_muons` | Highlight the Z→μ⁺μ⁻ measurement and the Higgs recoil `p_H = p_beam − p_Z` (tau side ghosted) — "we measure the muons, which fixes the Higgs momentum…" |
| `03_boost_to_rest_frame` | "…and that lets us boost into the Higgs rest frame": τ's back-to-back, `|p| ≈ M_H/2` (muons still shown) |
| `04_higgs_rest_frame` | Muons removed — the Higgs decay in its own frame |
| `05_decay_planes` | Each τ decay spans a **plane**, shown with the real (displaced) decays sitting inside the two translucent planes |
| `06_acoplanarity` | The **single** acoplanarity angle φ between the two planes (3/4 "book" view) |
| `07_acoplanarity_axial` | The same planes viewed down the τ axis — the "clock face", one angle unmistakably |
| `08_impact_parameters` | **Zoom in**: each pion's impact parameter d, sitting inside its decay plane |
| `09_decay_locations` | The payoff: d and the opening angle α pin down where each τ decayed — `L = |d|/sin α`, planes shown |

Standalone `--shot measurable` and `--shot measure-muons` render those
lab-frame slides on their own.

Physics conventions match the analysis (`spin_analysis.py`): the common axis
k̂ is the τ⁻ direction in the Higgs frame, both pions' azimuths are measured
about it, and the displayed φ is exactly the analysis acoplanarity.  Labels are **generic** (no specific numbers) so the figures are a stand-in for any event, while still being this real event's geometry.  The
rest-frame impact parameters, opening angles and decay lengths are obtained by
Lorentz-boosting the reconstructed decay 4-positions into the Higgs frame
(`extract_event.py`); the geometric identity `L = |d|/sin α` holds exactly in
that frame too.  The displayed rest-frame space is rotated so the τ axis is
world-X (horizontal in frame, level horizon) and the decay planes open
symmetrically upward.

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
- Cycles 4K, **tiled to exactly two tiles per still** (`tile_size = 2160`).
- One consistent palette (τ⁻ teal, τ⁺ orange, μ blue, Higgs/vertex gold,
  ν pale grey, reconstruction magenta, angles warm amber); subjects composed
  on the left/right third for slide backgrounds.

## Running it

`render.sh` fetches Blender on the host, builds the image, regenerates the
event, and renders:

```bash
cd blender
./render.sh                       # all stills + storyboard + boost animation
./render.sh steps 160             # the 10-frame step-by-step storyboard
./render.sh stills 160            # the standalone still frames
./render.sh anims 96 GPU          # the animations, on the GPU
./render.sh planes 192            # a single still
```

Shots: `event`, `reco`, `rest`, `planes`, `planes-axial`, `event-angular`
(stills, plus `measure-muons`) · `steps` (the 10-frame storyboard) · `event-anim`, `reco-anim`,
`rest-anim`, `planes-anim`, `boost`, `zoom` (animations) · `stills`, `anims`,
`all`.

### GPU rendering (for the finals)

Pass `GPU` as the third argument (needs the NVIDIA container runtime), or:

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

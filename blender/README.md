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
| `05_decay_planes` (.png / .mp4) | The two **translucent τ decay planes** and the **acoplanarity angle φ** between them — the heart of the angular analysis — *with* the resolved decay displacement. |
| `06_decay_planes_angular` (.png) | The **same event with the displacement left unresolved**: only the π directions from the PV and the two decay planes + φ. This is the pure angular analysis when the τ separation can't be resolved. |
| `07_event_angular` (.png) | The event display without the resolved displacement (pions drawn straight from the PV). |
| `step_00 … step_10` (.png) | A **numbered pedagogical storyboard** — one rendered frame per reconstruction step (see below). |
| `tautau_event.blend` | The full Blender scene, for opening / tweaking by hand. |

### The step-by-step storyboard (`--shot steps`)

A clean diagram per step of the method, in narrative order.  **No step-number
captions** — each frame carries only its physics labels, so the sequence can be
reordered and narrated however you like.  The flow: first boost into the Higgs
rest frame from the measured Z→μμ; then drop the muons (we're now just looking
at the Higgs); then build up the impact-parameter reconstruction of one τ, one
element at a time.

| Frame | Diagram |
|---|---|
| `01_higgs_rest_frame` | Boost into the Higgs rest frame from the measured Z→μ⁺μ⁻; τ's back-to-back, `|p| ≈ M_H/2` (muons shown) |
| `02_higgs_to_tautau` | Muons dropped — just the Higgs → τ⁺τ⁻ |
| `03_impact_parameter` | The measured π track misses the PV by the impact parameter **d** |
| `04_track_plane` | `p_τ` lies in the track plane span(π̂, d̂) |
| `05_alpha` | Parameterise the τ direction: `τ̂ = cos α·π̂ + sin α·d̂` |
| `06_mass_constraint` | The τ-mass constraint `m_τ² = (p_π+p_ν)²` locks `|p_τ|` |
| `07_decay_length` | Decay length from geometry: `L = |d|/sin α` |
| `08_decay_vertex` | Decay vertex `x = PV + L·τ̂` and proper time `t = L/βc` |
| `09_missing_momentum` | Resolve the ambiguity: `p_ν₁+p_ν₂ = p_H − p_π₁ − p_π₂` |
| `10_decay_planes` | Both τ's done → decay planes and the acoplanarity angle **φ** |

The single-τ frames (03–08) build up the *same* hero view of the τ⁺ track plane
element by element.

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
./render.sh steps 160             # the 11-frame step-by-step storyboard
./render.sh stills 160            # the standalone still frames
./render.sh anims 96 GPU          # the animations, on the GPU
./render.sh planes 192            # a single still
```

Shots: `event`, `reco`, `rest`, `planes`, `planes-angular`, `event-angular`
(stills) · `steps` (the 11-frame storyboard) · `event-anim`, `reco-anim`,
`rest-anim`, `planes-anim`, `boost` (animations) · `stills`, `anims`, `all`.

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

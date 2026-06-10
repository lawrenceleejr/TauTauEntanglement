# Blender event display — H → ττ reconstruction

A cinematic 3-D visualisation of the reconstruction method used in this
analysis, built from a **real Monte-Carlo event** (not a cartoon) and rendered
with Cycles in 4K inside a Blender Docker image.

The shots tell the reconstruction story in three beats:

| Output | Beat |
|---|---|
| `01_event_display` (.png / .mp4) | The lab-frame event: `e+e- → ZH → μ⁺μ⁻ τ⁺τ⁻`. The Z→μμ "tag" fixes the Higgs 4-momentum; each τ flies a short distance and decays to `π ν`. |
| `02_reconstruction_geometry` (.png / .mp4) | The Jeans impact-parameter method for one τ: the measured π track misses the primary vertex by the impact parameter **d**; the τ flight, the opening angle **α**, and the decay length **L = \|d\|/sin α** form a right triangle in the track plane. |
| `03_higgs_rest_frame` (.png / .mp4) | Boosted into the Higgs rest frame the two τ's are exactly **back-to-back**, each carrying ≈ M_H/2. This constraint, with the per-τ track planes and the τ-mass constraint, pins down *where* each τ decayed. |
| `04_boost_to_rest_frame.mp4` | The boost itself: the two lab-frame τ momenta morph into the back-to-back rest-frame configuration under a slowly drifting camera. |
| `tautau_event.blend` | The full Blender scene, for opening / tweaking by hand. |

Stills are composed for slide backgrounds (subject pushed to the left or right
third via lens shift); each still also has an animated version with a slow
ease-in-out camera arc.

## How it is built

1. **`extract_event.py`** re-uses the analysis' own `parse_hepmc` parser and
   `tau_reconstruction.reconstruct_event` to pick the cleanest `π ν × π ν`
   event from the HepMC sample (currently event **#205** of
   `EventSample1002.hepmc`), then dumps every kinematic and geometric quantity
   the scene needs — lab frame **and** Higgs rest frame — to `data/event.json`.

2. **`build_scene.py`** runs inside Blender and constructs the scene from that
   JSON, with the look baked in:
   - the whole scene lives inside a **softly lit dome** — a giant sphere whose
     interior emission gradient acts as a wraparound softbox *and* as a clean,
     seamless cyclorama backdrop (no floor, no horizon line);
   - every material is **satin-matte**: Principled BSDF, zero metalness, real
     surface roughness, plus a whisper of clearcoat and sheen so the dome
     reads as one long soft highlight on every track (automotive-matte);
   - **cinematic cameras**: shallow photographic depth of field with 9-blade
     bokeh (f/2.0–f/4 depending on the shot), slow arc/dolly moves with
     ease-in-out, and **motion blur on** for every animation;
   - Cycles, 4K (3840×2160), **tiled so each still needs exactly two tiles**
     (`tile_size = 2160` ⇒ `ceil(3840/2160)·ceil(2160/2160) = 2`);
   - one **consistent colour palette** across all shots
     (τ⁻ = teal, τ⁺ = orange, μ = blue, Higgs/vertex = gold, ν = pale grey,
     reconstruction accents = magenta).

## Running it

Everything is driven by `render.sh` (fetches Blender on the host, builds the
image, regenerates the event, renders):

```bash
cd blender
./render.sh                       # everything: 3 stills + 4 animations
./render.sh stills 160            # the three 4K stills
./render.sh anims 96 GPU          # the four animations, on the GPU
./render.sh reco 128              # a single still
```

Shots: `event`, `reco`, `rest` (stills) · `event-anim`, `reco-anim`,
`rest-anim`, `boost` (animations) · `stills`, `anims`, `all` (groups).

### GPU rendering

The final renders are meant to run on a GPU box. Pass `GPU` as the third
argument (requires the NVIDIA container runtime), or directly:

```bash
docker run --rm --gpus all -v "$PWD/blender":/work tautau-blender \
    -b -P /work/build_scene.py -- --shot anims --samples 128 --device GPU
```

`--device GPU` auto-selects OPTIX → CUDA → HIP → METAL → ONEAPI and falls
back to CPU if no GPU is found.

### `build_scene.py` arguments (after the `--`)

| Arg | Default | Meaning |
|---|---|---|
| `--shot` | `all` | shot name or group (see above) |
| `--samples` | `128` | Cycles samples (adaptive + OpenImageDenoise on top) |
| `--res-percent` | `100` | resolution scale (use ~30 for quick previews) |
| `--device` | `CPU` | `CPU` or `GPU` |
| `--frame-start` / `--frame-end` | `1` / `96` | animation frame range |
| `--fps` | `24` | animation frame rate (96 frames = 4 s) |

## Notes

- CPU preview renders work fine at reduced `--res-percent`; the 4K finals and
  especially the four ~100-frame animations are GPU territory.
- The Blender tarball (`blender.tar.xz`) and everything under `output/` are
  git-ignored; rerun `render.sh` to regenerate them.

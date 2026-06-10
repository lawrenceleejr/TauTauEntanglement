#!/usr/bin/env bash
#
# Build the Blender docker image and render the tau-tau reconstruction
# event display.  All output lands in blender/output/.
#
# Usage:
#   ./render.sh                       # everything: 3 stills + 4 animations
#   ./render.sh stills 160            # just the three 4K stills
#   ./render.sh anims 96 GPU          # all four animations on the GPU
#   ./render.sh event 96              # a single shot
#   ./render.sh boost 64 GPU          # just the boost animation
#
# Shots: event | reco | rest | planes | planes-axial | event-angular | measurable | measure-muons
#        steps  (the 10-frame storyboard)
#        event-anim | reco-anim | rest-anim | planes-anim | boost | zoom
#        stills | anims | all
#
set -euo pipefail

SHOT="${1:-all}"
SAMPLES="${2:-128}"
DEVICE="${3:-CPU}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="tautau-blender"

# Blender >= 4.5 is required for the native light colour-temperature controls.
BLENDER_VERSION="${BLENDER_VERSION:-4.5.10}"
BLENDER_SERIES="$(echo "$BLENDER_VERSION" | cut -d. -f1,2)"
TARBALL="$HERE/blender.tar.xz"

if [ ! -f "$TARBALL" ]; then
    echo "==> Fetching Blender ${BLENDER_VERSION} (host network)"
    curl -fL -o "$TARBALL" \
        "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
fi

EVENT_JSON="$HERE/data/event.json"
if [ ! -f "$EVENT_JSON" ]; then
    echo "==> Generating the real-MC event payload"
    python3 "$HERE/extract_event.py"
else
    echo "==> Using existing event payload ($EVENT_JSON)  [pass REGEN=1 to force]"
fi
if [ "${REGEN:-0}" = "1" ]; then
    echo "==> Regenerating the real-MC event payload (REGEN=1)"
    python3 "$HERE/extract_event.py"
fi

echo "==> Building docker image '$IMAGE'"
docker build -t "$IMAGE" "$HERE"

# GPU rendering needs the NVIDIA container runtime; harmless to omit on CPU.
GPU_FLAGS=()
if [ "$DEVICE" = "GPU" ]; then
    GPU_FLAGS=(--gpus all)
fi

echo "==> Rendering shot='$SHOT' samples='$SAMPLES' device='$DEVICE'"
docker run --rm "${GPU_FLAGS[@]}" \
    -v "$HERE":/work \
    "$IMAGE" \
    -b -P /work/build_scene.py -- \
    --shot "$SHOT" --samples "$SAMPLES" --device "$DEVICE"

echo "==> Output:"
ls -la "$HERE/output"

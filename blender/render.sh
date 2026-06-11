#!/usr/bin/env bash
#
# Render the tau-tau reconstruction event display.  By default this builds a
# Blender Docker image and renders inside it; pass --host to skip Docker and
# use a Blender already installed on this machine instead.  All output lands
# in blender/output/.
#
# Usage:
#   ./render.sh                       # everything: 3 stills + 4 animations
#   ./render.sh stills 160            # just the three 4K stills
#   ./render.sh anims 96 GPU          # all four animations on the GPU
#   ./render.sh event 96              # a single shot
#   ./render.sh boost 64 GPU          # just the boost animation
#   ./render.sh steps 200 GPU --host  # render on the host's own Blender (no Docker)
#
# Options:
#   --host           run on the Blender installed on this machine, not Docker.
#                    Override the binary with BLENDER_BIN=/path/to/blender;
#                    otherwise the script looks on PATH and at the standard
#                    macOS app location.
#
# Env:
#   REGEN=1          re-extract data/event.json (needs numpy/pyhepmc on host)
#   BLENDER_BIN=...  explicit Blender binary for --host mode
#
# Shots: event | reco | rest | planes | planes-axial | event-angular | measurable | measure-muons
#        steps  (the 10-frame storyboard)
#        event-anim | reco-anim | rest-anim | planes-anim | boost | zoom
#        stills | anims | all
#
set -euo pipefail

# Pull --host (and any other flags) out of the argument list, leaving the
# positional shot/samples/device arguments behind.
USE_HOST=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --host) USE_HOST=1 ;;
        *)      POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

SHOT="${1:-all}"
SAMPLES="${2:-128}"
DEVICE="${3:-CPU}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="tautau-blender"

# Blender >= 4.5 is required for the native light colour-temperature controls.
BLENDER_VERSION="${BLENDER_VERSION:-4.5.10}"
BLENDER_SERIES="$(echo "$BLENDER_VERSION" | cut -d. -f1,2)"
TARBALL="$HERE/blender.tar.xz"

# The Docker image bundles Blender from a tarball fetched on the host; in
# --host mode we use the machine's own Blender, so neither is needed.
if [ "$USE_HOST" -eq 0 ] && [ ! -f "$TARBALL" ]; then
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

if [ "$USE_HOST" -eq 1 ]; then
    # Locate a Blender on this machine: explicit override, then PATH, then the
    # standard macOS app bundle.
    BLENDER="${BLENDER_BIN:-}"
    if [ -z "$BLENDER" ]; then
        if command -v blender >/dev/null 2>&1; then
            BLENDER="$(command -v blender)"
        elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
            BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
        fi
    fi
    if [ -z "$BLENDER" ] || [ ! -x "$BLENDER" ]; then
        echo "ERROR: --host needs a Blender on this machine, but none was found." >&2
        echo "       Install Blender >= 4.5, or set BLENDER_BIN=/path/to/blender." >&2
        exit 1
    fi
    echo "==> Using host Blender: $BLENDER"
    "$BLENDER" --version | head -1

    echo "==> Rendering shot='$SHOT' samples='$SAMPLES' device='$DEVICE' (host)"
    "$BLENDER" -b -P "$HERE/build_scene.py" -- \
        --shot "$SHOT" --samples "$SAMPLES" --device "$DEVICE"
else
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
fi

echo "==> Output:"
ls -la "$HERE/output"

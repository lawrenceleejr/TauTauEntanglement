# Blender headless renderer for the tau-tau reconstruction event display.
#
# Everything renders on CPU with Cycles, so no GPU is required inside the
# container.  The Blender tarball is fetched on the host (see render.sh /
# fetch_blender.sh) and COPYed in, so the build itself needs no network.
#
#   ./render.sh all 128
#
# is equivalent to:
#
#   docker build -t tautau-blender blender/
#   docker run --rm -v "$PWD/blender":/work tautau-blender \
#          -b -P /work/build_scene.py -- --shot all --samples 128
#
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    BLENDER_HOME=/opt/blender

RUN apt-get update && apt-get install -y --no-install-recommends \
        xz-utils \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 \
        libegl1 libxkbcommon0 libsm6 libice6 libxext6 \
        libgomp1 libtbb12 libopenexr-3-1-30 \
    && rm -rf /var/lib/apt/lists/*

COPY blender.tar.xz /tmp/blender.tar.xz
RUN mkdir -p ${BLENDER_HOME} \
    && tar -xf /tmp/blender.tar.xz -C ${BLENDER_HOME} --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && ln -s ${BLENDER_HOME}/blender /usr/local/bin/blender

# A modern label typeface (Inter), bundled so renders are reproducible.
COPY fonts/label.ttf /opt/fonts/label.ttf

WORKDIR /work
ENTRYPOINT ["blender"]

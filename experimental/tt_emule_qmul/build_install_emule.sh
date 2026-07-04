#!/usr/bin/env bash
# Build and install the configured tt-metal/build_emule tree.
#
# Run this from an x86-64 Linux environment after configure_build_emule.sh:
#
# docker run --rm --platform linux/amd64 \
#   -v /Users/home/Documents:/work \
#   -w /work/tt-metal \
#   -e TT_METAL_HOME=/work/tt-metal \
#   -e TT_EMULE_HOME=/work/tt-emule \
#   -e JOBS=2 \
#   ubuntu:24.04 \
#   bash -lc 'bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh && \
#             bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/build_install_emule.sh'

set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-/work/tt-metal}"
BUILD_DIR="${TT_METAL_BUILD_EMULE_DIR:-$TT_METAL_HOME/build_emule}"
JOBS="${JOBS:-2}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "build_emule directory does not exist: $BUILD_DIR" >&2
    echo "Run experimental/tt_emule_qmul/configure_build_emule.sh first." >&2
    exit 2
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake was not found on PATH; run inside the configured Linux build environment." >&2
    exit 2
fi

cmake --build "$BUILD_DIR" --target install -j"$JOBS"

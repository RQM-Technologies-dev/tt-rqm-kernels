#!/usr/bin/env bash
# Configure tt-metal/build_emule for the experimental qmul candidate.
#
# Intended usage from the host:
#
# docker run --rm --platform linux/amd64 \
#   -v /Users/home/Documents:/work \
#   -w /work/tt-metal \
#   -e TT_METAL_HOME=/work/tt-metal \
#   -e TT_EMULE_HOME=/work/tt-emule \
#   ubuntu:24.04 \
#   bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh

set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-/work/tt-metal}"
TT_EMULE_HOME="${TT_EMULE_HOME:-/work/tt-emule}"
BUILD_DIR="${TT_METAL_BUILD_EMULE_DIR:-$TT_METAL_HOME/build_emule}"

if [[ ! -d "$TT_METAL_HOME" ]]; then
    echo "TT_METAL_HOME does not exist: $TT_METAL_HOME" >&2
    exit 2
fi
if [[ ! -d "$TT_EMULE_HOME" ]]; then
    echo "TT_EMULE_HOME does not exist: $TT_EMULE_HOME" >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    gnupg \
    gpg \
    jq \
    lsb-release \
    software-properties-common \
    wget

"$TT_METAL_HOME/install_dependencies.sh" --docker
"$TT_METAL_HOME/install_dependencies.sh" --sfpi

cmake -S "$TT_METAL_HOME" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$TT_METAL_HOME/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH="$TT_EMULE_HOME" \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
    -DWITH_PYTHON_BINDINGS=OFF \
    -DTT_METAL_BUILD_TESTS=OFF \
    -DTTNN_BUILD_TESTS=OFF \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=ON \
    -DBUILD_PROGRAMMING_EXAMPLES=ON

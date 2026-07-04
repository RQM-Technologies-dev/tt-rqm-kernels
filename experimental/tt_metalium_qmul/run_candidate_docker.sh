#!/usr/bin/env bash
# Run the built TT-Metalium qmul candidate inside a Linux Docker container.
#
# This script is intended to be used as a StructuredBench external-qmul command
# from the macOS host. The benchmark runner creates the input work directory on
# the host; this wrapper mounts that directory into Docker at /external_qmul so
# the Linux candidate can read a.bin/b.bin/manifest.json and write out.bin plus
# metrics.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCUMENTS_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

HOST_WORKDIR="${TT_RQM_EXTERNAL_QMUL_DIR:-}"
HOST_MANIFEST="${TT_RQM_EXTERNAL_QMUL_MANIFEST:-}"
TT_METAL_ROOT="${TT_METAL_HOME:-$DOCUMENTS_ROOT/tt-metal}"
TT_EMULE_ROOT="${TT_EMULE_HOME:-$DOCUMENTS_ROOT/tt-emule}"
DOCKER_IMAGE="${TT_RQM_QMUL_DOCKER_IMAGE:-ubuntu:24.04}"
CONTAINER_WORKDIR="/external_qmul"
CONTAINER_BINARY="/work/tt-rqm-kernels/experimental/tt_metalium_qmul/build_emule_candidate/tt_rqm_metalium_qmul_candidate"
DEFAULT_CLUSTER_DESC_REL="tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
HOST_MOCK_CLUSTER_DESC="${TT_METAL_MOCK_CLUSTER_DESC_PATH:-$TT_METAL_ROOT/$DEFAULT_CLUSTER_DESC_REL}"

if [[ -z "$HOST_WORKDIR" || -z "$HOST_MANIFEST" ]]; then
    echo "external-qmul environment missing: TT_RQM_EXTERNAL_QMUL_DIR and TT_RQM_EXTERNAL_QMUL_MANIFEST are required." >&2
    exit 2
fi
if [[ ! -d "$HOST_WORKDIR" ]]; then
    echo "external-qmul work directory does not exist: $HOST_WORKDIR" >&2
    exit 2
fi
if [[ ! -f "$HOST_MANIFEST" ]]; then
    echo "external-qmul manifest does not exist: $HOST_MANIFEST" >&2
    exit 2
fi
if [[ ! -d "$TT_METAL_ROOT" ]]; then
    echo "TT_METAL_HOME does not exist: $TT_METAL_ROOT" >&2
    exit 2
fi
if [[ ! -d "$TT_EMULE_ROOT" ]]; then
    echo "TT_EMULE_HOME does not exist: $TT_EMULE_ROOT" >&2
    exit 2
fi
if [[ ! -x "$REPO_ROOT/experimental/tt_metalium_qmul/build_emule_candidate/tt_rqm_metalium_qmul_candidate" ]]; then
    echo "Built TT-Metalium qmul candidate is missing. Run build_candidate.py first." >&2
    exit 2
fi
if [[ ! -f "$HOST_MOCK_CLUSTER_DESC" ]]; then
    echo "TT_METAL_MOCK_CLUSTER_DESC_PATH does not exist: $HOST_MOCK_CLUSTER_DESC" >&2
    exit 2
fi
if [[ "$HOST_MOCK_CLUSTER_DESC" != "$TT_METAL_ROOT/"* ]]; then
    echo "TT_METAL_MOCK_CLUSTER_DESC_PATH must live under TT_METAL_HOME for this Docker wrapper: $HOST_MOCK_CLUSTER_DESC" >&2
    exit 2
fi
CONTAINER_MOCK_CLUSTER_DESC="/work/tt-metal/${HOST_MOCK_CLUSTER_DESC#"$TT_METAL_ROOT/"}"

docker run --rm --platform linux/amd64 \
    -v "$REPO_ROOT:/work/tt-rqm-kernels" \
    -v "$TT_METAL_ROOT:/work/tt-metal" \
    -v "$TT_EMULE_ROOT:/work/tt-emule" \
    -v "$HOST_WORKDIR:$CONTAINER_WORKDIR" \
    -w /work/tt-rqm-kernels \
    -e TT_RQM_EXTERNAL_QMUL_DIR="$CONTAINER_WORKDIR" \
    -e TT_RQM_EXTERNAL_QMUL_MANIFEST="$CONTAINER_WORKDIR/manifest.json" \
    -e TT_METAL_HOME=/work/tt-metal \
    -e TT_METAL_RUNTIME_ROOT=/work/tt-metal \
    -e TT_METAL_MOCK_CLUSTER_DESC_PATH="$CONTAINER_MOCK_CLUSTER_DESC" \
    -e TT_EMULE_HOME=/work/tt-emule \
    -e TT_METAL_EMULE_MODE=1 \
    -e TT_METAL_SLOW_DISPATCH_MODE=1 \
    -e LD_LIBRARY_PATH="/work/tt-metal/build_emule/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib" \
    "$DOCKER_IMAGE" \
    bash -lc '
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update >/tmp/tt_rqm_candidate_apt_update.log 2>&1
        apt-get install -y --no-install-recommends ca-certificates gnupg gpg jq lsb-release software-properties-common wget >/tmp/tt_rqm_candidate_apt_install.log 2>&1
        /work/tt-metal/install_dependencies.sh --docker >/tmp/tt_rqm_candidate_docker_deps_install.log 2>&1
        /work/tt-metal/install_dependencies.sh --sfpi >/tmp/tt_rqm_candidate_sfpi_install.log 2>&1
        export TT_EMULE_JIT_CACHE_DIR="${TT_EMULE_JIT_CACHE_DIR:-$(mktemp -d /tmp/tt_rqm_jit_cache_XXXXXX)}"
        exec '"$CONTAINER_BINARY"'
    '

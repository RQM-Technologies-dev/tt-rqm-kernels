#!/usr/bin/env python3
"""Run the compensated H2A candidate without mutating the original variant."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experimental.tt_metalium_hamiltonian_lowering import run_candidate as baseline
from tt_rqm_kernels.hamiltonian_lowering_source_identity import (
    source_bundle_sha256 as canonical_source_bundle_sha256,
)

PACKAGE = Path(__file__).resolve().parent
DEFAULT_BINARY = PACKAGE / "build" / "tt_rqm_metalium_hamiltonian_lowering_compensated_candidate"


def source_bundle_sha256() -> str:
    return canonical_source_bundle_sha256(REPO)


def main() -> int:
    baseline.DEFAULT_BINARY = DEFAULT_BINARY
    baseline.source_bundle_sha256 = source_bundle_sha256
    status = baseline.main()
    if status != 0:
        return status
    work = os.environ.get("TT_RQM_H2A_DIR")
    if not work:
        return 2
    metrics_path = Path(work) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = metrics["candidate_metadata"]
    metadata.update(
        {
            "implementation_class": "single_core_tensix_sfpu_h2a_compensated_b",
            "angle_representation": "two-product high/low retained through split-2pi reduction",
            "period_reduction": "device-side nearest-multiple split-2pi reduction",
            "product_mode": "Dekker split TwoProduct with 4097 FP32 splitter",
            "fma_mode": "audited and compile-probed; rejected for quantized product residual",
            "magnitude_compensation": "none; ordinary FP32 r2 and sqrt retained",
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

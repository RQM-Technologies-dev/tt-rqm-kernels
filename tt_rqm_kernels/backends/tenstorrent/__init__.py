"""Tenstorrent-facing adapters for external structured-kernel candidates.

This package is intentionally small. It wraps the existing StructuredBench
`external-qmul` protocol and environment checks; it is not a TT-NN integration
and does not claim hardware availability by itself.
"""

from tt_rqm_kernels.backends.tenstorrent.availability import (
    DEFAULT_HARDWARE_COMMAND_ENV,
    ExecutionPath,
    TenstorrentReadiness,
    check_readiness,
    resolve_execution_path,
)
from tt_rqm_kernels.backends.tenstorrent.qmul_external import (
    ExternalQmulRun,
    TenstorrentAdapterError,
    run_configured_qmul,
    run_external_qmul_inputs,
    run_structuredbench_qmul,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    ReportLabelError,
    validate_external_qmul_label,
    write_structuredbench_report,
)

__all__ = [
    "DEFAULT_HARDWARE_COMMAND_ENV",
    "ExecutionPath",
    "ExternalQmulRun",
    "ReportLabelError",
    "TenstorrentAdapterError",
    "TenstorrentReadiness",
    "check_readiness",
    "resolve_execution_path",
    "run_configured_qmul",
    "run_external_qmul_inputs",
    "run_structuredbench_qmul",
    "validate_external_qmul_label",
    "write_structuredbench_report",
]

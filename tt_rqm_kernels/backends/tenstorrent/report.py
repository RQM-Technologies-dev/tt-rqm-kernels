"""Report helpers for Tenstorrent-facing external-qmul runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from tt_rqm_kernels.structuredbench import EXECUTION_LABELS, render_markdown_report

ExecutionLabel = Literal["cpu", "simulator", "emulation", "hardware"]


class ReportLabelError(ValueError):
    """Raised when a report label would misrepresent the execution environment."""


def validate_external_qmul_label(
    execution_label: str,
    *,
    command: str | None = None,
) -> ExecutionLabel:
    """Validate labels for an external-qmul candidate report."""

    if execution_label not in EXECUTION_LABELS:
        raise ReportLabelError(
            "execution_label must be one of: " + ", ".join(EXECUTION_LABELS)
        )
    if execution_label == "simulator":
        raise ReportLabelError(
            "external-qmul reports should use cpu, emulation, or hardware; "
            "use tt-lang-sim for simulator reports"
        )
    if execution_label == "hardware" and command is not None:
        lowered = command.lower()
        if "tt-emule" in lowered or "emule" in lowered or "run_candidate_docker" in lowered:
            raise ReportLabelError(
                "hardware reports require a real Tenstorrent hardware command; "
                "tt-emule or Docker emulation commands must use execution_label=emulation"
            )
        if "docker" in lowered or "podman" in lowered:
            raise ReportLabelError(
                "hardware reports require a real Tenstorrent hardware command; "
                "Docker/container commands must not be labeled as hardware"
            )
    return execution_label  # type: ignore[return-value]


def validate_stable_benchmark(
    execution_label: ExecutionLabel,
    *,
    stable_benchmark: bool,
) -> None:
    """Reject stable benchmark labels for non-hardware external candidates."""

    if stable_benchmark and execution_label != "hardware":
        raise ReportLabelError(
            "stable benchmark reports are only allowed for real hardware runs; "
            f"execution_label={execution_label} must use stable_benchmark=false"
        )


def methodology_note_for_label(
    execution_label: ExecutionLabel,
    *,
    stable_benchmark: bool = False,
) -> str:
    """Return a conservative default methodology note."""

    if execution_label == "hardware":
        if stable_benchmark:
            return (
                "Configured Tenstorrent hardware external-qmul run marked "
                "stable_benchmark=true by the operator; verify hardware, SDK, "
                "clocking, command, input sizes, and methodology before treating "
                "this as a stable benchmark."
            )
        return (
            "Configured Tenstorrent hardware external-qmul run; first samples "
            "should not be treated as stable benchmark results unless separately "
            "validated."
        )
    if execution_label == "emulation":
        return (
            "Tenstorrent tt-emule external-qmul run; this is emulation evidence, "
            "not hardware performance."
        )
    return "CPU external-qmul validation run; not a hardware performance result."


def write_structuredbench_report(
    report: dict[str, object],
    *,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
) -> None:
    """Write StructuredBench JSON and Markdown report files when requested."""

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown_report(report), encoding="utf-8")

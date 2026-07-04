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
    return execution_label  # type: ignore[return-value]


def methodology_note_for_label(execution_label: ExecutionLabel) -> str:
    """Return a conservative default methodology note."""

    if execution_label == "hardware":
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

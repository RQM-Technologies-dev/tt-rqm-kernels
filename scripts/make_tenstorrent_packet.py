from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Tenstorrent outreach packet from a StructuredBench JSON report."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/structuredbench_latest.json"),
        help="StructuredBench JSON report path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tenstorrent_packet.md"),
        help="Markdown packet output path.",
    )
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    packet = render_packet(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(packet, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def render_packet(report: dict[str, object]) -> str:
    results = report["results"]
    if not isinstance(results, list):
        raise TypeError("report results must be a list")

    benchmark_rows = [
        [
            str(result["workload"]),
            str(result["items"]),
            str(result["iterations"]),
            f"{float(result['latency_ms']):.4f}",
            f"{float(result['throughput']):.2f}",
            str(result["throughput_unit"]),
            f"{float(result['max_abs_error']):.3e}",
        ]
        for result in results
        if isinstance(result, dict)
    ]
    hardware_rows = [
        [
            str(result["workload"]),
            str(result["items"]),
            str(result["estimated_flops"]),
            f"{float(result['estimated_flops_per_s']):.3e}",
            str(result["estimated_total_bytes"]),
            f"{float(result['effective_gb_per_s']):.3f}",
            f"{float(result['arithmetic_intensity_flops_per_byte']):.3f}",
        ]
        for result in results
        if isinstance(result, dict)
    ]

    return "\n".join(
        [
            "# Tenstorrent Outreach Packet",
            "",
            "## Project Summary",
            "",
            (
                "`tt-rqm-kernels` is an independent RQM Technologies LLC project "
                "for structured quaternion, rotor, and phase-aware tensor kernels "
                "represented inside ordinary floating-point tensors. StructuredBench "
                "provides a CPU/PyTorch reference benchmark contract intended to "
                "compare future TT-Metalium and TT-NN backend implementations."
            ),
            "",
            (
                "Committed reports are sample CPU/PyTorch reference outputs. They "
                "are included to show the report shape and outreach packet format, "
                "not to claim stable hardware performance."
            ),
            "",
            "Report labels:",
            "",
            "```text",
            f"execution_label: {report.get('execution_label', 'unknown')}",
            f"stable_benchmark: {str(report.get('stable_benchmark', False)).lower()}",
            f"methodology_note: {report.get('methodology_note', 'not provided')}",
            "```",
            "",
            "## Why Tenstorrent Developers Should Care",
            "",
            (
                "StructuredBench gives Tenstorrent a compact benchmark class between "
                "scalar elementwise ops and large matmul. It focuses on structured "
                "4-lane tensor values that carry rotation, phase, orientation, "
                "direction, and geometric state inside ordinary floating-point tensors."
            ),
            "",
            (
                "The first target is `qmul` over `[N, 4]` tensors. It is small enough "
                "to validate with CPU/PyTorch and scalar references, but structured "
                "enough to exercise cross-lane dependencies, fixed multiply/add/sign "
                "patterns, data movement, fusion, register reuse, and arithmetic "
                "intensity. No native quaternion datatype, new silicon feature, or "
                "hardware change is required."
            ),
            "",
            "Proof path:",
            "",
            "```text",
            "CPU/PyTorch qmul reference",
            "-> scalar correctness check",
            "-> TT-Lang simulator qmul for [N, 4]",
            "-> tt-emule run of real TT-Metalium qmul candidate",
            "-> real TT-Metalium / Tenstorrent hardware report",
            "-> compare throughput, latency, numerical error, FLOPs/sec, GB/sec, and arithmetic intensity",
            "```",
            "",
            "## Long-Term Direction: QuantumIR for Classical AI Compute",
            "",
            (
                "QuantumIR here means a classical/AI accelerator front end for "
                "selected quantum-mechanics workloads, not a quantum-hardware "
                "proposal. The immediate ask remains narrow: placement guidance "
                "for a minimal `[N, 4]` structured `qmul` kernel path."
            ),
            "",
            (
                "Longer term, RQM Technologies is exploring QuantumIR as a "
                "domain-facing layer above these kernels. It would lower selected "
                "quantum-mechanics workloads on classical Tenstorrent/AI "
                "accelerators, including SU(2) rotations, unitary composition, "
                "Hamiltonian evolution, phase/coherence updates, and AI "
                "augmentation use cases, into the same structured quaternion, "
                "rotor, phase, and tensor operators used by StructuredBench."
            ),
            "",
            (
                "This does not claim that arbitrary quantum computation is "
                "efficiently classically simulable, does not ask Tenstorrent for "
                "native quaternion hardware, and does not replace the signal "
                "processing, physical AI, imaging, wave simulation, and "
                "scientific computing kernel story. It is a future front end built "
                "on the same kernel foundation."
            ),
            "",
            "## Benchmark Table",
            "",
            _markdown_table(
                [
                    "workload",
                    "items",
                    "iters",
                    "latency_ms",
                    "throughput",
                    "unit",
                    "max_abs_err",
                ],
                benchmark_rows,
            ),
            "",
            "## Hardware Metrics Table",
            "",
            _markdown_table(
                [
                    "workload",
                    "items",
                    "estimated_flops",
                    "estimated_flops_per_s",
                    "estimated_total_bytes",
                    "effective_gb_per_s",
                    "arithmetic_intensity",
                ],
                hardware_rows,
            ),
            "",
            "## Proposed First TT-Metalium Target",
            "",
            "Proposed first TT-Metalium target: `qmul` for `[N, 4]` quaternion tensors.",
            "",
            "## Proposed Second Target",
            "",
            "Proposed second target: `qrotate_vector` for streamed unit-rotor/vector rotation.",
            "",
            "## Relevant Docs",
            "",
            "- [docs/operator-contracts.md](../docs/operator-contracts.md)",
            "- [docs/structuredbench-spec.md](../docs/structuredbench-spec.md)",
            "- [docs/tenstorrent-rfc.md](../docs/tenstorrent-rfc.md)",
            "- [docs/quantum-ir.md](../docs/quantum-ir.md)",
            "- [docs/quantum-ir-roadmap.md](../docs/quantum-ir-roadmap.md)",
            "- [docs/quantum-ir-operator-mapping.md](../docs/quantum-ir-operator-mapping.md)",
            "",
            "## Suggested GitHub Discussion Text",
            "",
            "```text",
            "Hi Tenstorrent maintainers,",
            "",
            "RQM Technologies has a CPU/PyTorch reference benchmark for structured quaternion and rotor tensor kernels, with qmul as the proposed first [N, 4] TT-Metalium target.",
            "",
            "Where should a minimal TT-Metalium qmul example for [N, 4] structured tensors live?",
            "",
            "Secondary questions: if a TT-Metalium programming example is not the right starting point, is there a preferred TT-NN custom-op path? Would a TT-MLIR representation be useful later, after there is a concrete lower-stack qmul example?",
            "",
            "The benchmark reports throughput, latency, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity, with scalar-reference spot checks for correctness.",
            "```",
            "",
            "## Suggested Discord Post",
            "",
            "```text",
            "Hi Tenstorrent community, RQM Technologies is building an independent structured-kernel benchmark for quaternion and rotor tensor operators represented inside ordinary floating-point tensors.",
            "",
            "For a first external structured-kernel contribution, should we target a TT-Metalium programming example or a TT-NN custom-op path?",
            "",
            "Repo: https://github.com/RQM-Technologies-dev/tt-rqm-kernels",
            "```",
            "",
        ]
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

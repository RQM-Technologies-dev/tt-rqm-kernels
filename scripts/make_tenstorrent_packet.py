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
                "provides conformance-gated benchmark contracts, CPU/PyTorch references, "
                "simulator and emulator paths, and reproducible Wormhole/N300 evidence "
                "for `qmul`, fused SU(2) composition, and H2A Hamiltonian lowering."
            ),
            "",
            (
                "The table below is generated from the committed CPU/PyTorch reference "
                "report. Separate linked artifacts contain real Tenstorrent hardware "
                "evidence with their own provenance, claim levels, and limitations."
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
            "CPU/PyTorch qmul reference: complete",
            "-> scalar correctness check: complete",
            "-> TT-Lang simulator and tt-emule TT-Metalium paths: complete",
            "-> N300 Stage A silicon conformance: complete",
            "-> multicore Tensix/SFPU Stage B evidence: complete",
            "-> three-session one-device qmul stability qualification: Claim Level 2",
            "-> current-main upstream-shaped example: correctness validated; placement pending",
            "```",
            "",
            "## Immediate Ask",
            "",
            (
                "The remaining external request is maintainer placement guidance in "
                "[tenstorrent/tt-metal#49887]"
                "(https://github.com/tenstorrent/tt-metal/issues/49887): should the "
                "minimal FP32 `[N,4]` Hamilton-product example live as a contributed "
                "TT-Metalium programming example or an experimental TT-NN operation?"
            ),
            "",
            (
                "A self-contained current-main programming example is available on the "
                "public review branch. Its N=128 and N=4096 N300 runs are correctness "
                "evidence only; the port does not inherit the protected external "
                "candidate's Claim Level 2 status and makes no acceleration claim."
            ),
            "",
            "## Long-Term Direction: QuantumIR for Classical AI Compute",
            "",
            (
                "QuantumIR here means a classical/AI accelerator front end for "
                "selected quantum-mechanics workloads, not a quantum-hardware "
                "proposal. The immediate ask remains the narrow placement decision for "
                "the existing current-main `[N, 4]` `qmul` example."
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
            "## CPU/PyTorch Reference Table",
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
            "## Reference Workload-Shape Metrics",
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
            "## Current TT-Metalium Result",
            "",
            (
                "The scalar RISC-V Stage A baseline and multicore Tensix/SFPU Stage B "
                "candidate both ran on Wormhole. Three qualified device-0 sessions "
                "support the protected aggregate qmul Claim Level 2 release; this is "
                "stable one-device performance evidence, not a CPU or application "
                "acceleration claim."
            ),
            "",
            "## Proposed Second Target",
            "",
            "Proposed second target: `qrotate_vector` for streamed unit-rotor/vector rotation.",
            "",
            "## Relevant Docs",
            "",
            "- [docs/tenstorrent-landing.md](../docs/tenstorrent-landing.md)",
            "- [docs/benchmarks/wormhole-qmul.md](../docs/benchmarks/wormhole-qmul.md)",
            "- [docs/benchmarks/wormhole-qmul-hardware-evidence.md](../docs/benchmarks/wormhole-qmul-hardware-evidence.md)",
            "- [docs/upstream/current-main-qmul-port.md](../docs/upstream/current-main-qmul-port.md)",
            "- [docs/operator-contracts.md](../docs/operator-contracts.md)",
            "- [docs/structuredbench-spec.md](../docs/structuredbench-spec.md)",
            "- [docs/tenstorrent-rfc.md](../docs/tenstorrent-rfc.md)",
            "- [reports/tt_emule_qmul_candidate.md](tt_emule_qmul_candidate.md)",
            "- [docs/quantum-ir.md](../docs/quantum-ir.md)",
            "- [docs/quantum-ir-roadmap.md](../docs/quantum-ir-roadmap.md)",
            "- [docs/quantum-ir-operator-mapping.md](../docs/quantum-ir-operator-mapping.md)",
            "",
            "## Suggested GitHub Discussion Text",
            "",
            "```text",
            "Hi Tenstorrent maintainers,",
            "",
            "RQM Technologies maintains an independent structured-kernel benchmark with CPU/PyTorch references and real Wormhole/N300 evidence for FP32 [N,4] qmul.",
            "",
            "The external multicore Tensix/SFPU candidate has whole-output validation, profiler diagnostics, and a three-session one-device Claim Level 2 stability qualification. A separate self-contained example was refreshed against tt-metal main and passed N=128 and N=4096 N300 correctness runs; those two runs are correctness-only and do not inherit the protected release status.",
            "",
            "Would Tenstorrent prefer the minimal current-main example as a contributed TT-Metalium programming example, or should it remain external / move to an experimental TT-NN ProgramDescriptor operation?",
            "",
            "No upstream implementation PR will be opened until maintainers establish the preferred placement and layout boundary.",
            "```",
            "",
            "## Suggested Discord Post",
            "",
            "```text",
            "Hi Tenstorrent community, RQM Technologies maintains an independent structured-kernel benchmark for quaternion and rotor tensor operators represented inside ordinary floating-point tensors.",
            "",
            "The qmul path now has real Wormhole/N300 correctness, profiler, and one-device stability evidence. The remaining ask is placement guidance for the self-contained current-main example; it is not a request for a new datatype, silicon feature, or endorsement.",
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

"""Persistent Wormhole execution and validation for SU2ComposeBench."""

from __future__ import annotations

from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Mapping, MutableMapping

import torch

from tt_rqm_kernels.benchmark_integrity import (
    IntegrityError,
    command_sha256,
    repository_commit,
    timing_summary,
)
from tt_rqm_kernels.hamiltonian import (
    compose_hamiltonian_matrices,
    lower_two_level_hamiltonian,
    su2_compose_chain,
    u2_matrix_from_rotor_phase,
)


PROTOCOL = "tt-rqm-external-su2-compose-persistent.v1"
METRICS_SCHEMA = "tt-rqm-external-su2-compose-persistent-metrics.v1"
REPORT_SCHEMA = "tt-rqm-su2-compose-report.v1"
DEVICE = "tenstorrent/wormhole-device-0"
IMPLEMENTATION = "fused_tensix_sfpu_su2_compose"
TT_METAL_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
PERFORMANCE_CASES = (
    (32768, 8),
    (8192, 32),
    (2048, 128),
    (512, 512),
    (1024, 128),
    (4096, 128),
    (16384, 128),
    (65536, 128),
)


def _case_specs(stage: str) -> tuple[tuple[int, int, int, int, int], ...]:
    if stage == "conformance":
        return ((32, 8, 1, 0, 1), (2048, 8, 1, 0, 1))
    if stage == "performance":
        return tuple(
            (batch, steps, max(1, math.ceil(2_621_440 / (batch * steps))), 2, 10)
            for batch, steps in PERFORMANCE_CASES
        )
    raise IntegrityError("SU2ComposeBench stage must be conformance or performance")


def _coefficients(batch: int, steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + batch * 1009 + steps)
    value = torch.randn((batch, steps, 4), generator=generator, dtype=torch.float64)
    if batch >= 6:
        value[0, :, 1:] = 0.0
        value[1, :, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        value[2, :, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        value[3, :, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        value[4, 0::2, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        value[4, 1::2, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        value[5, 0::2, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        value[5, 1::2, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    return value


def run_su2_compose(
    *,
    command: str,
    stage: str,
    methodology_note: str,
    seed: int = 0,
    expected_candidate_sha256: str | None = None,
    expected_repository_commit: str | None = None,
    process_capture: MutableMapping[str, str] | None = None,
) -> dict[str, object]:
    """Run both hardware paths and validate every returned value."""

    if not methodology_note.strip():
        raise IntegrityError("SU2ComposeBench reports require a methodology note")
    command_tokens = shlex.split(command)
    if not command_tokens or Path(command_tokens[0]).name == "env":
        raise IntegrityError(
            "SU2ComposeBench command must name the candidate directly so its SHA-256 identifies the binary"
        )
    specs = _case_specs(stage)
    candidate_hash = command_sha256(command, Path.cwd())
    if expected_candidate_sha256 is not None and candidate_hash != expected_candidate_sha256:
        raise IntegrityError("SU2ComposeBench candidate SHA-256 differs from frozen identity")
    source_commit = expected_repository_commit or repository_commit(Path.cwd())
    prepared: list[tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]] = []

    with tempfile.TemporaryDirectory(prefix="tt-rqm-su2-compose-") as temp:
        workdir = Path(temp)
        cases: list[dict[str, object]] = []
        for batch, steps, repeats, warmups, samples in specs:
            coefficients = _coefficients(batch, steps, seed)
            rotors64, phases64 = lower_two_level_hamiltonian(coefficients, 0.05)
            rotors = rotors64.float()
            phases = phases64.float()
            rotor_name = f"rotors_b{batch}_k{steps}.bin"
            phase_name = f"phases_b{batch}_k{steps}.bin"
            _write_float32(workdir / rotor_name, rotors)
            _write_float32(workdir / phase_name, phases)
            rotor_hash = _sha256(workdir / rotor_name)
            phase_hash = _sha256(workdir / phase_name)
            case_id = f"su2-f32-seed-{seed}-b-{batch}-k-{steps}-{rotor_hash[:12]}-{phase_hash[:12]}"
            outputs = {
                "fused_rotors": f"fused_rotors_b{batch}_k{steps}.bin",
                "fused_phases": f"fused_phases_b{batch}_k{steps}.bin",
                "unfused_rotors": f"unfused_rotors_b{batch}_k{steps}.bin",
                "unfused_phases": f"unfused_phases_b{batch}_k{steps}.bin",
            }
            case = {
                "case_id": case_id,
                "B": batch,
                "K": steps,
                "repeat_count": repeats,
                "warmup_pairs": warmups,
                "samples": samples,
                "inputs": {
                    "rotors": rotor_name,
                    "phases": phase_name,
                    "rotors_sha256": rotor_hash,
                    "phases_sha256": phase_hash,
                },
                "outputs": outputs,
            }
            cases.append(case)
            prepared.append((case, coefficients, rotors, phases))

        manifest = {
            "schema": PROTOCOL,
            "workload": "su2_compose",
            "dtype": "float32",
            "device_id": 0,
            "seed": seed,
            "cases": cases,
            "outputs": {"metrics": "metrics.json"},
        }
        manifest_path = workdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        host_start = time.perf_counter()
        _run_candidate(
            command,
            workdir,
            manifest_path,
            candidate_hash,
            source_commit,
            process_capture=process_capture,
        )
        host_s = time.perf_counter() - host_start
        metrics = json.loads((workdir / "metrics.json").read_text())
        normalized = validate_su2_metrics(metrics, manifest, candidate_hash, host_s)
        if normalized["provenance"]["repository_commit"] != source_commit:
            raise IntegrityError("SU2ComposeBench execution-source commit mismatch")
        if stage == "performance" and not normalized["performance_eligible"]:
            raise IntegrityError("performance collection requires performance_eligible=true")

        results: list[dict[str, object]] = []
        for (case, coefficients, rotors, phases), candidate_case in zip(
            prepared, normalized["cases"], strict=True
        ):
            batch, steps = int(case["B"]), int(case["K"])
            output = case["outputs"]
            assert isinstance(output, dict)
            fused_rotor = _read_float32(workdir / str(output["fused_rotors"]), (batch, 4))
            fused_phase = _read_float32(workdir / str(output["fused_phases"]), (batch, 2))
            unfused_rotor = _read_float32(workdir / str(output["unfused_rotors"]), (batch, 4))
            unfused_phase = _read_float32(workdir / str(output["unfused_phases"]), (batch, 2))
            reference_rotor, reference_phase = su2_compose_chain(rotors.double(), phases.double())
            fused_correctness = _correctness(
                fused_rotor, fused_phase, reference_rotor, reference_phase
            )
            unfused_correctness = _correctness(
                unfused_rotor, unfused_phase, reference_rotor, reference_phase
            )
            oracle_error, end_to_end_error = _oracle_errors(coefficients, fused_rotor, fused_phase)
            if oracle_error > 1e-11:
                raise IntegrityError(
                    f"CPU oracles disagree for B={batch}, K={steps}: {oracle_error}"
                )
            timing = candidate_case["timings_s"]
            repeats = int(case["repeat_count"])
            fused_samples = [float(value) / repeats for value in timing["fused_samples"]]
            unfused_samples = [float(value) / repeats for value in timing["unfused_samples"]]
            fused_summary = timing_summary(fused_samples)
            unfused_summary = timing_summary(unfused_samples)
            results.append(
                {
                    "case_id": case["case_id"],
                    "B": batch,
                    "K": steps,
                    "input_hashes": {
                        "rotors_sha256": case["inputs"]["rotors_sha256"],
                        "phases_sha256": case["inputs"]["phases_sha256"],
                    },
                    "repeat_count": repeats,
                    "warmup_pairs": case["warmup_pairs"],
                    "samples": case["samples"],
                    "stable_benchmark": False,
                    "performance_eligible": normalized["performance_eligible"],
                    "fused": {"timing_s": fused_summary, "correctness": fused_correctness},
                    "unfused": {"timing_s": unfused_summary, "correctness": unfused_correctness},
                    "comparison": {
                        "fused_over_unfused_median": float(fused_summary["median"])
                        / float(unfused_summary["median"]),
                        "steps_per_s_fused": batch * steps / float(fused_summary["median"]),
                        "trajectories_per_s_fused": batch / float(fused_summary["median"]),
                        "qmul_per_s_fused": batch * (steps - 1) / float(fused_summary["median"]),
                        "fused_logical_bytes": 24 * batch * steps + 24 * batch,
                        "unfused_logical_bytes": 72 * batch * (steps - 1),
                    },
                    "cpu_oracle_max_abs_error": oracle_error,
                    "end_to_end_matrix_max_abs_error": end_to_end_error,
                    "candidate_metadata": candidate_case["work"],
                    "raw_candidate_timings_s": timing,
                }
            )

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "SU2ComposeBench",
        "family": "SU2HamiltonianBench",
        "benchmark_stage": stage,
        "protocol": PROTOCOL,
        "device": DEVICE,
        "dtype": "float32",
        "seed": seed,
        "execution_label": "hardware",
        "stable_benchmark": False,
        "performance_eligible": normalized["performance_eligible"],
        "methodology_note": methodology_note,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "lifecycle": normalized["lifecycle"],
        "session_timing": {**normalized["session_timings_s"], "host_process_end_to_end": host_s},
        "provenance": {
            "repository_commit": source_commit,
            "candidate_sha256": candidate_hash,
            "candidate": normalized["provenance"],
        },
        "results": results,
        "nonclaims": [
            "no_stability_claim",
            "no_acceleration_claim",
            "no_cpu_comparison",
            "no_measured_bandwidth_claim",
            "no_full_device_side_hamiltonian_lowering_claim",
        ],
    }
    return report


def _correctness(
    rotor: torch.Tensor,
    phase: torch.Tensor,
    reference_rotor: torch.Tensor,
    reference_phase: torch.Tensor,
) -> dict[str, object]:
    rotor_diff = rotor.double() - reference_rotor
    phase_diff = phase.double() - reference_phase
    values = torch.cat((rotor_diff.reshape(-1), phase_diff.reshape(-1)))
    actual = torch.cat((rotor.double().reshape(-1), phase.double().reshape(-1)))
    expected = torch.cat((reference_rotor.reshape(-1), reference_phase.reshape(-1)))
    finite = torch.isfinite(actual)
    close = torch.isclose(actual, expected, atol=1e-4, rtol=1e-4)
    failures = int((~close | ~finite).sum().item())
    if failures:
        raise IntegrityError(f"SU2ComposeBench output has {failures} failing values")
    matrix = u2_matrix_from_rotor_phase(rotor.double(), phase.double())
    identity = torch.eye(2, dtype=torch.complex128).expand(matrix.shape)
    phase_complex = torch.complex(phase[:, 0].double(), phase[:, 1].double())
    rotor_identity_phase = torch.zeros_like(phase.double())
    rotor_identity_phase[:, 0] = 1.0
    su2 = u2_matrix_from_rotor_phase(rotor.double(), rotor_identity_phase)
    state = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
    evolved = matrix @ state
    x = 2.0 * torch.real(torch.conj(evolved[:, 0]) * evolved[:, 1])
    y = 2.0 * torch.imag(torch.conj(evolved[:, 0]) * evolved[:, 1])
    z = evolved[:, 0].abs().square() - evolved[:, 1].abs().square()
    return {
        "validated_values": int(actual.numel()),
        "failing_values": failures,
        "nonfinite_values": int((~finite).sum().item()),
        "max_abs_error": float(values.abs().max().item()),
        "rms_error": float(torch.sqrt(torch.mean(values.square())).item()),
        "quaternion_norm_drift": float(
            (torch.linalg.vector_norm(rotor.double(), dim=-1) - 1).abs().max().item()
        ),
        "phase_norm_drift": float(
            (torch.linalg.vector_norm(phase.double(), dim=-1) - 1).abs().max().item()
        ),
        "unitarity_frobenius_error": float(
            torch.linalg.matrix_norm(matrix.mH @ matrix - identity).max().item()
        ),
        "su2_determinant_error": float((torch.linalg.det(su2) - 1).abs().max().item()),
        "u2_phase_consistency_error": float(
            (torch.linalg.det(matrix) - phase_complex.square()).abs().max().item()
        ),
        "bloch_norm_drift": float(
            (torch.sqrt(x.square() + y.square() + z.square()) - 1).abs().max().item()
        ),
    }


def _oracle_errors(
    coefficients: torch.Tensor,
    hardware_rotor: torch.Tensor,
    hardware_phase: torch.Tensor,
) -> tuple[float, float]:
    oracle_error = 0.0
    end_to_end_error = 0.0
    for start in range(0, coefficients.shape[0], 256):
        stop = min(start + 256, coefficients.shape[0])
        matrix_oracle = compose_hamiltonian_matrices(coefficients[start:stop], 0.05)
        rotor64, phase64 = lower_two_level_hamiltonian(coefficients[start:stop], 0.05)
        total_rotor64, total_phase64 = su2_compose_chain(rotor64, phase64)
        quaternion_oracle = u2_matrix_from_rotor_phase(total_rotor64, total_phase64)
        hardware = u2_matrix_from_rotor_phase(
            hardware_rotor[start:stop].double(), hardware_phase[start:stop].double()
        )
        oracle_error = max(
            oracle_error, float((matrix_oracle - quaternion_oracle).abs().max().item())
        )
        end_to_end_error = max(
            end_to_end_error, float((matrix_oracle - hardware).abs().max().item())
        )
    return oracle_error, end_to_end_error


def validate_su2_metrics(
    metrics: object, manifest: Mapping[str, object], candidate_hash: str, host_s: float
) -> dict[str, object]:
    if not isinstance(metrics, dict):
        raise IntegrityError("SU2ComposeBench metrics must be an object")
    expected = {
        "schema": METRICS_SCHEMA,
        "protocol": PROTOCOL,
        "device": DEVICE,
        "dtype": "float32",
        "execution_kind": "hardware",
        "implementation_class": IMPLEMENTATION,
        "stable_benchmark": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise IntegrityError(f"SU2ComposeBench metrics {key} mismatch")
    if not isinstance(metrics.get("performance_eligible"), bool):
        raise IntegrityError("performance_eligible must be boolean")
    if metrics.get("lifecycle") != {
        "device_count": 1,
        "device_id": 0,
        "create_count": 1,
        "close_count": 1,
    }:
        raise IntegrityError("SU2ComposeBench requires one device-0 lifecycle")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("candidate_sha256") != candidate_hash:
        raise IntegrityError("candidate hash mismatch")
    for key in (
        "chip_type",
        "tt_metal_commit",
        "compiler_version",
        "runtime_version",
        "build_id",
        "repository_commit",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or value.lower() in {"", "unknown", "unset", "none", "n/a"}:
            raise IntegrityError(f"missing provenance.{key}")
    if provenance.get("tt_metal_commit") != TT_METAL_COMMIT:
        raise IntegrityError("TT-Metal commit mismatch")
    session = metrics.get("session_timings_s")
    if (
        not isinstance(session, dict)
        or float(session.get("candidate_session", 0)) <= 0
        or float(session["candidate_session"]) > host_s * 1.05 + 1e-6
    ):
        raise IntegrityError("invalid candidate session timing")
    expected_cases, actual_cases = manifest.get("cases"), metrics.get("cases")
    if (
        not isinstance(expected_cases, list)
        or not isinstance(actual_cases, list)
        or len(expected_cases) != len(actual_cases)
    ):
        raise IntegrityError("metrics case count mismatch")
    for expected_case, actual in zip(expected_cases, actual_cases, strict=True):
        if not isinstance(expected_case, dict) or not isinstance(actual, dict):
            raise IntegrityError("case must be an object")
        for key in ("case_id", "B", "K", "repeat_count", "warmup_pairs", "samples"):
            if actual.get(key) != expected_case.get(key):
                raise IntegrityError(f"case {key} mismatch")
        timings = actual.get("timings_s")
        if not isinstance(timings, dict):
            raise IntegrityError("case timings missing")
        for key in ("fused_samples", "unfused_samples", "paired_order"):
            if not isinstance(timings.get(key), list) or len(timings[key]) != int(
                expected_case["samples"]
            ):
                raise IntegrityError(f"case {key} count mismatch")
        work = actual.get("work")
        if not isinstance(work, dict):
            raise IntegrityError("case work metadata missing")
        tiles = (int(expected_case["B"]) + 1023) // 1024
        available = int(work.get("available_core_count", 0))
        if (
            work.get("core_count") != min(tiles, available)
            or work.get("device_count") != 1
            or work.get("device_id") != 0
        ):
            raise IntegrityError("case core/device metadata mismatch")
        if (
            work.get("fused_dispatches_per_chain") != 1
            or work.get("unfused_dispatches_per_chain") != int(expected_case["K"]) - 1
        ):
            raise IntegrityError("case dispatch metadata mismatch")
    return metrics


def render_su2_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole",
        "",
        "> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**",
        "",
        "H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.",
        "",
        f"Stage: `{report['benchmark_stage']}`  ",
        f"Performance eligible: `{str(report['performance_eligible']).lower()}`  ",
        "Stable benchmark: `false`",
        "",
        str(report["methodology_note"]),
        "",
        "## Results",
        "",
        "| B | K | values/path | fused median s | unfused median s | ratio | fused max error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:  # type: ignore[union-attr]
        lines.append(
            f"| {result['B']} | {result['K']} | {result['fused']['correctness']['validated_values']} | "
            f"{result['fused']['timing_s']['median']:.9f} | {result['unfused']['timing_s']['median']:.9f} | "
            f"{result['comparison']['fused_over_unfused_median']:.6f} | {result['fused']['correctness']['max_abs_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_candidate(
    command: str,
    workdir: Path,
    manifest: Path,
    candidate_hash: str,
    source_commit: str,
    *,
    process_capture: MutableMapping[str, str] | None = None,
) -> None:
    tokens = shlex.split(command)
    if not tokens or any(word in command.lower() for word in ("emule", "docker", "reference")):
        raise IntegrityError("SU2ComposeBench hardware command is invalid")
    env = os.environ.copy()
    env.update(
        {
            "TT_RQM_SU2_COMPOSE_DIR": str(workdir),
            "TT_RQM_CANDIDATE_SHA256": candidate_hash,
            "TT_RQM_BUILD_ID": candidate_hash,
            "TT_RQM_REPOSITORY_COMMIT": source_commit,
        }
    )
    completed = subprocess.run(
        [*tokens, "--workdir", str(workdir), "--manifest", str(manifest), "--device", "0"],
        capture_output=True,
        text=True,
        env=env,
    )
    if process_capture is not None:
        process_capture["stdout"] = completed.stdout
        process_capture["stderr"] = completed.stderr
    if completed.returncode:
        raise IntegrityError(
            f"SU2ComposeBench candidate failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _write_float32(path: Path, tensor: torch.Tensor) -> None:
    values = array("f", tensor.contiguous().view(-1).tolist())
    if sys.byteorder != "little":
        values.byteswap()
    path.write_bytes(values.tobytes())


def _read_float32(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != shape[0] * shape[1]:
        raise IntegrityError(f"output length mismatch for {path.name}")
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

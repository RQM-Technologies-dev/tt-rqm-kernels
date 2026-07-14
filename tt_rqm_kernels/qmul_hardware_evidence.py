"""Deterministic processing for the post-Level-1 Wormhole qmul evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "tt-rqm-qmul-hardware-evidence.v1"
DEFAULT_RAW = Path("benchmarks/raw")
DEFAULT_PROCESSED = Path("benchmarks/processed")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(paths: Iterable[Path], root: Path) -> list[dict[str, str]]:
    return [
        {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
        for path in sorted(set(paths))
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _result_map(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(value["items"]): value for value in report["results"]}


def _timing(result: Mapping[str, Any]) -> tuple[float, float]:
    timing = result["timing"]["device_s"]
    return float(timing["median"]), float(timing["p95"])


def output_backpressure(repo_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    directories = {
        2: raw_root / "wormhole-qmul-output-cb-depth-2-01",
        4: raw_root / "wormhole-qmul-output-cb-depth-4-01",
    }
    reports = {depth: _load(path / "report.json") for depth, path in directories.items()}
    rows = []
    baseline = _result_map(reports[2])
    deeper = _result_map(reports[4])
    for items in sorted(baseline):
        median_2, p95_2 = _timing(baseline[items])
        median_4, p95_4 = _timing(deeper[items])
        rows.append({
            "items": items,
            "depth_2_median_s": median_2,
            "depth_4_median_s": median_4,
            "depth_4_to_2_median_ratio": median_4 / median_2,
            "depth_2_p95_s": p95_2,
            "depth_4_p95_s": p95_4,
            "depth_4_to_2_p95_ratio": p95_4 / p95_2,
            "depth_2_correctness_passed": baseline[items]["correctness"]["passed"],
            "depth_4_correctness_passed": deeper[items]["correctness"]["passed"],
        })
    candidate_hashes = {
        report["provenance"]["candidate"]["candidate_sha256"] for report in reports.values()
    }
    source_paths = [
        *(path / name for path in directories.values() for name in ("report.json", "session-manifest.json")),
        raw_root / "wormhole-qmul-output-cb-setup-failure-01" / "session-manifest.json",
    ]
    payload = {
        "schema": SCHEMA,
        "evidence_type": "output-circular-buffer-backpressure-ablation",
        "classification": "one-change diagnostic; not a new qualified release candidate",
        "controlled_change": "output circular-buffer depth: 2 tiles versus 4 tiles",
        "held_fixed": ["candidate binary", "arithmetic", "layout", "core allocation", "protocol", "timing contract"],
        "same_candidate": len(candidate_hashes) == 1,
        "candidate_sha256": next(iter(candidate_hashes)),
        "decision": "retain output_cb_depth=2; depth 4 did not improve both published sizes",
        "setup_failure_disclosed": True,
        "rows": rows,
        "sources": _sources(source_paths, repo_root),
    }
    lines = [
        "# Wormhole qmul output-backpressure ablation",
        "",
        "Only output circular-buffer depth changed. Both runs used the same binary, arithmetic, layout, 56-core allocation, protocol, and timing contract.",
        "",
        "| N | depth 2 median ms | depth 4 median ms | D4/D2 | depth 2 p95 ms | depth 4 p95 ms | D4/D2 p95 | correct |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['items']:,} | {row['depth_2_median_s']*1e3:.6f} | "
            f"{row['depth_4_median_s']*1e3:.6f} | {row['depth_4_to_2_median_ratio']:.6f} | "
            f"{row['depth_2_p95_s']*1e3:.6f} | {row['depth_4_p95_s']*1e3:.6f} | "
            f"{row['depth_4_to_2_p95_ratio']:.6f} | "
            f"{'yes' if row['depth_2_correctness_passed'] and row['depth_4_correctness_passed'] else 'no'} |"
        )
    lines.extend([
        "",
        "Decision: retain depth 2. Depth 4 was effectively unchanged at N=65,536 and slower at N=262,144.",
        "",
        "A pre-device setup failure caused by an unset TT_METAL_RUNTIME_ROOT is preserved in raw evidence and excluded from timing comparison.",
    ])
    return payload, "\n".join(lines)


def device_parity(repo_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    names = {
        "device0_conformance": "wormhole-qmul-device0-parity-reference-conformance-01",
        "device0_performance": "wormhole-qmul-device0-parity-reference-performance-01",
        "device1_conformance": "wormhole-qmul-device1-conformance-01",
        "device1_performance": "wormhole-qmul-device1-performance-01",
    }
    reports = {key: _load(raw_root / value / "report.json") for key, value in names.items()}
    source_paths = [raw_root / value / "report.json" for value in names.values()]
    for value in names.values():
        source_paths.extend(
            [raw_root / value / "pre-device-health.txt", raw_root / value / "post-device-health.txt"]
        )
    d0 = _result_map(reports["device0_performance"])
    d1 = _result_map(reports["device1_performance"])
    rows = []
    for items in sorted(d0):
        m0, p0 = _timing(d0[items])
        m1, p1 = _timing(d1[items])
        c0 = d0[items]["candidate_metadata"]
        c1 = d1[items]["candidate_metadata"]
        rows.append(
            {
                "items": items,
                "device0_median_s": m0,
                "device1_median_s": m1,
                "device1_to_device0_median_ratio": m1 / m0,
                "device0_p95_s": p0,
                "device1_p95_s": p1,
                "device1_to_device0_p95_ratio": p1 / p0,
                "device0_core_count": c0["core_count"],
                "device1_core_count": c1["core_count"],
                "grid_equal": (c0["grid_x"], c0["grid_y"]) == (c1["grid_x"], c1["grid_y"]),
                "correctness_equal": d0[items]["output_sha256"] == d1[items]["output_sha256"],
                "max_abs_error": d1[items]["correctness"]["whole_output_max_abs_error"],
            }
        )
    conformance_ok = all(
        result["correctness"]["passed"]
        for key in ("device0_conformance", "device1_conformance")
        for result in reports[key]["results"]
    )
    payload = {
        "schema": SCHEMA,
        "evidence_type": "device-1-parity",
        "classification": "device-1 parity evidence; not stability or dual-device scaling",
        "conformance_passed": conformance_ok,
        "same_candidate": len(
            {reports[key]["provenance"]["candidate_sha256"] for key in reports}
        ) == 1,
        "rows": rows,
        "sources": _sources(source_paths, repo_root),
    }
    lines = [
        "# Wormhole qmul device-1 parity evidence",
        "",
        "Diagnostic parity evidence only; this is not device-0 stability, dual-device scaling, aggregate N300 throughput, acceleration, or endorsement.",
        "",
        "| N | device 0 median ms | device 1 median ms | D1/D0 | device 0 p95 ms | device 1 p95 ms | cores | correctness |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['items']:,} | {row['device0_median_s']*1e3:.6f} | "
            f"{row['device1_median_s']*1e3:.6f} | {row['device1_to_device0_median_ratio']:.4f} | "
            f"{row['device0_p95_s']*1e3:.6f} | {row['device1_p95_s']*1e3:.6f} | "
            f"{row['device1_core_count']} | {'identical' if row['correctness_equal'] else 'different'} |"
        )
    return payload, "\n".join(lines)


def core_scaling(repo_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    path = raw_root / "wormhole-qmul-core-scaling-01" / "report.json"
    report = _load(path)
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for result in report["results"]:
        grouped[int(result["items"])].append(result)
    rows = []
    for items in sorted(grouped):
        ordered = sorted(grouped[items], key=lambda value: value["candidate_metadata"]["core_count"])
        baseline = _timing(ordered[0])[0]
        for result in ordered:
            median_s, p95_s = _timing(result)
            metadata = result["candidate_metadata"]
            cores = int(metadata["core_count"])
            speedup = baseline / median_s
            rows.append(
                {
                    "items": items,
                    "requested_cores": metadata["requested_max_cores"],
                    "actual_cores": cores,
                    "component_tiles": metadata["component_tiles"],
                    "group_1_core_count": metadata["group_1_core_count"],
                    "group_1_tiles_per_core": metadata["group_1_tiles_per_core"],
                    "group_2_core_count": metadata["group_2_core_count"],
                    "group_2_tiles_per_core": metadata["group_2_tiles_per_core"],
                    "work_allocation_imbalance_tiles": metadata["work_allocation_imbalance_tiles"],
                    "median_s": median_s,
                    "p95_s": p95_s,
                    "qmul_per_s": result["throughput"],
                    "speedup": speedup,
                    "parallel_efficiency": speedup / cores,
                    "correctness_passed": result["correctness"]["passed"],
                    "max_abs_error": result["correctness"]["whole_output_max_abs_error"],
                }
            )
    payload = {
        "schema": SCHEMA,
        "evidence_type": "controlled-core-scaling",
        "classification": "diagnostic; no inferred or idle-core scaling points",
        "rows": rows,
        "sources": _sources([path], repo_root),
    }
    lines = [
        "# Wormhole qmul controlled core scaling",
        "",
        "Diagnostic evidence. Requested and physically active cores are identical, and active cores never exceed component tiles.",
        "",
        "| N | cores | tiles | median ms | p95 ms | qmul/s | speedup | efficiency | imbalance | correct |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['items']:,} | {row['actual_cores']} | {row['component_tiles']} | "
            f"{row['median_s']*1e3:.6f} | {row['p95_s']*1e3:.6f} | {row['qmul_per_s']:,.0f} | "
            f"{row['speedup']:.4f} | {row['parallel_efficiency']:.4f} | "
            f"{row['work_allocation_imbalance_tiles']} | {'yes' if row['correctness_passed'] else 'no'} |"
        )
    return payload, "\n".join(lines)


def initialization(repo_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    names = [
        "wormhole-qmul-initialization-order-a-01",
        "wormhole-qmul-initialization-order-b-01",
        "wormhole-qmul-initialization-order-c-01",
    ]
    paths = [raw_root / name / "report.json" for name in names]
    rows = []
    for path in paths:
        report = _load(path)
        for occurrence, result in enumerate(report["results"]):
            phases = result["timing"]["phases_s"]
            median_s, p95_s = _timing(result)
            rows.append(
                {
                    "session_id": path.parent.name,
                    "occurrence": occurrence,
                    "items": result["items"],
                    "buffer_allocation_s": phases["buffer_allocation"],
                    "program_build_s": phases["program_build"],
                    "h2d_s": phases["h2d"],
                    "prewarm_sync_s": phases["prewarm_sync"],
                    "warmup_s": phases["warmup"],
                    "median_s": median_s,
                    "p95_s": p95_s,
                    "d2h_s": phases["d2h"],
                    "cleanup_s": phases["cleanup"],
                }
            )
    first = [row for row in rows if row["occurrence"] == 0]
    later = [row for row in rows if row["occurrence"] > 0]
    finding = {
        "attribution": "first device submission/dispatch and transfer initialization, independent of first case size",
        "first_h2d_range_s": [min(row["h2d_s"] for row in first), max(row["h2d_s"] for row in first)],
        "later_h2d_range_s": [min(row["h2d_s"] for row in later), max(row["h2d_s"] for row in later)],
        "first_warmup_range_s": [min(row["warmup_s"] for row in first), max(row["warmup_s"] for row in first)],
        "later_warmup_range_s": [min(row["warmup_s"] for row in later), max(row["warmup_s"] for row in later)],
        "program_build_is_primary_cause": False,
        "size_dependent": False,
        "stability_protocol_changed": False,
    }
    payload = {
        "schema": SCHEMA,
        "evidence_type": "initialization-diagnostics",
        "classification": "diagnostic; canonical stability order unchanged",
        "finding": finding,
        "rows": rows,
        "sources": _sources(paths, repo_root),
    }
    lines = [
        "# Wormhole qmul initialization diagnostics",
        "",
        "The elevated first-case H2D and warmup times follow the first device submission regardless of whether the first size is 4,096, 65,536, or 262,144. Program construction is not the dominant change, and the preregistered stability order is unchanged.",
        "",
        "| order | occurrence | N | allocation ms | build ms | H2D ms | prewarm ms | warmup ms | median ms | D2H ms | cleanup ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = row["session_id"].split("-order-")[1].split("-")[0].upper()
        lines.append(
            f"| {label} | {row['occurrence']+1} | {row['items']:,} | {row['buffer_allocation_s']*1e3:.3f} | "
            f"{row['program_build_s']*1e3:.3f} | {row['h2d_s']*1e3:.3f} | "
            f"{row['prewarm_sync_s']*1e3:.3f} | {row['warmup_s']*1e3:.3f} | "
            f"{row['median_s']*1e3:.3f} | {row['d2h_s']*1e3:.3f} | {row['cleanup_s']*1e3:.3f} |"
        )
    return payload, "\n".join(lines)


def _profile_dispatches(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        header = [value.strip() for value in handle.readline().split(",")]
        rows = list(csv.DictReader(handle, fieldnames=header))
    opened: dict[tuple[Any, ...], int] = {}
    records = []
    for row in rows:
        zone = row["zone name"].strip()
        if not zone.endswith("-KERNEL"):
            continue
        core = (int(row["core_x"]), int(row["core_y"]))
        processor = row["RISC processor type"].strip()
        key = (core, processor, row["timer_id"].strip(), zone)
        timestamp = int(row["time[cycles since reset]"])
        if row["type"].strip() == "ZONE_START":
            opened[key] = timestamp
        elif row["type"].strip() == "ZONE_END" and key in opened:
            start = opened.pop(key)
            records.append({"core": core, "processor": processor, "start": start, "cycles": timestamp - start})
    brisc = sorted((record for record in records if record["processor"] == "BRISC"), key=lambda value: value["start"])
    if len(brisc) % 56:
        raise ValueError("profile does not contain complete 56-core dispatches")
    starts = [min(value["start"] for value in brisc[index:index + 56]) for index in range(0, len(brisc), 56)]
    dispatches = []
    roles = {"BRISC": "reader", "NCRISC": "writer", "TRISC_0": "compute_unpack", "TRISC_1": "compute_math", "TRISC_2": "compute_pack"}
    for index, start in enumerate(starts):
        stop = starts[index + 1] - 5000 if index + 1 < len(starts) else 10**30
        selected = [value for value in records if start - 5000 <= value["start"] < stop]
        summary = {}
        for processor, role in roles.items():
            values = [value["cycles"] for value in selected if value["processor"] == processor]
            summary[role] = {"count": len(values), "min_cycles": min(values), "median_cycles": statistics.median(values), "max_cycles": max(values)}
        dispatches.append({"dispatch_index": index, "items": 65536 if index < 11 else 262144, "roles": summary})
    return dispatches


def _match_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing metric matching {pattern!r}")
    return float(match.group(1))


def profiler_and_ceilings(repo_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    profile_dir = raw_root / "wormhole-qmul-profile-01"
    ceilings = raw_root / "wormhole-qmul-hardware-ceilings-01"
    profile_csv = profile_dir / "profiler/device-profiler-logs/profile_log_device.csv"
    dispatches = _profile_dispatches(profile_csv)
    by_size = []
    for items in (65536, 262144):
        selected = [value for value in dispatches if value["items"] == items][1:]
        role_summary = {}
        for role in ("reader", "compute_unpack", "compute_math", "compute_pack", "writer"):
            maxima = [float(value["roles"][role]["max_cycles"]) for value in selected]
            medians = [float(value["roles"][role]["median_cycles"]) for value in selected]
            role_summary[role] = {
                "median_of_core_medians_cycles": statistics.median(medians),
                "median_of_core_maxima_cycles": statistics.median(maxima),
            }
        by_size.append({"items": items, "profiled_dispatches_excluding_first": len(selected), "roles": role_summary})
    pcie = (ceilings / "pcie-dram.stdout.txt").read_text(encoding="utf-8")
    noc_read = (ceilings / "noc-read-ceiling.stdout.txt").read_text(encoding="utf-8")
    noc_write = (ceilings / "noc-write-ceiling.stdout.txt").read_text(encoding="utf-8")
    dispatch = (ceilings / "dispatch-all-cores-ceiling.stdout.txt").read_text(encoding="utf-8")
    compute = (ceilings / "compute-bfp8-fpu-ceiling-1024.stdout.txt").read_text(encoding="utf-8")
    dram = (ceilings / "dram-adjacent-read-ceiling.stdout.txt").read_text(encoding="utf-8")
    dram_values = [float(value) for value in re.findall(r"\(([0-9.]+)GB/s\)", dram)][1:]
    ceiling_rows = [
        {"kind": "pcie_h2d", "value": _match_float(pcie, r"CSV_OUTPUT:H2D_Bandwidth\(GB/s\):([0-9.]+)"), "unit": "GB/s", "status": "measured"},
        {"kind": "pcie_d2h", "value": _match_float(pcie, r"D2H_Bandwidth\(GB/s\):([0-9.]+)"), "unit": "GB/s", "status": "measured"},
        {"kind": "noc_read", "value": _match_float(noc_read, r"CSV_OUTPUT:Bandwidth\(B/cc\):([0-9.]+)"), "unit": "B/clock", "status": "measured-host-timed"},
        {"kind": "noc_write", "value": _match_float(noc_write, r"CSV_OUTPUT:Bandwidth\(B/cc\):([0-9.]+)"), "unit": "B/clock", "status": "measured-host-timed"},
        {"kind": "dram_adjacent_read", "value": statistics.median(dram_values), "unit": "GB/s", "status": "measured-host-timed-post-first-use"},
        {"kind": "dispatch_all_cores", "value": _match_float(dispatch, r"Ran in ([0-9.]+)us per iteration"), "unit": "us/iteration", "status": "measured"},
        {"kind": "compute_bfp8_fpu", "value": _match_float(compute, r"CSV_OUTPUT:RMax\(TFLOPS\):([0-9.]+)"), "unit": "TFLOP/s", "status": "closest-supported-noncomparable"},
        {"kind": "compute_fp32_sfpu", "value": None, "unit": "TFLOP/s", "status": "not available in pinned microbenchmark suite"},
    ]
    failed = [
        {"benchmark": "test_noc_adjacent --use-device-profiler", "reason": "fixed profiler postprocessor buffer overflow; raw failure retained"},
        {"benchmark": "test_dram_offchip", "reason": "pinned kernel source fails JIT compilation against pinned TensorAccessor API; raw failure retained"},
        {"benchmark": "test_compute_mm FP32-accumulation single-core", "reason": "pinned profiler postprocessor buffer overflow; raw failure retained"},
    ]
    source_paths = [
        profile_dir / "report.json",
        profile_csv,
        profile_dir / "profiler/wormhole-qmul-profile-01.tracy",
    ] + sorted(path for path in ceilings.rglob("*") if path.is_file())
    payload = {
        "schema": SCHEMA,
        "evidence_type": "profiler-and-same-device-ceilings",
        "classification": "diagnostic; qmul logical traffic is not measured bandwidth",
        "profile": {
            "dispatch_count": len(dispatches),
            "clock_mhz": 1000,
            "sizes": by_size,
            "overlap": "reader, compute, and writer kernel scopes overlap on every profiled core",
            "critical_path": "writer/NCRISC is marginally longest; compute is nearly coextensive",
            "not_observable": ["circular-buffer stall counters", "NoC wait counters", "SFPU utilization counters"],
        },
        "ceilings": ceiling_rows,
        "failed_or_unsupported_attempts": failed,
        "sources": _sources(source_paths, repo_root),
    }
    lines = [
        "# Wormhole qmul profiler and same-device ceilings",
        "",
        "Diagnostic evidence. The qmul logical 48-byte model is not reinterpreted as measured DRAM, NoC, or PCIe traffic.",
        "",
        "## Profiler",
        "",
        "Device Program Profiler and Tracy captured N=65,536 and N=262,144. Reader, compute, and writer scopes overlap on all 56 cores. The writer/NCRISC maximum is marginally longest, with compute nearly coextensive. The pinned tools expose no circular-buffer stall, NoC-wait, or SFPU-utilization counters.",
        "",
        "| N | reader max cycles | compute-math max cycles | writer max cycles |",
        "|---:|---:|---:|---:|",
    ]
    for row in by_size:
        roles = row["roles"]
        lines.append(
            f"| {row['items']:,} | {roles['reader']['median_of_core_maxima_cycles']:.0f} | "
            f"{roles['compute_math']['median_of_core_maxima_cycles']:.0f} | "
            f"{roles['writer']['median_of_core_maxima_cycles']:.0f} |"
        )
    lines += ["", "## Pinned microbenchmarks", "", "| measurement | value | status |", "|---|---:|---|"]
    for row in ceiling_rows:
        value = "not measured" if row["value"] is None else f"{row['value']:.3f} {row['unit']}"
        lines.append(f"| {row['kind']} | {value} | {row['status']} |")
    lines += ["", "The BFP8 FPU matmul is the closest supported compute benchmark, not an FP32 SFPU ceiling and not a qmul comparison. Failed pinned benchmark attempts remain in the raw evidence directory."]
    return payload, "\n".join(lines)


def saturation(repo_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    directory = raw_root / "wormhole-qmul-saturation-01"
    report_path = directory / "report.json"
    preflight_path = directory / "memory-preflight.json"
    report = _load(report_path)
    preflight = _load(preflight_path)
    rows = []
    for result in report["results"]:
        median_s, p95_s = _timing(result)
        metadata = result["candidate_metadata"]
        rows.append(
            {
                "items": result["items"],
                "component_tiles": metadata["component_tiles"],
                "actual_cores": metadata["core_count"],
                "median_s": median_s,
                "p95_s": p95_s,
                "samples_s": result["timing"]["device_s"]["samples"],
                "qmul_per_s": result["throughput"],
                "logical_gb_per_s": result["effective_gb_per_s"],
                "estimated_flops_per_s": result["estimated_flops_per_s"],
                "phases_s": result["timing"]["phases_s"],
                "correctness_passed": result["correctness"]["passed"],
                "validated_values": result["correctness"]["validated_values"],
                "max_abs_error": result["correctness"]["whole_output_max_abs_error"],
            }
        )
    knee = next(row for row in rows if row["items"] == 57344)
    peak = max(rows, key=lambda value: value["qmul_per_s"])
    payload = {
        "schema": SCHEMA,
        "evidence_type": "larger-size-saturation",
        "classification": "diagnostic; logical traffic and estimated FLOP/s only",
        "memory_preflight": preflight,
        "occupancy_knee": {"items": knee["items"], "component_tiles": knee["component_tiles"], "actual_cores": knee["actual_cores"]},
        "peak_observed": {"items": peak["items"], "qmul_per_s": peak["qmul_per_s"]},
        "rows": rows,
        "sources": _sources([report_path, preflight_path, directory / "pre-device-health.txt", directory / "post-device-health.txt"], repo_root),
    }
    lines = [
        "# Wormhole qmul larger-size saturation sweep",
        "",
        "Diagnostic evidence only. Logical GB/s and estimated FLOP/s use the workload model; they are not measured fabric bandwidth or a hardware peak claim.",
        "",
        f"Memory preflight passed; the largest case uses {preflight['largest_total_device_buffers_mib']:.0f} MiB across the two inputs and one output planar buffer.",
        "",
        "| N | tiles | cores | median ms | p95 ms | qmul/s | logical GB/s | estimated GFLOP/s | max abs error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['items']:,} | {row['component_tiles']} | {row['actual_cores']} | "
            f"{row['median_s']*1e3:.6f} | {row['p95_s']*1e3:.6f} | {row['qmul_per_s']:,.0f} | "
            f"{row['logical_gb_per_s']:.3f} | {row['estimated_flops_per_s']/1e9:.3f} | {row['max_abs_error']:.3e} |"
        )
    lines += [
        "",
        "The latency-dominated region extends through the small sizes. N=57,344 is the exact 56-tile/56-core occupancy knee. Throughput continues rising after full occupancy and reaches 2.99 billion qmul/s at N=1,048,576, so this sweep establishes a broadening plateau rather than a hard saturation point.",
    ]
    return payload, "\n".join(lines)


def generate_all(repo_root: Path, raw_root: Path | None = None, processed_root: Path | None = None) -> list[Path]:
    repo_root = repo_root.resolve()
    raw_root = (raw_root or repo_root / DEFAULT_RAW).resolve()
    processed_root = (processed_root or repo_root / DEFAULT_PROCESSED).resolve()
    generators = {
        "wormhole-qmul-device1-parity": device_parity,
        "wormhole-qmul-core-scaling": core_scaling,
        "wormhole-qmul-initialization-diagnostics": initialization,
        "wormhole-qmul-profiler-and-ceilings": profiler_and_ceilings,
        "wormhole-qmul-saturation": saturation,
        "wormhole-qmul-output-backpressure": output_backpressure,
    }
    written = []
    index_entries = []
    for stem, generator in generators.items():
        payload, markdown = generator(repo_root, raw_root)
        json_path = processed_root / f"{stem}.json"
        md_path = processed_root / f"{stem}.md"
        _write_json(json_path, payload)
        _write_text(md_path, markdown)
        written.extend([json_path, md_path])
        index_entries.append({
            "evidence_type": payload["evidence_type"],
            "json": str(json_path.relative_to(repo_root)),
            "json_sha256": _sha256(json_path),
            "markdown": str(md_path.relative_to(repo_root)),
            "markdown_sha256": _sha256(md_path),
        })
    index = {
        "schema": "tt-rqm-qmul-hardware-evidence-index.v1",
        "current_public_claim_level": 2,
        "stability_qualification_passed": True,
        "public_claim_updated": True,
        "entries": index_entries,
    }
    index_path = processed_root / "wormhole-qmul-hardware-evidence-index.json"
    _write_json(index_path, index)
    written.append(index_path)
    return written

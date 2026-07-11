# Tenstorrent Hardware Report Template

This template is for a future real Tenstorrent execution of StructuredBench
`qmul`. It is not a current hardware result.

## Status

```text
status: template
execution_label: cpu | simulator | emulation | hardware
benchmark_stage: conformance | performance
stable_benchmark: false
```

For first hardware samples, keep:

```text
This is an initial hardware sample for methodology validation. It is not a stable performance benchmark.
```

## Environment

```text
repo_commit:
tt_metal_commit:
tt_metal_home:
cloud_or_host:
device_type:
device_count:
software_stack_version:
python_version:
torch_version:
candidate_command:
candidate_sha256:
compiler_version:
runtime_version:
build_id:
timer_scope:
```

## Workload

```text
suite: qmul
workload: qmul
shape: [N, 4] x [N, 4] -> [N, 4]
lane_order: [real, i, j, k]
dtype: float32
seed:
items:
iterations:
warmup:
```

## Commands

CPU/PyTorch reference:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/qmul_cpu_reference.json \
  --markdown-output reports/qmul_cpu_reference.md
```

TT-Lang simulator, if used:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/tt_lang_qmul_sim.json \
  --markdown-output reports/tt_lang_qmul_sim.md
```

Stage A TT-Metalium candidate:

```bash
python experimental/tt_metalium_qmul/validate_candidate.py \
  --candidate-command "/path/to/tt_metalium_qmul_candidate" \
  --execution-label hardware \
  --benchmark-stage conformance \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/tt_metalium_qmul_candidate.json \
  --markdown-output reports/tt_metalium_qmul_candidate.md
```

## StructuredBench Result Fields

Copy the relevant `structuredbench.v1` fields here:

| field | value |
| --- | --- |
| schema |  |
| backend |  |
| device |  |
| dtype |  |
| items |  |
| iterations |  |
| warmup |  |
| latency_ms |  |
| throughput |  |
| throughput_unit | qmul/s |
| max_abs_error |  |
| max_rel_error |  |
| rms_error |  |
| scalar_reference_max_abs_error |  |
| correctness.passed |  |
| correctness.validated_values |  |
| timing.setup_s |  |
| timing.device_s |  |
| timing.end_to_end_s |  |
| implementation_class |  |
| performance_eligible | false |
| estimated_flops_per_s |  |
| effective_gb_per_s |  |
| arithmetic_intensity_flops_per_byte |  |
| checksum |  |

## Methodology Notes

- State whether this result is CPU, simulator, emulation, or hardware.
- State whether the candidate command ran on real Tenstorrent hardware.
- State whether hardware counters were used. If not, FLOP and byte metrics are
  StructuredBench logical estimates.
- State whether this is a single sample or a repeated run.
- Do not present first samples as stable hardware performance.

## Interpretation

```text
correctness_status:
scalar_reference_status:
report_label:
known_limitations:
next_action:
```

## Non-Claims

- This report does not imply Tenstorrent endorsement.
- This report does not request native quaternion hardware.
- This report does not claim stable hardware performance unless repeated runs
  and methodology are documented separately.

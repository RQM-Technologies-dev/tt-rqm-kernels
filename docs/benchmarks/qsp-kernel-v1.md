# QKERNEL-1 matched computational-efficiency benchmark

Status: source and CPU correctness harness implemented; N300 evidence pending.

This benchmark extends the existing evidence architecture without modifying or
repurposing historical qmul or SU(2) results. It registers qmul, quaternion dot
or structured matvec, SU(2) frame transport, selective q + q^j filtering,
fused frame transport plus filtering, and ordered rotor composition.

Every run uses identical inputs, Float32 precision, batch shapes, thread/core
counts, warmups, repetitions, synchronization/materialization boundaries, and
whole-output tolerances across unrestricted, materialized-structured,
direct-structured-real, complex-pair where applicable, and quaternion
formulations. CPU evidence uses vectorized PyTorch in one harness and records
raw samples, affinity when available, thread count, compiler, CPU model, and
runtime. It is development evidence, not a Tenstorrent or application claim.

The structured gate requires geometric-mean throughput of at least 1.25x the
unrestricted comparator, at least 2x coefficient storage or traffic reduction,
improvement at 80% of registered sizes, p95 non-regression, and full
correctness. The native gate requires at least 1.15x throughput versus the best
direct structured-real control (or 15% lower device time), a favorable paired
interval, CPU performance at least 0.95x matched control, full correctness, and
three qualified N300 cold-start sessions.

No existing stable qmul result satisfies these gates. A kernel-only result
cannot establish a signal-processing product advantage. Application
acceleration remains prohibited.

## Development CPU command

```bash
python scripts/run_qsp_kernel_benchmark.py \
  --stage cpu --sizes 256 4096 --warmups 3 --repetitions 10 \
  --output reports/development/qsp-kernel-cpu.json
```

## Exact N300 collection sequence

Configure `TT_RQM_QSP_HARDWARE_COMMAND` to the audited real-device executable
implementing protocol `tt-rqm-qsp-kernel-benchmark.v1`; Docker, emulation,
simulators, and Python reference commands are rejected. Run each performance
session from a fresh host process/device lifecycle:

```bash
python scripts/run_qsp_kernel_benchmark.py --stage conformance --device 0 \
  --session-id qsp-n300-conformance \
  --output benchmarks/qsp-decision-v1/n300-conformance.json
python scripts/run_qsp_kernel_benchmark.py --stage performance --device 0 \
  --session-id qsp-n300-01 --output benchmarks/qsp-decision-v1/n300-session-01.json
python scripts/run_qsp_kernel_benchmark.py --stage performance --device 0 \
  --session-id qsp-n300-02 --output benchmarks/qsp-decision-v1/n300-session-02.json
python scripts/run_qsp_kernel_benchmark.py --stage performance --device 0 \
  --session-id qsp-n300-03 --output benchmarks/qsp-decision-v1/n300-session-03.json
python scripts/qualify_qsp_kernel_benchmark.py \
  benchmarks/qsp-decision-v1/n300-session-01.json \
  benchmarks/qsp-decision-v1/n300-session-02.json \
  benchmarks/qsp-decision-v1/n300-session-03.json \
  --output benchmarks/processed/qsp-kernel-v1-qualification.json
```

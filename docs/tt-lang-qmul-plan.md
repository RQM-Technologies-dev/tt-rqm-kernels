# TT-Lang qmul Prototype Plan

Status update: TT-Lang simulation is a completed simulator milestone, not the
latest validation stage. This page documents the simulator path for `qmul`.
For the current lower-stack evidence, see
[docs/tt-emule-qmul-validation-plan.md](tt-emule-qmul-validation-plan.md) and
[reports/tt_emule_qmul_candidate.md](../reports/tt_emule_qmul_candidate.md).

This milestone creates the first Tenstorrent-adjacent implementation path for
`tt-rqm-kernels`: quaternion multiply over ordinary `[N, 4]` float tensors in
the TT-Lang functional simulator.

## Current Target

```text
PyTorch qmul reference
-> scalar correctness spot check
-> TT-Lang simulated qmul
-> StructuredBench-compatible simulation report
-> tt-emule validation of the experimental TT-Metalium candidate
-> future Tenstorrent hardware report through external-qmul
```

The simulator path is intentionally not a hardware result. It validates kernel
logic, data layout, and report shape before any TT-Metalium or TT-NN work.
The next backend evidence target is a real Tenstorrent hardware run of the
existing `external-qmul` candidate path, not a simulator or emulation result
presented as hardware.

## Operator Contract

Inputs:

```text
a: [N, 4] float32 = [real, i, j, k]
b: [N, 4] float32 = [real, i, j, k]
```

Output:

```text
out: [N, 4] float32
```

Operation:

```text
out.real = a.real*b.real - a.i*b.i - a.j*b.j - a.k*b.k
out.i    = a.real*b.i    + a.i*b.real + a.j*b.k - a.k*b.j
out.j    = a.real*b.j    - a.i*b.k    + a.j*b.real + a.k*b.i
out.k    = a.real*b.k    + a.i*b.j    - a.j*b.i + a.k*b.real
```

## Simulator Layout

The default prototype uses row-major tensors and processes a fixed number of
quaternions per block. Each lane is copied as a `[block, 1]` slice, which makes
the cross-lane dependencies explicit in the compute kernel while keeping the
public tensor contract as `[N, 4]`.

The experimental raw-element variant uses row-major `[block, 4]` transfers and
TT-Lang `raw_element_read` / `raw_element_write` calls inside a data-movement
kernel. It is intended to expose the scalar lane equations directly in simulator
traces and statistics. It is not a recommended hardware layout and it is not
TT-Metalium source.

This is a simulator-first shape, not a claim that this is the final best
TT-Metalium or TT-NN layout.

## Run

```bash
python scripts/run_ttlang_qmul_smoke.py --check
```

If `tt-lang-sim` is unavailable:

```bash
python3.12 -m venv --prompt ttlang ttlang-venv
source ttlang-venv/bin/activate
python -m pip install tt-lang-sim
tt-lang-setup
python scripts/run_ttlang_qmul_smoke.py --items 128
```

For the committed report artifact, use the deterministic smoke configuration:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/tt_lang_qmul_sim.json \
  --markdown-output reports/tt_lang_qmul_sim.md
```

Run the experimental raw-element variant:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --variant raw \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0
```

Optional simulator trace/stat capture:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --trace-output reports/tt_lang_qmul_trace.jsonl \
  --stats-output reports/tt_lang_qmul_stats.txt \
  --json-output reports/tt_lang_qmul_sim.json \
  --markdown-output reports/tt_lang_qmul_sim.md
```

Trace capture runs `tt-lang-sim` with `--trace`. When `tt-lang-sim-stats` is
available, the runner post-processes the trace and records the text summary in
the `tt_lang_sim` report metadata. If stats tooling is unavailable or fails,
the simulator benchmark still succeeds and reports `stats_error` instead of
turning the correctness run into a failure.

The same backend can be reached through StructuredBench:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --tt-lang-variant raw \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0
```

StructuredBench trace/stat capture uses TT-Lang-specific flags:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --tt-lang-trace-output reports/tt_lang_qmul_trace.jsonl \
  --tt-lang-stats-output reports/tt_lang_qmul_stats.txt
```

Successful runs write:

```text
reports/tt_lang_qmul_sim.json
reports/tt_lang_qmul_sim.md
```

The report uses `structuredbench.v1`, `backend="tt-lang-sim"`, and
`simulation=true`. It also records the deterministic seed, benchmark shape, and
simulator metadata when the `tt-lang-sim` CLI reports it. Trace/stat fields are
stored under `tt_lang_sim` as simulator-only diagnostics:

```text
tt_lang_sim.variant
tt_lang_sim.trace_enabled
tt_lang_sim.trace_path
tt_lang_sim.stats_available
tt_lang_sim.stats_summary
tt_lang_sim.stats_error
```

Timing values remain environment-dependent simulator measurements. Trace and
statistics output can help inspect functional simulator activity, but it is not
hardware performance evidence.

## Acceptance Criteria

- deterministic `[N, 4]` sample inputs
- TT-Lang simulator execution
- default block/slice variant and experimental raw-element variant are clearly
  labeled in report metadata
- CPU/PyTorch comparison
- independent scalar reference spot check
- JSON and Markdown reports compatible with StructuredBench
- optional trace/stat capture is recorded under simulator-only metadata
- clear report language that these are simulator/reference outputs, not
  hardware performance claims

## References

- TT-Lang overview: <https://docs.tenstorrent.com/tt-lang/overview.html>
- TT-Lang simulator docs: <https://docs.tenstorrent.com/tt-lang/simulator.html>
- TT-Lang specification: <https://docs.tenstorrent.com/tt-lang/specs/TTLangSpecification.html>

# TT-Lang qmul Prototype Plan

This milestone creates the first Tenstorrent-adjacent implementation path for
`tt-rqm-kernels`: quaternion multiply over ordinary `[N, 4]` float tensors in
the TT-Lang functional simulator.

## Current Target

```text
PyTorch qmul reference
-> scalar correctness spot check
-> TT-Lang simulated qmul
-> StructuredBench-compatible simulation report
```

The simulator path is intentionally not a hardware result. It validates kernel
logic, data layout, and report shape before any TT-Metalium or TT-NN work.

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

The prototype uses row-major tensors and processes a fixed number of
quaternions per block. Each lane is copied as a `[block, 1]` slice, which makes
the cross-lane dependencies explicit in the compute kernel while keeping the
public tensor contract as `[N, 4]`.

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

Successful runs write:

```text
reports/tt_lang_qmul_sim.json
reports/tt_lang_qmul_sim.md
```

The report uses `structuredbench.v1`, `backend="tt-lang-sim"`, and
`simulation=true`.

## Acceptance Criteria

- deterministic `[N, 4]` sample inputs
- TT-Lang simulator execution
- CPU/PyTorch comparison
- independent scalar reference spot check
- JSON and Markdown reports compatible with StructuredBench
- clear report language that these are simulator/reference outputs, not
  hardware performance claims

## References

- TT-Lang overview: <https://docs.tenstorrent.com/tt-lang/overview.html>
- TT-Lang simulator docs: <https://docs.tenstorrent.com/tt-lang/simulator.html>
- TT-Lang specification: <https://docs.tenstorrent.com/tt-lang/specs/TTLangSpecification.html>

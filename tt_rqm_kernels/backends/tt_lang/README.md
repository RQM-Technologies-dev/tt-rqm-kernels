# TT-Lang qmul Prototype

This package contains the optional TT-Lang simulator path for the first
Tenstorrent-adjacent `qmul` prototype.

The contract is intentionally narrow:

```text
input a: [N, 4] float32 = [real, i, j, k]
input b: [N, 4] float32 = [real, i, j, k]
output:  [N, 4] float32 = Hamilton product a * b
```

The default implementation uses row-major tensors and splits the four lanes
into `[block, 1]` slices inside TT-Lang. The compute kernel then forms each
output lane from the fixed Hamilton-product multiply/add/sign pattern.

An experimental raw-element variant also exists. It transfers `[block, 4]`
chunks and uses TT-Lang raw element reads/writes in the simulator to expose the
scalar lane equations directly. This variant is for simulator comparison only;
it is not a hardware layout recommendation.

## Status

- Backend target: TT-Lang functional simulator.
- Hardware target: not implemented.
- Report target: `structuredbench.v1` with `backend="tt-lang-sim"` and
  `simulation=true`.
- Default package dependency: none. Install `tt-lang-sim` separately.

## Run

```bash
python3.12 -m venv --prompt ttlang ttlang-venv
source ttlang-venv/bin/activate
python -m pip install tt-lang-sim
tt-lang-setup
python scripts/run_ttlang_qmul_smoke.py --items 128
```

Run the raw-element simulator variant:

```bash
python scripts/run_ttlang_qmul_smoke.py --variant raw --items 128
```

## Non-Goals

- No fake TT-Metalium, TT-NN, or hardware backend.
- No native quaternion datatype or hardware feature request.
- No hardware performance claim from simulator results.

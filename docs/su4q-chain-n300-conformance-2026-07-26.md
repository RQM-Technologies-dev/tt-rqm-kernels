# `su4q-chain-n300-conformance`

Status: **passed, internal correctness evidence only**

This experiment tested whether proof-gated quaternion–Cartan `su4q` blocks remain numerically correct when composed as a simulation process. It is not a performance benchmark or acceleration claim.

## Method

Each session started with 128 two-qubit states and a depth-major FP32 block tensor. N300 device 0 received the blocks and initial states once, applied eight device stages per block, retained every intermediate state in device DRAM, and returned only the final states. Depths ran in the required order: 1, 8, 32, then 128.

The corpus combined six compiler-selected, proof-verified `su4q` blocks with explicitly labeled convention sentinels. Chains covered repeated and mixed compiler blocks, reversed noncommutative order, inverse cancellation, local-only transforms, identity, iSWAP-class structure, and accumulated global phase. The complex128 oracle composed the original source-circuit unitaries in execution order; compiler fallbacks remained separate from hardware correctness.

## Results

All comparisons used unaligned output with `atol=1e-4`, `rtol=1e-4`.

| Depth | Failed values | Nonfinite values | Maximum absolute error | Maximum norm drift | Transfer contract | Clean close |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 0 | 0 | `1.955462722857959e-7` | `1.8590227646164692e-7` | yes | yes |
| 8 | 0 | 0 | `7.213143653483911e-7` | `8.638757094114879e-7` | yes | yes |
| 32 | 0 | 0 | `2.4827369807800537e-6` | `3.031551712551206e-6` | yes | yes |
| 128 | 0 | 0 | `1.1740932621240319e-5` | `1.0587272758932897e-5` | yes | yes |

The error increased with composition depth but remained within the fixed acceptance envelope. At depth 128 the worst unaligned component error was about 8.5 times below `1e-4`. This is FP32 correctness and stability evidence only.

Every session had complete case coverage, matching input hashes and provenance, one device creation, one clean closure, one initial block upload, one initial-state upload, zero intermediate host transfers, and one final-state download. Preflight and postflight both showed healthy DRAM and `FAULTS=0x0`; device 1 was not used.

## Verification and provenance

- `rqm-compiler`: `38408fac8bec95f5e243727395161360af5223b6`
- `rqm-entanglement`: `b5fa69f6a57eccc0f43765341cf84203c790920a`
- `tt-rqm-kernels` base: `fffa30784a00656a1a26ee89406633fecc9574ed`
- chain implementation: `709ce601764f3344201463ac6042229c6fa0ff96`
- `tt-metal`: `9802b80464cfd213b1146189753d8aedf86193fe`
- focused source, protocol, serialization, ordering, malformed-input, oracle, and transfer-contract tests: 20 passed
- all four host-only depth checks: passed
- narrow TT-Metalium target: built with `-Wall -Wextra -Werror`

Per-depth manifests contain compiler routing evidence and input hashes. Metrics contain source/binary hashes, program count, transfer counts, and device identity. Validations contain the final output hashes and numerical results.

The branch was not pushed, no issue or Discord post was updated, no performance sweep was run, and no acceleration claim is supported by this experiment.

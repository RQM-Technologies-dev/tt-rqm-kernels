# Fused multi-qubit `su4q` N300 conformance debrief

Status: **incomplete; failed at the first hardware gate**. This is a
correctness-only result and is not performance eligible.

## Good

The compiler/export/oracle side worked. All seven host sessions passed, from
3 qubits at depth 1 through four 15-qubit states at depth 128. The largest host
case composed 128 changing-pair operations over 131,072 complex amplitudes
with zero failed or nonfinite values. Its maximum absolute error was
`2.1143e-7`; maximum norm drift was `6.6908e-7`.

The narrow TT-Metalium target also built against pinned `tt-metal`
`9802b804` with `-Wall -Wextra -Werror`. The implementation has the intended
one-program-per-operation and device-resident ping-pong structure in source.
Compiler-selected cases remain distinguished from direct convention
sentinels, and their references come from original source-circuit unitaries.

## Bad

The N300 did not complete even the 3-qubit, depth-1 baseline, so there is no
hardware numerical result. Watcher-guided attempts first exposed unsupported
4-byte and 16-byte DRAM reads. After those were replaced with aligned 32-byte
bursts, a reader scratch circular-buffer assertion was found and removed. The
final candidate then stalled at the first reader/compute circular-buffer
handoff (`CRBW`/`UPAD`) and produced no output.

Because the process did not close the device normally, the clean-closure gate
also failed. Device 0 was reset afterward and its sanitized health snapshot
was normal. In accordance with the ladder contract, the six larger hardware
sessions were not run.

## Why this test mattered

The host results show that the updated compiler can supply real proof-gated
quaternion-Cartan blocks for changing qubit pairs, and that the scheduling,
little-endian pair conventions, reversed targets, accumulated phase, and
independent complex128 oracle all agree across the intended 15-qubit/depth-128
workload.

The hardware result identifies the next real engineering boundary: the
multi-qubit math has not yet been proven in the fused N300 dataflow. The
remaining blocker is the Tensix reader/compute handoff, not evidence of a bad
quaternion decomposition or excessive FP32 numerical drift.

## Still unknown

- N300 output correctness at any multi-qubit size.
- Whether the fused path actually completes with exactly `depth` programs and
  no intermediate host transfers.
- Practical runtime or speedup versus CPU/GPU simulators.
- Numerical growth on N300 at 15 qubits and depth 128.
- Repeated-run stability, larger statevectors, and cross-device behavior.

## Recommendation

Do **not** start a performance phase yet. First replace the ad hoc parameter
and state circular-buffer choreography with a minimal, watcher-instrumented
single-tile handshake test. Once that test closes cleanly, reintroduce one
fused `su4q` stage and rerun only 3 qubits/depth 1. The performance phase
should require the complete correctness ladder to pass first.

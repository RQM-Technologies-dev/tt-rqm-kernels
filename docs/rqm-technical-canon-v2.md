# RQM Technical Canon v2 Alignment

`tt-rqm-kernels` is a correctness, portability, and benchmark-replication
target for standard-compatible quaternion and structured kernels.

## Active direction

Align future fused quaternion APIs with the planned RQM Quaternion Kernel Pack:

```text
q_compose_normalize
q_rotate_batch
q_compose_rotate
q_compose_rotate_normalize
q_integrate_angular_velocity
```

EXP-009 measured a narrow advantage only for a NumPy float64 path on one
x86-64 CPU: 49.6% to 74.1% lower median latency at held-out batches 1,024
through 131,072, while batch 32 was 286.6% slower. The 144-versus-264-byte
result is a representation model, not hardware-counter evidence.

Existing Tenstorrent reports retain their exact device, precision, timer,
stability, and revision qualifiers. Simulator, emulation, correctness, or
source-level results are not accelerator performance evidence. No universal,
Tenstorrent, RISC-V, or embedded speedup follows from EXP-009.

Evidence authority:
`RQM-Technologies-dev/rqm-experiments/docs/RQM_TECHNICAL_CANON_V2.md`.

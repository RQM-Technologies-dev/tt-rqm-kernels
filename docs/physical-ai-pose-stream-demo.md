# Physical-AI Pose Stream Demo

## Purpose

This document defines the first reproducible physical-AI pose stream demo for
`tt-rqm-kernels`.

Tracker issue:
<https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/12>

The demo shows how existing quaternion and rotor kernels can represent a
compact stream of orientation updates and vector rotations inside ordinary
floating-point tensors. It is a benchmark/demo, not a robotics application.

## Demo Contract

The demo uses existing reference kernels:

```text
orientation = qnormalize(qmul(delta_rotor, base_orientation))
world_vector = qrotate_vector(orientation, body_vector)
```

Tensor shapes:

```text
base_orientation: [N, 4]
delta_rotor:      [N, 4]
body_vector:      [N, 3]
world_vector:     [N, 3]
```

Quaternion lane order remains:

```text
[real, i, j, k]
```

## Run

```bash
python examples/physical_ai_pose_stream.py \
  --items 1024 \
  --iters 5 \
  --warmup 2
```

Write report artifacts:

```bash
python examples/physical_ai_pose_stream.py \
  --items 1024 \
  --iters 5 \
  --warmup 2 \
  --json-output reports/physical_ai_pose_stream.json \
  --markdown-output reports/physical_ai_pose_stream.md
```

## Report Fields

The script emits a StructuredBench-style CPU/PyTorch reference report:

- schema
- backend
- device
- dtype
- seed
- items
- iterations
- warmup
- structured shape
- latency
- throughput
- unit-rotor max absolute error
- vector-norm preservation max absolute error
- checksum

The report is intentionally labeled CPU/PyTorch reference output. It is not
Tenstorrent hardware performance.

## Tenstorrent Relevance

This demo is useful after `qmul` because `qrotate_vector` builds on Hamilton
products and exposes a more complete stream pattern:

- update orientation state
- preserve unit-rotor stability
- rotate body-frame vectors into world-frame vectors
- measure vector-norm preservation
- report a compact stream throughput number

That makes it a practical second benchmark/demo lane for robotics, sensing,
graphics, and physical-AI state streams after the lower-stack `qmul` path is
proven.

## Non-Goals

- No full robotics application.
- No sensor-fusion stack.
- No new math operator.
- No fake Tenstorrent backend.
- No hardware-performance claim.
- No defense-first framing.

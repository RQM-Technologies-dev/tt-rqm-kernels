# Future Alignment: Coherence-Aware State Integrity for Physical AI

## Purpose and status

This document records a future product and technical alignment for
`tt-rqm-kernels`: **RQM State Engine**, a proposed coherence-aware
state-estimation and integrity engine for physical AI.

It is documentation and strategic alignment only. The separate
`rqm-state-engine` repository does not yet exist, and this repository does not
currently implement an estimator, sensor ingestion, failure classifier, PX4 or
ROS 2 integration, new kernel, or Tenstorrent backend for this direction.

The immediate Stage A priority is complete: the scalar TT-Metalium `qmul`
candidate has a real N300 hardware-labeled conformance report. The next lower-
stack milestone is a separately validated, performance-eligible Stage B
implementation. This document describes a credible application layer that
could later use the resulting primitives; it does not change that kernel-
validation sequence.

## The physical-AI state-integrity problem

Physical-AI systems must form a spatial state from imperfect, asynchronous,
and sometimes conflicting sensor streams. A pose estimate alone does not say
whether the inputs remain mutually consistent, whether a periodic disturbance
is present, or how an application should respond when a sensor is delayed,
biased, degraded, or unavailable.

The future direction is a trusted spatial-state layer that continuously
measures cross-sensor coherence, looks for documented spectral signatures of
sensor corruption, and produces a pose estimate together with interpretable
integrity information. This is a supervision and evaluation direction, not a
claim that the proposed features improve navigation accuracy.

## Why ordinary quaternion positioning is insufficient

Quaternions are already a standard way to represent orientation; using them is
not the differentiator. For example, PX4 EKF2 includes a quaternion orientation
state, fuses multiple sensors, tracks innovations, and exposes sensor-quality
and vibration-related diagnostics. [PX4 EKF2 documentation](https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf)
also describes delays, bias, GPS-quality checks, sensor inconsistencies, and
observation rejection.

The proposed RQM State Engine would sit around an existing estimator rather
than replace it. Its future value proposition is coherence-aware integrity:
relating residuals, timing, spectral features, and sensor agreement to the
trustworthiness of an otherwise conventional spatial-state estimate. Whether
that provides a measurable benefit over an unmodified estimator remains a
benchmark question.

## Proposed RQM State Engine

RQM State Engine is a proposed separate application-layer repository. It would
ingest sensor and estimator information, maintain coherence and sensor-health
features, supervise an existing estimator, and report state-integrity results.
It is not proposed as a new navigation filter in this repository.

Potential inputs include IMU, magnetometer, GNSS, optical flow, odometry,
radar, lidar, cameras, RF, or other time-aligned observation streams where
available. Its first role would be to observe an estimator's predicted state,
measurements, residuals, and status—not to take over safety-critical control.

## Potential outputs

The proposed engine could expose the following documented outputs:

- position
- velocity
- orientation
- uncertainty/confidence
- per-sensor health
- cross-sensor coherence
- detected or suspected failure mode
- degraded-navigation/integrity status

These outputs are proposed interface fields, not guaranteed capabilities or
validated performance claims.

## Relationship to `tt-rqm-kernels`

`tt-rqm-kernels` can provide accelerated numerical primitives beneath a future
RQM State Engine. It should remain a correctness-first kernel and benchmark
repository: compact tensor contracts, CPU/PyTorch references, explicit
execution labels, and backend-comparable reports.

The proposed application layer would compose those primitives with data
ingestion, estimator supervision, failure detection, sensor weighting, and
evaluation. It should keep application policy out of this repository and only
propose a backend after a measured workload justifies one.

## Candidate kernel map

The first four entries below correspond to existing operators or an existing
composition direction. The remaining entries are future candidates; none is a
commitment to implement a new kernel or hardware backend.

| State-engine operation | Existing or future kernel direction |
| --- | --- |
| Orientation composition | Existing `qmul` |
| Frame transformation | Existing `qrotate_vector` |
| Rotor stabilization | Existing `qnormalize` |
| Phase tracking | Existing `phase_update` |
| Pose-state streams | Existing rotor primitives composed into a future fused pose update |
| Vibration/interference analysis | Future FFT, wavelet, or spectral kernels |
| Cross-sensor agreement | Future correlation/coherence kernels |
| Innovation monitoring | Future reductions and statistical kernels |
| Adaptive sensor weighting | Future fused score-and-weight kernels |
| Learned failure classification | Future TT-NN inference integration |

`qmul` remains the lower-stack proof point. The existing
[physical-AI pose-stream demo](physical-ai-pose-stream-demo.md) is a compact
orientation-stream benchmark, not a state-estimation or integrity engine.

## Why this matters to Tenstorrent

This is a potential path from isolated structured kernels to an end-to-end
physical-AI workload. The broader workload could combine geometric state,
high-rate sensor streams, windowed spectral and coherence processing, learned
residual classification, multiple simultaneous hypotheses, fused
perception/state estimation, and fleet-scale analysis.

A small conventional EKF on one vehicle may not justify an AI accelerator by
itself. The accelerator-shaped opportunity becomes stronger when radar, lidar,
camera, RF, optical, and inertial streams are processed alongside integrity
features and learned perception. This frames the opportunity as accelerating a
spatial-intelligence and integrity pipeline around physical-AI inference—not
as accelerating a small Kalman filter.

Industry materials show why high-bandwidth multisensor pipelines and
deterministic edge handling are relevant design constraints, while not implying
that this project is connected to or endorsed by their vendors. NVIDIA's
[physical-AI ecosystem update](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)
describes an expanding robotics deployment ecosystem; its [IGX Thor overview](https://developer.nvidia.com/blog/nvidia-igx-thor-powers-industrial-medical-and-robotics-edge-ai-applications/)
discusses real-time, sensor-intensive edge workloads and high-bandwidth
camera, lidar, and radar ingestion.

No native quaternion hardware, new silicon feature, or hardware change is
required for this direction. Values can remain ordinary floating-point tensors;
any Tenstorrent backend is a future measurement-driven choice.

## First achievable demonstration

The first demonstration should be conservative and offline:

1. Replay public or user-provided vehicle sensor logs.
2. Read IMU, magnetometer, GNSS, optical-flow, and odometry residuals where
   available.
3. Inject or identify vibration, bias drift, latency/time misalignment, sensor
   disagreement, periodic interference, and GNSS degradation.
4. Compute documented sensor-health or coherence features.
5. Supervise an existing estimator rather than replacing it.
6. Compare navigation error, false alarms, detection time, isolation time, and
   recovery time against an unmodified baseline.

This begins as offline log analysis before simulation, hardware-in-the-loop,
or any safety-critical real-time control integration. The success condition is
a reproducible evaluation record, not a navigation-performance claim.

## Staged technical roadmap

1. **Phase 0 — definition:** Define datasets, fault models, metrics, and
   baseline estimators.
2. **Phase 1 — offline evidence:** Build offline log replay and deterministic
   fault injection.
3. **Phase 2 — supervision:** Add a coherence-aware estimator supervisor and
   health scoring.
4. **Phase 3 — compatible integration:** Add ROS 2/PX4-compatible integration
   in simulation or hardware-in-the-loop.
5. **Phase 4 — workload evidence:** Profile candidate kernels and identify
   accelerator-shaped workloads.
6. **Phase 5 — measured backend work:** Implement selected Tenstorrent
   backends only where measurements justify them.
7. **Phase 6 — visualization and reproducibility:** Integrate visualization
   through WaveEngine and publish reproducible physical-AI benchmarks.

## Repository boundaries

| Component | Scope |
| --- | --- |
| `tt-rqm-kernels` | Accelerated quaternion, rotor, phase, spectral, coherence, reduction, pose-stream, and integrity-scoring primitives. |
| Proposed future `rqm-state-engine` | Sensor ingestion, estimator supervision, failure detection, sensor weighting, PX4/ROS 2 integration, fault-injection benchmarks, and state-integrity evaluation. |
| `spectral-core` | Hardware-independent spectral and coherence primitives. |
| WaveEngine | Visualization, sensor-health diagnostics, and log replay. |
| RQM compiler work | Lowering the estimator/integrity graph to Tenstorrent and other edge targets. |

These boundaries preserve a small, reusable kernel surface here and avoid
turning this repository into an estimator, flight stack, or product monorepo.

## Evidence and external context

- [PX4 EKF2](https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf)
  documents a practical baseline: quaternion orientation state, multi-sensor
  fusion, delayed measurements, innovation monitoring, vibration/aliasing
  concerns, bias, and GPS-quality checks. It motivates measurable baselines,
  not a claim that a new supervisor is needed or superior.
- [NVIDIA's physical-AI ecosystem update](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)
  is external context that physical-AI deployment, simulation, and robotics
  tooling are active industry areas; it is not evidence about RQM performance.
- [NVIDIA IGX Thor](https://developer.nvidia.com/blog/nvidia-igx-thor-powers-industrial-medical-and-robotics-edge-ai-applications/)
  illustrates the edge-computing relevance of deterministic handling and
  high-bandwidth multisensor pipelines. It does not imply hardware equivalence
  or a Tenstorrent endorsement.
- [DARPA RACER](https://www.darpa.mil/news/2026/racer-finish-line) reports a
  reusable autonomy stack operating in challenging off-road environments
  without GPS or pre-mapped routes. It provides downstream degraded-navigation
  context, not a defense-first product claim or a relationship with DARPA.

## Guardrails and non-claims

- RQM State Engine is proposed; the separate repository does not yet exist.
- This repository does not currently improve navigation accuracy or provide
  coherence-aware estimator supervision.
- Cross-sensor coherence is not claimed to outperform an EKF unless controlled
  benchmarks demonstrate it.
- No hardware performance is claimed without a real, explicitly labeled
  Tenstorrent hardware result.
- This direction does not imply Tenstorrent endorsement or require native
  quaternion hardware.
- PX4/ROS 2, estimator algorithms, sensor weighting, fault-injection systems,
  and safety-critical control integration remain outside this repository.
- Physical AI is the primary framing. Degraded navigation and defense-adjacent
  applications are downstream contexts, not the lead pitch.
- The active `qmul`, TT-Metalium, `tt-emule`, and hardware-report path remains
  the immediate priority.

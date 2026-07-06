# Tenstorrent Console Access Request

Status update:

- Request `CUST-812` was acknowledged by Tenstorrent support for RQM
  Technologies TT-Cloud access for StructuredBench `qmul` hardware validation.
- The acknowledgement confirms receipt only. The repo should still wait for an
  enabled Resource Type, explicit no-cost access, or delegated hardware-run
  instructions before attempting hardware execution.

Hi Tenstorrent team,

RQM Technologies maintains `tt-rqm-kernels`, an independent open-source
structured-kernel benchmark and operator pack for quaternion, rotor, phase,
signal-processing, physical-AI, and scientific-computing tensor workloads on
ordinary floating-point tensors.

The repo currently includes:

- CPU/PyTorch reference kernels
- StructuredBench reports
- TT-Lang simulator checks
- tt-emule evidence
- an `external-qmul` protocol for candidate commands
- a safe quickstart path for hardware-labeled reports once real Tenstorrent
  access exists

In Console, the relevant path appears to be:

```text
Compute -> Resources -> Request Capacity
```

The form currently opens, but the `Resource Type` dropdown has no selectable
options, so `Submit Request` remains disabled.

We are looking for one of two paths:

1. Enable a Resource Type / capacity allocation for the `RQM-Technologies-dev`
   Console organization so we can run one small Tenstorrent Cloud Console
   validation of StructuredBench `qmul` over `[N, 4]` float32 tensors.
2. Use a maintainer-run validation path where a Tenstorrent engineer runs the
   repo's hardware handoff packet in a Tenstorrent-controlled environment and
   returns the report artifacts.

The desired output is one real hardware report:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

This is not a request for native quaternion hardware, a new chip feature, or
Tenstorrent endorsement of RQM theory. The goal is a small, conservative
hardware validation artifact for a structured tensor kernel.

We want to avoid payment-backed provisioning unless no-cost access is confirmed
or per-use spend is explicitly approved by the repo owner. Could you grant
capacity for this one small `[N, 4]` `qmul` hardware report, enable the missing
Resource Type, or point us to a no-cost Console evaluation, sponsored
open-source contributor, or maintainer-run path that fits this validation?

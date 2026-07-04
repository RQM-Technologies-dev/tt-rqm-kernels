# Tenstorrent Console Access Request

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

We are looking for capacity to run one small Tenstorrent Cloud Console
validation of StructuredBench `qmul` over `[N, 4]` float32 tensors. A
maintainer-run validation path would also work. The desired output is one real
hardware report:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

This is not a request for native quaternion hardware, a new chip feature, or
Tenstorrent endorsement of RQM theory. The goal is a small, conservative
hardware validation artifact for a structured tensor kernel.

We want to avoid payment-backed provisioning unless no-cost access is confirmed
or per-use spend is explicitly approved by the repo owner. Could you grant
capacity for this one small `[N, 4]` `qmul` hardware report, or point us to a
no-cost Console evaluation, sponsored open-source contributor, or maintainer-run
path that fits this validation?

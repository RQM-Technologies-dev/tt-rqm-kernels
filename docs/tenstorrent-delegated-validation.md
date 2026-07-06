# Tenstorrent Delegated Validation

This packet is for a Tenstorrent maintainer or engineer who can run the
hardware path in a Tenstorrent-controlled environment and return report
artifacts.

This is the preferred no-billing-exposure path for RQM Technologies. RQM does
not provision cloud resources, does not provide payment-backed access, and does
not store credentials in this repo.

The immediate ask is one real hardware-labeled StructuredBench `qmul` report,
not placement guidance.

If the validation uses the RQM Tenstorrent Console organization, the observed
path is:

```text
Compute -> Resources -> Request Capacity
```

The request should be for capacity to run one small `[N, 4]` StructuredBench
`qmul` hardware report. After access is granted, either a managed VSCode/browser
instance or SSH baremetal shell can run the same commands.

If Console capacity remains blocked because `Request Capacity` has no selectable
Resource Type, use the engineer handoff:
[docs/tenstorrent-engineer-copy-paste-packet.md](tenstorrent-engineer-copy-paste-packet.md).

## Copy/Paste Sequence

Use the one-block sequence in:

```text
docs/tenstorrent-engineer-copy-paste-packet.md
```

That sequence covers:

- clone and editable install
- readiness check
- focused CPU/adapter checks
- CPU StructuredBench smoke
- `external-qmul` CPU reference validation
- experimental TT-Metalium candidate build
- hardware-labeled StructuredBench validation
- returned JSON, Markdown, and environment notes

## Required Outputs

Please return:

- `reports/tt_hardware_qmul_quickstart.json`
- `reports/tt_hardware_qmul_quickstart.md`
- `reports/tt_hardware_qmul_environment.txt`
- any build/runtime log if the sequence fails

The hardware command must implement:
[docs/tenstorrent-hardware-command-contract.md](tenstorrent-hardware-command-contract.md).

## Label Requirements

Real hardware results must use:

```text
execution_label=hardware
stable_benchmark=false
```

Use `execution_label=hardware` only when the candidate actually ran on real
Tenstorrent hardware. Do not return TT-Lang simulator, CPU, Docker, or tt-emule
output as hardware.

Only mark a future result stable if the hardware, software stack, clocking,
input sizes, command, and methodology have been separately validated.

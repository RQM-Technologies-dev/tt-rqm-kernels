# Tenstorrent Cloud Access Plan

This document defines a no-surprise-billing path for running
`tt-rqm-kernels` on Tenstorrent Cloud or Tenstorrent-maintainer hardware.

The repo does not contain cloud credentials, tokens, payment setup, instance
provisioning logic, or provider-specific billing behavior. It does not assume
Tenstorrent Cloud is free. Hardware execution should happen only after explicit
access is provided by the repo owner, Tenstorrent, or another authorized
environment owner.

## Current Console Fit

The logged-in Console flow observed for RQM Technologies shows:

- API inference available.
- Billing and usage visible.
- Compute visible.
- Resources visible, with no dedicated hardware allocation observed.
- Instances blocked until access is granted.
- Baremetal blocked until access is granted.

The correct capacity request path for `tt-rqm-kernels` is:

```text
Compute -> Resources -> Request Capacity
```

The request should be for one small `[N, 4]` StructuredBench `qmul` hardware
report. After access is granted, this repo supports two future execution
surfaces:

- managed VSCode/browser instance copy/paste run
- SSH baremetal run

## Route A: Local / Free Validation

Use local modes first:

- CPU/PyTorch reference kernels
- StructuredBench CPU reports
- TT-Lang functional simulator when installed
- tt-emule when locally configured
- external-qmul protocol validation

Commands:

```bash
python scripts/rqm_tt_quickstart.py --check
python -m tt_rqm_kernels.structuredbench --suite smoke --items 128 --iters 1 --warmup 0
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

These paths should use `execution_label=cpu`, `execution_label=simulator`, or
`execution_label=emulation` according to the real environment. They must not be
reported as hardware performance.

## Route B: Tenstorrent-Maintainer-Run Validation

This is the preferred no-billing-exposure hardware path.

RQM provides:

- public repo URL
- command contract
- candidate input/output protocol
- expected report files
- CPU/PyTorch and external-qmul reference validation commands

Tenstorrent runs:

- the hardware command in a Tenstorrent-controlled environment
- `scripts/rqm_tt_quickstart.py --mode hardware`
- report generation under `reports/`

Tenstorrent returns:

- `reports/tt_hardware_qmul_quickstart.json`
- `reports/tt_hardware_qmul_quickstart.md`
- environment notes: hardware kind, SDK version, `tt-metal` commit, host, exact
  command

This route has no RQM cloud billing exposure because RQM does not provision or
pay for the execution environment.

## Route C: Requested No-Cost Tenstorrent Cloud Evaluation Access

If maintainer-run validation is not available, RQM can request no-cost
evaluation or sponsored open-source contributor access.

Acceptable contact routes:

- Tenstorrent Console, after confirming no payment-backed provisioning is
  required
- Tenstorrent support
- a focused GitHub issue or Discussion reply
- Tenstorrent community Discord

Do not proceed with payment-backed provisioning unless explicitly approved by
the repo owner. Do not enter credentials, tokens, payment methods, or account
secrets into this repo.

## Sample No-Cost Access Note

```text
Hi Tenstorrent team,

RQM Technologies maintains tt-rqm-kernels, an independent open-source benchmark
and operator pack for structured quaternion/rotor/phase tensor kernels on
ordinary floating-point tensors.

We have CPU/PyTorch references, StructuredBench reports, TT-Lang simulator
checks, tt-emule evidence, and an external-qmul protocol for candidate
commands. The next narrow validation step is one small real-hardware report for
qmul over [N, 4] float32 tensors.

In Console, the visible path appears to be Compute -> Resources -> Request
Capacity. Could you grant capacity for one small StructuredBench qmul hardware
validation over [N, 4] float32 tensors, or point us to the right maintainer-run
path? The desired returned artifacts are:

- reports/tt_hardware_qmul_quickstart.json
- reports/tt_hardware_qmul_quickstart.md
- hardware/software environment notes

We are not asking for native quaternion hardware, a new chip feature, or a
paid cloud provisioning path. The immediate goal is a conservative hardware
validation artifact for a compact structured-kernel benchmark.
```

## Runner Scaffold

Use:

```bash
python scripts/rqm_tt_cloud_runner.py --check
python scripts/rqm_tt_cloud_runner.py --mode local
python scripts/rqm_tt_cloud_runner.py --mode vscode --print-instructions
python scripts/rqm_tt_cloud_runner.py --mode delegated --print-instructions
python scripts/rqm_tt_cloud_runner.py \
  --mode ssh \
  --host <host> \
  --remote-dir <repo-dir> \
  --remote-command "<real hardware qmul command>"
```

VSCode mode prints copy/paste commands for a granted browser instance. SSH mode
is a dry run by default. It prints the command that would run and does not
connect unless `--execute` is supplied.

The runner is not a cloud API client and does not create cloud resources.

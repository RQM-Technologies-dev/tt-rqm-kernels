# Tenstorrent Console Access Plan

This plan defines a no-surprise-billing path for using Tenstorrent Cloud Console
as a browser-based route to real Tenstorrent silicon for `qmul` validation.

Purpose:

- reach real Tenstorrent silicon through the browser when explicit access is
  available
- run one small `[N, 4]` `qmul` hardware validation through the existing
  `external-qmul` / StructuredBench path
- avoid AWS-style surprise billing, payment-backed provisioning, credential
  storage, or paid API calls from this repo

Tenstorrent Cloud Console:

```text
https://console.tenstorrent.com/
```

## Why This Path

The browser Console is the easiest candidate path to reach Tenstorrent silicon
without buying local hardware. It may also let a maintainer or sponsored
evaluation environment run the existing hardware command and return reports.

This repository does not assume Tenstorrent Cloud is free. Console currently
shows Usage and Billing surfaces, including credit/cost tracking. The repo
provides only a safe runbook and copy/paste commands after explicit no-cost,
sponsored, maintainer-run, or consciously approved per-use access exists.

## Observed Console Workflow

The logged-in Console flow observed for the RQM organization is:

- API inference is available through the Models/API pages.
- Usage and Billing are visible.
- Compute is visible.
- Resources is visible, but no dedicated hardware allocation was observed.
- Instances is blocked until access is granted.
- Baremetal is blocked until access is granted.
- `Request Capacity` opens, but the `Resource Type` dropdown currently has no
  selectable options, so `Submit Request` is disabled.
- Tenstorrent support acknowledged `CUST-812` for TT-Cloud access for
  StructuredBench `qmul` hardware validation. This records that the request was
  received; it does not yet indicate that capacity or hardware access is
  available.

For real `qmul` execution, the path is:

```text
Compute -> Resources -> Request Capacity
```

After capacity is granted, use one of two execution surfaces:

- `Instances`: managed VSCode/browser instance for copy/paste commands.
- `Baremetal`: SSH access to an existing Tenstorrent host.

The Models/API path is useful for hosted inference, but it is not the target for
arbitrary StructuredBench `qmul` execution.

If the Resource Type blocker remains after `CUST-812`, use the delegated
engineer handoff in
[docs/tenstorrent-engineer-copy-paste-packet.md](tenstorrent-engineer-copy-paste-packet.md).

## No-Surprise-Billing Rules

Do not proceed if any of these are unclear:

- Do not enter payment information.
- Do not approve paid or metered resources without explicit repo-owner approval.
- Do not create instances if pricing or per-use terms are unclear.
- Stop if payment method, subscription, or unapproved spend approval appears.
- Proceed only with explicit no-cost evaluation, sponsored access,
  maintainer-run validation, or consciously approved per-use spend by the repo
  owner.

The repo must not:

- store credentials, tokens, account secrets, or payment data
- call paid cloud APIs
- create cloud resources
- infer that Console access is free
- label simulator or emulation output as hardware

## Browser Workflow

1. Open Tenstorrent Cloud Console:

   ```text
   https://console.tenstorrent.com/
   ```

2. Sign in or create an account.
3. Confirm Usage and Billing are visible and credit/spend is understood.
4. Open `Compute -> Resources`.
5. If no allocation exists, use `Request Capacity` for one small
   StructuredBench `qmul` run.
6. After access is granted, use either:
   - `Instances` for a managed VSCode/browser shell.
   - `Baremetal` for SSH access.
7. Confirm the hardware type if the Console shows one.
8. Clone `tt-rqm-kernels`.
9. Run CPU smoke tests.
10. Run the quickstart environment check.
11. Configure or run the real hardware `qmul` command.
12. Export JSON and Markdown report artifacts.

## Expected Artifacts

Return:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

Include environment notes:

- Console workspace label
- hardware kind if shown
- software stack version
- `tt-metal` commit if available
- exact hardware command
- whether the run was no-cost, sponsored, maintainer-run, or consciously
  approved

## Blocked States

Stop and record the blocker if:

- no account access exists
- no no-cost workspace is available
- no browser shell or remote access is available
- `Request Capacity` has no selectable Resource Type
- Instances and Baremetal remain blocked after the capacity request
- no TT-Metalium stack is available
- no real hardware `qmul` command exists yet
- any unapproved billing or payment-backed provisioning step appears

## Hardware Report Command

Once a real hardware command exists in the Console environment:

```bash
export TT_RQM_HARDWARE_QMUL_COMMAND="<TENSTORRENT_HARDWARE_QMUL_COMMAND>"
python scripts/rqm_tt_quickstart.py \
  --mode hardware \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_hardware_qmul_quickstart.json \
  --markdown-output reports/tt_hardware_qmul_quickstart.md
```

`<TENSTORRENT_HARDWARE_QMUL_COMMAND>` must be a real command in the Console
environment that implements the external-qmul protocol.

Real hardware reports must use:

```text
execution_label=hardware
```

Do not use a CPU reference command, TT-Lang simulator command, or tt-emule
wrapper as the hardware command.

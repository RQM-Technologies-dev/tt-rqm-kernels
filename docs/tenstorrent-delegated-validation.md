# Tenstorrent Delegated Validation

This packet is for a Tenstorrent maintainer or engineer who can run the
hardware path in a Tenstorrent-controlled environment and return report
artifacts.

This is the preferred no-billing-exposure path for RQM Technologies. RQM does
not provision cloud resources, does not provide payment-backed access, and does
not store credentials in this repo.

If the validation uses the RQM Tenstorrent Console organization, the observed
path is:

```text
Compute -> Resources -> Request Capacity
```

The request should be for capacity to run one small `[N, 4]` StructuredBench
`qmul` hardware report. After access is granted, either a managed VSCode/browser
instance or SSH baremetal shell can run the same commands below.

## Copy/Paste Sequence

```bash
git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git
cd tt-rqm-kernels
python -m pip install -e ".[dev]"
python -m pytest
```

Run CPU smoke:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite smoke \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Run external-qmul CPU reference:

```bash
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Run the real hardware command:

```bash
python scripts/rqm_tt_quickstart.py \
  --mode hardware \
  --command "<real Tenstorrent hardware qmul command>" \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_hardware_qmul_quickstart.json \
  --markdown-output reports/tt_hardware_qmul_quickstart.md
```

The hardware command must implement the contract in
[docs/tenstorrent-hardware-command-contract.md](tenstorrent-hardware-command-contract.md).

## Return Artifacts

Please return:

- `reports/tt_hardware_qmul_quickstart.json`
- `reports/tt_hardware_qmul_quickstart.md`
- environment notes

Environment notes should include:

- hardware kind
- host or environment label
- TT-Metalium / SDK version
- `tt-metal` commit if applicable
- exact hardware command
- any relevant runtime notes

## Label Requirements

Real hardware results must use:

```text
execution_label=hardware
```

Do not return TT-Lang simulator, CPU, or tt-emule output as hardware.

First hardware samples should normally keep:

```text
stable_benchmark=false
```

Only mark a result stable if the hardware, software stack, clocking, input
sizes, command, and methodology have been separately validated.

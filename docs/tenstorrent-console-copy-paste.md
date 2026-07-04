# Tenstorrent Console Copy/Paste Commands

Use these commands inside a granted Tenstorrent Cloud Console VSCode/browser
instance or SSH baremetal shell after explicit no-cost, sponsored,
maintainer-run, or consciously approved per-use access is confirmed.

The observed Console path for this repo is:

```text
Compute -> Resources -> Request Capacity
```

Ask for capacity to run one small `[N, 4]` StructuredBench `qmul` hardware
report. Do not launch Instances or Baremetal runs until access and spend terms
are explicit.

Stop if payment method, subscription, or unapproved spend approval appears.

## A. Clone And Install

```bash
git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git
cd tt-rqm-kernels
python -m pip install -e ".[dev]"
```

## B. CPU Reference

```bash
python -m pytest
python -m tt_rqm_kernels.structuredbench --suite smoke
```

## C. Environment Check

```bash
python scripts/rqm_tt_quickstart.py --check
```

## D. Hardware Command Placeholder

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
environment that implements the `external-qmul` protocol:

```text
float32 [N, 4] x [N, 4] -> [N, 4]
lane order: [real, i, j, k]
operation: Hamilton product
```

The command must read:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
a.bin
b.bin
manifest.json
```

and write:

```text
out.bin
metrics.json
```

The hardware report artifacts are:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

Do not use this hardware mode with a CPU reference, TT-Lang simulator, or
tt-emule command.

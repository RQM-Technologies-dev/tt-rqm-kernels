# H2B Development Runtime-Isolation Ladder

Contract-v1 Session 2 remains retained and did not pass. Its first evidenced
failure was TT-Metal device/runtime initialization with active Ethernet
dispatch cores and unexpected run-mailbox value `0x40`. It produced no
numerical output, so it is not evidence against H2A arithmetic, protected H1
composition, device-DRAM layout, ordering, or whole-chain correctness.

This D0-D5 ladder is a newly versioned development diagnostic. Every invocation
runs exactly one stage in a fresh process and output directory. Results use
`tt-rqm-h2b-runtime-isolation-result.v1`, are development-only, and are never
benchmark evidence, performance-eligible, stable, or claim-bearing.

## Progression rule

Run stages in order and stop at the first failure. Do not retry or replace a
result. A later stage may be attempted only after the previous stage completed
validation, closed the device cleanly, and produced a structurally valid result.

| Stage | Isolates | Does not prove |
| --- | --- | --- |
| D0 | Unit-mesh creation, queue acquisition, and shutdown. | DMA or kernel execution. |
| D1 | Small bitwise DRAM write/sync/read loopback. | RQM arithmetic. |
| D2 | Compensated H2A only with an identity-oracle case. | Protected H1 or chaining. |
| D3 | Protected fused H1 only with host-generated identity steps. | H2A or chaining. |
| D4 | One-device sequential H2A then H1 through device DRAM. | External protocol integration. |
| D5 | One development case through the existing external H2B protocol binary. | A pilot, claim, stability, or performance. |

The D0-D4 executable emits immediately flushed `H2B_RUNTIME_EVENT` JSON lines
before and after lifecycle boundaries. D5 runs the existing external-protocol
candidate as a separate child process and records only the truthful
`external_protocol` and independent-validation boundaries; it does not invent
internal lifecycle events after execution. The runner retains stdout, stderr,
an events JSONL file, a final JSON result, and a Markdown summary. It uses a
fresh TT-Metal cache and never invokes the frozen pilot collector.

## Running one stage

Build `tools/h2b_n300_runtime_diagnostic` against the pinned clean source and
TT-Metal checkout. Then configure the runtime roots and run D0:

```bash
export TT_METAL_HOME=/path/to/pinned/tt-metal
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
python scripts/run_h2b_runtime_diagnostic.py \
  --stage d0 \
  --command /path/to/tt_rqm_h2b_runtime_diagnostic \
  --output-root reports/development/h2b-runtime
```

D2-D5 additionally require `--acknowledge-development-only`. For D5 the runner
creates a new one-case identity manifest and inputs inside the unique diagnostic
directory; it never reads a frozen pilot case.

After each preceding stage passes, use a new invocation:

```bash
python scripts/run_h2b_runtime_diagnostic.py --stage d1 --command /path/to/tt_rqm_h2b_runtime_diagnostic
python scripts/run_h2b_runtime_diagnostic.py --stage d2 --acknowledge-development-only --command /path/to/tt_rqm_h2b_runtime_diagnostic
python scripts/run_h2b_runtime_diagnostic.py --stage d3 --acknowledge-development-only --command /path/to/tt_rqm_h2b_runtime_diagnostic
python scripts/run_h2b_runtime_diagnostic.py --stage d4 --acknowledge-development-only --command /path/to/tt_rqm_h2b_runtime_diagnostic
python scripts/run_h2b_runtime_diagnostic.py \
  --stage d5 \
  --acknowledge-development-only \
  --external-command /path/to/tt_rqm_metalium_hamiltonian_evolution_candidate
```

The runner generates a unique subdirectory beneath
`reports/development/h2b-runtime` for every command and refuses dirty source,
missing roots, unresolved shared libraries, or unavailable device visibility.

## Failure interpretation

The runner classifies only the first incomplete boundary: `environment`,
`device_initialization`, `buffer_allocation`, `host_to_device`, `program_build`,
`h2a_execution`, `h1_execution`, `device_to_host`, `external_protocol`,
`numerical_validation`, `device_shutdown`, `event_validation`, or `unknown`.
A failure never implies a later layer failed.

Potential operator-level causes include stale fast-dispatch firmware, incomplete
device or Ethernet-dispatch teardown, unit-mesh discovery of both N300 chips,
runtime/firmware/UMD/KMD incompatibility, shared-memory state, cache state, or
repeated process initialization. The repository does not automate reset,
reboot, firmware mutation, or force-kill actions; those remain host-operator
decisions.

A separately versioned H2B pilot contract may be prepared only after D0-D5 pass
in order in fresh invocations with valid provenance and clean device shutdown.

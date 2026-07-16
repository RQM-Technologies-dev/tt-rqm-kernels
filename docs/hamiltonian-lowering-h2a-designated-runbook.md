# H2A designated Claim Level 0 collection runbook

The compensated candidate, source manifest, binary, compiler/runtime identity,
and nine serialized inputs are frozen. The current status is
`frozen_not_collected`; this runbook is not evidence that a session exists or
will pass.

Use two clean checkouts: the current governance checkout containing the frozen
contract, and a detached source checkout exactly at implementation commit
`225cb213ae79df7acd43d6056841c3eae7b5fc40`. The candidate binary must hash to
`b12063fd8ff73ff7372713eeb3fbdea31c56462c94e314713909a1f07e225979`, and
TT-Metal must be clean at `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`.

First run the non-mutating preflight. It opens no session and performs no
device work:

```bash
python scripts/collect_hamiltonian_lowering_h2a_designated.py \
  --dry-run \
  --governance-root . \
  --source-repo "$H2A_SOURCE_REPO" \
  --tt-metal-root "$TT_METAL_ROOT" \
  --candidate-binary "$H2A_CANDIDATE"
```

In the later authorized collection task, choose the one prescheduled session
identity and a new output path, then run exactly:

```bash
python scripts/collect_hamiltonian_lowering_h2a_designated.py \
  --governance-root . \
  --source-repo "$H2A_SOURCE_REPO" \
  --tt-metal-root "$TT_METAL_ROOT" \
  --candidate-binary "$H2A_CANDIDATE" \
  --session-id "$H2A_SESSION_ID" \
  --output-root "$H2A_SESSION_OUTPUT"
```

Do not skip, reorder, retry, replace, or discard a case. The collector retains
all nine attempts even after a failure and never promotes the result. After
inspection, qualify the untouched package offline:

```bash
python scripts/qualify_hamiltonian_lowering_h2a_designated.py \
  --session-root "$H2A_SESSION_OUTPUT"
```

Qualification may report `qualification_passed=true` for the retained session,
but still leaves `claim_level=null`, `stable_benchmark=false`, and
`performance_eligible=false`. A later publication task owns any Claim Level 0
release decision.

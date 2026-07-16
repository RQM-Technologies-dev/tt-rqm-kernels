# H2A Device-Side Hamiltonian Coefficient Lowering on Wormhole

HamiltonianLoweringBench H2A has a public **Claim Level 0 — silicon
conformance** release. One designated N300 device-0 session passed the exact
nine-case contract frozen before collection. This is a correctness and
provenance result, not a performance or stability result.

## Public claim

| Field | Released value |
|---|---|
| Claim level | 0 — silicon conformance |
| Device scope | one N300, device 0, one Tensix core |
| Designated sessions | 1 |
| Frozen cases | 9 |
| Attempts per case | 1 |
| Retries or replacements | 0 |
| `stable_benchmark` | `false` |
| `performance_eligible` | `false` |

The session retained every case result. All nine cases report zero failing
values, zero nonfinite values, one device create, and one device close. The
largest observed errors across the campaign were:

| Metric | Maximum |
|---|---:|
| Rotor absolute error | `1.1688396223238917e-04` |
| Phase absolute error | `7.396376489055001e-08` |
| Rotor norm drift | `5.99417120383805e-08` |
| Phase norm drift | `7.368516952155346e-08` |
| Complex-matrix reconstruction error | `1.2605900036450937e-04` |

The release uses the frozen combined whole-output tolerance of `atol=1e-4`
and `rtol=1e-4`; the large-angle case passes that combined contract.

## Immutable identity

- Candidate source commit: `225cb213ae79df7acd43d6056841c3eae7b5fc40`
- Candidate binary SHA-256:
  `b12063fd8ff73ff7372713eeb3fbdea31c56462c94e314713909a1f07e225979`
- Source bundle SHA-256:
  `519b2b9ffb7341893aed1574604ce3c0021b9c47830ca9c297d03d69b7cf80d5`
- TT-Metal commit: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`
- Designated session:
  `h2a-designated-conformance-n300-20260716-session-1`
- Original 46-file package tar SHA-256:
  `c68151223dbf3d789635338d641a4211334719b4a8b17c7ca7701d6b319fe746`

The [release manifest](../../benchmarks/manifests/wormhole-hamiltonian-lowering.json)
hash-binds the frozen contract, source identity, serialized-input manifest,
untouched session manifest, complete 46-file inventory, preflight and device
health, collector records, CPU validation, and offline qualification. The
[processed summary](../../benchmarks/processed/wormhole-hamiltonian-lowering-h2a-summary.json)
is regenerated deterministically from those committed artifacts.

## Reproduce the release gate

```bash
python scripts/validate_hamiltonian_lowering_release.py
python scripts/reproduce_wormhole_hamiltonian_lowering.py --check
```

## Nonclaims

This release does not establish performance, benchmark stability, acceleration,
CPU comparison, measured bandwidth, energy efficiency, dual-device scaling,
H2B integration, a complete device-resident H2 pipeline, inheritance from H1,
or Tenstorrent endorsement. The immutable source session and all nine case
reports remain `claim_level=null`, `stable_benchmark=false`, and
`performance_eligible=false`.

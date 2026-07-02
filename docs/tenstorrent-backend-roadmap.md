# Tenstorrent Backend Roadmap

This project starts with CPU/PyTorch reference kernels and then moves selected operators downward into the Tenstorrent stack when correctness and benchmark surfaces are stable.

## Phase 1: CPU/PyTorch Reference Kernels

Goals:

- implement readable PyTorch reference operators
- enforce the quaternion tensor convention `[..., 4]`
- validate shape and broadcasting behavior
- test Hamilton product identities, associativity tolerance, inverses, and unit-rotor norm preservation
- provide benchmark scripts with stable inputs and simple metrics
- provide StructuredBench reports for repeatable backend comparisons

Initial operators:

- `qmul`
- `qconj`
- `qnorm`
- `qnormalize`
- `qinverse`
- `qdot`
- `qrotate_vector`
- phase and orientation tracking helpers
- StructuredBench `smoke`, `full`, `qmul`, and `qrotate` suites

## Phase 2: TT-Metalium `qmul` and `qrotate` Kernels

Goals:

- port `qmul` to an explicit TT-Metalium kernel
- port `qrotate_vector` for streamed vector rotation
- map quaternion final-dimension layout onto Tenstorrent tile and memory movement constraints
- compare TT-Metalium outputs against the PyTorch reference kernels
- report throughput, latency, and numerical error against CPU reference outputs
- emit the `structuredbench.v1` report fields for CPU/PyTorch and TT-Metalium runs

Candidate work:

- tile-level layout experiments for final dimension `4`
- fused Hamilton product expressions
- streamed rotor-vector-rotor-conjugate evaluation
- benchmark parity with `benchmarks/qmul_throughput.py` and `benchmarks/qrotate_stream.py`

## Phase 3: TT-NN Operator Wrappers

Goals:

- expose selected structured kernels through TT-NN-style operator wrappers
- keep PyTorch reference behavior as the correctness contract
- define dtype, layout, shape, and broadcasting expectations
- make unit tests usable across CPU reference and TT-NN paths

Candidate wrappers:

- `tt_rqm.qmul`
- `tt_rqm.qrotate_vector`
- `tt_rqm.qnormalize`
- `tt_rqm.phase_delta`

## Phase 4: TT-Forge / TT-MLIR Lowering Exploration

Goals:

- explore lowering structured tensor operators from graph-level representations
- identify when quaternion operations should lower as fused kernels rather than scalar op expansions
- document shape contracts and graph rewrite opportunities
- evaluate portability across model compilation paths

This phase is exploratory. It should be driven by working Phase 1 reference tests and at least one measurable lower-stack kernel from Phase 2.

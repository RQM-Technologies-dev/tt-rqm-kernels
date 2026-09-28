# SU4Q N300 conformance

This private experimental target consumes proof-gated `su4q` compiler blocks
and applies the complete quaternion-Cartan operation to ordinary FP32
two-qubit state vectors on one Wormhole device.

The eight device-resident stages are right `q0`, right `q1`, Cartan `XX`,
Cartan `YY`, Cartan `ZZ`, left `q0`, left `q1`, and global phase. Intermediate
states remain in ping-pong device DRAM buffers. The only host transfers are the
initial block/state upload and final state download.

This is a correctness experiment. Its metrics set `performance_eligible` to
`false`; timings are diagnostic and do not support an acceleration claim.

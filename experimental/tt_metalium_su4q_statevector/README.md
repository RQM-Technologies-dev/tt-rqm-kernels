# Fused multi-qubit SU4Q conformance

Private correctness-only TT-Metalium experiment for applying proof-gated
quaternion-Cartan blocks to arbitrary pairs in a device-resident statevector.
The fused program gathers each `|q1 q0>` quartet, performs the complete SU4Q
transform in Tensix L1, and scatters it to the next device buffer.

All metrics set `performance_eligible=false`. Timings are diagnostic only.

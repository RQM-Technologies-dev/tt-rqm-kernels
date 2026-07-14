Hardware evidence collection and Level 2 publication are complete on the local
branch; no issue change has been posted.

- Three distinct device-0 cold-start sessions passed the preregistered
  `tt-rqm-benchmark-stability.v1` qualification. Maximum within/cross-session
  deviations were 6.2083%/3.2133% at N=4096, 2.0413%/2.4194% at N=65536, and
  1.7059%/0.9190% at N=262144.
- Individual reports remain `stable_benchmark=false`; only the qualification
  artifact is true.
- Device-1 parity, corrected physical core scaling, initialization diagnostics,
  Device Program Profiler plus Tracy, supported pinned hardware ceilings, and
  the N=1024 through 1048576 saturation sweep are also complete as diagnostics.
- The new release manifest publishes Claim Level 2 without changing any
  immutable single-session report. No acceleration, CPU, dual-device,
  application, hardware-bandwidth, or endorsement claim is proposed.

Recommended tracker action: check the three-session stability criterion and
link the Level 2 manifest, qualification, and processed evidence index.

from __future__ import annotations

import torch

from tt_rqm_kernels import phase_difference, smooth_phase, unwrap_phase


measurements = torch.tensor([3.00, 3.08, -3.12, -3.02, -2.94])
unwrapped = unwrap_phase(measurements)
deltas = phase_difference(measurements[1:], measurements[:-1])
smoothed = smooth_phase(measurements[:-1], measurements[1:], alpha=0.25)

print("unwrapped", unwrapped)
print("wrapped_deltas", deltas)
print("smoothed", smoothed)

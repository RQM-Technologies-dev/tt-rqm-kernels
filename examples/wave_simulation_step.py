from __future__ import annotations

import torch

from tt_rqm_kernels import integrate_phase, phase_to_unit_vector


amplitude = torch.tensor([1.0, 0.75, 0.5])
phase = torch.tensor([0.0, 1.0, 2.0])
angular_rate = torch.tensor([0.1, 0.2, -0.1])
dt = 0.016

next_phase = integrate_phase(phase, angular_rate, dt)
state = amplitude.unsqueeze(-1) * phase_to_unit_vector(next_phase)

print("next_phase", next_phase)
print("state_cos_sin", state)

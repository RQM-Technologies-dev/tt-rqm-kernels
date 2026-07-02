from __future__ import annotations

import torch

from tt_rqm_kernels import qmul, qnormalize, qrotate_vector


def rotor_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    half = angle.unsqueeze(-1) / 2.0
    return qnormalize(torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1))


axis = torch.tensor([[0.0, 0.0, 1.0]])
delta_yaw = torch.tensor([0.05])
orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
forward_body = torch.tensor([[1.0, 0.0, 0.0]])

delta_rotor = rotor_from_axis_angle(axis, delta_yaw)
orientation = qnormalize(qmul(delta_rotor, orientation))
forward_world = qrotate_vector(orientation, forward_body)

print("orientation", orientation)
print("forward_world", forward_world)

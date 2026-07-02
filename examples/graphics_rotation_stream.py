from __future__ import annotations

import torch

from tt_rqm_kernels import qnormalize, qrotate_vector


cube_vertices = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ]
)

rotor = qnormalize(torch.tensor([0.9659258, 0.0, 0.2588190, 0.0]))
rotated_vertices = qrotate_vector(rotor, cube_vertices)

print(rotated_vertices)

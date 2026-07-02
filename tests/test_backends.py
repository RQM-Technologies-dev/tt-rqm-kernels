from __future__ import annotations

import torch

from tt_rqm_kernels.backends import scalar_reference, tenstorrent_stub
from tt_rqm_kernels.quaternion_ops import qmul, qnormalize
from tt_rqm_kernels.rotor_ops import qrotate_vector


def test_scalar_reference_agrees_with_torch_qmul() -> None:
    a = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.25, 0.75, 1.0],
            [-0.1, 0.2, -0.3, 0.4],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor(
        [
            [0.25, 0.5, -0.75, 1.25],
            [1.0, 2.0, 3.0, 4.0],
            [0.8, -0.6, 0.4, -0.2],
        ],
        dtype=torch.float64,
    )

    torch_out = qmul(a, b)
    scalar_out = torch.tensor(
        [
            scalar_reference.qmul_scalar(a[index].tolist(), b[index].tolist())
            for index in range(a.shape[0])
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(torch_out, scalar_out, rtol=1e-12, atol=1e-12)


def test_scalar_reference_agrees_with_torch_qrotate() -> None:
    rotors = qnormalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9238795325, 0.0, 0.0, 0.3826834324],
                [0.8660254038, 0.5, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
    )
    vectors = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch_out = qrotate_vector(rotors, vectors)
    scalar_out = torch.tensor(
        [
            scalar_reference.qrotate_vector_scalar(
                rotors[index].tolist(),
                vectors[index].tolist(),
            )
            for index in range(rotors.shape[0])
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(torch_out, scalar_out, rtol=1e-12, atol=1e-12)


def test_tenstorrent_stub_fails_gracefully() -> None:
    try:
        tenstorrent_stub.qmul(None, None)
    except NotImplementedError as exc:
        assert tenstorrent_stub.NOT_IMPLEMENTED_MESSAGE in str(exc)
    else:
        raise AssertionError("Tenstorrent stub should fail with NotImplementedError")

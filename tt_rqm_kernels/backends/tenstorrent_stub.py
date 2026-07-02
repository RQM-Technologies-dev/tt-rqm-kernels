"""Future Tenstorrent backend contract for StructuredBench."""

from __future__ import annotations

NOT_IMPLEMENTED_MESSAGE = (
    "Tenstorrent backend is not implemented yet. This placeholder defines the "
    "expected backend contract for future TT-Metalium / TT-NN ports."
)

name = "tenstorrent"


def _raise_not_implemented(*args: object, **kwargs: object) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


def qmul(*args: object, **kwargs: object) -> None:
    """Future `[N, 4] x [N, 4] -> [N, 4] Hamilton product backend entrypoint."""

    _raise_not_implemented(*args, **kwargs)


def qrotate_vector(*args: object, **kwargs: object) -> None:
    """Future rotor/vector rotation backend entrypoint."""

    _raise_not_implemented(*args, **kwargs)


def qnormalize(*args: object, **kwargs: object) -> None:
    """Future quaternion normalization backend entrypoint."""

    _raise_not_implemented(*args, **kwargs)


def qinverse(*args: object, **kwargs: object) -> None:
    """Future quaternion inverse backend entrypoint."""

    _raise_not_implemented(*args, **kwargs)


def phase_update(*args: object, **kwargs: object) -> None:
    """Future phase update backend entrypoint."""

    _raise_not_implemented(*args, **kwargs)

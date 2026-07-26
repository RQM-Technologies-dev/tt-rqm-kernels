from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tt_rqm_kernels.su4q_statevector_conformance import (
    BATCH,
    DEPTHS,
    PROTOCOL,
    QUBITS,
    _schedule,
    apply_pair,
    quartet_indices,
)

ROOT = Path("experimental/tt_metalium_su4q_statevector")


def test_protocol_surface_is_frozen() -> None:
    assert PROTOCOL == "tt-rqm-su4q-statevector-conformance.v1"
    assert QUBITS == (3, 4, 6, 10, 15)
    assert DEPTHS == (1, 8, 32, 128)
    assert BATCH == 4


def test_quartet_indices_use_qiskit_q1_q0_order() -> None:
    actual = quartet_indices(3, 0, 2)
    assert actual.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7]]
    reversed_pair = quartet_indices(3, 2, 0)
    assert reversed_pair.tolist() == [[0, 4, 1, 5], [2, 6, 3, 7]]


@pytest.mark.parametrize(
    "args",
    ((3, 0, 0), (3, -1, 1), (3, 0, 3), (5, 0, 1)),
)
def test_quartet_indices_reject_malformed_pairs(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        quartet_indices(*args)


def test_apply_pair_respects_reversed_target_order() -> None:
    states = np.zeros((1, 8), dtype=np.complex128)
    states[0, 1] = 1
    x_on_q0 = np.asarray(
        ((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)),
        dtype=np.complex128,
    )
    forward = apply_pair(states, x_on_q0, 0, 2, 3)
    reversed_result = apply_pair(states, x_on_q0, 2, 0, 3)
    assert np.argmax(np.abs(forward[0])) == 0
    assert np.argmax(np.abs(reversed_result[0])) == 5


def test_schedule_covers_compiler_adjacent_nonadjacent_and_reversed() -> None:
    schedule = _schedule(6, 32, 11)
    assert {item["case_index"] for item in schedule} >= set(range(6))
    assert {item["pair_class"] for item in schedule} == {
        "adjacent",
        "non_adjacent",
        "reversed_order",
    }


def test_fused_source_contract() -> None:
    source = (ROOT / "src/statevector_candidate.cpp").read_text()
    compute = (ROOT / "kernels/fused_compute.cpp").read_text()
    reader = (ROOT / "kernels/gather_reader.cpp").read_text()
    writer = (ROOT / "kernels/scatter_writer.cpp").read_text()
    assert '"program_count", depth' in source
    assert source.count("EnqueueWriteMeshBuffer") == 2
    assert source.count("EnqueueReadMeshBuffer") == 1
    assert '"intermediate_h2d_count", 0' in source
    assert '"intermediate_d2h_count", 0' in source
    assert "apply_local(bank_a, bank_b, 0)" in compute
    assert "apply_cartan(bank_a, bank_b, 0)" in compute
    assert "apply_phase(bank_b)" in compute
    assert "amplitude_out & ~(1U << q0) & ~(1U << q1)" in reader
    assert "((amplitude >> q0) & 1U)" in writer
    assert "4 * sizeof(uint32_t)" in reader
    assert "noc_async_write_page" in writer
    assert "block_to_unitary" not in source


def test_build_is_narrow_and_warnings_are_errors() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "tt_rqm_metalium_su4q_statevector_conformance" in cmake
    assert "-Wall -Wextra -Werror" in cmake

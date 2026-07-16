from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from tt_rqm_kernels.hamiltonian_lowering_benchmark import reference_cases
from tt_rqm_kernels.hamiltonian_lowering_candidate import HARDWARE_ATOL, HARDWARE_RTOL
from tt_rqm_kernels.hamiltonian_lowering_compensation import (
    INV_TWO_PI,
    TWO_PI_HI,
    TWO_PI_LO,
    compensated_angle,
    compensated_sum,
    fp32,
    reduce_pair,
    split_two_product,
)
from tt_rqm_kernels.hamiltonian_lowering_pilot import frozen_case_input_hashes

ROOT = Path(__file__).resolve().parents[1]
COMPENSATED = ROOT / "experimental" / "tt_metalium_hamiltonian_lowering_compensated"
ORIGINAL = ROOT / "experimental" / "tt_metalium_hamiltonian_lowering"


def _source_digest(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".cpp", ".h", ".py", ".txt"}:
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    ((53851.6484375, fp32(math.pi / 7)), (10000.0, fp32(math.pi / 7)), (-20000.0, fp32(math.pi / 7))),
)
def test_split_two_product_recovers_exact_fp32_input_product(lhs: float, rhs: float) -> None:
    pair = split_two_product(lhs, rhs)
    assert pair.value == fp32(lhs) * fp32(rhs)


def test_compensated_sum_retains_low_component() -> None:
    pair = compensated_sum((1.0e8, 1.0, -1.0e8))
    assert pair.value == 1.0


def test_split_period_constants_reconstruct_two_pi() -> None:
    assert TWO_PI_HI == fp32(2.0 * math.pi)
    assert TWO_PI_LO < 0.0
    assert TWO_PI_HI + TWO_PI_LO == pytest.approx(2.0 * math.pi, abs=1e-14)
    assert INV_TWO_PI == fp32(1.0 / (2.0 * math.pi))


@pytest.mark.parametrize("coefficient", (0.0, 1.0, -1.0, 10000.0, -20000.0, 53851.6484375))
def test_compensated_reduction_preserves_sign_and_quadrant(coefficient: float) -> None:
    scaled_dt = fp32(math.pi / 7)
    pair, reduced = compensated_angle(coefficient, scaled_dt)
    expected = math.remainder(pair.value, 2.0 * math.pi)
    assert reduced == pytest.approx(expected, abs=8e-7)
    assert math.sin(reduced) == pytest.approx(math.sin(pair.value), abs=8e-7)
    assert math.cos(reduced) == pytest.approx(math.cos(pair.value), abs=8e-7)


def test_low_component_is_collapsed_only_after_reduction() -> None:
    pair = split_two_product(53851.6484375, fp32(math.pi / 7))
    collapsed_first = fp32(pair.hi + pair.lo)
    reduced = reduce_pair(pair)
    expected = math.remainder(pair.value, 2.0 * math.pi)
    assert abs(reduced - expected) < abs(math.remainder(collapsed_first, 2.0 * math.pi) - expected)


def test_exact_zero_remains_exact() -> None:
    pair, reduced = compensated_angle(0.0, fp32(math.pi / 7))
    assert pair.hi == pair.lo == reduced == 0.0


def test_frozen_large_inputs_and_hashes_are_deterministic() -> None:
    first = frozen_case_input_hashes()
    second = frozen_case_input_hashes()
    assert first == second
    large = next(case for case in reference_cases(seed=0) if case["id"] == "large_angles")
    assert hashlib.sha256(large["hamiltonians"].numpy().tobytes()).hexdigest() == first["large_angles"]["hamiltonians_sha256"]


def test_candidate_accepts_original_inputs_not_host_reduced_angles() -> None:
    manifest_source = (ROOT / "tt_rqm_kernels" / "hamiltonian_lowering_candidate.py").read_text()
    kernel_source = (COMPENSATED / "kernels" / "compute_hamiltonian_lowering_compensated.cpp").read_text()
    assert '"hamiltonians"' in manifest_source and '"dt"' in manifest_source
    assert "reduced_theta" not in manifest_source
    assert "reduced_angle" not in manifest_source
    assert "angle_pair(r,step_hi,step_lo" in kernel_source
    assert "angle_pair(h0,step_hi,step_lo" in kernel_source


def test_original_and_compensated_source_identities_differ() -> None:
    assert _source_digest(ORIGINAL) != _source_digest(COMPENSATED)


def test_frozen_tolerances_and_pilot_nonclaim_are_unchanged() -> None:
    assert HARDWARE_ATOL == HARDWARE_RTOL == 1e-4
    pilot_source = (ROOT / "tt_rqm_kernels" / "hamiltonian_lowering_pilot.py").read_text()
    assert '"stable_benchmark": False' in pilot_source
    assert '"claim_level": None' in pilot_source

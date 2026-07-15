from __future__ import annotations

from pathlib import Path

import pytest

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import run_su2_compose
from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.su2_profile import (
    DEFAULT_EVIDENCE_MANIFEST,
    SU2ProfileError,
    parse_device_profile,
    parse_tracy_statistics,
    validate_profile_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


PROCESSORS = ("BRISC", "TRISC_0", "TRISC_1", "TRISC_2", "NCRISC")


def _write_device_profile(
    path: Path, *, dispatches: int, omit: tuple[int, str] | None = None
) -> None:
    lines = [
        "ARCH: wormhole_b0",
        "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],stat value,Run ID,zone name,type,source line,source file",
    ]
    for dispatch in range(dispatches):
        start = 1000 + dispatch * 1000
        for index, processor in enumerate(PROCESSORS):
            if omit == (dispatch, processor):
                continue
            zone = f"{processor}-KERNEL"
            lines.append(f"0,0,0,{processor},0,{start + index},0,1,{zone},ZONE_START,1,kernel.cpp")
            lines.append(
                f"0,0,0,{processor},0,{start + index + 100 + index},0,1,{zone},ZONE_END,1,kernel.cpp"
            )
    path.write_text("\n".join(lines) + "\n")


def test_device_profile_maps_fused_and_unfused_roles(tmp_path: Path) -> None:
    profile = tmp_path / "profile.csv"
    _write_device_profile(profile, dispatches=2)
    parsed = parse_device_profile(profile, steps=2, core_count=1)
    assert parsed["dispatch_count"] == 2
    assert parsed["paths"]["fused"]["dispatch_count"] == 1
    assert parsed["paths"]["unfused"]["dispatch_count"] == 1
    assert parsed["paths"]["fused"]["critical_device_role"] == "writer"
    assert parsed["circular_buffer_waits"] == "not_observable"


def test_device_profile_rejects_incomplete_role_markers(tmp_path: Path) -> None:
    profile = tmp_path / "profile.csv"
    _write_device_profile(profile, dispatches=2, omit=(1, "NCRISC"))
    with pytest.raises(SU2ProfileError, match="incomplete NCRISC"):
        parse_device_profile(profile, steps=2, core_count=1)


def test_tracy_dispatch_and_finish_fraction(tmp_path: Path) -> None:
    stats = tmp_path / "stats.csv"
    stats.write_text(
        "name,src_file,src_line,total_ns,total_perc,counts,mean_ns,min_ns,max_ns,std_ns\n"
        "EnqueueProgram,a.cpp,1,100,0,2,50,40,60,1\n"
        "FDMeshCommandQueue::finish,b.cpp,2,300,0,2,150,140,160,1\n"
        "FDMeshCommandQueue::finish_nolock,b.cpp,3,500,0,3,166,100,200,1\n"
    )
    parsed = parse_tracy_statistics(stats, timed_pair_s=1e-6)
    assert parsed["zones"]["EnqueueProgram"]["count"] == 2
    assert parsed["direct_dispatch_and_finish_fraction_of_timed_pair"] == 0.4


def test_custom_su2_cases_are_profile_only() -> None:
    with pytest.raises(IntegrityError, match="diagnostic profile cases only"):
        run_su2_compose(
            command="/bin/true",
            stage="performance",
            methodology_note="test",
            case_specs=((32, 8, 1, 0, 1),),
        )


def test_retained_su2_profile_evidence_validates() -> None:
    manifest = validate_profile_evidence(ROOT / DEFAULT_EVIDENCE_MANIFEST, repo_root=ROOT)
    assert [attempt["status"] for attempt in manifest["attempts"]] == [
        "failed",
        "failed",
        "passed",
    ]

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/h2b_n300_runtime_diagnostic/probe.cpp"
RUNNER = ROOT / "scripts/run_h2b_runtime_diagnostic.py"


def stage_body(source: str, name: str, next_name: str) -> str:
    return source.split(f"int {name}", 1)[1].split(f"int {next_name}", 1)[0]


def test_d0_and_d1_do_not_build_or_execute_rqm_programs() -> None:
    source = SOURCE.read_text()
    d0 = stage_body(source, "run_d0", "run_d1")
    d1 = stage_body(source, "run_d1", "run_kernel_stage")
    for body in (d0, d1):
        assert "build_h2a_program" not in body
        assert "build_h1_program" not in body
        assert "EnqueueMeshWorkload" not in body


def test_d2_and_d3_are_mutually_isolated_by_stage_guards() -> None:
    source = SOURCE.read_text()
    body = stage_body(source, "run_kernel_stage", "run_d5")
    assert 'const bool use_h2a = stage != "d3"' in body
    assert 'const bool use_h1 = stage != "d2"' in body
    assert "build_h2a_program(device, input, intermediate" in body
    assert "build_h1_program(device, intermediate, final_output" in body


def test_d4_uses_one_device_and_device_resident_intermediate() -> None:
    source = SOURCE.read_text()
    body = stage_body(source, "run_kernel_stage", "run_d5")
    assert body.count("MeshDevice::create_unit_mesh(0)") == 1
    assert "build_h2a_program(device, input, intermediate" in body
    assert "build_h1_program(device, intermediate, final_output" in body
    assert "EnqueueReadMeshBuffer(queue, output, source" in body


def test_d5_is_runner_owned_and_never_fabricated_by_probe() -> None:
    source = SOURCE.read_text()
    runner = RUNNER.read_text()
    assert "run_d5" not in source
    assert '"external_protocol_begin"' in runner
    assert "validate_d5_outputs" in runner
    assert "--external-command" in runner
    assert "collect_pilot" not in source
    assert "collect_pilot" not in runner


def test_events_are_json_prefixed_and_immediately_flushed() -> None:
    source = SOURCE.read_text()
    assert '"H2B_RUNTIME_EVENT "' in source
    assert "std::flush" in source
    assert '"unknown H2B runtime-isolation stage"' in source


def test_runner_uses_fresh_output_cache_and_never_retries() -> None:
    runner = RUNNER.read_text()
    assert "uuid.uuid4().hex[:8]" in runner
    assert "output_dir.mkdir(parents=True, exist_ok=False)" in runner
    assert 'cache = output_dir / "tt-metal-cache"' in runner
    assert (
        '"attempt_count": 1'
        in (ROOT / "tt_rqm_kernels/hamiltonian_evolution_runtime_isolation.py").read_text()
    )
    assert (
        '"retry_count": 0'
        in (ROOT / "tt_rqm_kernels/hamiltonian_evolution_runtime_isolation.py").read_text()
    )
    assert "for attempt" not in runner


def test_runner_preflight_is_non_device_creating_and_fail_closed() -> None:
    runner = RUNNER.read_text()
    preflight = runner.split("def preflight", 1)[1].split("def prepare_d5_case", 1)[0]
    assert "MeshDevice" not in preflight
    assert "TT_METAL_HOME" in preflight
    assert "TT_METAL_RUNTIME_ROOT" in preflight
    assert "device 0 visibility preflight failed" in preflight
    assert "shared libraries do not resolve" in preflight
    assert "diagnostic source checkout must be clean" in preflight
    assert "TT-Metal checkout must be clean" in preflight

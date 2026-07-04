"""Environment discovery for Tenstorrent-facing qmul candidate paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import shutil
from typing import Literal, Mapping

Mode = Literal["emule", "hardware"]

DEFAULT_HARDWARE_COMMAND_ENV = "TT_RQM_HARDWARE_QMUL_COMMAND"
TT_METAL_ENV_VARS = ("TT_METAL_HOME", "TT_METALIUM_HOME")
TT_EMULE_ENV_VAR = "TT_EMULE_HOME"
EMULE_COMMAND = "bash experimental/tt_metalium_qmul/run_candidate_docker.sh"
EMULE_SCRIPT_REL = Path("experimental/tt_metalium_qmul/run_candidate_docker.sh")
EMULE_BINARY_REL = Path(
    "experimental/tt_metalium_qmul/build_emule_candidate/"
    "tt_rqm_metalium_qmul_candidate"
)


@dataclass(frozen=True)
class ReadinessItem:
    """One quickstart readiness check."""

    name: str
    available: bool
    detail: str
    path: str | None = None


@dataclass(frozen=True)
class TenstorrentReadiness:
    """Current Tenstorrent adapter readiness summary."""

    repo_root: str
    report_dir: str
    python_package_available: bool
    torch_available: bool
    tt_metal_home: str | None
    tt_emule_home: str | None
    docker_available: bool
    emule_candidate_script: str
    emule_candidate_script_present: bool
    emule_candidate_binary: str
    emule_candidate_binary_present: bool
    emule_ready: bool
    hardware_command: str | None
    hardware_ready: bool
    checks: tuple[ReadinessItem, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class ExecutionPath:
    """Resolved external-qmul command and label for one execution mode."""

    mode: Mode
    command: str | None
    execution_label: Literal["emulation", "hardware"]
    available: bool
    reason: str


def check_readiness(
    *,
    repo_root: Path | None = None,
    report_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    hardware_command: str | None = None,
) -> TenstorrentReadiness:
    """Inspect local readiness without running a Tenstorrent candidate."""

    root = (repo_root or default_repo_root()).resolve()
    environment = env or os.environ
    reports = (report_dir or root / "reports").resolve()
    tt_metal = _resolve_root(TT_METAL_ENV_VARS, "tt-metal", root, environment)
    tt_emule = _resolve_root((TT_EMULE_ENV_VAR,), "tt-emule", root, environment)
    script = root / EMULE_SCRIPT_REL
    binary = root / EMULE_BINARY_REL
    docker_path = shutil.which("docker")
    resolved_hardware_command = hardware_command or environment.get(
        DEFAULT_HARDWARE_COMMAND_ENV
    )

    package_available = importlib.util.find_spec("tt_rqm_kernels") is not None
    torch_available = importlib.util.find_spec("torch") is not None
    script_present = script.exists()
    binary_present = binary.exists()
    docker_available = docker_path is not None
    emule_ready = all(
        [
            script_present,
            binary_present,
            tt_metal is not None,
            tt_emule is not None,
            docker_available,
        ]
    )
    hardware_ready = bool(resolved_hardware_command)

    checks = (
        ReadinessItem(
            "Python package import",
            package_available,
            "tt_rqm_kernels import spec found"
            if package_available
            else "tt_rqm_kernels import spec not found",
        ),
        ReadinessItem(
            "torch availability",
            torch_available,
            "torch import spec found" if torch_available else "torch is unavailable",
        ),
        ReadinessItem("repo root", root.exists(), str(root), str(root)),
        ReadinessItem(
            "TT_METAL_HOME",
            tt_metal is not None,
            _env_or_sibling_detail(TT_METAL_ENV_VARS, "tt-metal", tt_metal, environment),
            str(tt_metal) if tt_metal is not None else None,
        ),
        ReadinessItem(
            "TT_EMULE_HOME",
            tt_emule is not None,
            _env_or_sibling_detail((TT_EMULE_ENV_VAR,), "tt-emule", tt_emule, environment),
            str(tt_emule) if tt_emule is not None else None,
        ),
        ReadinessItem(
            "Docker",
            docker_available,
            docker_path or "docker not found on PATH",
            docker_path,
        ),
        ReadinessItem(
            "emule candidate script",
            script_present,
            "present" if script_present else "missing",
            str(script),
        ),
        ReadinessItem(
            "emule candidate binary",
            binary_present,
            "present" if binary_present else "missing; build the candidate first",
            str(binary),
        ),
        ReadinessItem(
            "report output directory",
            reports.exists(),
            str(reports),
            str(reports),
        ),
        ReadinessItem(
            "hardware command",
            hardware_ready,
            (
                f"configured via command or {DEFAULT_HARDWARE_COMMAND_ENV}"
                if hardware_ready
                else f"not configured; set {DEFAULT_HARDWARE_COMMAND_ENV} or pass --command"
            ),
        ),
    )

    return TenstorrentReadiness(
        repo_root=str(root),
        report_dir=str(reports),
        python_package_available=package_available,
        torch_available=torch_available,
        tt_metal_home=str(tt_metal) if tt_metal is not None else None,
        tt_emule_home=str(tt_emule) if tt_emule is not None else None,
        docker_available=docker_available,
        emule_candidate_script=str(script),
        emule_candidate_script_present=script_present,
        emule_candidate_binary=str(binary),
        emule_candidate_binary_present=binary_present,
        emule_ready=emule_ready,
        hardware_command=resolved_hardware_command,
        hardware_ready=hardware_ready,
        checks=checks,
    )


def resolve_execution_path(
    mode: Mode,
    *,
    command: str | None = None,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ExecutionPath:
    """Resolve the command and execution label for emulation or hardware."""

    if mode == "emule":
        readiness = check_readiness(repo_root=repo_root, env=env)
        if not readiness.emule_ready:
            missing = [
                item.name
                for item in readiness.checks
                if item.name
                in {
                    "TT_METAL_HOME",
                    "TT_EMULE_HOME",
                    "Docker",
                    "emule candidate script",
                    "emule candidate binary",
                }
                and not item.available
            ]
            return ExecutionPath(
                mode=mode,
                command=command or EMULE_COMMAND,
                execution_label="emulation",
                available=False,
                reason="emulation path unavailable: " + ", ".join(missing),
            )
        return ExecutionPath(
            mode=mode,
            command=command or EMULE_COMMAND,
            execution_label="emulation",
            available=True,
            reason="emulation path available",
        )

    environment = env or os.environ
    resolved_command = command or environment.get(DEFAULT_HARDWARE_COMMAND_ENV)
    if not resolved_command:
        return ExecutionPath(
            mode=mode,
            command=None,
            execution_label="hardware",
            available=False,
            reason=(
                "hardware command is not configured; pass --command or set "
                f"{DEFAULT_HARDWARE_COMMAND_ENV}"
            ),
        )
    return ExecutionPath(
        mode=mode,
        command=resolved_command,
        execution_label="hardware",
        available=True,
        reason="hardware command configured",
    )


def default_repo_root() -> Path:
    """Return the repository root from this installed source tree."""

    return Path(__file__).resolve().parents[3]


def _resolve_root(
    env_names: tuple[str, ...],
    sibling_name: str,
    repo_root: Path,
    env: Mapping[str, str],
) -> Path | None:
    for name in env_names:
        value = env.get(name)
        if value:
            path = Path(value).expanduser()
            return path.resolve() if path.exists() else None
    sibling = repo_root.parent / sibling_name
    return sibling.resolve() if sibling.exists() else None


def _env_or_sibling_detail(
    env_names: tuple[str, ...],
    sibling_name: str,
    path: Path | None,
    env: Mapping[str, str],
) -> str:
    for name in env_names:
        value = env.get(name)
        if value:
            return f"{name}={value}" if path is not None else f"{name} does not exist: {value}"
    if path is not None:
        return f"{env_names[0]} unset; sibling {sibling_name} checkout found at {path}"
    return f"{env_names[0]} unset and sibling {sibling_name} checkout not found"

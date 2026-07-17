from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h2b_n300_remote", ROOT / "scripts/h2b_n300_remote.py"
)
assert SPEC is not None and SPEC.loader is not None
REMOTE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REMOTE)


def _config() -> object:
    return REMOTE.Config(
        helper=Path("/private/helper.py"),
        base="/private/base",
        metal="/private/tt-metal",
        mpi="/private/mpi",
        source="/private/source",
        binary="/private/candidate",
        cache="/private/cache/session-2",
    )


def test_run_command_propagates_both_runtime_roots() -> None:
    config = _config()
    work, cache = REMOTE._case_remote_paths(config, "identity_k1")
    command = REMOTE._run_command(config, work, cache)
    assert "export TT_METAL_HOME=/private/tt-metal" in command
    assert "export TT_METAL_RUNTIME_ROOT=/private/tt-metal" in command
    assert "export TT_METAL_CACHE=/private/cache/session-2/identity_k1" in command


def test_case_paths_isolate_runtime_caches_and_sanitizer_removes_paths() -> None:
    config = _config()
    first = REMOTE._case_remote_paths(config, "identity_k1")
    second = REMOTE._case_remote_paths(config, "identity_k2")
    assert first[1] != second[1]
    sanitized = REMOTE._sanitize("/private/candidate /private/tt-metal", config)
    assert "/private/" not in sanitized
    assert sanitized == "<candidate-binary> <tt-metal-root>"

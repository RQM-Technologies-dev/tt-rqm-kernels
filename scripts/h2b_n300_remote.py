#!/usr/bin/env python3
"""Private-endpoint-neutral remote launcher for H2B N300 pilot collection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import NamedTuple


class Config(NamedTuple):
    helper: Path
    base: str
    metal: str
    mpi: str
    source: str
    binary: str
    cache: str


def _config() -> Config:
    required = {
        "helper": os.environ.get("TT_RQM_H2B_REMOTE_HELPER", ""),
        "base": os.environ.get("TT_RQM_H2B_REMOTE_BASE", ""),
        "metal": os.environ.get("TT_RQM_H2B_REMOTE_TT_METAL_ROOT", ""),
        "mpi": os.environ.get("TT_RQM_H2B_REMOTE_MPI_ROOT", ""),
        "source": os.environ.get("TT_RQM_H2B_REMOTE_SOURCE_ROOT", ""),
        "binary": os.environ.get("TT_RQM_H2B_REMOTE_BINARY", ""),
        "cache": os.environ.get("TT_RQM_H2B_REMOTE_CACHE_ROOT", ""),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing remote launcher configuration: {', '.join(missing)}")
    helper = Path(required["helper"]).resolve()
    if not helper.is_file():
        raise ValueError("remote helper is not a file")
    return Config(
        helper=helper, **{key: value for key, value in required.items() if key != "helper"}
    )


def _helper(config: Config, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(config.helper), *args], capture_output=True, text=True
    )


def _remote(config: Config, command: str) -> subprocess.CompletedProcess[str]:
    return _helper(config, "exec", command)


def _sanitize(value: str, config: Config) -> str:
    replacements = {
        config.binary: "<candidate-binary>",
        config.source: "<candidate-source>",
        config.cache: "<runtime-cache-root>",
        config.metal: "<tt-metal-root>",
        config.mpi: "<mpi-root>",
        config.base: "<remote-session-root>",
    }
    for private, public in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        value = value.replace(private, public)
    return re.sub(r"[^\s]+@[^\s:]+:", "<remote-endpoint>:", value)


def _preflight(config: Config) -> int:
    script = r"""import hashlib,importlib,json,os,platform,subprocess,sys
from pathlib import Path
base=Path(os.environ["BASE"]).resolve()
home_value=os.environ.get("TT_METAL_HOME","")
runtime_value=os.environ.get("TT_METAL_RUNTIME_ROOT","")
metal=Path(os.environ["EXPECTED_METAL"]).resolve()
home=Path(home_value).resolve() if home_value else None
runtime=Path(runtime_value).resolve() if runtime_value else None
source=Path(os.environ["SOURCE"]).resolve()
binary=Path(os.environ["BINARY"]).resolve()
cache=Path(os.environ["CACHE"]).resolve()
mpi=Path(os.environ["MPI"]).resolve()
def git(path,*args):
    result=subprocess.run(["git","-C",str(path),*args],capture_output=True,text=True)
    return result.returncode,result.stdout.strip()
source_rc,source_commit=git(source,"rev-parse","HEAD")
source_status_rc,source_status=git(source,"status","--porcelain")
metal_rc,metal_commit=git(metal,"rev-parse","HEAD")
metal_status_rc,metal_status=git(metal,"status","--porcelain")
bundle=None
if source_rc==0:
    sys.path.insert(0,str(source))
    module=importlib.import_module("experimental.tt_metalium_hamiltonian_evolution.run_candidate")
    bundle=module.source_bundle_sha256()
candidate_sha=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None
ldd_env=os.environ.copy()
ldd_env["LD_LIBRARY_PATH"]=str(mpi/"lib")+":"+str(metal/"build_n300/lib")+":"+ldd_env.get("LD_LIBRARY_PATH","")
ldd=subprocess.run(["ldd",str(binary)],capture_output=True,text=True,env=ldd_env) if binary.is_file() else None
ldd_text="" if ldd is None else ldd.stdout+ldd.stderr
unresolved=[] if ldd is None else [line.strip() for line in ldd_text.splitlines() if "not found" in line]
health_result=subprocess.run(["tt-smi","-s"],capture_output=True,text=True)
health=json.loads(health_result.stdout) if health_result.returncode==0 else {}
metal_library_lines=[line for line in ldd_text.splitlines() if "libtt_" in line]
expected_metal_library=bool(metal_library_lines) and all(str(metal) in line for line in metal_library_lines)
payload={
 "schema":"tt-rqm-hamiltonian-evolution-pilot-preflight.v1",
 "tt_metal_home_set":bool(home_value),
 "tt_metal_runtime_root_set":bool(runtime_value),
 "runtime_roots_resolve_same":home==runtime==metal,
 "runtime_root_exists":metal.is_dir(),
 "runtime_discoverable":(metal/"tt_metal").is_dir() and (metal/"build_n300/lib/libtt_metal.so").is_file(),
 "candidate_exists":binary.is_file(),
 "candidate_executable":binary.is_file() and os.access(binary,os.X_OK),
 "candidate_sha256":candidate_sha,
 "source_exists":source.is_dir(),
 "source_commit":source_commit if source_rc==0 else None,
 "source_tree_clean":source_status_rc==0 and not bool(source_status),
 "source_bundle_sha256":bundle,
 "tt_metal_commit":metal_commit if metal_rc==0 else None,
 "tt_metal_tree_clean":metal_status_rc==0 and not bool(metal_status),
 "shared_libraries_resolved":ldd is not None and ldd.returncode==0 and not unresolved,
 "unresolved_shared_library_count":len(unresolved),
 "tt_metal_library_from_expected_root":expected_metal_library,
 "runtime_cache_parent_writable":base.is_dir() and os.access(base,os.W_OK),
 "runtime_cache_session_root_new":not cache.exists(),
 "device_health":health,
 "device_id":0,
 "host_identity_sha256":hashlib.sha256(platform.node().encode()).hexdigest(),
 "compiler_version":subprocess.run([str(mpi/"bin/mpicxx"),"--version"],capture_output=True,text=True).stdout.splitlines()[0],
 "tt_smi_version":subprocess.run(["tt-smi","--version"],capture_output=True,text=True).stdout.strip().splitlines()[0],
}
payload["passed"]=all(payload[key] is True for key in (
 "tt_metal_home_set","tt_metal_runtime_root_set","runtime_roots_resolve_same",
 "runtime_root_exists","runtime_discoverable","candidate_exists","candidate_executable",
 "source_exists","source_tree_clean","tt_metal_tree_clean","shared_libraries_resolved",
 "tt_metal_library_from_expected_root","runtime_cache_parent_writable",
 "runtime_cache_session_root_new")) and health_result.returncode==0
print(json.dumps(payload,sort_keys=True))"""
    env = {
        "BASE": config.base,
        "EXPECTED_METAL": config.metal,
        "TT_METAL_HOME": config.metal,
        "TT_METAL_RUNTIME_ROOT": config.metal,
        "SOURCE": config.source,
        "BINARY": config.binary,
        "CACHE": config.cache,
        "MPI": config.mpi,
    }
    command = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    result = _remote(config, f"{command} python3 -c {shlex.quote(script)}")
    sys.stdout.write(_sanitize(result.stdout, config))
    sys.stderr.write(_sanitize(result.stderr, config))
    return result.returncode


def _health(config: Config) -> int:
    result = _remote(config, "tt-smi -s")
    sys.stdout.write(_sanitize(result.stdout, config))
    sys.stderr.write(_sanitize(result.stderr, config))
    return result.returncode


def _environment(config: Config) -> int:
    script = r"""import hashlib,json,os,platform,subprocess
from pathlib import Path
source=Path(os.environ["SOURCE"])
metal=Path(os.environ["METAL"])
binary=Path(os.environ["BINARY"])
def first(command):
    result=subprocess.run(command,capture_output=True,text=True)
    return result.stdout.splitlines()[0] if result.returncode==0 and result.stdout else "unavailable"
def git(path,*args):
    return subprocess.run(["git","-C",str(path),*args],check=True,capture_output=True,text=True).stdout.strip()
print(json.dumps({
 "schema":"tt-rqm-hamiltonian-evolution-pilot-environment.v1",
 "host_identity_sha256":hashlib.sha256(platform.node().encode()).hexdigest(),
 "platform":platform.platform(),
 "cpu_affinity":sorted(os.sched_getaffinity(0)),
 "numa_online":open("/sys/devices/system/node/online").read().strip(),
 "python_version":platform.python_version(),
 "compiler_version":first([os.environ["MPI"]+"/bin/mpicxx","--version"]),
 "cmake_version":first(["cmake","--version"]),
 "tt_smi_version":first(["tt-smi","--version"]),
 "source_commit":git(source,"rev-parse","HEAD"),
 "source_tree_clean":not bool(git(source,"status","--porcelain")),
 "tt_metal_commit":git(metal,"rev-parse","HEAD"),
 "tt_metal_tree_clean":not bool(git(metal,"status","--porcelain")),
 "candidate_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
 "runtime_cache_policy":"fresh empty TT_METAL_CACHE per case",
 "tt_metal_home_set":True,
 "tt_metal_runtime_root_set":True,
 "runtime_roots_resolve_same":True,
 "device_id":0,
 "performance_collection":False,
 "profiler_enabled":False,
 "watcher_enabled":False,
},sort_keys=True))"""
    env = {
        "METAL": config.metal,
        "SOURCE": config.source,
        "BINARY": config.binary,
        "MPI": config.mpi,
    }
    command = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    result = _remote(config, f"{command} python3 -c {shlex.quote(script)}")
    sys.stdout.write(_sanitize(result.stdout, config))
    sys.stderr.write(_sanitize(result.stderr, config))
    return result.returncode


def _run(config: Config) -> int:
    work = Path(os.environ["TT_RQM_H2B_DIR"]).resolve()
    manifest = Path(os.environ["TT_RQM_H2B_MANIFEST"]).resolve()
    case_id = work.parent.name
    remote_work, remote_cache = _case_remote_paths(config, case_id)
    setup = _remote(
        config,
        " && ".join(
            (
                f"mkdir -p {shlex.quote(config.base + '/session-2-runs')}",
                f"mkdir -p {shlex.quote(config.cache)}",
                f"mkdir {shlex.quote(remote_work)}",
                f"mkdir {shlex.quote(remote_cache)}",
                f"test -w {shlex.quote(remote_cache)}",
            )
        ),
    )
    if setup.returncode != 0:
        (work / "_remote_stderr.txt").write_text(_sanitize(setup.stderr, config), encoding="utf-8")
        return setup.returncode
    for local in (work / "hamiltonians.bin", work / "dt.bin", manifest):
        uploaded = _helper(config, "upload", str(local), f"{remote_work}/{local.name}")
        if uploaded.returncode != 0:
            (work / "_remote_stderr.txt").write_text(
                _sanitize(uploaded.stderr, config), encoding="utf-8"
            )
            return uploaded.returncode
    command = _run_command(config, remote_work, remote_cache)
    result = _remote(config, command)
    stdout = _sanitize(result.stdout, config)
    stderr = _sanitize(result.stderr, config)
    (work / "_remote_stdout.txt").write_text(stdout, encoding="utf-8")
    (work / "_remote_stderr.txt").write_text(stderr, encoding="utf-8")
    returncode = result.returncode
    for name in ("metrics.json", "final_rotors.bin", "final_phases.bin"):
        downloaded = _helper(config, "download", f"{remote_work}/{name}", str(work / name))
        if downloaded.returncode != 0 and returncode == 0:
            returncode = downloaded.returncode
            stderr += _sanitize(downloaded.stderr, config)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return returncode


def _case_remote_paths(config: Config, case_id: str) -> tuple[str, str]:
    if not case_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id):
        raise ValueError("invalid H2B case id")
    return f"{config.base}/session-2-runs/{case_id}", f"{config.cache}/{case_id}"


def _run_command(config: Config, remote_work: str, remote_cache: str) -> str:
    return " && ".join(
        (
            f"cd {shlex.quote(config.source)}",
            f"export PATH={shlex.quote(config.mpi + '/bin')}:$PATH",
            f"export LD_LIBRARY_PATH={shlex.quote(config.mpi + '/lib')}:{shlex.quote(config.metal + '/build_n300/lib')}:$LD_LIBRARY_PATH",
            f"export CXX={shlex.quote(config.mpi + '/bin/mpicxx')}",
            f"export TT_METAL_HOME={shlex.quote(config.metal)}",
            f"export TT_METAL_RUNTIME_ROOT={shlex.quote(config.metal)}",
            f"export TT_METAL_CACHE={shlex.quote(remote_cache)}",
            f"export TT_RQM_H2B_DIR={shlex.quote(remote_work)}",
            f"export TT_RQM_H2B_MANIFEST={shlex.quote(remote_work + '/manifest.json')}",
            f"export TT_RQM_H2B_BINARY={shlex.quote(config.binary)}",
            "python3 experimental/tt_metalium_hamiltonian_evolution/run_candidate.py",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "health", "environment", "run"))
    args = parser.parse_args()
    try:
        config = _config()
        return {
            "preflight": _preflight,
            "health": _health,
            "environment": _environment,
            "run": _run,
        }[args.action](config)
    except (KeyError, OSError, ValueError) as exc:
        print(f"H2B remote launcher failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

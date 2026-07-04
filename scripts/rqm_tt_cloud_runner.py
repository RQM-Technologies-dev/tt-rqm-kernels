#!/usr/bin/env python3
"""No-surprise Tenstorrent Cloud / maintainer-hardware runner scaffold."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import PurePosixPath


DEFAULT_JSON_REPORT = "reports/tt_hardware_qmul_quickstart.json"
DEFAULT_MARKDOWN_REPORT = "reports/tt_hardware_qmul_quickstart.md"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local, delegated, or explicit SSH Tenstorrent qmul runs. "
            "This script is not a cloud API client and never provisions paid "
            "resources."
        )
    )
    parser.add_argument("--check", action="store_true", help="Print configured state.")
    parser.add_argument(
        "--mode",
        choices=("local", "vscode", "ssh", "delegated"),
        default=None,
    )
    parser.add_argument("--host", default=None, help="SSH host for an existing machine.")
    parser.add_argument(
        "--remote-dir",
        default=None,
        help="Remote repo checkout directory for SSH mode.",
    )
    parser.add_argument(
        "--remote-command",
        default=None,
        help="Hardware qmul command already available on the remote machine.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run SSH commands. Omit for dry-run command printing.",
    )
    parser.add_argument(
        "--print-instructions",
        action="store_true",
        help="Print delegated maintainer-run instructions.",
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.check:
        _print_check(args)
        return 0

    if args.mode is None:
        parser.error("--mode is required unless --check is set")

    if args.mode == "local":
        return _run_local_check()
    if args.mode == "vscode":
        if not args.print_instructions:
            print(
                "vscode mode is copy/paste instruction-only; pass --print-instructions.",
                file=sys.stderr,
            )
            return 2
        _print_vscode_instructions(args)
        return 0
    if args.mode == "delegated":
        if not args.print_instructions:
            print(
                "delegated mode is instruction-only; pass --print-instructions.",
                file=sys.stderr,
            )
            return 2
        _print_delegated_instructions(args)
        return 0
    if args.mode == "ssh":
        return _run_ssh_mode(args)

    raise AssertionError(f"unsupported mode: {args.mode}")


def _print_check(args: argparse.Namespace) -> None:
    print("RQM Tenstorrent Cloud runner check")
    print("cloud API client: not implemented")
    print("paid provisioning: not implemented")
    print("credentials written to disk: never")
    print("default execution: local/free checks only")
    print("observed Console status: API inference available")
    print("observed Console status: billing/usage visible")
    print("observed Console status: compute visible")
    print("observed Console status: resources available, no allocation observed")
    print("observed Console status: instances/baremetal blocked until access")
    print("capacity request path: Compute -> Resources -> Request Capacity")
    print("")
    print(f"mode: {args.mode or 'not selected'}")
    print(f"host configured: {str(bool(args.host)).lower()}")
    print(f"remote dir configured: {str(bool(args.remote_dir)).lower()}")
    print(f"remote hardware command configured: {str(bool(args.remote_command)).lower()}")
    print(f"ssh execute requested: {str(args.execute).lower()}")
    print("")
    print("Available safe routes:")
    print("- local: CPU/PyTorch, TT-Lang simulator, tt-emule, external-qmul checks")
    print("- vscode: copy/paste run inside a granted Console VSCode/browser instance")
    print("- delegated: Tenstorrent maintainer runs hardware command and returns reports")
    print("- ssh: dry-run by default; requires --execute to run on an existing host")


def _run_local_check() -> int:
    print("Local/free validation paths:")
    print("- CPU/PyTorch StructuredBench")
    print("- TT-Lang simulator when installed")
    print("- tt-emule when locally configured")
    print("- external-qmul protocol checks")
    print("")
    return subprocess.run(
        [sys.executable, "scripts/rqm_tt_quickstart.py", "--check"],
        check=False,
    ).returncode


def _run_ssh_mode(args: argparse.Namespace) -> int:
    missing = [
        name
        for name, value in (
            ("--host", args.host),
            ("--remote-dir", args.remote_dir),
            ("--remote-command", args.remote_command),
        )
        if not value
    ]
    if missing:
        print("ssh mode requires " + ", ".join(missing), file=sys.stderr)
        return 2

    remote_command = _remote_hardware_command(args)
    ssh_command = ["ssh", args.host, remote_command]
    print("SSH hardware run command:")
    print(_shell_join(ssh_command))
    print("")
    print("This assumes the repo is already cloned and configured on the remote host.")
    print("This script does not provision cloud resources and does not store credentials.")
    print("")
    if not args.execute:
        print("Dry run only. Add --execute to run this SSH command.")
        return 0

    print("Executing SSH command because --execute was provided.")
    completed = subprocess.run(ssh_command, check=False)
    return completed.returncode


def _print_vscode_instructions(args: argparse.Namespace) -> None:
    print("Tenstorrent Console VSCode/browser instance instructions")
    print("")
    print("Use this only after Console access is granted through:")
    print("Compute -> Resources -> Request Capacity")
    print("")
    print("Open a managed Instance / VSCode browser shell, then paste:")
    print("")
    print("git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git")
    print("cd tt-rqm-kernels")
    print('python -m pip install -e ".[dev]"')
    print("python -m pytest")
    print("python -m tt_rqm_kernels.structuredbench --suite smoke --items 128 --iters 1 --warmup 0")
    print(
        "python scripts/validate_qmul_candidate.py "
        "--command \"python scripts/qmul_external_reference.py\" "
        "--items 128 --iters 1 --warmup 0"
    )
    print("")
    print("After a real Tenstorrent qmul command exists in that environment:")
    print("")
    print('export TT_RQM_HARDWARE_QMUL_COMMAND="<real Tenstorrent hardware qmul command>"')
    print(
        "python scripts/rqm_tt_quickstart.py "
        "--mode hardware "
        f"--items {args.items} --iters {args.iters} --warmup {args.warmup} --seed {args.seed} "
        f"--json-output {DEFAULT_JSON_REPORT} "
        f"--markdown-output {DEFAULT_MARKDOWN_REPORT}"
    )
    print("")
    print("Return:")
    print(f"- {DEFAULT_JSON_REPORT}")
    print(f"- {DEFAULT_MARKDOWN_REPORT}")
    print("- environment notes: Console instance label, hardware kind, SDK version, command")


def _remote_hardware_command(args: argparse.Namespace) -> str:
    remote_dir = _quote_remote_path(args.remote_dir)
    hardware_command = shlex.quote(args.remote_command)
    return (
        f"cd {remote_dir} && "
        "python scripts/rqm_tt_quickstart.py "
        "--mode hardware "
        f"--command {hardware_command} "
        f"--items {args.items} "
        f"--iters {args.iters} "
        f"--warmup {args.warmup} "
        f"--seed {args.seed} "
        f"--json-output {DEFAULT_JSON_REPORT} "
        f"--markdown-output {DEFAULT_MARKDOWN_REPORT}"
    )


def _print_delegated_instructions(args: argparse.Namespace) -> None:
    print("Delegated Tenstorrent hardware validation instructions")
    print("")
    print("RQM provides the public repo, command contract, and report format.")
    print("A Tenstorrent engineer runs the hardware command in their own environment.")
    print("No RQM cloud billing or provisioning is required.")
    print("Console path if using RQM's org: Compute -> Resources -> Request Capacity.")
    print("Execution surface after access: VSCode/browser instance or SSH baremetal.")
    print("")
    print("Copy/paste sequence:")
    print("")
    print("git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git")
    print("cd tt-rqm-kernels")
    print('python -m pip install -e ".[dev]"')
    print("python -m pytest")
    print("python -m tt_rqm_kernels.structuredbench --suite smoke --items 128 --iters 1 --warmup 0")
    print(
        "python scripts/validate_qmul_candidate.py "
        "--command \"python scripts/qmul_external_reference.py\" "
        "--items 128 --iters 1 --warmup 0"
    )
    print(
        "python scripts/rqm_tt_quickstart.py "
        "--mode hardware "
        "--command \"<real Tenstorrent hardware qmul command>\" "
        f"--items {args.items} --iters {args.iters} --warmup {args.warmup} --seed {args.seed} "
        f"--json-output {DEFAULT_JSON_REPORT} "
        f"--markdown-output {DEFAULT_MARKDOWN_REPORT}"
    )
    print("")
    print("Return:")
    print(f"- {DEFAULT_JSON_REPORT}")
    print(f"- {DEFAULT_MARKDOWN_REPORT}")
    print("- environment notes: hardware kind, host, SDK version, tt-metal commit, command")


def _quote_remote_path(value: str) -> str:
    # Treat remote-dir as a POSIX path because SSH targets are expected to be Linux.
    return shlex.quote(str(PurePosixPath(value)))


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

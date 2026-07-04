#!/usr/bin/env python3
"""Safe Tenstorrent Cloud Console helper.

This script prints browser/runbook guidance only. It does not open a browser,
ask for credentials, call Tenstorrent APIs, provision cloud resources, or write
secrets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


CONSOLE_URL = "https://console.tenstorrent.com/"
REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPO_ROOT / "scripts" / "rqm_tt_quickstart.py"
DOCS = [
    REPO_ROOT / "docs" / "tenstorrent-console-access-plan.md",
    REPO_ROOT / "docs" / "tenstorrent-console-copy-paste.md",
    REPO_ROOT / "docs" / "tenstorrent-hardware-command-contract.md",
    REPO_ROOT / "docs" / "tenstorrent-delegated-validation.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print safe Tenstorrent Cloud Console access guidance. No cloud APIs, "
            "no provisioning, and no credential storage."
        )
    )
    parser.add_argument("--check", action="store_true", help="Print local readiness.")
    parser.add_argument(
        "--print-browser-steps",
        action="store_true",
        help="Print the no-surprise-billing browser workflow.",
    )
    parser.add_argument(
        "--print-copy-paste-commands",
        action="store_true",
        help="Print commands for a Console browser shell.",
    )
    parser.add_argument(
        "--open-url-text",
        action="store_true",
        help="Print the Tenstorrent Cloud Console URL as text only.",
    )
    args = parser.parse_args()

    if args.check:
        _print_check()
    if args.print_browser_steps:
        _print_browser_steps()
    if args.print_copy_paste_commands:
        _print_copy_paste_commands()
    if args.open_url_text:
        print(CONSOLE_URL)
    if not (
        args.check
        or args.print_browser_steps
        or args.print_copy_paste_commands
        or args.open_url_text
    ):
        parser.error(
            "select at least one of --check, --print-browser-steps, "
            "--print-copy-paste-commands, or --open-url-text"
        )
    return 0


def _print_check() -> None:
    print("RQM Tenstorrent Console runner check")
    print(f"repo root: {REPO_ROOT}")
    print(f"git status: {_git_status_summary()}")
    print(f"quickstart script: {_present(QUICKSTART)}")
    for doc in DOCS:
        print(f"{doc.relative_to(REPO_ROOT)}: {_present(doc)}")
    configured = bool(os.environ.get("TT_RQM_HARDWARE_QMUL_COMMAND"))
    print(f"TT_RQM_HARDWARE_QMUL_COMMAND configured: {str(configured).lower()}")
    print("billing/provisioning: not implemented")
    print("network calls: not performed")
    print("credentials required: no")
    print("credentials written to disk: never")
    print("browser auto-open: disabled")
    print("")
    print("Observed Console workflow status:")
    print("API inference available: yes")
    print("billing/usage visible: yes")
    print("compute visible: yes")
    print("resources page available: yes")
    print("dedicated hardware allocation: none observed")
    print("instances access: blocked until access is granted")
    print("baremetal access: blocked until access is granted")
    print("capacity request path: Compute -> Resources -> Request Capacity")
    print("hardware run modes after access: VSCode/browser instance or SSH baremetal")


def _print_browser_steps() -> None:
    print("Tenstorrent Cloud Console browser workflow")
    print("")
    print("1. Open Tenstorrent Cloud Console:")
    print(f"   {CONSOLE_URL}")
    print("2. Sign in or create an account.")
    print("3. Confirm Usage and Billing are visible and credit/spend is understood.")
    print("4. Open Compute -> Resources.")
    print("5. If no allocation exists, use Request Capacity for one small [N, 4] StructuredBench qmul report.")
    print("6. After access is granted, choose one execution surface:")
    print("   - Instances: managed VSCode/browser shell copy-paste run")
    print("   - Baremetal: SSH run on an existing Tenstorrent host")
    print("7. Confirm hardware type if shown.")
    print("8. Clone tt-rqm-kernels.")
    print("9. Run CPU smoke.")
    print("10. Run python scripts/rqm_tt_quickstart.py --check.")
    print("11. Configure a real hardware qmul command only after access is explicit.")
    print("12. Export JSON/Markdown artifacts from reports/.")
    print("")
    print("No cloud resources are created by this repo.")
    print("No credentials are requested or stored by this repo.")
    print("No capacity request is submitted by this repo.")


def _print_copy_paste_commands() -> None:
    print("Tenstorrent Console copy/paste commands")
    print("")
    print("# Run these inside a granted Console VSCode/browser instance or SSH baremetal shell.")
    print("# First request access at Compute -> Resources -> Request Capacity if needed.")
    print("")
    print("# A. Clone and install")
    print("git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git")
    print("cd tt-rqm-kernels")
    print('python -m pip install -e ".[dev]"')
    print("")
    print("# B. CPU reference")
    print("python -m pytest")
    print("python -m tt_rqm_kernels.structuredbench --suite smoke")
    print("")
    print("# C. Environment check")
    print("python scripts/rqm_tt_quickstart.py --check")
    print("")
    print("# D. Hardware command placeholder")
    print('export TT_RQM_HARDWARE_QMUL_COMMAND="<TENSTORRENT_HARDWARE_QMUL_COMMAND>"')
    print("python scripts/rqm_tt_quickstart.py \\")
    print("  --mode hardware \\")
    print("  --items 128 \\")
    print("  --iters 1 \\")
    print("  --warmup 0 \\")
    print("  --json-output reports/tt_hardware_qmul_quickstart.json \\")
    print("  --markdown-output reports/tt_hardware_qmul_quickstart.md")
    print("")
    print(
        "<TENSTORRENT_HARDWARE_QMUL_COMMAND> must implement the external-qmul "
        "protocol in the Console environment."
    )


def _present(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _git_status_summary() -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unavailable"
    if not completed.stdout.strip():
        return "clean"
    return "dirty"


if __name__ == "__main__":
    raise SystemExit(main())

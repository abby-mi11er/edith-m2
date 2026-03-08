#!/usr/bin/env python3
"""
Install a launchd agent that watches vault Incoming ZIPs and auto-syncs.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install Edith Vault watch launchd agent")
    p.add_argument("--python-bin", default="", help="Interpreter path (default: .venv/bin/python)")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--settle-seconds", type=float, default=20.0)
    p.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-smoketest", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def run_launchctl(args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()
        return True, (proc.stdout or "").strip()
    except Exception as e:
        return False, str(e)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    py = Path(args.python_bin).expanduser().resolve() if args.python_bin else (root / ".venv" / "bin" / "python").resolve()
    if not py.exists():
        raise SystemExit(f"Python interpreter not found: {py}")

    label = "com.edith.vault.watch"
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    logs_dir = Path.home() / "Library" / "Logs" / "Edith"
    launch_agents.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{label}.plist"

    cmd = [
        str(py),
        str((root / "scripts" / "watch_vault_exports.py").resolve()),
        "--poll-seconds",
        str(max(2.0, float(args.poll_seconds))),
        "--settle-seconds",
        str(max(2.0, float(args.settle_seconds))),
    ]
    if args.no_smoketest:
        cmd.append("--no-smoketest")

    payload = {
        "Label": label,
        "ProgramArguments": cmd,
        "WorkingDirectory": str(root.resolve()),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str((logs_dir / f"{label}.out.log").resolve()),
        "StandardErrorPath": str((logs_dir / f"{label}.err.log").resolve()),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
        },
        "ProcessType": "Background",
    }

    with plist_path.open("wb") as f:
        plistlib.dump(payload, f, fmt=plistlib.FMT_XML)
    os.chmod(plist_path, 0o644)

    load_results = []
    if args.load:
        uid = os.getuid()
        domain = f"gui/{uid}"
        run_launchctl(["launchctl", "bootout", domain, str(plist_path)])
        ok_bootstrap, msg_bootstrap = run_launchctl(["launchctl", "bootstrap", domain, str(plist_path)])
        ok_enable, msg_enable = run_launchctl(["launchctl", "enable", f"{domain}/{label}"])
        load_results.append(
            {
                "label": label,
                "bootstrap_ok": ok_bootstrap,
                "bootstrap_detail": msg_bootstrap,
                "enable_ok": ok_enable,
                "enable_detail": msg_enable,
            }
        )

    print(
        json.dumps(
            {
                "ok": True,
                "label": label,
                "plist": str(plist_path),
                "python_bin": str(py),
                "logs_dir": str(logs_dir),
                "load_results": load_results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

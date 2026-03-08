#!/usr/bin/env python3
"""
Install launchd agent for nightly Edith maintenance.
Runs nightly_maintenance.py at 02:00 local time.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install Edith nightly maintenance launchd agent")
    p.add_argument("--python-bin", default="", help="Interpreter path (default: sys python)")
    p.add_argument("--dotenv-path", default="", help="Optional .env path")
    p.add_argument("--app-data-dir", default="", help="Optional EDITH_APP_DATA_DIR")
    p.add_argument("--hour", type=int, default=2, help="Start hour (0-23)")
    p.add_argument("--minute", type=int, default=0, help="Start minute (0-59)")
    p.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
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
    py = Path(args.python_bin).expanduser().resolve() if args.python_bin else Path(os.sys.executable).resolve()
    if not py.exists():
        raise SystemExit(f"Python interpreter not found: {py}")
    task = (root / "scripts" / "nightly_maintenance.py").resolve()
    if not task.exists():
        raise SystemExit(f"nightly_maintenance.py not found: {task}")

    dotenv_path = ""
    if args.dotenv_path:
        dp = Path(args.dotenv_path).expanduser().resolve()
        if not dp.exists():
            raise SystemExit(f"--dotenv-path not found: {dp}")
        dotenv_path = str(dp)

    app_data_dir = ""
    if args.app_data_dir:
        app_data_dir = str(Path(args.app_data_dir).expanduser().resolve())

    hour = min(23, max(0, int(args.hour)))
    minute = min(59, max(0, int(args.minute)))

    label = "com.edith.nightly.maintenance"
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    logs_dir = Path.home() / "Library" / "Logs" / "Edith"
    launch_agents.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{label}.plist"

    env = {
        "PYTHONUNBUFFERED": "1",
    }
    if dotenv_path:
        env["EDITH_DOTENV_PATH"] = dotenv_path
    if app_data_dir:
        env["EDITH_APP_DATA_DIR"] = app_data_dir

    payload = {
        "Label": label,
        "ProgramArguments": [str(py), str(task)],
        "WorkingDirectory": str(root.resolve()),
        "RunAtLoad": False,
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "StandardOutPath": str((logs_dir / f"{label}.out.log").resolve()),
        "StandardErrorPath": str((logs_dir / f"{label}.err.log").resolve()),
        "EnvironmentVariables": env,
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
                "task": str(task),
                "dotenv_path": dotenv_path,
                "app_data_dir": app_data_dir,
                "start": {"hour": hour, "minute": minute},
                "logs_dir": str(logs_dir),
                "load_results": load_results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

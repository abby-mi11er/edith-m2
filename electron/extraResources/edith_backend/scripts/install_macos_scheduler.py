#!/usr/bin/env python3
"""
Install launchd jobs for unattended Edith practice/eval loops on macOS.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Edith launchd automation jobs")
    parser.add_argument("--python-bin", default="", help="Python interpreter path (default: .venv/bin/python)")
    parser.add_argument("--mode", default="Files only", choices=["Files only", "Web only", "Files + Web"])
    parser.add_argument("--backend", default="chroma", choices=["google", "chroma"])
    parser.add_argument("--nightly-hour", type=int, default=2)
    parser.add_argument("--nightly-minute", type=int, default=15)
    parser.add_argument("--weekly-weekday", type=int, default=0, help="0=Sunday ... 6=Saturday")
    parser.add_argument("--weekly-hour", type=int, default=4)
    parser.add_argument("--weekly-minute", type=int, default=30)
    parser.add_argument("--weekly-fine-tune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_launchctl(args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()
        return True, (proc.stdout or "").strip()
    except Exception as e:
        return False, str(e)


def build_job(label: str, python_bin: str, root: Path, mode: str, backend: str, log_dir: Path, schedule: dict, weekly_fine_tune: bool) -> dict:
    cmd = [
        python_bin,
        str((root / "scripts" / "run_practice_loop.py").resolve()),
        "--mode",
        mode,
        "--backend",
        backend,
        "--generate-cases",
        "--export-sft",
    ]
    if weekly_fine_tune:
        cmd.append("--fine-tune")

    return {
        "Label": label,
        "ProgramArguments": cmd,
        "WorkingDirectory": str(root.resolve()),
        "RunAtLoad": False,
        "StartCalendarInterval": schedule,
        "StandardOutPath": str((log_dir / f"{label}.out.log").resolve()),
        "StandardErrorPath": str((log_dir / f"{label}.err.log").resolve()),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
        },
        "ProcessType": "Background",
    }


def main() -> int:
    if os.name != "posix":
        raise SystemExit("This installer is for macOS launchd.")

    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    python_bin = (args.python_bin or str(root / ".venv" / "bin" / "python")).strip()
    python_path = Path(python_bin).expanduser().resolve()
    if not python_path.exists():
        raise SystemExit(f"Python interpreter not found: {python_path}")

    launch_agents = Path.home() / "Library" / "LaunchAgents"
    log_dir = Path.home() / "Library" / "Logs" / "Edith"
    launch_agents.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    nightly_label = "com.edith.practice.nightly"
    weekly_label = "com.edith.practice.weekly"

    nightly_job = build_job(
        label=nightly_label,
        python_bin=str(python_path),
        root=root,
        mode=args.mode,
        backend=args.backend,
        log_dir=log_dir,
        schedule={"Hour": int(args.nightly_hour), "Minute": int(args.nightly_minute)},
        weekly_fine_tune=False,
    )

    weekly_schedule = {"Weekday": int(args.weekly_weekday), "Hour": int(args.weekly_hour), "Minute": int(args.weekly_minute)}
    weekly_job = build_job(
        label=weekly_label,
        python_bin=str(python_path),
        root=root,
        mode=args.mode,
        backend=args.backend,
        log_dir=log_dir,
        schedule=weekly_schedule,
        weekly_fine_tune=bool(args.weekly_fine_tune),
    )

    nightly_plist = launch_agents / f"{nightly_label}.plist"
    weekly_plist = launch_agents / f"{weekly_label}.plist"

    with nightly_plist.open("wb") as f:
        plistlib.dump(nightly_job, f, fmt=plistlib.FMT_XML)
    with weekly_plist.open("wb") as f:
        plistlib.dump(weekly_job, f, fmt=plistlib.FMT_XML)

    os.chmod(nightly_plist, 0o644)
    os.chmod(weekly_plist, 0o644)

    load_results = []
    if args.load:
        uid = os.getuid()
        domain = f"gui/{uid}"
        for label, plist in ((nightly_label, nightly_plist), (weekly_label, weekly_plist)):
            run_launchctl(["launchctl", "bootout", domain, str(plist)])
            ok_bootstrap, msg_bootstrap = run_launchctl(["launchctl", "bootstrap", domain, str(plist)])
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

    result = {
        "ok": True,
        "launch_agents_dir": str(launch_agents),
        "nightly_plist": str(nightly_plist),
        "weekly_plist": str(weekly_plist),
        "python_bin": str(python_path),
        "mode": args.mode,
        "backend": args.backend,
        "load_results": load_results,
        "logs_dir": str(log_dir),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

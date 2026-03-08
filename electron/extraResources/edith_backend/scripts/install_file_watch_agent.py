#!/usr/bin/env python3
"""
Install a launchd agent that watches EDITH_DATA_ROOT and auto-indexes file changes.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install Edith local file watch launchd agent")
    p.add_argument("--python-bin", default="", help="Interpreter path (default: .venv/bin/python)")
    p.add_argument("--data-root", default="", help="Folder to watch (default: EDITH_DATA_ROOT)")
    p.add_argument("--backend", default="google", choices=["google", "chroma"])
    p.add_argument("--dotenv-path", default="", help="Optional .env path for GOOGLE_API_KEY/EDITH_STORE_*")
    p.add_argument("--app-data-dir", default="", help="Optional EDITH_APP_DATA_DIR path")
    p.add_argument("--debounce", type=float, default=2.0, help="EDITH_WATCH_DEBOUNCE seconds")
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
    py = Path(args.python_bin).expanduser().resolve() if args.python_bin else (root / ".venv" / "bin" / "python").resolve()
    if not py.exists():
        raise SystemExit(f"Python interpreter not found: {py}")

    watch_script = (root / "watch_files.py").resolve()
    if not watch_script.exists():
        raise SystemExit(f"watch_files.py not found: {watch_script}")

    data_root_raw = (args.data_root or os.getenv("EDITH_DATA_ROOT") or "").strip()
    if not data_root_raw:
        raise SystemExit("Missing --data-root (or EDITH_DATA_ROOT).")
    data_root = Path(data_root_raw).expanduser().resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    dotenv_path = ""
    if args.dotenv_path:
        dp = Path(args.dotenv_path).expanduser().resolve()
        if not dp.exists():
            raise SystemExit(f"--dotenv-path not found: {dp}")
        dotenv_path = str(dp)

    app_data_dir = ""
    if args.app_data_dir:
        app_data_dir = str(Path(args.app_data_dir).expanduser().resolve())

    label = "com.edith.files.watch"
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    logs_dir = Path.home() / "Library" / "Logs" / "Edith"
    launch_agents.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{label}.plist"

    env = {
        "PYTHONUNBUFFERED": "1",
        "EDITH_DATA_ROOT": str(data_root),
        "EDITH_RETRIEVAL_BACKEND": args.backend,
        "EDITH_WATCH_DEBOUNCE": str(max(0.5, float(args.debounce))),
    }
    if dotenv_path:
        env["EDITH_DOTENV_PATH"] = dotenv_path
    if app_data_dir:
        env["EDITH_APP_DATA_DIR"] = app_data_dir

    payload = {
        "Label": label,
        "ProgramArguments": [str(py), str(watch_script)],
        "WorkingDirectory": str(root.resolve()),
        "RunAtLoad": True,
        "KeepAlive": True,
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
                "watch_script": str(watch_script),
                "data_root": str(data_root),
                "backend": args.backend,
                "dotenv_path": dotenv_path,
                "app_data_dir": app_data_dir,
                "logs_dir": str(logs_dir),
                "load_results": load_results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


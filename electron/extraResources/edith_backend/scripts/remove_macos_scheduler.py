#!/usr/bin/env python3
"""
Remove Edith launchd automation jobs from macOS user LaunchAgents.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove Edith launchd automation jobs")
    parser.add_argument("--delete-plists", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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
    uid = os.getuid()
    domain = f"gui/{uid}"
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    labels = ["com.edith.practice.nightly", "com.edith.practice.weekly"]

    out = {"ok": True, "jobs": []}
    for label in labels:
        plist = launch_agents / f"{label}.plist"
        ok_bootout, msg_bootout = run_launchctl(["launchctl", "bootout", domain, str(plist)])
        removed = False
        if args.delete_plists and plist.exists():
            try:
                plist.unlink()
                removed = True
            except Exception:
                removed = False
        out["jobs"].append(
            {
                "label": label,
                "plist": str(plist),
                "bootout_ok": ok_bootout,
                "bootout_detail": msg_bootout,
                "plist_removed": removed,
            }
        )

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

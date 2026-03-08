#!/usr/bin/env python3
"""
Remove launchd agent for Edith nightly maintenance.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def run_cmd(args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()
        return True, (proc.stdout or "").strip()
    except Exception as e:
        return False, str(e)


def main() -> int:
    label = "com.edith.nightly.maintenance"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    uid = os.getuid()
    domain = f"gui/{uid}"

    ok_bootout, msg_bootout = run_cmd(["launchctl", "bootout", domain, str(plist)])
    ok_disable, msg_disable = run_cmd(["launchctl", "disable", f"{domain}/{label}"])
    removed = False
    if plist.exists():
        try:
            plist.unlink()
            removed = True
        except Exception:
            removed = False

    print(
        json.dumps(
            {
                "ok": True,
                "label": label,
                "plist": str(plist),
                "bootout_ok": ok_bootout,
                "bootout_detail": msg_bootout,
                "disable_ok": ok_disable,
                "disable_detail": msg_disable,
                "plist_removed": removed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Remove the launchd agent that watches EDITH_DATA_ROOT and auto-indexes file changes.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def run_launchctl(args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()
        return True, (proc.stdout or "").strip()
    except Exception as e:
        return False, str(e)


def main() -> int:
    label = "com.edith.files.watch"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    uid = os.getuid()
    domain = f"gui/{uid}"

    results = []
    ok_bootout, msg_bootout = run_launchctl(["launchctl", "bootout", domain, str(plist)])
    results.append({"step": "bootout", "ok": ok_bootout, "detail": msg_bootout})
    ok_disable, msg_disable = run_launchctl(["launchctl", "disable", f"{domain}/{label}"])
    results.append({"step": "disable", "ok": ok_disable, "detail": msg_disable})

    removed = False
    if plist.exists():
        plist.unlink()
        removed = True

    print(
        json.dumps(
            {
                "ok": True,
                "label": label,
                "plist_removed": removed,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


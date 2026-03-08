#!/usr/bin/env python3
"""
Remove the Edith vault watch launchd agent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove Edith Vault watch launchd agent")
    p.add_argument("--delete-plist", action=argparse.BooleanOptionalAction, default=True)
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
    label = "com.edith.vault.watch"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    uid = os.getuid()
    domain = f"gui/{uid}"

    results = []
    ok_disable, msg_disable = run_launchctl(["launchctl", "disable", f"{domain}/{label}"])
    results.append({"step": "disable", "ok": ok_disable, "detail": msg_disable})
    ok_bootout, msg_bootout = run_launchctl(["launchctl", "bootout", domain, str(plist)])
    results.append({"step": "bootout", "ok": ok_bootout, "detail": msg_bootout})

    deleted = False
    if args.delete_plist and plist.exists():
        try:
            plist.unlink()
            deleted = True
        except Exception:
            deleted = False

    print(
        json.dumps(
            {
                "ok": True,
                "label": label,
                "plist": str(plist),
                "plist_deleted": deleted,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

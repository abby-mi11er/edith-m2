#!/usr/bin/env python3
"""
Watch EDITH_VAULT_EXPORT_DIR/Incoming for new ZIPs and trigger sync.
Uses polling to avoid extra runtime dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def load_env():
    if not load_dotenv:
        return
    root = Path(__file__).resolve().parent.parent
    candidates = []
    override = os.environ.get("EDITH_DOTENV_PATH")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            root / ".env",
            Path.cwd() / ".env",
            Path.home() / "Library" / "Application Support" / "Edith" / ".env",
        ]
    )
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def parse_args():
    p = argparse.ArgumentParser(description="Watch vault export inbox and auto-run sync")
    p.add_argument("--poll-seconds", type=float, default=float(os.getenv("EDITH_VAULT_WATCH_POLL_SECONDS", "15")))
    p.add_argument("--settle-seconds", type=float, default=float(os.getenv("EDITH_VAULT_WATCH_SETTLE_SECONDS", "20")))
    p.add_argument("--once", action="store_true", help="Run one scan/sync cycle and exit")
    p.add_argument("--no-smoketest", action="store_true", help="Pass through to sync runner")
    return p.parse_args()


def inbox_dir(vault_root: Path):
    incoming = vault_root / "Incoming"
    return incoming if incoming.exists() else vault_root


def zip_signature(root: Path):
    rows = []
    if not root.exists():
        return rows
    for p in root.rglob("*.zip"):
        if not p.is_file():
            continue
        if any(part in {"Archived", "Failed", "Quarantine"} for part in p.parts):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rows.append((str(p.relative_to(root)), int(st.st_size), int(st.st_mtime)))
    rows.sort()
    return rows


def all_settled(root: Path, settle_seconds: float):
    now = time.time()
    for p in root.rglob("*.zip"):
        if not p.is_file():
            continue
        if any(part in {"Archived", "Failed", "Quarantine"} for part in p.parts):
            continue
        try:
            if (now - p.stat().st_mtime) < settle_seconds:
                return False
        except OSError:
            return False
    return True


def run_sync(project_root: Path, no_smoketest: bool):
    cmd = [sys.executable, str(project_root / "scripts" / "sync_vault_exports.py")]
    if no_smoketest:
        cmd.append("--no-smoketest")
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    payload = {}
    if out:
        try:
            payload = json.loads(out)
        except Exception:
            payload = {}
    print(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "code": int(proc.returncode),
                "ok": proc.returncode in (0, 2),
                "summary": payload,
                "stderr": err[:300],
            },
            ensure_ascii=False,
        )
    )
    return proc.returncode


def main() -> int:
    load_env()
    args = parse_args()

    raw = (os.getenv("EDITH_VAULT_EXPORT_DIR") or "").strip()
    if not raw:
        raise SystemExit("EDITH_VAULT_EXPORT_DIR is required.")
    vault_root = Path(raw).expanduser().resolve()
    if not vault_root.exists():
        raise SystemExit(f"Vault export directory not found: {vault_root}")
    watch_root = inbox_dir(vault_root)
    project_root = Path(__file__).resolve().parent.parent

    last_sig = None
    while True:
        sig = zip_signature(watch_root)
        changed = sig != last_sig
        should_run = changed and len(sig) > 0 and all_settled(watch_root, float(max(2.0, args.settle_seconds)))
        if should_run:
            run_sync(project_root, no_smoketest=bool(args.no_smoketest))
            last_sig = zip_signature(watch_root)
        elif changed:
            last_sig = sig

        if args.once:
            break
        time.sleep(max(2.0, float(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

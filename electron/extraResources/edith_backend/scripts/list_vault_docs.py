#!/usr/bin/env python3
"""
List documents currently stored in an Edith Google File Search store.

Usage:
  python scripts/list_vault_docs.py
  python scripts/list_vault_docs.py --limit 200
  python scripts/list_vault_docs.py --contains "policy"
  python scripts/list_vault_docs.py --store fileSearchStores/your_store_id
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai


def load_env() -> None:
    app_home = Path.home() / "Library" / "Application Support" / "Edith"
    candidates = []
    override = os.environ.get("EDITH_DOTENV_PATH")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent / ".env",
            Path.cwd() / ".env",
            app_home / ".env",
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


def normalize_store_id(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    if s.startswith("fileSearchStores/"):
        return s
    return f"fileSearchStores/{s}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List docs in Edith Google file search store")
    p.add_argument("--store", default="", help="Store id (fileSearchStores/...).")
    p.add_argument("--api-key", default="", help="Google API key. If omitted, env is used, then secure prompt.")
    p.add_argument("--no-prompt", action="store_true", help="Do not prompt for missing API key.")
    p.add_argument("--limit", type=int, default=100, help="Max rows to print.")
    p.add_argument("--contains", default="", help="Filter rows by case-insensitive text in display/name.")
    return p.parse_args()


def store_from_config() -> str:
    candidates = [
        Path.home() / "Library" / "Application Support" / "edith-desktop-shell" / "config.json",
        Path.home() / "Library" / "Application Support" / "Edith" / "config.json",
    ]
    for cfg in candidates:
        if not cfg.exists():
            continue
        try:
            payload = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw = (
            payload.get("store_id")
            or payload.get("storeId")
            or payload.get("vault_id")
            or payload.get("vaultId")
            or ""
        )
        store = normalize_store_id(str(raw))
        if store:
            return store
    return ""


def main() -> int:
    load_env()
    args = parse_args()

    api_key = (args.api_key or os.environ.get("GOOGLE_API_KEY") or "").strip()
    store = normalize_store_id(
        args.store
        or os.environ.get("EDITH_STORE_ID")
        or os.environ.get("EDITH_STORE_MAIN")
        or os.environ.get("EDITH_VAULT_ID")
        or store_from_config()
        or ""
    )
    limit = max(1, int(args.limit or 100))
    needle = (args.contains or "").strip().lower()

    if not api_key and not args.no_prompt:
        try:
            api_key = getpass.getpass("GOOGLE_API_KEY (hidden): ").strip()
        except Exception:
            api_key = ""
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY (set env, pass --api-key, or allow prompt).")
    if not store:
        raise SystemExit("Missing store id. Set EDITH_STORE_ID/EDITH_STORE_MAIN/EDITH_VAULT_ID or pass --store.")

    client = genai.Client(api_key=api_key)

    rows = []
    try:
        for doc in client.file_search_stores.documents.list(parent=store):
            display = str(getattr(doc, "display_name", "") or "")
            name = str(getattr(doc, "name", "") or "")
            if needle:
                hay = f"{display} {name}".lower()
                if needle not in hay:
                    continue
            rows.append((display, name))
            if len(rows) >= limit:
                break
    except httpx.ConnectError:
        print("ERROR: Unable to reach Google API endpoint (network/DNS unavailable).")
        return 2
    except Exception as exc:
        msg = str(exc)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            print("ERROR: API key invalid. Update GOOGLE_API_KEY and retry.")
            return 2
        if "PERMISSION_DENIED" in msg:
            print("ERROR: Permission denied for this store id. Verify store ownership/access.")
            return 2
        print(f"ERROR: Vault list failed: {msg}")
        return 2

    print(f"Store: {store}")
    print(f"Shown: {len(rows)}")
    if not rows:
        return 0

    for idx, (display, name) in enumerate(rows, start=1):
        label = display or name
        print(f"{idx:>4}. {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

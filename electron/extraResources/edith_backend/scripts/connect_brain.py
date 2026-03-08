#!/usr/bin/env python3
"""
Resolve and persist the active Edith File Search store id.

Behavior:
1) Try explicit target id (arg or env).
2) Scan existing stores by display-name keywords.
3) Optionally create a store if nothing is found.

Writes EDITH_VAULT_ID, EDITH_STORE_MAIN, and EDITH_STORE_ID into .env.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google import genai


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve and save Edith store id")
    p.add_argument("--env", default="", help="Optional .env path to read/write")
    p.add_argument("--target-id", default="", help="Preferred store id (fileSearchStores/...)")
    p.add_argument("--keywords", default="edith,academic", help="Comma-separated fallback keywords")
    p.add_argument("--create-display", default="edith-main-brain", help="Display name for auto-create")
    p.add_argument("--no-create", action="store_true", help="Do not create a store if none found")
    p.add_argument("--dry-run", action="store_true", help="Resolve only; do not modify .env")
    return p.parse_args()


def normalize_store_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if not raw.startswith("fileSearchStores/"):
        raw = f"fileSearchStores/{raw}"
    return raw


def discover_env_path(arg_env: str) -> Path:
    if arg_env:
        return Path(arg_env).expanduser().resolve()

    project_root = Path(__file__).resolve().parent.parent
    project_env = project_root / ".env"
    if project_env.exists():
        return project_env

    app_env = Path.home() / "Library" / "Application Support" / "Edith" / ".env"
    if app_env.exists():
        return app_env

    return project_env


def load_environment(env_path: Path) -> None:
    candidates = [
        env_path,
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            try:
                load_dotenv(dotenv_path=c, override=False)
            except Exception:
                continue


def choose_backup_store(stores: Iterable, keywords: list[str]):
    keywords = [k for k in keywords if k]
    best = None
    best_score = -1
    for store in stores:
        name = (getattr(store, "display_name", "") or "").strip()
        lname = name.lower()
        score = 0
        for kw in keywords:
            if kw in lname:
                score += 1
        if score > best_score and score > 0:
            best = store
            best_score = score
    return best


def resolve_store(client: genai.Client, target_id: str, keywords: list[str], create_display: str, allow_create: bool):
    candidate = normalize_store_id(target_id)

    if candidate:
        try:
            store = client.file_search_stores.get(name=candidate)
            return store.name, f"Using target id ({store.display_name})"
        except Exception as exc:  # noqa: BLE001
            print(f"Target id not available: {exc}")

    stores = list(client.file_search_stores.list())
    backup = choose_backup_store(stores, keywords)
    if backup is not None:
        return backup.name, f"Using backup match ({backup.display_name})"

    if allow_create:
        created = client.file_search_stores.create(config={"display_name": create_display})
        return created.name, f"Created new store ({created.display_name})"

    return "", "No matching store found and creation disabled."


def upsert_env_values(path: Path, updates: dict[str, str]) -> None:
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    keys = set(updates.keys())
    kept: list[str] = []
    for line in existing_lines:
        striped = line.strip()
        if not striped or striped.startswith("#") or "=" not in striped:
            kept.append(line)
            continue
        key = striped.split("=", 1)[0].strip()
        if key in keys:
            continue
        kept.append(line)

    if kept and not kept[-1].endswith("\n"):
        kept[-1] = kept[-1] + "\n"

    for k, v in updates.items():
        kept.append(f"{k}={v}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(kept), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def main() -> int:
    args = parse_args()
    env_path = discover_env_path(args.env)
    load_environment(env_path)

    api_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        print(f"FAIL: GOOGLE_API_KEY missing. Expected in: {env_path}")
        return 2

    target_id = (
        args.target_id
        or os.environ.get("EDITH_VAULT_ID")
        or os.environ.get("EDITH_STORE_ID")
        or os.environ.get("EDITH_STORE_MAIN")
        or ""
    )
    keywords = [x.strip().lower() for x in (args.keywords or "").split(",") if x.strip()]
    if not keywords:
        keywords = ["edith", "academic"]

    client = genai.Client(api_key=api_key)
    store_id, reason = resolve_store(
        client=client,
        target_id=target_id,
        keywords=keywords,
        create_display=args.create_display.strip() or "edith-main-brain",
        allow_create=not args.no_create,
    )
    if not store_id:
        print(f"FAIL: {reason}")
        return 1

    print(f"OK: {reason}")
    print(f"Resolved store id: {store_id}")

    updates = {
        "EDITH_VAULT_ID": store_id,
        "EDITH_STORE_MAIN": store_id,
        "EDITH_STORE_ID": store_id,
    }

    if args.dry_run:
        print("Dry run: no .env changes written.")
        for k, v in updates.items():
            print(f"{k}={v}")
        return 0

    upsert_env_values(env_path, updates)
    print(f"Wrote updates to: {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

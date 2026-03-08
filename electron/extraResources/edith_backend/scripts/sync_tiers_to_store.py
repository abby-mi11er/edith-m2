#!/usr/bin/env python3
"""
Sync tiered local folders (canon/inbox/projects) to a Google File Search store.

Key behavior:
- Dedupe by file content hash (sqlite manifest)
- Parallel uploads with retries
- Safe metadata tagging (tier/project/rel_path/sha256)
- Supports dry-run and explicit tier selection
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google import genai

IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
DEFAULT_TIERS = ("canon", "inbox", "projects")
DEFAULT_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt", ".epub", ".tex",
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".xml", ".dta", ".do", ".sav", ".gph", ".sas7bdat",
    ".py", ".ipynb", ".js", ".html", ".css", ".sql", ".sh", ".r", ".cpp", ".c", ".java",
    ".jpg", ".jpeg", ".png", ".tiff", ".mp3", ".mp4", ".wav", ".m4a",
}

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class UploadTask:
    path: Path
    file_hash: str
    tier: str
    rel_path: str
    project: str


@dataclass(frozen=True)
class UploadResult:
    task: UploadTask
    status: str  # success | exists | fail
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync tier folders into a File Search store")
    parser.add_argument("--root", default="", help="Root folder containing canon/inbox/projects")
    parser.add_argument("--env", default="", help="Optional .env path")
    parser.add_argument("--store-id", default="", help="File Search store id override")
    parser.add_argument("--db", default="edith_memory.sqlite3", help="Manifest sqlite path")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent uploads")
    parser.add_argument("--max-file-mb", type=int, default=200, help="Skip files larger than this")
    parser.add_argument("--tiers", default=",".join(DEFAULT_TIERS), help="Comma-separated tiers")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without uploading")
    parser.add_argument("--retry", type=int, default=4, help="Retries per file on transient errors")
    return parser.parse_args()


def load_environment(env_path: str) -> None:
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    else:
        candidates.append(Path(__file__).resolve().parent.parent / ".env")
    candidates.append(Path.cwd() / ".env")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            try:
                load_dotenv(dotenv_path=candidate, override=False)
            except Exception:
                continue


def normalize_store_id(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if not value.startswith("fileSearchStores/"):
        value = f"fileSearchStores/{value}"
    return value


def normalize_tiers(raw: str) -> list[str]:
    out: list[str] = []
    for item in (raw or "").split(","):
        tier = item.strip().strip("/").strip("\\")
        if not tier:
            continue
        if "/" in tier or "\\" in tier or tier in {".", ".."}:
            continue
        out.append(tier)
    return out


def compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            file_hash TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            tier TEXT NOT NULL,
            project TEXT,
            updated_at INTEGER NOT NULL
        )
        """
    )
    con.commit()
    return con


def known_hashes(con: sqlite3.Connection) -> set[str]:
    cur = con.cursor()
    cur.execute("SELECT file_hash FROM memory")
    return {row[0] for row in cur.fetchall()}


def parse_project(rel_path: Path) -> str:
    parts = rel_path.parts
    if not parts:
        return ""
    if parts[0] != "projects" or len(parts) < 2:
        return ""
    return parts[1]


def iter_tasks(root: Path, tiers: list[str], valid_exts: set[str], max_bytes: int) -> Iterable[UploadTask]:
    for tier in tiers:
        tier_root = root / tier
        if not tier_root.exists() or not tier_root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(tier_root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for filename in filenames:
                path = Path(dirpath) / filename
                if not path.is_file():
                    continue
                if path.suffix.lower() not in valid_exts:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size <= 0 or size > max_bytes:
                    continue

                rel = path.relative_to(root)
                rel_str = rel.as_posix()
                project = parse_project(rel)
                try:
                    digest = compute_hash(path)
                except OSError:
                    continue
                yield UploadTask(path=path, file_hash=digest, tier=tier, rel_path=rel_str, project=project)


def is_transient_error(msg: str) -> bool:
    low = (msg or "").lower()
    tokens = ("429", "500", "503", "timeout", "deadline", "temporar", "connection reset", "unavailable")
    return any(t in low for t in tokens)


def get_client(api_key: str) -> genai.Client:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = genai.Client(api_key=api_key)
        _THREAD_LOCAL.client = client
    return client


def check_store_access(api_key: str, store_id: str) -> tuple[bool, str]:
    try:
        probe = genai.Client(api_key=api_key)
        probe.file_search_stores.get(name=store_id)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def wait_operation(client: genai.Client, op, poll_s: float = 2.0) -> None:
    current = op
    if not hasattr(current, "done"):
        return
    while not getattr(current, "done", False):
        time.sleep(poll_s)
        current = client.operations.get(current)
    err = getattr(current, "error", None)
    if err:
        raise RuntimeError(str(err))


def upload_to_store(client: genai.Client, store_id: str, task: UploadTask, metadata: dict):
    custom_metadata = []
    for key, value in metadata.items():
        text = str(value or "").strip()
        if not text:
            continue
        custom_metadata.append({"key": str(key), "string_value": text})

    cfg = {"display_name": task.path.name}
    if custom_metadata:
        cfg["custom_metadata"] = custom_metadata

    # Prefer the modern method/parameter supported by current SDKs.
    try:
        return client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_id,
            file=str(task.path),
            config=cfg,
        )
    except TypeError as exc:
        # Backward compatibility for older SDK variants.
        msg = str(exc).lower()
        if "unexpected keyword argument" not in msg:
            raise

    for key in ("local_file_path", "file_path"):
        try:
            return client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_id,
                config=cfg,
                **{key: str(task.path)},
            )
        except TypeError as exc:
            msg = str(exc).lower()
            if "unexpected keyword argument" in msg:
                continue
            raise

    # As a last resort, try import_file using file_name semantics.
    return client.file_search_stores.import_file(
        file_search_store_name=store_id,
        file_name=str(task.path),
        config=cfg,
    )


def upload_file_with_retry(task: UploadTask, api_key: str, store_id: str, retries: int) -> UploadResult:
    client = get_client(api_key)
    backoff = 2

    metadata = {
        "tier": task.tier,
        "project": task.project,
        "rel_path": task.rel_path,
        "sha256": task.file_hash,
    }

    for attempt in range(1, retries + 1):
        try:
            op = upload_to_store(client, store_id, task, metadata)
            wait_operation(client, op)
            return UploadResult(task=task, status="success")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            low = msg.lower()
            if "already exists" in low or "duplicate" in low:
                return UploadResult(task=task, status="exists", detail=msg)
            if attempt < retries and is_transient_error(msg):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            return UploadResult(task=task, status="fail", detail=msg)

    return UploadResult(task=task, status="fail", detail="exhausted retries")


def persist_success(con: sqlite3.Connection, result: UploadResult) -> None:
    now = int(time.time())
    con.execute(
        """
        INSERT OR REPLACE INTO memory(file_hash, path, tier, project, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (result.task.file_hash, str(result.task.path), result.task.tier, result.task.project, now),
    )


def main() -> int:
    args = parse_args()
    load_environment(args.env)

    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent.parent
    if not root.exists() or not root.is_dir():
        print(f"FAIL: root not found: {root}")
        return 2

    tiers = normalize_tiers(args.tiers)
    if not tiers:
        tiers = list(DEFAULT_TIERS)

    db_path = Path(args.db).expanduser().resolve()
    max_bytes = max(1, int(args.max_file_mb)) * 1024 * 1024
    workers = max(1, min(32, int(args.workers)))
    retries = max(1, min(10, int(args.retry)))

    con = init_db(db_path)
    known = known_hashes(con)

    pending: list[UploadTask] = []
    skipped_hash = 0
    for task in iter_tasks(root, tiers, DEFAULT_EXTENSIONS, max_bytes):
        if task.file_hash in known:
            skipped_hash += 1
            continue
        pending.append(task)

    print("Store:  (not required for dry-run)" if args.dry_run else "Store:  resolving from env/args")
    print(f"Root:  {root}")
    print(f"Tiers: {', '.join(tiers)}")
    print(f"Found new files: {len(pending)} | skipped by hash: {skipped_hash}")

    if args.dry_run:
        for task in pending[:20]:
            print(f"DRY-RUN {task.tier:8s} {task.rel_path}")
        if len(pending) > 20:
            print(f"... and {len(pending) - 20} more")
        con.close()
        return 0

    api_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    store_id = normalize_store_id(
        args.store_id
        or os.environ.get("EDITH_STORE_ID")
        or os.environ.get("EDITH_STORE_MAIN")
        or os.environ.get("EDITH_VAULT_ID")
    )
    if not api_key:
        con.close()
        print("FAIL: GOOGLE_API_KEY missing")
        return 2
    if not store_id:
        con.close()
        print("FAIL: store id missing (use --store-id or set EDITH_STORE_ID/EDITH_STORE_MAIN/EDITH_VAULT_ID)")
        return 2

    ok_store, store_msg = check_store_access(api_key, store_id)
    if not ok_store:
        con.close()
        print(f"FAIL: store check failed: {store_msg}")
        return 2

    print(f"Store: {store_id}")

    if not pending:
        con.close()
        print("No new files to upload.")
        return 0

    stats = {"success": 0, "exists": 0, "fail": 0}
    failures: list[UploadResult] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload_file_with_retry, task, api_key, store_id, retries) for task in pending]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                stats["fail"] += 1
                failures.append(
                    UploadResult(
                        task=UploadTask(path=Path("<unknown>"), file_hash="", tier="", rel_path="<unknown>", project=""),
                        status="fail",
                        detail=str(exc),
                    )
                )
                continue

            stats[result.status] = stats.get(result.status, 0) + 1
            if result.status in {"success", "exists"}:
                persist_success(con, result)
            if result.status == "fail":
                failures.append(result)

    con.commit()
    con.close()

    print(
        "Done: "
        f"added={stats['success']} "
        f"already_exists={stats['exists']} "
        f"failed={stats['fail']}"
    )

    if failures:
        print("Sample failures:")
        for row in failures[:10]:
            print(f"- {row.task.rel_path}: {row.detail[:300]}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

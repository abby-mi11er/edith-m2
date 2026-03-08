#!/usr/bin/env python3
"""
Create a reproducible corpus snapshot for Edith.

Snapshot includes:
- file manifest (rel path + sha256 + size + mtime)
- corpus hash
- optional Chroma directory hash
- selected config/model settings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Edith corpus snapshot")
    p.add_argument("--docs-root", default="", help="Override EDITH_DATA_ROOT")
    p.add_argument("--app-data-dir", default="", help="Override EDITH_APP_DATA_DIR")
    p.add_argument("--out", default="", help="Optional explicit snapshot output path")
    p.add_argument("--include-chroma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-files", type=int, default=20000)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env():
    if not load_dotenv:
        return
    root = project_root()
    candidates = [
        root / ".env",
        Path.home() / "Library" / "Application Support" / "Edith" / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def app_data_dir(override: str) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    raw = (os.getenv("EDITH_APP_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "Library" / "Application Support" / "Edith").resolve()


def docs_root(override: str) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    raw = (os.getenv("EDITH_DATA_ROOT") or "").strip()
    # Keep snapshot behavior explicit: if EDITH_DATA_ROOT is unset, treat docs as missing
    # instead of silently hashing the current working directory.
    return Path(raw).expanduser().resolve() if raw else (Path.home() / ".edith_missing_data_root")


def chroma_dir(app_data: Path) -> Path:
    raw = (os.getenv("EDITH_CHROMA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (app_data / "chroma").resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path):
    if not root.exists():
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            yield Path(base) / fn


def manifest_for_root(root: Path, max_files: int):
    rows = []
    if not root.exists():
        return rows
    count = 0
    for p in iter_files(root):
        try:
            st = p.stat()
            rel = str(p.relative_to(root))
            rows.append(
                {
                    "rel_path": rel,
                    "sha256": sha256_file(p),
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                }
            )
            count += 1
            if count >= max_files:
                break
        except Exception:
            continue
    rows.sort(key=lambda x: x["rel_path"])
    return rows


def hash_manifest(rows):
    h = hashlib.sha256()
    for row in rows:
        payload = f"{row.get('rel_path','')}|{row.get('sha256','')}|{row.get('size',0)}|{row.get('mtime',0)}\n"
        h.update(payload.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def selected_env_config():
    keys = [
        "EDITH_MODEL",
        "EDITH_MODEL_PROFILE",
        "EDITH_MODEL_FALLBACKS",
        "EDITH_RETRIEVAL_BACKEND",
        "EDITH_SOURCE_MODE",
        "EDITH_HYBRID_FILE_POLICY",
        "EDITH_REQUIRE_CITATIONS",
        "EDITH_QUERY_REWRITE",
        "EDITH_SUPPORT_AUDIT",
        "EDITH_CONFIDENCE_ROUTING",
        "EDITH_MULTI_PASS",
        "EDITH_CONTRADICTION_CHECK",
        "EDITH_CHROMA_COLLECTION",
        "EDITH_EMBED_MODEL",
    ]
    cfg = {}
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> int:
    load_env()
    args = parse_args()

    app_data = app_data_dir(args.app_data_dir)
    docs = docs_root(args.docs_root)
    chroma = chroma_dir(app_data)
    snapshots = app_data / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    manifest = manifest_for_root(docs, max_files=max(500, int(args.max_files)))
    corpus_hash = hash_manifest(manifest)

    chroma_manifest = []
    chroma_hash = ""
    if args.include_chroma and chroma.exists():
        chroma_manifest = manifest_for_root(chroma, max_files=max(500, int(args.max_files)))
        chroma_hash = hash_manifest(chroma_manifest)

    retrieval_profile = {}
    profile_path = app_data / "retrieval_profile.json"
    if profile_path.exists():
        try:
            retrieval_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            retrieval_profile = {}

    snapshot = {
        "generated_at": now_utc(),
        "docs_root": str(docs),
        "docs_configured": bool((os.getenv("EDITH_DATA_ROOT") or "").strip() or args.docs_root),
        "docs_exists": bool(docs.exists()),
        "docs_file_count": len(manifest),
        "corpus_hash": corpus_hash,
        "manifest": manifest,
        "include_chroma": bool(args.include_chroma),
        "chroma_dir": str(chroma),
        "chroma_hash": chroma_hash,
        "chroma_file_count": len(chroma_manifest),
        "config": selected_env_config(),
        "retrieval_profile": retrieval_profile if isinstance(retrieval_profile, dict) else {},
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out).expanduser().resolve() if args.out else (snapshots / f"snapshot_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    (snapshots / "latest_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "snapshot": str(out_path),
                "docs_file_count": len(manifest),
                "corpus_hash": corpus_hash,
                "chroma_hash": chroma_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

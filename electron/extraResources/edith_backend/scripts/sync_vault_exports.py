#!/usr/bin/env python3
"""
Sync Google Vault export ZIPs into Edith local data root.

Features:
- Inbox/Archive/Failed workflow
- Incremental ZIP processing via manifest hash tracking
- Supported-file copy with size allowlist and quarantine flow
- Basic MBOX -> TXT conversion for indexability
- Dedup by content hash across repeated export snapshots
- Sync report + last-run summary JSON
- Optional index + smoketest execution
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mailbox
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


SUPPORTED_EXTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".rtf",
    ".odt",
    ".tex",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
    ".py",
    ".ipynb",
    ".js",
    ".ts",
    ".sql",
    ".r",
    ".R",
    ".eml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_slug(value: str, default: str = "item") -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text).strip("_")
    return text[:120] if text else default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync Google Vault exports into Edith data root")
    p.add_argument("--no-index", action="store_true", help="Skip indexer run after sync")
    p.add_argument("--no-smoketest", action="store_true", help="Skip post-index smoketest queries")
    p.add_argument("--force-index", action="store_true", help="Run index even if no file changes")
    p.add_argument("--max-files", type=int, default=0, help="Optional limit for copied files per run (0=unlimited)")
    p.add_argument("--dry-run", action="store_true", help="Plan and report without mutating files")
    return p.parse_args()


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
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)


def app_state_dir() -> Path:
    raw = (os.getenv("EDITH_APP_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "Library" / "Application Support" / "Edith").resolve()


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(default, dict) and isinstance(obj, dict):
            return obj
        if isinstance(default, list) and isinstance(obj, list):
            return obj
    except Exception:
        pass
    return default


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
            if pid and pid_alive(pid):
                return False, f"sync already running (pid {pid})"
        except Exception:
            pass
        try:
            lock_path.unlink()
        except Exception:
            return False, "could not clear stale lock file"
    data = {"pid": os.getpid(), "started_at": utc_now()}
    lock_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return True, "ok"


def release_lock(lock_path: Path):
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def infer_export_meta(zip_path: Path, incoming_root: Path):
    rel = str(zip_path.relative_to(incoming_root))
    stem = zip_path.stem
    export_id = safe_slug(stem, "vault_export")
    date_match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    export_date = ""
    if date_match:
        export_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

    custodian = ""
    m = re.search(r"(?:custodian|user|account)[-_]([A-Za-z0-9._@-]+)", rel, flags=re.I)
    if m:
        custodian = clean_text(m.group(1))
    if not custodian:
        email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", rel)
        if email:
            custodian = email.group(0)

    matter_name = ""
    parts = list(zip_path.relative_to(incoming_root).parts)
    if len(parts) >= 2:
        maybe = clean_text(parts[0])
        if maybe and maybe.lower() not in {"incoming", "exports"}:
            matter_name = maybe
    if not matter_name:
        m2 = re.search(r"(?:matter|case)[-_]([A-Za-z0-9._-]+)", rel, flags=re.I)
        if m2:
            matter_name = clean_text(m2.group(1))

    return {
        "vault_export_id": export_id,
        "vault_export_date": export_date,
        "vault_custodian": custodian,
        "vault_matter_name": matter_name,
        "source_zip_rel": rel,
    }


def safe_extract_zip(
    zip_path: Path,
    dst: Path,
    max_entries: int = 50000,
    max_total_bytes: int = 2 * 1024 * 1024 * 1024,
    max_member_bytes: int = 300 * 1024 * 1024,
    max_compression_ratio: float = 200.0,
):
    extracted = []
    base_real = os.path.realpath(str(dst))
    total_bytes = 0
    entry_count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            entry_count += 1
            if entry_count > max_entries:
                break
            # Zip-bomb guardrails.
            member_size = int(max(0, info.file_size or 0))
            compressed_size = int(max(0, info.compress_size or 0))
            if member_size > max_member_bytes:
                continue
            if compressed_size <= 0 and member_size > 0:
                continue
            if compressed_size > 0 and (member_size / compressed_size) > max_compression_ratio:
                continue
            total_bytes += member_size
            if total_bytes > max_total_bytes:
                break
            name = str(info.filename or "")
            # Normalize and block traversal.
            normalized = Path(name)
            if normalized.is_absolute():
                continue
            target = dst / normalized
            target_real = os.path.realpath(str(target))
            if os.path.commonpath([base_real, target_real]) != base_real:
                continue
            target_path = Path(target_real)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target_path.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target_path)
    return extracted


def decode_payload(msg) -> str:
    payload = msg.get_payload(decode=True)
    if payload is None:
        text = msg.get_payload()
        if isinstance(text, str):
            return text
        return ""
    charset = msg.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="ignore")
    except Exception:
        return payload.decode("utf-8", errors="ignore")


def mbox_messages_to_txt(mbox_path: Path, out_dir: Path, max_messages: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    box = mailbox.mbox(str(mbox_path))
    count = 0
    for msg in box:
        if max_messages > 0 and count >= max_messages:
            break
        count += 1
        subject = clean_text(msg.get("Subject", ""))
        from_ = clean_text(msg.get("From", ""))
        to_ = clean_text(msg.get("To", ""))
        date_ = clean_text(msg.get("Date", ""))
        message_id = clean_text(msg.get("Message-ID", "")).strip("<>")
        body = ""
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                ctype = (part.get_content_type() or "").lower()
                disp = clean_text(part.get("Content-Disposition", "")).lower()
                if "attachment" in disp:
                    continue
                if ctype in {"text/plain", "text/html"}:
                    parts.append(decode_payload(part))
            body = "\n\n".join([p for p in parts if p])
        else:
            body = decode_payload(msg)
        body = clean_text(body)
        if not message_id:
            message_id = hashlib.sha1(f"{subject}|{from_}|{to_}|{date_}|{body[:500]}".encode("utf-8")).hexdigest()
        file_id = safe_slug(message_id, f"msg_{count}")
        out_path = out_dir / f"{file_id}.txt"
        txt = (
            f"Subject: {subject}\n"
            f"From: {from_}\n"
            f"To: {to_}\n"
            f"Date: {date_}\n"
            f"Message-ID: {message_id}\n\n"
            f"{body}\n"
        )
        out_path.write_text(txt, encoding="utf-8", errors="ignore")
        rows.append(
            {
                "path": out_path,
                "meta": {
                    "mail_subject": subject,
                    "mail_from": from_,
                    "mail_to": to_,
                    "mail_date": date_,
                    "mail_message_id": message_id,
                },
            }
        )
    return rows


def move_with_unique_name(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    suffix = src.suffix
    candidate = dst_dir / src.name
    idx = 1
    while candidate.exists():
        candidate = dst_dir / f"{stem}_{idx}{suffix}"
        idx += 1
    shutil.move(str(src), str(candidate))
    return candidate


def format_md_report(summary: dict):
    lines = []
    lines.append(f"# Vault Sync Report — {summary.get('finished_at', '')}")
    lines.append("")
    lines.append("## Totals")
    totals = summary.get("totals") or {}
    for key in [
        "zip_seen",
        "zip_processed",
        "zip_skipped_unchanged",
        "zip_failed",
        "files_new",
        "files_updated",
        "files_deduped",
        "files_unsupported",
        "files_too_large",
        "files_quarantined",
        "mbox_messages_converted",
        "index_ran",
        "smoketest_ran",
        "smoketest_failed",
    ]:
        lines.append(f"- {key}: {totals.get(key, 0)}")
    lines.append("")

    by_type = summary.get("by_filetype") or {}
    if by_type:
        lines.append("## Added/Updated by Filetype")
        for ext, count in sorted(by_type.items(), key=lambda x: (-int(x[1]), x[0])):
            lines.append(f"- {ext}: {count}")
        lines.append("")

    by_export = summary.get("by_export") or {}
    if by_export:
        lines.append("## Added/Updated by Export")
        for exp, count in sorted(by_export.items(), key=lambda x: (-int(x[1]), x[0])):
            lines.append(f"- {exp}: {count}")
        lines.append("")

    changed = summary.get("changed_files") or []
    if changed:
        lines.append("## Changed Files")
        for rel in changed[:200]:
            lines.append(f"- `{rel}`")
        lines.append("")

    skipped = summary.get("skipped_examples") or []
    if skipped:
        lines.append("## Skipped Examples")
        for row in skipped[:120]:
            lines.append(f"- {row}")
        lines.append("")

    zip_results = summary.get("zip_results") or []
    if zip_results:
        lines.append("## ZIP Processing")
        for row in zip_results[:200]:
            lines.append(
                f"- {row.get('zip')}: {row.get('status')} "
                f"(new={row.get('files_new', 0)}, updated={row.get('files_updated', 0)}, dedup={row.get('files_deduped', 0)})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_cmd(cmd: list[str], cwd: Path, env: dict, timeout_s: int):
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "code": int(proc.returncode),
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "code": 124,
            "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "") if isinstance(e.stderr, str) else f"Timed out after {timeout_s}s",
        }
    except Exception as e:
        return {"ok": False, "code": 1, "stdout": "", "stderr": str(e)}


def parse_smoketest_queries(default_queries: list[str]):
    raw = (os.getenv("EDITH_VAULT_SMOKETEST_QUERIES") or "").strip()
    if not raw:
        return default_queries[:3]
    out = []
    for part in raw.split("||"):
        q = clean_text(part)
        if q:
            out.append(q)
    return out[:8] if out else default_queries[:3]


def main() -> int:
    load_env()
    args = parse_args()

    vault_root_raw = (os.getenv("EDITH_VAULT_EXPORT_DIR") or "").strip()
    data_root_raw = (os.getenv("EDITH_DATA_ROOT") or "").strip()
    if not vault_root_raw:
        raise SystemExit("Missing EDITH_VAULT_EXPORT_DIR.")
    if not data_root_raw:
        raise SystemExit("Missing EDITH_DATA_ROOT.")

    vault_root = Path(vault_root_raw).expanduser().resolve()
    data_root = Path(data_root_raw).expanduser().resolve()
    state_dir = app_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / ".vault_sync.lock"
    ok_lock, lock_msg = acquire_lock(lock_path)
    if not ok_lock:
        print(json.dumps({"ok": False, "reason": lock_msg}, indent=2))
        return 1

    started_at = utc_now()
    try:
        incoming = vault_root / "Incoming" if (vault_root / "Incoming").exists() else vault_root
        archived = vault_root / "Archived"
        failed = vault_root / "Failed"
        quarantine = vault_root / "Quarantine"
        archived.mkdir(parents=True, exist_ok=True)
        failed.mkdir(parents=True, exist_ok=True)
        quarantine.mkdir(parents=True, exist_ok=True)

        sync_root = data_root / "vault_sync"
        manifests_dir = sync_root / "_manifests"
        reports_dir = sync_root / "_reports"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        sync_root.mkdir(parents=True, exist_ok=True)

        manifest_path = manifests_dir / "vault_sync_manifest.json"
        file_manifest_path = manifests_dir / "vault_file_manifest.json"
        manifest = load_json(
            manifest_path,
            {"version": 2, "processed_zip_hashes": {}, "hash_to_rel": {}, "updated_at": ""},
        )
        file_manifest = load_json(file_manifest_path, {})
        processed_zip_hashes = manifest.get("processed_zip_hashes") or {}
        hash_to_rel = manifest.get("hash_to_rel") or {}
        if not isinstance(processed_zip_hashes, dict):
            processed_zip_hashes = {}
        if not isinstance(hash_to_rel, dict):
            hash_to_rel = {}
        if not isinstance(file_manifest, dict):
            file_manifest = {}

        # Clean stale hash mappings.
        stale_hashes = []
        for h, rel in hash_to_rel.items():
            if not isinstance(rel, str):
                stale_hashes.append(h)
                continue
            p = data_root / rel
            if not p.exists():
                stale_hashes.append(h)
        for h in stale_hashes:
            hash_to_rel.pop(h, None)

        max_file_mb = int(os.getenv("EDITH_MAX_FILE_MB", "50"))
        max_bytes = max(1, max_file_mb) * 1024 * 1024
        include_loose = os.getenv("EDITH_VAULT_SYNC_INCLUDE_LOOSE_FILES", "true").lower() == "true"
        quarantine_unknown = os.getenv("EDITH_VAULT_SYNC_QUARANTINE_UNKNOWN", "true").lower() == "true"
        mbox_max_messages = int(os.getenv("EDITH_VAULT_MBOX_MAX_MESSAGES", "20000"))
        run_smoketest = (
            os.getenv("EDITH_VAULT_SYNC_SMOKETEST", "true").lower() == "true" and not args.no_smoketest
        )
        force_index = bool(args.force_index)
        reindex_timeout = int(os.getenv("EDITH_REINDEX_TIMEOUT_SECONDS", "1800"))

        totals = {
            "zip_seen": 0,
            "zip_processed": 0,
            "zip_skipped_unchanged": 0,
            "zip_failed": 0,
            "files_new": 0,
            "files_updated": 0,
            "files_deduped": 0,
            "files_unsupported": 0,
            "files_too_large": 0,
            "files_quarantined": 0,
            "mbox_messages_converted": 0,
            "index_ran": 0,
            "smoketest_ran": 0,
            "smoketest_failed": 0,
        }
        by_filetype = Counter()
        by_export = Counter()
        changed_files = []
        skipped_examples = []
        zip_results = []

        max_files_left = int(args.max_files) if int(args.max_files or 0) > 0 else 0

        def register_file(
            src_file: Path,
            rel_inside_export: str,
            export_meta: dict,
            zip_hash: str,
            extra_meta: dict | None = None,
        ):
            nonlocal max_files_left
            ext = src_file.suffix.lower()
            size = int(src_file.stat().st_size)
            rel_inside_export = str(Path(rel_inside_export))
            if ext not in SUPPORTED_EXTS:
                totals["files_unsupported"] += 1
                if quarantine_unknown and not args.dry_run:
                    qdst = quarantine / export_meta["vault_export_id"] / rel_inside_export
                    qdst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, qdst)
                    totals["files_quarantined"] += 1
                if len(skipped_examples) < 80:
                    skipped_examples.append(f"unsupported: {src_file.name}")
                return {"status": "unsupported"}
            if size > max_bytes:
                totals["files_too_large"] += 1
                if len(skipped_examples) < 80:
                    skipped_examples.append(f"too_large: {src_file.name} ({size} bytes)")
                return {"status": "too_large"}
            if max_files_left > 0 and len(changed_files) >= max_files_left:
                if len(skipped_examples) < 80:
                    skipped_examples.append(f"max_files_limit: {src_file.name}")
                return {"status": "max_limit"}

            fhash = sha256_file(src_file)
            existing_rel = hash_to_rel.get(fhash, "")
            if existing_rel:
                existing_path = data_root / existing_rel
                if existing_path.exists():
                    totals["files_deduped"] += 1
                    row = file_manifest.get(existing_rel) or {}
                    ids = set(row.get("vault_export_ids") or [])
                    ids.add(export_meta.get("vault_export_id", ""))
                    row["vault_export_ids"] = sorted([x for x in ids if x])
                    custodians = set(str(x) for x in (row.get("vault_custodians") or []))
                    if export_meta.get("vault_custodian"):
                        custodians.add(str(export_meta["vault_custodian"]))
                    row["vault_custodians"] = sorted([x for x in custodians if x])
                    matters = set(str(x) for x in (row.get("vault_matter_names") or []))
                    if export_meta.get("vault_matter_name"):
                        matters.add(str(export_meta["vault_matter_name"]))
                    row["vault_matter_names"] = sorted([x for x in matters if x])
                    row["last_seen_at"] = utc_now()
                    file_manifest[existing_rel] = row
                    return {"status": "dedup", "rel_path": existing_rel}

            rel_dst = str(Path("vault_sync") / export_meta["vault_export_id"] / rel_inside_export)
            dst = data_root / rel_dst
            status = "new"
            if dst.exists():
                try:
                    old_hash = sha256_file(dst)
                    if old_hash != fhash:
                        status = "updated"
                except Exception:
                    status = "updated"

            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst)

            hash_to_rel[fhash] = rel_dst
            row = {
                "rel_path": rel_dst,
                "content_sha256": fhash,
                "file_size": size,
                "vault_export_id": export_meta.get("vault_export_id", ""),
                "vault_export_ids": [export_meta.get("vault_export_id", "")] if export_meta.get("vault_export_id") else [],
                "vault_export_date": export_meta.get("vault_export_date", ""),
                "vault_custodian": export_meta.get("vault_custodian", ""),
                "vault_custodians": [export_meta.get("vault_custodian", "")] if export_meta.get("vault_custodian") else [],
                "vault_matter_name": export_meta.get("vault_matter_name", ""),
                "vault_matter_names": [export_meta.get("vault_matter_name", "")] if export_meta.get("vault_matter_name") else [],
                "source_zip_hash": zip_hash,
                "source_zip_rel": export_meta.get("source_zip_rel", ""),
                "source_rel_path": rel_inside_export,
                "ingested_at": utc_now(),
                "last_seen_at": utc_now(),
            }
            if extra_meta:
                row.update({k: clean_text(v) if isinstance(v, str) else v for k, v in extra_meta.items()})
            file_manifest[rel_dst] = row

            if status == "new":
                totals["files_new"] += 1
            else:
                totals["files_updated"] += 1
            by_filetype[ext or ""] += 1
            by_export[export_meta.get("vault_export_id", "unknown")] += 1
            changed_files.append(rel_dst)
            return {"status": status, "rel_path": rel_dst}

        zip_files = []
        for p in incoming.rglob("*.zip"):
            if not p.is_file():
                continue
            if any(part in {"Archived", "Failed", "Quarantine"} for part in p.parts):
                continue
            zip_files.append(p)
        zip_files.sort(key=lambda x: x.stat().st_mtime)
        totals["zip_seen"] = len(zip_files)

        for zip_path in zip_files:
            try:
                zhash = sha256_file(zip_path)
            except Exception as e:
                totals["zip_failed"] += 1
                zip_results.append({"zip": str(zip_path.name), "status": f"hash_error: {e}"})
                continue

            export_meta = infer_export_meta(zip_path, incoming)
            if zhash in processed_zip_hashes:
                totals["zip_skipped_unchanged"] += 1
                if not args.dry_run:
                    moved = move_with_unique_name(zip_path, archived)
                    processed_zip_hashes[zhash]["duplicate_archived_as"] = moved.name
                    processed_zip_hashes[zhash]["last_seen_at"] = utc_now()
                zip_results.append({"zip": str(zip_path.name), "status": "unchanged_duplicate"})
                continue

            row_stats = {"new": 0, "updated": 0, "dedup": 0}
            try:
                with tempfile.TemporaryDirectory(prefix="edith_vault_") as td:
                    tmp_root = Path(td)
                    try:
                        extract_max_mb = int(os.getenv("EDITH_VAULT_MAX_EXTRACT_MB", "2048"))
                    except ValueError:
                        extract_max_mb = 2048
                    if extract_max_mb < 64:
                        extract_max_mb = 64
                    try:
                        max_zip_entries = int(os.getenv("EDITH_VAULT_MAX_ZIP_ENTRIES", "50000"))
                    except ValueError:
                        max_zip_entries = 50000
                    try:
                        max_zip_ratio = float(os.getenv("EDITH_VAULT_MAX_COMPRESSION_RATIO", "200"))
                    except ValueError:
                        max_zip_ratio = 200.0
                    extracted = safe_extract_zip(
                        zip_path,
                        tmp_root,
                        max_entries=max_zip_entries,
                        max_total_bytes=extract_max_mb * 1024 * 1024,
                        max_member_bytes=max_bytes,
                        max_compression_ratio=max_zip_ratio,
                    )
                    for src in extracted:
                        rel_src = os.path.relpath(os.path.realpath(str(src)), os.path.realpath(str(tmp_root)))
                        ext = src.suffix.lower()
                        if ext == ".mbox":
                            mail_dir = tmp_root / "_converted_mail" / safe_slug(Path(rel_src).stem, "mbox")
                            mail_rows = mbox_messages_to_txt(src, mail_dir, max_messages=mbox_max_messages)
                            totals["mbox_messages_converted"] += len(mail_rows)
                            for msg_row in mail_rows:
                                out = register_file(
                                    msg_row["path"],
                                    str(Path("_mail") / Path(rel_src).stem / msg_row["path"].name),
                                    export_meta,
                                    zhash,
                                    extra_meta=msg_row.get("meta") or {},
                                )
                                st = out.get("status")
                                if st in row_stats:
                                    row_stats[st] += 1
                        else:
                            out = register_file(src, rel_src, export_meta, zhash)
                            st = out.get("status")
                            if st in row_stats:
                                row_stats[st] += 1

                totals["zip_processed"] += 1
                processed_zip_hashes[zhash] = {
                    "zip_name": zip_path.name,
                    "source_zip_rel": export_meta.get("source_zip_rel", ""),
                    "vault_export_id": export_meta.get("vault_export_id", ""),
                    "vault_export_date": export_meta.get("vault_export_date", ""),
                    "vault_custodian": export_meta.get("vault_custodian", ""),
                    "vault_matter_name": export_meta.get("vault_matter_name", ""),
                    "processed_at": utc_now(),
                    "status": "success",
                    "files_new": row_stats["new"],
                    "files_updated": row_stats["updated"],
                    "files_deduped": row_stats["dedup"],
                }
                if not args.dry_run:
                    moved = move_with_unique_name(zip_path, archived)
                    processed_zip_hashes[zhash]["archived_as"] = moved.name
                zip_results.append(
                    {
                        "zip": zip_path.name,
                        "status": "processed",
                        "files_new": row_stats["new"],
                        "files_updated": row_stats["updated"],
                        "files_deduped": row_stats["dedup"],
                    }
                )
            except Exception as e:
                totals["zip_failed"] += 1
                processed_zip_hashes[zhash] = {
                    "zip_name": zip_path.name,
                    "source_zip_rel": export_meta.get("source_zip_rel", ""),
                    "vault_export_id": export_meta.get("vault_export_id", ""),
                    "processed_at": utc_now(),
                    "status": "failed",
                    "error": str(e),
                }
                if not args.dry_run and zip_path.exists():
                    move_with_unique_name(zip_path, failed)
                zip_results.append({"zip": zip_path.name, "status": f"failed: {e}"})

        if include_loose:
            for src in incoming.rglob("*"):
                if not src.is_file():
                    continue
                if src.suffix.lower() == ".zip":
                    continue
                if any(part in {"Archived", "Failed", "Quarantine"} for part in src.parts):
                    continue
                rel_src = str(src.relative_to(incoming))
                export_meta = {
                    "vault_export_id": "raw_files",
                    "vault_export_date": "",
                    "vault_custodian": "",
                    "vault_matter_name": "",
                    "source_zip_rel": "raw_files",
                }
                register_file(src, str(Path("raw_files") / rel_src), export_meta, zip_hash="raw_files")

        # Remove stale file_manifest entries for missing files.
        removed_files = []
        for rel in list(file_manifest.keys()):
            p = data_root / rel
            if not p.exists():
                removed_files.append(rel)
                file_manifest.pop(rel, None)
        for h, rel in list(hash_to_rel.items()):
            if rel in removed_files:
                hash_to_rel.pop(h, None)

        manifest["version"] = 2
        manifest["updated_at"] = utc_now()
        manifest["processed_zip_hashes"] = processed_zip_hashes
        manifest["hash_to_rel"] = hash_to_rel
        if not args.dry_run:
            save_json(manifest_path, manifest)
            save_json(file_manifest_path, file_manifest)

        root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env.setdefault("EDITH_APP_DATA_DIR", str(state_dir))
        backend = (os.getenv("EDITH_RETRIEVAL_BACKEND") or "google").strip().lower()
        index_script = "chroma_index.py" if backend == "chroma" else "index_files.py"
        index_result = {"ok": True, "code": 0, "stdout": "", "stderr": "", "skipped": True}
        changed_count = int(totals["files_new"] + totals["files_updated"] + totals["mbox_messages_converted"])
        if not args.no_index and (changed_count > 0 or force_index):
            if not args.dry_run:
                index_result = run_cmd(
                    [sys.executable, str(root / index_script)],
                    cwd=root,
                    env=env,
                    timeout_s=max(60, reindex_timeout),
                )
            else:
                index_result = {"ok": True, "code": 0, "stdout": "dry-run index skipped", "stderr": "", "skipped": True}
            totals["index_ran"] = 1

        smoketest_results = []
        if run_smoketest and totals["index_ran"] == 1 and index_result.get("ok"):
            default_q = []
            for rel in changed_files[:3]:
                stem = clean_text(Path(rel).stem.replace("_", " ").replace("-", " "))
                if stem:
                    default_q.append(stem[:180])
            queries = parse_smoketest_queries(default_q)
            if queries and not args.dry_run:
                for q in queries[:5]:
                    cmd = [
                        sys.executable,
                        str(root / "smoketest_query.py"),
                        q,
                        "--mode",
                        "Files only",
                        "--backend",
                        backend,
                    ]
                    res = run_cmd(cmd, cwd=root, env=env, timeout_s=240)
                    ok = bool(res.get("ok"))
                    if not ok:
                        totals["smoketest_failed"] += 1
                    smoketest_results.append(
                        {
                            "query": q,
                            "ok": ok,
                            "code": int(res.get("code", 1)),
                            "stderr": (res.get("stderr") or "")[:280],
                        }
                    )
                totals["smoketest_ran"] = 1

        summary = {
            "ok": True,
            "started_at": started_at,
            "finished_at": utc_now(),
            "vault_root": str(vault_root),
            "incoming_dir": str(incoming),
            "archived_dir": str(archived),
            "failed_dir": str(failed),
            "quarantine_dir": str(quarantine),
            "sync_root": str(sync_root),
            "totals": totals,
            "by_filetype": dict(by_filetype),
            "by_export": dict(by_export),
            "changed_files": changed_files[:600],
            "removed_files": removed_files[:400],
            "skipped_examples": skipped_examples,
            "zip_results": zip_results,
            "index_result": index_result,
            "smoketest_results": smoketest_results,
            "degraded": bool(totals["smoketest_failed"] > 0 or totals["zip_failed"] > 0),
        }

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        md_path = reports_dir / f"{ts}.md"
        json_path = reports_dir / f"{ts}.json"
        latest_json = reports_dir / "last_sync_summary.json"
        if not args.dry_run:
            md_path.write_text(format_md_report(summary), encoding="utf-8")
            save_json(json_path, summary)
            save_json(latest_json, summary)

        print(
            json.dumps(
                {
                    "ok": True,
                    "degraded": summary["degraded"],
                    "totals": totals,
                    "report_md": str(md_path),
                    "report_json": str(json_path),
                    "latest": str(latest_json),
                },
                indent=2,
            )
        )
        return 0 if not summary["degraded"] else 2
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())

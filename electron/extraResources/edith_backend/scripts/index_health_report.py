#!/usr/bin/env python3
"""
Generate index health report for nightly/weekly maintenance.

Outputs JSON to:
  <EDITH_APP_DATA_DIR>/index_health_report.json
Tracks prior snapshot in:
  <EDITH_APP_DATA_DIR>/index_health_state.json
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_env():
    if not load_dotenv:
        return
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / ".env",
        Path.home() / "Library" / "Application Support" / "Edith" / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def app_data_dir() -> Path:
    raw = (os.getenv("EDITH_APP_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "Library" / "Application Support" / "Edith").resolve()


def load_index_rows(report_path: Path):
    if not report_path.exists():
        return []
    rows = []
    with report_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def normalize_rel_path(row: dict):
    return str(row.get("rel_path") or row.get("file_name") or "").strip()


def load_cipher(app_dir: Path):
    if Fernet is None:
        return None
    key = (os.getenv("EDITH_CHAT_ENCRYPTION_KEY") or "").strip()
    if not key:
        key_file = app_dir / "chat_history" / ".chat.key"
        if key_file.exists():
            try:
                key = key_file.read_text(encoding="utf-8").strip()
            except Exception:
                key = ""
    if not key:
        return None
    try:
        return Fernet(key.encode("utf-8"))
    except Exception:
        return None


def load_run_records(run_ledger_path: Path, cipher):
    rows = []
    if not run_ledger_path.exists():
        return rows
    with run_ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if "enc" in obj:
                token = str(obj.get("enc") or "")
                if not token or cipher is None:
                    continue
                try:
                    raw = cipher.decrypt(token.encode("utf-8"))
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def retrieved_rel_paths(run_records):
    hits = Counter()
    for rec in run_records:
        for src in rec.get("sources") or []:
            if not isinstance(src, dict):
                continue
            rel = str(src.get("rel_path") or src.get("uri") or "").strip()
            if rel:
                hits[rel] += 1
    return hits


def load_state(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    load_env()
    app_dir = app_data_dir()
    app_dir.mkdir(parents=True, exist_ok=True)

    report_path = app_dir / "edith_index_report.csv"
    run_ledger_path = Path(
        os.getenv("EDITH_RUN_LEDGER_PATH", str(app_dir / "run_ledger.jsonl"))
    ).expanduser()
    out_path = app_dir / "index_health_report.json"
    state_path = app_dir / "index_health_state.json"

    rows = load_index_rows(report_path)
    rel_paths = [normalize_rel_path(r) for r in rows if normalize_rel_path(r)]
    rel_set = set(rel_paths)

    prev = load_state(state_path)
    prev_rel = set(prev.get("rel_paths") or [])
    new_files = sorted(rel_set - prev_rel)
    removed_files = sorted(prev_rel - rel_set)

    family_counts = Counter(str(r.get("doc_family") or "").strip() for r in rows if str(r.get("doc_family") or "").strip())
    duplicate_families = [{"doc_family": fam, "count": int(count)} for fam, count in family_counts.items() if int(count) > 1]
    duplicate_families.sort(key=lambda x: x["count"], reverse=True)

    missing_rows = []
    for r in rows:
        missing = []
        if not str(r.get("title_guess") or "").strip():
            missing.append("title")
        if not str(r.get("author_guess") or "").strip():
            missing.append("author")
        if not str(r.get("year_guess") or "").strip():
            missing.append("year")
        if not str(r.get("doc_type") or "").strip():
            missing.append("doc_type")
        if missing:
            missing_rows.append(
                {
                    "rel_path": normalize_rel_path(r),
                    "missing": missing,
                }
            )

    cipher = load_cipher(app_dir)
    run_records = load_run_records(run_ledger_path, cipher)
    retrieved_counts = retrieved_rel_paths(run_records)
    never_retrieved = sorted([p for p in rel_set if retrieved_counts.get(p, 0) == 0])
    low_retrieved = sorted(
        [{"rel_path": p, "count": int(retrieved_counts.get(p, 0))} for p in rel_set if 0 < int(retrieved_counts.get(p, 0)) <= 1],
        key=lambda x: x["rel_path"],
    )

    report = {
        "generated_at": now_utc(),
        "index_report": str(report_path),
        "run_ledger": str(run_ledger_path),
        "totals": {
            "docs_indexed": len(rows),
            "new_files": len(new_files),
            "removed_files": len(removed_files),
            "duplicate_families": len(duplicate_families),
            "missing_metadata_docs": len(missing_rows),
            "never_retrieved_docs": len(never_retrieved),
            "low_retrieved_docs": len(low_retrieved),
        },
        "new_files": new_files[:300],
        "removed_files": removed_files[:300],
        "duplicate_families": duplicate_families[:200],
        "missing_metadata": missing_rows[:500],
        "never_retrieved": never_retrieved[:500],
        "low_retrieved": low_retrieved[:500],
    }

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_state(
        state_path,
        {
            "generated_at": report["generated_at"],
            "rel_paths": sorted(rel_set),
        },
    )
    print(json.dumps({"ok": True, "report": str(out_path), "totals": report["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

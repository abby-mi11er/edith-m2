#!/usr/bin/env python3
"""
Export a reproducibility bundle for Edith.

Bundle includes:
- eval cases
- retrieval profile + index report + feedback DB + run ledger
- optional chat history
- optional secrets (.env/config)
- optional local Chroma DB
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import hashlib
import subprocess
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


REDACT_KEYS = {
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "EDITH_APP_PASSWORD",
    "EDITH_APP_PASSWORD_HASH",
    "EDITH_CHAT_ENCRYPTION_KEY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Edith reproducibility bundle")
    p.add_argument("--out", default="", help="Output .tar.gz path")
    p.add_argument("--include-secrets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-chat-history", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-chroma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--portability-mode", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-snapshot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--encrypt-run-ledger", action=argparse.BooleanOptionalAction, default=False)
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
    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)


def app_data_dir() -> Path:
    raw = (os.getenv("EDITH_APP_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "Library" / "Application Support" / "Edith").resolve()


def chroma_dir(app_data: Path) -> Path:
    raw = (os.getenv("EDITH_CHROMA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (app_data / "chroma").resolve()


def safe_copy(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def safe_copytree(src: Path, dst: Path):
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def redact_env_file(src: Path, dst: Path):
    if not src.exists():
        return False
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key, value = line.split("=", 1)
        k = key.strip()
        if k in REDACT_KEYS:
            out.append(f"{k}=[REDACTED]")
        else:
            out.append(f"{k}={value}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def safe_file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_dir(src: Path, dst: Path):
    if not src.exists() or not src.is_dir():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def try_encrypt_run_ledger(src: Path, dst: Path):
    if not src.exists():
        return {"ok": False, "reason": "run_ledger_missing"}
    if Fernet is None:
        return {"ok": False, "reason": "cryptography_missing"}
    key = (os.getenv("EDITH_CHAT_ENCRYPTION_KEY") or "").strip()
    if not key:
        key_path = src.parent / "chat_history" / ".chat.key"
        if key_path.exists():
            try:
                key = key_path.read_text(encoding="utf-8").strip()
            except Exception:
                key = ""
    if not key:
        return {"ok": False, "reason": "encryption_key_missing"}
    try:
        cipher = Fernet(key.encode("utf-8"))
        data = src.read_bytes()
        token = cipher.encrypt(data)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(token)
        return {
            "ok": True,
            "encrypted_path": str(dst),
            "plaintext_sha256": safe_file_hash(src),
            "ciphertext_sha256": safe_file_hash(dst),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def main() -> int:
    load_env()
    args = parse_args()
    root = project_root()
    app_data = app_data_dir()
    chroma = chroma_dir(app_data)
    portability_mode = bool(args.portability_mode)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out).expanduser().resolve() if args.out else (root / "backups" / f"edith_repro_{ts}.tar.gz").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="edith_repro_") as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir(parents=True, exist_ok=True)

        copied = {}

        # Project-level artifacts
        copied["eval_cases"] = safe_copy(root / "eval" / "cases.jsonl", stage / "project" / "eval" / "cases.jsonl")
        copied["eval_traps"] = safe_copy(
            root / "eval" / "hallucination_traps.jsonl",
            stage / "project" / "eval" / "hallucination_traps.jsonl",
        )

        env_src = root / ".env"
        if args.include_secrets:
            copied["project_env"] = safe_copy(env_src, stage / "project" / ".env")
        else:
            copied["project_env_redacted"] = redact_env_file(env_src, stage / "project" / ".env.redacted")

        # App-state artifacts
        copied["retrieval_profile"] = safe_copy(app_data / "retrieval_profile.json", stage / "app_state" / "retrieval_profile.json")
        copied["index_report"] = safe_copy(app_data / "edith_index_report.csv", stage / "app_state" / "edith_index_report.csv")
        copied["feedback_db"] = safe_copy(app_data / "feedback.sqlite3", stage / "app_state" / "feedback.sqlite3")
        if args.encrypt_run_ledger:
            copied["run_ledger"] = False
            copied["run_ledger_encrypted"] = try_encrypt_run_ledger(
                app_data / "run_ledger.jsonl",
                stage / "app_state" / "run_ledger.jsonl.enc",
            )
        else:
            copied["run_ledger"] = safe_copy(app_data / "run_ledger.jsonl", stage / "app_state" / "run_ledger.jsonl")
        copied["desktop_config"] = safe_copy(app_data / "config.json", stage / "app_state" / "config.json")
        copied["glossary_graph"] = safe_copy(app_data / "glossary_graph.json", stage / "app_state" / "glossary_graph.json")
        copied["citation_graph"] = safe_copy(app_data / "citation_graph.json", stage / "app_state" / "citation_graph.json")
        copied["chapter_anchors"] = safe_copy(app_data / "chapter_anchors.json", stage / "app_state" / "chapter_anchors.json")
        copied["claim_inventory"] = safe_copy(app_data / "claim_inventory.json", stage / "app_state" / "claim_inventory.json")
        copied["experiment_ledger"] = safe_copy(app_data / "experiment_ledger.json", stage / "app_state" / "experiment_ledger.json")
        copied["bibliography_db"] = safe_copy(app_data / "bibliography_db.json", stage / "app_state" / "bibliography_db.json")
        copied["entity_timeline"] = safe_copy(app_data / "entity_timeline.json", stage / "app_state" / "entity_timeline.json")
        copied["index_health_report"] = safe_copy(app_data / "index_health_report.json", stage / "app_state" / "index_health_report.json")

        if args.include_chat_history:
            copied["chat_history"] = safe_copytree(app_data / "chat_history", stage / "app_state" / "chat_history")
        else:
            copied["chat_history"] = False

        if args.include_chroma:
            copied["chroma_dir"] = safe_copytree(chroma, stage / "app_state" / "chroma")
        else:
            copied["chroma_dir"] = False

        if args.include_snapshot:
            snap_script = root / "scripts" / "corpus_snapshot.py"
            if snap_script.exists():
                snap_out = app_data / "snapshots" / f"snapshot_{ts}.json"
                try:
                    proc = subprocess.run(
                        [os.getenv("PYTHON", "python3"), str(snap_script), "--out", str(snap_out), "--include-chroma" if args.include_chroma else "--no-include-chroma"],
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                        timeout=1800,
                    )
                    copied["snapshot_generated"] = proc.returncode == 0
                    copied["snapshot_generation_output"] = (proc.stdout or proc.stderr or "").strip()[:2000]
                except Exception as e:
                    copied["snapshot_generated"] = False
                    copied["snapshot_generation_output"] = str(e)
            copied["latest_snapshot"] = safe_copy(
                app_data / "snapshots" / "latest_snapshot.json",
                stage / "app_state" / "snapshots" / "latest_snapshot.json",
            )
            copied["snapshot_dir"] = copy_if_dir(
                app_data / "snapshots",
                stage / "app_state" / "snapshots",
            )
        else:
            copied["latest_snapshot"] = False
            copied["snapshot_dir"] = False

        if portability_mode:
            copied["templates_dir"] = copy_if_dir(root / "templates", stage / "project" / "templates")
            copied["ontology_dir"] = copy_if_dir(root / "ontology", stage / "project" / "ontology")
            copied["skills_dir"] = copy_if_dir(root / "skills", stage / "project" / "skills")
        else:
            copied["templates_dir"] = False
            copied["ontology_dir"] = False
            copied["skills_dir"] = False

        manifest = {
            "timestamp_utc": ts,
            "project_root": str(root),
            "app_data_dir": str(app_data),
            "chroma_dir": str(chroma),
            "include_secrets": bool(args.include_secrets),
            "include_chat_history": bool(args.include_chat_history),
            "include_chroma": bool(args.include_chroma),
            "include_snapshot": bool(args.include_snapshot),
            "encrypt_run_ledger": bool(args.encrypt_run_ledger),
            "portability_mode": portability_mode,
            "copied": copied,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(stage, arcname="edith_repro_bundle")

    print(
        json.dumps(
            {
                "ok": True,
                "bundle": str(out_path),
                "include_secrets": bool(args.include_secrets),
                "include_chat_history": bool(args.include_chat_history),
                "include_chroma": bool(args.include_chroma),
                "include_snapshot": bool(args.include_snapshot),
                "encrypt_run_ledger": bool(args.encrypt_run_ledger),
                "portability_mode": portability_mode,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Create an OpenAI supervised fine-tuning job from JSONL files.

Usage:
  python fine_tune_sft.py --train edith_train.jsonl --val edith_val.jsonl --base-model gpt-4.1-mini

Environment:
  OPENAI_API_KEY=...
  OPENAI_BASE_URL=https://api.openai.com/v1  (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create OpenAI fine-tuning job for Edith SFT JSONL data")
    p.add_argument("--train", required=True, help="Path to training JSONL file")
    p.add_argument("--val", default="", help="Path to validation JSONL file")
    p.add_argument("--base-model", default=os.getenv("OPENAI_BASE_MODEL", "gpt-4.1-mini"), help="Base model")
    p.add_argument("--suffix", default="edith", help="Model suffix label")
    p.add_argument("--watch", action="store_true", help="Poll fine-tune status until terminal state")
    p.add_argument("--poll-seconds", type=int, default=15, help="Polling interval when --watch is set")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print payload only")
    return p.parse_args()


def require_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is required.")
    return key


def base_url() -> str:
    url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
    if not url:
        url = "https://api.openai.com/v1"
    return url


def auth_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def upload_file(path: Path, api_key: str, api_base: str) -> str:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"File not found: {path}")

    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "application/jsonl")}
        data = {"purpose": "fine-tune"}
        resp = requests.post(
            f"{api_base}/files",
            headers=auth_headers(api_key),
            files=files,
            data=data,
            timeout=120,
        )
    if resp.status_code >= 300:
        raise SystemExit(f"Upload failed for {path.name}: {resp.status_code} {resp.text}")

    payload = resp.json()
    file_id = payload.get("id")
    if not file_id:
        raise SystemExit(f"Upload response missing file id: {json.dumps(payload, ensure_ascii=False)}")
    return file_id


def create_job(train_file_id: str, val_file_id: str, model: str, suffix: str, api_key: str, api_base: str) -> dict:
    body = {
        "model": model,
        "training_file": train_file_id,
    }
    if val_file_id:
        body["validation_file"] = val_file_id
    if suffix:
        body["suffix"] = suffix

    resp = requests.post(
        f"{api_base}/fine_tuning/jobs",
        headers={**auth_headers(api_key), "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=120,
    )
    if resp.status_code >= 300:
        raise SystemExit(f"Fine-tune job creation failed: {resp.status_code} {resp.text}")
    return resp.json()


def get_job(job_id: str, api_key: str, api_base: str) -> dict:
    resp = requests.get(
        f"{api_base}/fine_tuning/jobs/{job_id}",
        headers=auth_headers(api_key),
        timeout=60,
    )
    if resp.status_code >= 300:
        raise SystemExit(f"Failed to fetch job status: {resp.status_code} {resp.text}")
    return resp.json()


def main() -> int:
    args = parse_args()
    api_key = require_api_key()
    api_base = base_url()

    train_path = Path(args.train).expanduser().resolve()
    val_path = Path(args.val).expanduser().resolve() if args.val else None

    if not train_path.exists():
        raise SystemExit(f"Training file not found: {train_path}")
    if val_path and not val_path.exists():
        raise SystemExit(f"Validation file not found: {val_path}")

    if args.dry_run:
        preview = {
            "base_model": args.base_model,
            "train": str(train_path),
            "val": str(val_path) if val_path else "",
            "suffix": args.suffix,
            "api_base": api_base,
        }
        print(json.dumps(preview, indent=2))
        return 0

    print(f"Uploading training file: {train_path.name}")
    train_file_id = upload_file(train_path, api_key, api_base)

    val_file_id = ""
    if val_path:
        print(f"Uploading validation file: {val_path.name}")
        val_file_id = upload_file(val_path, api_key, api_base)

    print("Creating fine-tune job...")
    job = create_job(
        train_file_id=train_file_id,
        val_file_id=val_file_id,
        model=args.base_model,
        suffix=args.suffix,
        api_key=api_key,
        api_base=api_base,
    )

    job_id = job.get("id", "")
    print(json.dumps({
        "job_id": job_id,
        "status": job.get("status"),
        "model": job.get("model"),
        "training_file": train_file_id,
        "validation_file": val_file_id,
    }, indent=2))

    if not args.watch or not job_id:
        return 0

    terminal = {"succeeded", "failed", "cancelled"}
    while True:
        time.sleep(max(5, int(args.poll_seconds)))
        status_payload = get_job(job_id, api_key, api_base)
        status = (status_payload.get("status") or "").lower()
        fine_tuned_model = status_payload.get("fine_tuned_model")
        msg = {
            "job_id": job_id,
            "status": status,
            "fine_tuned_model": fine_tuned_model,
        }
        print(json.dumps(msg, ensure_ascii=False))
        if status in terminal:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

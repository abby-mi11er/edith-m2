#!/usr/bin/env python3
"""
Export Edith chat history into OpenAI SFT JSONL datasets.

Reads chat history from:
  EDITH_APP_DATA_DIR/chat_history (default: ~/Library/Application Support/Edith/chat_history)

Outputs:
  edith_train.jsonl
  edith_val.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Edith SFT dataset from saved chats")
    p.add_argument("--chat-dir", default="", help="Override chat_history directory")
    p.add_argument("--train-out", default="edith_train.jsonl", help="Train output JSONL path")
    p.add_argument("--val-out", default="edith_val.jsonl", help="Validation output JSONL path")
    p.add_argument("--max-examples", type=int, default=800, help="Max examples to export")
    p.add_argument("--include-refusals", action="store_true", help="Include 'Not found in sources.' examples")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--redact-pii", action=argparse.BooleanOptionalAction, default=True, help="Redact likely PII in exported data")
    p.add_argument("--redact-token", default="[REDACTED]", help="Replacement token for redacted values")
    p.add_argument("--only-positive-feedback", action=argparse.BooleanOptionalAction, default=False, help="Only include assistant turns with positive feedback and no refusal/bad flags")
    p.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False, help="Also emit staged curriculum train files (recall, synthesis, production)")
    return p.parse_args()


def app_data_dir() -> Path:
    raw = (os.getenv("EDITH_APP_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Library" / "Application Support" / "Edith"


def resolve_chat_dir(override: str) -> Path:
    if override:
        return Path(override).expanduser()
    return app_data_dir() / "chat_history"


def load_cipher(chat_dir: Path):
    if Fernet is None:
        return None
    explicit = (os.getenv("EDITH_CHAT_ENCRYPTION_KEY") or "").strip()
    key = explicit
    if not key:
        key_file = chat_dir / ".chat.key"
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


def read_chat(path: Path, cipher):
    try:
        raw = path.read_bytes()
    except Exception:
        return None

    if path.suffix == ".enc":
        if cipher is None:
            return None
        try:
            raw = cipher.decrypt(raw)
        except Exception:
            return None

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def find_assistant_query(messages, assistant_idx: int):
    msg = messages[assistant_idx] if 0 <= assistant_idx < len(messages) else {}
    direct = (msg.get("query") or "").strip()
    if direct:
        return direct
    for i in range(assistant_idx - 1, -1, -1):
        m = messages[i] or {}
        if m.get("role") == "user":
            return (m.get("text") or "").strip()
    return ""


PII_PATTERNS = [
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\\-_]{20,}\b"),
]


def redact_pii_text(text: str, enabled: bool, token: str):
    out = str(text or "")
    if not enabled or not out:
        return out
    for rx in PII_PATTERNS:
        out = rx.sub(token, out)
    return out


def format_sources(sources, max_sources: int = 6, max_chars: int = 500, redact_pii: bool = True, redact_token: str = "[REDACTED]"):
    rows = []
    for idx, src in enumerate((sources or [])[:max_sources], start=1):
        if not isinstance(src, dict):
            continue
        title = redact_pii_text((src.get("title") or src.get("uri") or f"source_{idx}").strip(), enabled=redact_pii, token=redact_token)
        snippet = redact_pii_text((src.get("snippet") or "").strip(), enabled=redact_pii, token=redact_token)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        line = f"[S{idx}] {title}"
        if snippet:
            line += f"\nsnippet={snippet}"
        rows.append(line)
    return "\n\n".join(rows)


def build_example(query: str, answer: str, sources, redact_pii: bool, redact_token: str):
    q = redact_pii_text((query or "").strip(), enabled=redact_pii, token=redact_token)
    a = redact_pii_text((answer or "").strip(), enabled=redact_pii, token=redact_token)
    if not q or not a:
        return None

    source_text = format_sources(sources, redact_pii=redact_pii, redact_token=redact_token)
    system = (
        "You are Edith, a grounded research assistant. "
        "If facts are unsupported by SOURCES, answer exactly: Not found in sources. "
        "Preserve citation labels like [S1], [S2] when evidence is used."
    )
    user = f"QUESTION:\n{q}\n\nSOURCES:\n{source_text if source_text else '(none)'}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": a},
        ]
    }


def collect_examples(
    chat_dir: Path,
    max_examples: int,
    include_refusals: bool,
    redact_pii: bool,
    redact_token: str,
    allowed_run_ids=None,
):
    cipher = load_cipher(chat_dir)
    files = sorted(list(chat_dir.glob("*.json")) + list(chat_dir.glob("*.json.enc")))
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    out = []
    for path in files:
        chat = read_chat(path, cipher)
        if not isinstance(chat, dict):
            continue
        msgs = chat.get("messages") or []
        if not isinstance(msgs, list):
            continue

        for idx, msg in enumerate(msgs):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            run_id = str(msg.get("run_id") or "").strip()
            if isinstance(allowed_run_ids, set):
                if not run_id or run_id not in allowed_run_ids:
                    continue
            answer = (msg.get("text") or "").strip()
            if not answer:
                continue
            sources = msg.get("sources") or []
            refusal = answer.lower() == "not found in sources."
            if not sources and not (include_refusals and refusal):
                continue

            query = find_assistant_query(msgs, idx)
            ex = build_example(query, answer, sources, redact_pii=redact_pii, redact_token=redact_token)
            if ex:
                out.append(ex)
                if len(out) >= max_examples:
                    return out
    return out


def load_feedback_run_filter(chat_dir: Path, only_positive_feedback: bool):
    if not only_positive_feedback:
        return None
    db_path = chat_dir.parent / "feedback.sqlite3"
    if not db_path.exists():
        return set()
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                run_id,
                SUM(CASE WHEN feedback_type='answer' AND value>0 THEN 1 ELSE 0 END) AS good_answer,
                SUM(CASE WHEN feedback_type='answer' AND value<0 THEN 1 ELSE 0 END) AS bad_answer,
                SUM(CASE WHEN feedback_type='should_refuse' THEN 1 ELSE 0 END) AS should_refuse,
                SUM(CASE WHEN feedback_type='sources' AND value<0 THEN 1 ELSE 0 END) AS bad_sources,
                SUM(CASE WHEN feedback_type='bad_citation' THEN 1 ELSE 0 END) AS bad_citation
            FROM feedback_events
            WHERE run_id IS NOT NULL AND run_id <> ''
            GROUP BY run_id
            """
        )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return set()

    allowed = set()
    for run_id, good_answer, bad_answer, should_refuse, bad_sources, bad_citation in rows:
        if int(good_answer or 0) < 1:
            continue
        if int(bad_answer or 0) > 0:
            continue
        if int(should_refuse or 0) > 0:
            continue
        if int(bad_sources or 0) > 1:
            continue
        if int(bad_citation or 0) > 0:
            continue
        allowed.add(str(run_id))
    return allowed


def split_examples(examples, val_ratio: float):
    keyed = []
    for ex in examples:
        k = hashlib.sha256(json.dumps(ex, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        keyed.append((k, ex))
    keyed.sort(key=lambda x: x[0])

    every = max(2, int(round(1.0 / max(0.001, float(val_ratio)))))
    train = []
    val = []
    for i, (_, ex) in enumerate(keyed):
        if (i + 1) % every == 0:
            val.append(ex)
        else:
            train.append(ex)
    if not train and val:
        train.append(val.pop(0))
    return train, val


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def example_question(example: dict):
    try:
        user = str((example.get("messages") or [])[1].get("content") or "")
    except Exception:
        return ""
    m = re.search(r"QUESTION:\s*(.*?)\n\s*\nSOURCES:", user, flags=re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    return user.strip()


def classify_curriculum_stage(example: dict):
    q = example_question(example).lower()
    if any(k in q for k in ["production", "deploy", "deployment", "monitor", "rollout", "risk"]):
        return "production"
    if any(k in q for k in ["where did i say", "where is", "define", "definition", "quote", "verbatim"]):
        return "recall"
    return "synthesis"


def write_curriculum_files(train_out: Path, train_rows):
    buckets = {"recall": [], "synthesis": [], "production": []}
    for ex in train_rows or []:
        stage = classify_curriculum_stage(ex)
        buckets.setdefault(stage, []).append(ex)

    prefix = train_out.with_suffix("")
    paths = {
        "recall": Path(f"{prefix}_recall.jsonl"),
        "synthesis": Path(f"{prefix}_synthesis.jsonl"),
        "production": Path(f"{prefix}_production.jsonl"),
        "curriculum": Path(f"{prefix}_curriculum.jsonl"),
    }
    write_jsonl(paths["recall"], buckets["recall"])
    write_jsonl(paths["synthesis"], buckets["synthesis"])
    write_jsonl(paths["production"], buckets["production"])
    merged = list(buckets["recall"]) + list(buckets["synthesis"]) + list(buckets["production"])
    write_jsonl(paths["curriculum"], merged)
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "counts": {
            "recall": len(buckets["recall"]),
            "synthesis": len(buckets["synthesis"]),
            "production": len(buckets["production"]),
            "curriculum_total": len(merged),
        },
    }


def main() -> int:
    args = parse_args()
    chat_dir = resolve_chat_dir(args.chat_dir)
    if not chat_dir.exists():
        raise SystemExit(f"Chat directory not found: {chat_dir}")

    allowed_run_ids = load_feedback_run_filter(chat_dir, only_positive_feedback=bool(args.only_positive_feedback))
    examples = collect_examples(
        chat_dir=chat_dir,
        max_examples=max(1, int(args.max_examples)),
        include_refusals=bool(args.include_refusals),
        redact_pii=bool(args.redact_pii),
        redact_token=str(args.redact_token or "[REDACTED]"),
        allowed_run_ids=allowed_run_ids,
    )
    train, val = split_examples(examples, val_ratio=args.val_ratio)

    train_out = Path(args.train_out).expanduser().resolve()
    val_out = Path(args.val_out).expanduser().resolve()
    write_jsonl(train_out, train)
    write_jsonl(val_out, val)
    curriculum_summary = {}
    if bool(args.curriculum):
        curriculum_summary = write_curriculum_files(train_out, train)

    print(json.dumps({
        "chat_dir": str(chat_dir),
        "examples_total": len(examples),
        "feedback_filter_enabled": bool(args.only_positive_feedback),
        "allowed_run_ids": len(allowed_run_ids) if isinstance(allowed_run_ids, set) else None,
        "train_count": len(train),
        "val_count": len(val),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "curriculum": curriculum_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

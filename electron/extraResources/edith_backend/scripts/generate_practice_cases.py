#!/usr/bin/env python3
"""
Generate unattended "practice conversation" eval cases from local documents.

Output format matches eval/run.py JSONL case schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx"}
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__"}
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "about",
    "this",
    "that",
    "paper",
    "chapter",
    "draft",
    "final",
    "version",
    "notes",
    "doc",
    "document",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic practice cases from local docs.")
    parser.add_argument("--docs-root", default="", help="Override EDITH_DATA_ROOT")
    parser.add_argument("--out", default="eval/generated/practice_cases.jsonl", help="Output JSONL path")
    parser.add_argument("--max-docs", type=int, default=18, help="Max documents to sample")
    parser.add_argument("--mode", default="Files only", choices=["Files only", "Web only", "Files + Web"])
    parser.add_argument("--backend", default="chroma", choices=["google", "chroma"])
    parser.add_argument("--include-adversarial", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_docs_root(override: str) -> Path:
    raw = (override or "").strip()
    if not raw:
        raw = (os.getenv("EDITH_DATA_ROOT") or "").strip()
    if not raw:
        raw = str(project_root() / "data")
    return Path(raw).expanduser().resolve()


def clean_title(name: str) -> str:
    stem = Path(name).stem
    text = re.sub(r"[_\-]+", " ", stem)
    text = re.sub(r"\s+", " ", text).strip()
    return text or stem


def title_keywords(title: str, max_terms: int = 4) -> list[str]:
    tokens = []
    for tok in re.split(r"[^A-Za-z0-9]+", title.lower()):
        t = tok.strip()
        if len(t) < 3 or t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        tokens.append(t)
    out = []
    seen = set()
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def iter_docs(root: Path):
    if not root.exists():
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            p = Path(base) / fn
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
            yield p


def collect_docs(root: Path, max_docs: int) -> list[dict]:
    rows = []
    for p in iter_docs(root):
        try:
            rel = str(p.relative_to(root))
        except Exception:
            rel = p.name
        title = clean_title(p.name)
        kws = title_keywords(title)
        hint = kws[0] if kws else Path(p).stem.lower()
        rows.append({"path": p, "rel": rel, "title": title, "hint": hint, "keywords": kws})
    rows.sort(key=lambda r: r["rel"].lower())
    return rows[: max(1, int(max_docs))]


def add_case(rows: list[dict], payload: dict):
    p = dict(payload)
    p.setdefault("expected_sources", [])
    rows.append(p)


def build_cases(docs: list[dict], mode: str, backend: str, include_adversarial: bool) -> list[dict]:
    cases = []
    next_id = 1

    for doc in docs:
        hint = doc["hint"]
        title = doc["title"]

        add_case(
            cases,
            {
                "id": f"practice_{next_id:03d}_definition",
                "query": f'In "{title}", list the key definitions or terms and cite the source.',
                "expected_refusal": False,
                "expected_sources": [hint],
                "mode": mode,
                "backend": backend,
                "category": "definition_finder",
            },
        )
        next_id += 1

        add_case(
            cases,
            {
                "id": f"practice_{next_id:03d}_method",
                "query": f'According to "{title}", summarize method steps or procedures with citations per step.',
                "expected_refusal": False,
                "expected_sources": [hint],
                "mode": mode,
                "backend": backend,
                "category": "method_recall",
            },
        )
        next_id += 1

        add_case(
            cases,
            {
                "id": f"practice_{next_id:03d}_production",
                "query": (
                    f'Using "{title}", provide three sections: From sources (cited), '
                    "Proposed plan (clearly labeled), and Risks/unknowns."
                ),
                "expected_refusal": False,
                "expected_sources": [hint],
                "mode": mode,
                "backend": backend,
                "category": "production_translation",
            },
        )
        next_id += 1

    for i in range(0, max(0, len(docs) - 1), 2):
        left = docs[i]
        right = docs[i + 1]
        add_case(
            cases,
            {
                "id": f"practice_{next_id:03d}_cross",
                "query": (
                    f'Compare how "{left["title"]}" and "{right["title"]}" treat the same problem. '
                    "List key differences with citations."
                ),
                "expected_refusal": False,
                "expected_sources": [left["hint"], right["hint"]],
                "mode": mode,
                "backend": backend,
                "category": "cross_document_synthesis",
            },
        )
        next_id += 1

    if include_adversarial:
        add_case(
            cases,
            {
                "id": f"practice_{next_id:03d}_adversarial_gpu",
                "query": "What exact GPU model was used in the experiments? Answer only from sources.",
                "expected_refusal": True,
                "expected_sources": [],
                "mode": mode,
                "backend": backend,
                "category": "hallucination_trap",
            },
        )
        next_id += 1
        add_case(
            cases,
            {
                "id": f"practice_{next_id:03d}_adversarial_budget",
                "query": "What is the project budget amount in USD? Answer only from sources.",
                "expected_refusal": True,
                "expected_sources": [],
                "mode": mode,
                "backend": backend,
                "category": "hallucination_trap",
            },
        )

    return cases


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    if load_dotenv:
        root_env = project_root() / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=root_env, override=False)

    args = parse_args()
    docs_root = resolve_docs_root(args.docs_root)
    docs = collect_docs(docs_root, max_docs=args.max_docs)
    cases = build_cases(docs, mode=args.mode, backend=args.backend, include_adversarial=bool(args.include_adversarial))

    out_path = Path(args.out).expanduser().resolve()
    write_jsonl(out_path, cases)

    print(
        json.dumps(
            {
                "docs_root": str(docs_root),
                "docs_used": len(docs),
                "cases_written": len(cases),
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

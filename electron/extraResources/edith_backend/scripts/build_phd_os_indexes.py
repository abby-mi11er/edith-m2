#!/usr/bin/env python3
"""
Build PhD-OS knowledge artifacts from local files:
- glossary_graph.json
- citation_graph.json
- chapter_anchors.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None


SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx", ".bib", ".tex"}
IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PhD OS glossary/citation/chapter artifacts")
    p.add_argument("--docs-root", default="", help="Override EDITH_DATA_ROOT")
    p.add_argument("--app-data-dir", default="", help="Override EDITH_APP_DATA_DIR")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--max-chars-per-doc", type=int, default=120000)
    return p.parse_args()


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


def clean_text(text: str):
    s = str(text or "")
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def iter_docs(root: Path):
    if not root.exists():
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            p = Path(base) / fn
            if p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def read_doc_text(path: Path, max_chars: int):
    ext = path.suffix.lower()
    text = ""
    if ext in {".txt", ".md", ".bib", ".tex"}:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
    elif ext == ".pdf" and PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            parts = []
            for p in reader.pages[:120]:
                parts.append(p.extract_text() or "")
            text = "\n".join(parts)
        except Exception:
            text = ""
    elif ext == ".docx" and Document is not None:
        try:
            d = Document(str(path))
            text = "\n".join([x.text for x in d.paragraphs])
        except Exception:
            text = ""
    return clean_text(text)[: max(1000, int(max_chars))]


def infer_chapter(rel_path: str):
    rp = str(rel_path or "").lower()
    m = re.search(r"(chapter[\s_-]*\d+|ch[\s_-]*\d+)", rp)
    if m:
        return re.sub(r"[\s_-]+", " ", m.group(1)).strip().title()
    parts = Path(rel_path).parts
    if parts:
        first = parts[0].strip()
        if first:
            return first
    return "General"


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [clean_text(x) for x in parts if clean_text(x)]


def extract_terms_and_defs(text: str):
    terms = {}
    acronyms = {}
    synonyms = defaultdict(set)
    equations = []

    for m in re.finditer(r"\b([A-Z][A-Za-z0-9\- ]{3,90})\s+\(([A-Z]{2,12})\)", text):
        long_name = clean_text(m.group(1))
        ac = clean_text(m.group(2))
        if long_name and ac:
            acronyms[ac] = long_name
            terms.setdefault(long_name, {"definition": "", "synonyms": set(), "acronyms": set(), "equations": []})
            terms[long_name]["acronyms"].add(ac)

    for m in re.finditer(
        r"\b([A-Z][A-Za-z0-9\- ]{2,90})\s+(?:is|are|refers to|means)\s+([^.;]{20,260})",
        text,
        flags=re.I,
    ):
        term = clean_text(m.group(1))
        definition = clean_text(m.group(2))
        if not term or not definition:
            continue
        t = terms.setdefault(term, {"definition": "", "synonyms": set(), "acronyms": set(), "equations": []})
        if not t["definition"] or len(definition) < len(t["definition"]):
            t["definition"] = definition

    for m in re.finditer(r"\b([A-Z][A-Za-z0-9\- ]{2,80})\s+\((?:also called|aka|also known as)\s+([^)]+)\)", text, flags=re.I):
        a = clean_text(m.group(1))
        b = clean_text(m.group(2))
        if a and b:
            synonyms[a].add(b)
            synonyms[b].add(a)

    for m in re.finditer(r"([A-Za-z][A-Za-z0-9_]{0,20}\s*=\s*[^.;]{3,120})", text):
        eq = clean_text(m.group(1))
        if eq and eq not in equations:
            equations.append(eq)
        if len(equations) >= 20:
            break

    return terms, acronyms, synonyms, equations


def extract_intext_citations(text: str):
    cites = []
    patterns = [
        r"\(([A-Z][A-Za-z'`\-]+(?:\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?)\)",
        r"\b([A-Z][A-Za-z'`\-]+(?:\s+et al\.)?)\s*\((?:19|20)\d{2}[a-z]?\)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            c = clean_text(m.group(1))
            if c:
                cites.append(c)
    out = []
    for c in cites:
        if c not in out:
            out.append(c)
    return out[:120]


def split_authors(raw: str):
    text = clean_text(raw or "")
    if not text:
        return []
    parts = re.split(r"\s+and\s+|;", text, flags=re.I)
    out = []
    for p in parts:
        name = clean_text(p).strip(",")
        if name and name not in out:
            out.append(name)
    return out


def parse_bibtex_entries(text: str):
    """
    Lightweight BibTeX parser suitable for structured bibliography indexing.
    """
    raw = str(text or "")
    entries = []
    i = 0
    n = len(raw)
    while i < n:
        at = raw.find("@", i)
        if at < 0:
            break
        j = at + 1
        while j < n and raw[j].isalpha():
            j += 1
        entry_type = clean_text(raw[at + 1 : j]).lower()
        if not entry_type:
            i = at + 1
            continue
        while j < n and raw[j].isspace():
            j += 1
        if j >= n or raw[j] != "{":
            i = at + 1
            continue
        j += 1
        key_start = j
        while j < n and raw[j] not in {",", "}"}:
            j += 1
        cite_key = clean_text(raw[key_start:j]).strip(",")
        if j >= n or raw[j] != ",":
            i = at + 1
            continue
        j += 1
        depth = 1
        body_start = j
        while j < n and depth > 0:
            ch = raw[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        body = raw[body_start : max(body_start, j - 1)]
        fields = {}
        for fm in re.finditer(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*([{\"\n])", body):
            fname = clean_text(fm.group(1)).lower()
            start = fm.end()
            if not fname:
                continue
            opener = fm.group(2)
            if opener == "{":
                k = start
                d = 1
                while k < len(body) and d > 0:
                    if body[k] == "{":
                        d += 1
                    elif body[k] == "}":
                        d -= 1
                    k += 1
                value = body[start : max(start, k - 1)]
            else:
                k = start
                while k < len(body):
                    if body[k] == '"' and body[k - 1] != "\\":
                        break
                    k += 1
                value = body[start:k]
            fields[fname] = clean_text(value).strip(",")
        entries.append(
            {
                "entry_type": entry_type,
                "cite_key": cite_key,
                "title": clean_text(fields.get("title", "")),
                "authors": split_authors(fields.get("author", "")),
                "year": clean_text(fields.get("year", "")),
                "venue": clean_text(fields.get("journal", "") or fields.get("booktitle", "")),
                "keywords": [k.strip() for k in re.split(r",|;", fields.get("keywords", "")) if k.strip()],
                "fields": fields,
            }
        )
        i = j
    return entries


def parse_reference_lines(text: str):
    refs = []
    for m in re.finditer(
        r"\b([A-Z][A-Za-z'`\-]+(?:\s+et al\.)?)\s*[,(]?\s*((?:19|20)\d{2}[a-z]?)\)?[.,]?\s+([^.;]{10,220})",
        text or "",
    ):
        author = clean_text(m.group(1))
        year = clean_text(m.group(2))
        title = clean_text(m.group(3))
        if title and len(title) > 140:
            title = title[:140].rstrip() + "..."
        refs.append(
            {
                "cite_key": "",
                "title": title,
                "authors": [author] if author else [],
                "year": year,
                "venue": "",
                "keywords": [],
                "entry_type": "reference_line",
                "fields": {},
            }
        )
        if len(refs) >= 300:
            break
    return refs


def normalize_bibliography_key(entry: dict):
    cite_key = clean_text(entry.get("cite_key", ""))
    if cite_key:
        return cite_key.lower()
    authors = entry.get("authors") or []
    lead = clean_text(authors[0] if authors else "").split(" ")[0].lower() if authors else "unknown"
    year = clean_text(entry.get("year", "")).lower()
    title = clean_text(entry.get("title", "")).lower()
    head = re.sub(r"[^a-z0-9]+", "_", title)[:40].strip("_")
    return "_".join([x for x in [lead, year, head] if x])


def author_year_key(author: str, year: str):
    a = clean_text(author or "")
    y = clean_text(year or "")
    surname = re.split(r"[\s,;]+", a)[0] if a else ""
    return f"{surname.lower()}_{y.lower()}" if surname and y else ""


def build_bibliography_db(docs: list[dict], citation_edges: list[dict]):
    by_key = {}
    source_docs = {}
    author_year_index = {}
    for d in docs:
        rel = d.get("rel_path", "")
        text = d.get("text", "")
        ext = Path(rel).suffix.lower()
        entries = []
        if ext in {".bib", ".tex"}:
            entries.extend(parse_bibtex_entries(text))
        if not entries:
            entries.extend(parse_reference_lines(text))
        for entry in entries:
            key = normalize_bibliography_key(entry)
            if not key:
                continue
            row = by_key.setdefault(
                key,
                {
                    "id": key,
                    "cite_key": clean_text(entry.get("cite_key", "")),
                    "title": clean_text(entry.get("title", "")),
                    "authors": [],
                    "year": clean_text(entry.get("year", "")),
                    "venue": clean_text(entry.get("venue", "")),
                    "keywords": [],
                    "entry_type": clean_text(entry.get("entry_type", "")),
                    "source_docs": [],
                    "citation_mentions": 0,
                    "mentioned_in_docs": [],
                },
            )
            if not row["title"] and entry.get("title"):
                row["title"] = clean_text(entry.get("title", ""))
            if not row["year"] and entry.get("year"):
                row["year"] = clean_text(entry.get("year", ""))
            if not row["venue"] and entry.get("venue"):
                row["venue"] = clean_text(entry.get("venue", ""))
            for a in entry.get("authors") or []:
                name = clean_text(a)
                if name and name not in row["authors"]:
                    row["authors"].append(name)
            for kw in entry.get("keywords") or []:
                tag = clean_text(kw)
                if tag and tag.lower() not in {x.lower() for x in row["keywords"]}:
                    row["keywords"].append(tag)
            if rel and rel not in row["source_docs"]:
                row["source_docs"].append(rel)
            source_docs[key] = row["source_docs"]
            if row["authors"] and row.get("year"):
                ay = author_year_key(row["authors"][0], row["year"])
                if ay and ay not in author_year_index:
                    author_year_index[ay] = key

    for edge in citation_edges:
        cit = clean_text(edge.get("citation", ""))
        src_doc = clean_text(edge.get("source_doc", ""))
        if not cit:
            continue
        m = re.match(r"([A-Z][A-Za-z'`\-]+).*?((?:19|20)\d{2}[a-z]?)", cit)
        key = ""
        if m:
            ay = author_year_key(m.group(1), m.group(2))
            key = author_year_index.get(ay, "")
        if not key:
            # fallback fuzzy: first record with citation token in title/authors
            low = cit.lower()
            for rid, row in by_key.items():
                hay = " ".join([row.get("title", ""), " ".join(row.get("authors") or [])]).lower()
                if hay and (low.split(",")[0] in hay):
                    key = rid
                    break
        if key and key in by_key:
            row = by_key[key]
            row["citation_mentions"] = int(row.get("citation_mentions", 0)) + 1
            if src_doc and src_doc not in row["mentioned_in_docs"]:
                row["mentioned_in_docs"].append(src_doc)

    records = list(by_key.values())
    records.sort(key=lambda x: (-(int(x.get("citation_mentions", 0))), x.get("year", ""), x.get("title", "")))
    return {
        "generated_at": datetime_utc(),
        "records": records[:4000],
    }


def infer_doc_year(rel_path: str, text: str):
    m = re.search(r"(19|20)\d{2}", rel_path or "")
    if m:
        return m.group(0)
    m = re.search(r"\b(19|20)\d{2}\b", text or "")
    if m:
        return m.group(0)
    return ""


def extract_entity_events(text: str, rel_path: str, chapter: str):
    events = []
    sentences = split_sentences(text or "")
    year = infer_doc_year(rel_path, text)
    dataset_rx = re.compile(r"\b([A-Z][A-Za-z0-9_.\-]{1,40}\s+(?:dataset|corpus|benchmark))\b")
    system_rx = re.compile(r"\b([A-Z][A-Za-z0-9_.\-]{1,40}\s+(?:system|framework|pipeline|model))\b")
    version_rx = re.compile(r"\b([A-Za-z][A-Za-z0-9_.\-]{1,30}\s+v(?:ersion)?\s*\d+(?:\.\d+)*)\b", re.I)
    collab_name_rx = re.compile(r"\b([A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){1,2})\b")
    for idx, sent in enumerate(sentences):
        low = sent.lower()
        for rx, etype in ((dataset_rx, "dataset"), (system_rx, "system"), (version_rx, "version")):
            for m in rx.finditer(sent):
                entity = clean_text(m.group(1))
                if not entity:
                    continue
                events.append(
                    {
                        "entity": entity,
                        "entity_type": etype,
                        "year": year,
                        "chapter": chapter,
                        "source_doc": rel_path,
                        "sentence_index": idx,
                        "snippet": sent[:260],
                    }
                )
        if any(k in low for k in ["collaborat", "coauthor", "advisor", "with "]):
            for m in collab_name_rx.finditer(sent):
                entity = clean_text(m.group(1))
                if not entity:
                    continue
                events.append(
                    {
                        "entity": entity,
                        "entity_type": "collaborator",
                        "year": year,
                        "chapter": chapter,
                        "source_doc": rel_path,
                        "sentence_index": idx,
                        "snippet": sent[:260],
                    }
                )
    # de-dupe
    seen = set()
    out = []
    for e in events:
        key = (
            e.get("entity", "").lower(),
            e.get("entity_type", ""),
            e.get("year", ""),
            e.get("source_doc", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out[:400]


def build_entity_timeline(docs: list[dict]):
    events = []
    for d in docs:
        events.extend(extract_entity_events(d.get("text", ""), d.get("rel_path", ""), d.get("chapter", "General")))
    events = events[:10000]
    by_entity = defaultdict(list)
    for e in events:
        by_entity[e.get("entity", "unknown")].append(e)
    entities = []
    for name, rows in by_entity.items():
        rows.sort(key=lambda r: (r.get("year", ""), r.get("source_doc", ""), int(r.get("sentence_index", 0))))
        entities.append(
            {
                "entity": name,
                "entity_type": rows[0].get("entity_type", "entity"),
                "introduced_year": rows[0].get("year", ""),
                "introduced_in": rows[0].get("source_doc", ""),
                "event_count": len(rows),
            }
        )
    entities.sort(key=lambda x: (x.get("introduced_year", ""), x.get("entity", "")))
    return {
        "generated_at": datetime_utc(),
        "events": events,
        "entities": entities[:4000],
    }


def sentence_samples(sentences, needles, limit=3):
    out = []
    for s in sentences:
        low = s.lower()
        if any(n in low for n in needles):
            out.append(s)
            if len(out) >= limit:
                break
    return out


def extract_claim_inventory(doc_text: str, rel_path: str, chapter: str, max_claims: int = 40):
    sentences = split_sentences(doc_text or "")
    claim_markers = [
        "we show",
        "we find",
        "we demonstrate",
        "we propose",
        "we argue",
        "our results",
        "hypothesis",
        "assume",
        "assumption",
        "significant",
        "improved",
        "decrease",
        "increase",
    ]
    caveat_markers = ["limitation", "caveat", "however", "uncertain", "future work", "threat"]
    claims = []
    for idx, sent in enumerate(sentences):
        low = sent.lower()
        if len(sent) < 24:
            continue
        if not any(m in low for m in claim_markers):
            continue
        caveats = []
        if any(m in low for m in caveat_markers):
            caveats.append(sent)
        if idx + 1 < len(sentences):
            nxt = sentences[idx + 1]
            if any(m in nxt.lower() for m in caveat_markers):
                caveats.append(nxt)
        claims.append(
            {
                "claim": sent,
                "where": {"doc": rel_path, "chapter": chapter, "sentence_index": idx},
                "evidence": [sent],
                "caveats": caveats[:2],
            }
        )
        if len(claims) >= max_claims:
            break
    return claims


def extract_experiment_rows(doc_text: str, rel_path: str, chapter: str, max_rows: int = 60):
    sentences = split_sentences(doc_text or "")
    rows = []
    date_rx = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+20\d{2})\b", re.I)
    param_rx = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{1,24})\s*=\s*([A-Za-z0-9._:-]{1,32})\b")
    for idx, sent in enumerate(sentences):
        low = sent.lower()
        if not any(k in low for k in ["experiment", "run", "trial", "ablation", "epoch", "baseline"]):
            continue
        name = ""
        m = re.search(r"\b(experiment|run|trial)\s*[:#-]?\s*([A-Za-z0-9_.-]{2,60})", sent, flags=re.I)
        if m:
            name = clean_text(m.group(2))
        if not name:
            name = f"{Path(rel_path).stem}-row-{idx}"
        params = []
        for pm in param_rx.finditer(sent):
            params.append({"key": clean_text(pm.group(1)), "value": clean_text(pm.group(2))})
            if len(params) >= 8:
                break
        date_match = date_rx.search(sent)
        result_note = ""
        if any(k in low for k in ["accuracy", "f1", "auc", "rmse", "loss", "improved", "decrease", "increase"]):
            result_note = sent
        elif idx + 1 < len(sentences):
            nxt = sentences[idx + 1]
            nxt_low = nxt.lower()
            if any(k in nxt_low for k in ["accuracy", "f1", "auc", "rmse", "loss", "improved", "decrease", "increase"]):
                result_note = nxt
        rows.append(
            {
                "experiment": name,
                "date": clean_text(date_match.group(1)) if date_match else "",
                "parameters": params,
                "result": result_note,
                "source_doc": rel_path,
                "chapter": chapter,
                "sentence_index": idx,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def main() -> int:
    load_env()
    args = parse_args()
    docs_root = Path(args.docs_root).expanduser().resolve() if args.docs_root else Path(
        os.getenv("EDITH_DATA_ROOT", "")
    ).expanduser().resolve()
    if not docs_root.exists():
        raise SystemExit(f"EDITH_DATA_ROOT missing: {docs_root}")

    app_data = Path(args.app_data_dir).expanduser().resolve() if args.app_data_dir else Path(
        os.getenv("EDITH_APP_DATA_DIR", str(Path.home() / "Library" / "Application Support" / "Edith"))
    ).expanduser().resolve()
    app_data.mkdir(parents=True, exist_ok=True)

    docs = []
    for p in iter_docs(docs_root):
        rel = str(p.relative_to(docs_root))
        text = read_doc_text(p, max_chars=int(args.max_chars_per_doc))
        if not text:
            continue
        docs.append({"path": str(p), "rel_path": rel, "chapter": infer_chapter(rel), "text": text})
        if len(docs) >= int(args.max_docs):
            break

    glossary_terms = {}
    acronym_map = {}
    cooc = Counter()
    citation_edges = []
    chapter_docs = defaultdict(list)
    claim_inventory = []
    experiment_rows = []

    for d in docs:
        text = d["text"]
        chapter = d["chapter"]
        chapter_docs[chapter].append(d)
        terms, acronyms, synonyms, equations = extract_terms_and_defs(text)
        for ac, long_name in acronyms.items():
            acronym_map[ac] = long_name
        for term, meta in terms.items():
            row = glossary_terms.setdefault(
                term,
                {
                    "term": term,
                    "definition": "",
                    "synonyms": [],
                    "acronyms": [],
                    "equations": [],
                    "introduced_in": chapter,
                    "mentions": 0,
                    "docs": [],
                },
            )
            if meta.get("definition") and (not row["definition"] or len(meta["definition"]) < len(row["definition"])):
                row["definition"] = meta["definition"]
            row["mentions"] += 1
            if d["rel_path"] not in row["docs"]:
                row["docs"].append(d["rel_path"])
            for ac in meta.get("acronyms", set()):
                if ac not in row["acronyms"]:
                    row["acronyms"].append(ac)
            for eq in meta.get("equations", []):
                if eq not in row["equations"]:
                    row["equations"].append(eq)
        for a, syns in synonyms.items():
            if a in glossary_terms:
                for s in syns:
                    if s not in glossary_terms[a]["synonyms"]:
                        glossary_terms[a]["synonyms"].append(s)

        doc_cites = extract_intext_citations(text)
        for c in doc_cites:
            citation_edges.append({"source_doc": d["rel_path"], "citation": c, "chapter": chapter})

        claim_inventory.extend(extract_claim_inventory(text, d["rel_path"], chapter, max_claims=30))
        experiment_rows.extend(extract_experiment_rows(text, d["rel_path"], chapter, max_rows=40))

        # Co-occurrence graph for top terms inside doc.
        local_terms = [t for t in glossary_terms.keys() if t.lower() in text.lower()]
        local_terms = local_terms[:50]
        for i in range(len(local_terms)):
            for j in range(i + 1, len(local_terms)):
                a = local_terms[i]
                b = local_terms[j]
                key = "||".join(sorted([a, b]))
                cooc[key] += 1

    glossary_nodes = sorted(
        glossary_terms.values(),
        key=lambda x: (-(x.get("mentions") or 0), x.get("term", "")),
    )[:1200]

    concept_edges = []
    for key, w in cooc.most_common(2000):
        a, b = key.split("||", 1)
        concept_edges.append({"a": a, "b": b, "weight": int(w)})

    citation_counts = Counter(e["citation"] for e in citation_edges)
    citation_nodes = [{"citation": c, "count": int(n)} for c, n in citation_counts.most_common(1500)]

    chapter_anchors = {}
    for chapter, rows in chapter_docs.items():
        merged = " ".join(r["text"][:40000] for r in rows)
        sents = split_sentences(merged)
        if not sents:
            continue
        chapter_anchors[chapter] = {
            "thesis": sentence_samples(sents, ["we", "this chapter", "argue", "show"], limit=2),
            "contributions": sentence_samples(sents, ["contribution", "we propose", "we present"], limit=3),
            "assumptions_limitations": sentence_samples(
                sents,
                ["assumption", "limitation", "threat", "caveat"],
                limit=4,
            ),
            "key_results": sentence_samples(sents, ["result", "improved", "increase", "decrease", "significant"], limit=4),
            "open_questions": sentence_samples(sents, ["future work", "open", "unknown", "next step"], limit=3),
            "doc_count": len(rows),
            "docs": [r["rel_path"] for r in rows][:50],
        }

    glossary_graph = {
        "generated_at": datetime_utc(),
        "docs_scanned": len(docs),
        "nodes": glossary_nodes,
        "acronym_map": acronym_map,
        "edges": concept_edges,
    }
    citation_graph = {
        "generated_at": datetime_utc(),
        "docs_scanned": len(docs),
        "citations": citation_nodes,
        "edges": citation_edges,
    }
    chapter_summary = {
        "generated_at": datetime_utc(),
        "docs_scanned": len(docs),
        "chapters": chapter_anchors,
    }
    bibliography_db = build_bibliography_db(docs, citation_edges)
    entity_timeline = build_entity_timeline(docs)
    claim_inventory = claim_inventory[:3000]
    cites_by_doc = defaultdict(list)
    for edge in citation_edges:
        doc = clean_text(edge.get("source_doc", ""))
        cit = clean_text(edge.get("citation", ""))
        if doc and cit and cit not in cites_by_doc[doc]:
            cites_by_doc[doc].append(cit)
    for claim in claim_inventory:
        where = claim.get("where") or {}
        doc = clean_text(where.get("doc", ""))
        claim["support_citations"] = cites_by_doc.get(doc, [])[:8]
    experiment_rows = experiment_rows[:2500]
    claim_inventory_obj = {
        "generated_at": datetime_utc(),
        "docs_scanned": len(docs),
        "claims": claim_inventory,
    }
    experiment_ledger_obj = {
        "generated_at": datetime_utc(),
        "docs_scanned": len(docs),
        "experiments": experiment_rows,
    }

    glossary_path = app_data / "glossary_graph.json"
    citation_path = app_data / "citation_graph.json"
    anchors_path = app_data / "chapter_anchors.json"
    claim_path = app_data / "claim_inventory.json"
    experiment_path = app_data / "experiment_ledger.json"
    bibliography_path = app_data / "bibliography_db.json"
    timeline_path = app_data / "entity_timeline.json"
    glossary_path.write_text(json.dumps(glossary_graph, indent=2, ensure_ascii=False), encoding="utf-8")
    citation_path.write_text(json.dumps(citation_graph, indent=2, ensure_ascii=False), encoding="utf-8")
    anchors_path.write_text(json.dumps(chapter_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    claim_path.write_text(json.dumps(claim_inventory_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment_path.write_text(json.dumps(experiment_ledger_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    bibliography_path.write_text(json.dumps(bibliography_db, indent=2, ensure_ascii=False), encoding="utf-8")
    timeline_path.write_text(json.dumps(entity_timeline, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "docs_scanned": len(docs),
                "glossary_nodes": len(glossary_nodes),
                "concept_edges": len(concept_edges),
                "citation_nodes": len(citation_nodes),
                "citation_edges": len(citation_edges),
                "chapters": len(chapter_anchors),
                "claims": len(claim_inventory),
                "experiment_rows": len(experiment_rows),
                "bibliography_records": len(bibliography_db.get("records") or []),
                "timeline_events": len(entity_timeline.get("events") or []),
                "glossary_graph": str(glossary_path),
                "citation_graph": str(citation_path),
                "chapter_anchors": str(anchors_path),
                "claim_inventory": str(claim_path),
                "experiment_ledger": str(experiment_path),
                "bibliography_db": str(bibliography_path),
                "entity_timeline": str(timeline_path),
            },
            indent=2,
        )
    )
    return 0


def datetime_utc():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())

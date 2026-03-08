"""
§LIB-ANALYTICS: Library Analytics Routes
Extracted from main.py — 9 endpoints
All endpoints read _library_cache from server.main.
"""
import json
import logging
import re
import time as _time
from pathlib import Path
from collections import Counter, defaultdict
from fastapi import APIRouter, Request

router = APIRouter(tags=["Library Analytics"])
log = logging.getLogger("edith.library_analytics")


def _get_cache():
    """Import library cache lazily; snapshot under lock to avoid torn reads."""
    from server.server_state import library_cache as _library_cache, library_lock as _library_lock
    with _library_lock:
        return list(_library_cache)


@router.get("/api/library/gaps")
async def literature_gaps():
    """Identify topics with few papers (literature gaps)."""
    _library_cache = _get_cache()
    topic_counts = Counter()
    method_counts = Counter()
    for d in _library_cache:
        t = d.get("academic_topic", "").strip()
        m = d.get("method", "").strip()
        if t: topic_counts[t] += 1
        if m: method_counts[m] += 1

    sparse_topics = [{"topic": t, "count": c} for t, c in topic_counts.items() if c < 3]
    sparse_methods = [{"method": m, "count": c} for m, c in method_counts.items() if c < 3]
    well_covered = [{"topic": t, "count": c} for t, c in topic_counts.most_common(10)]
    return {
        "sparse_topics": sorted(sparse_topics, key=lambda x: x["count"]),
        "sparse_methods": sorted(sparse_methods, key=lambda x: x["count"]),
        "well_covered": well_covered,
        "total_topics": len(topic_counts),
        "total_papers": len(_library_cache),
    }


@router.get("/api/library/method-audit")
async def method_audit():
    """Compare methods used across topics."""
    _library_cache = _get_cache()
    topic_methods = defaultdict(lambda: defaultdict(int))
    for d in _library_cache:
        t = d.get("academic_topic", "General")
        m = d.get("method", "")
        if m:
            topic_methods[t][m] += 1

    matrix = []
    for topic, methods in sorted(topic_methods.items()):
        matrix.append({
            "topic": topic,
            "methods": dict(methods),
            "dominant": max(methods, key=methods.get) if methods else "",
            "diversity": len(methods),
        })
    return {"matrix": matrix}


@router.get("/api/library/data-overlap")
async def data_overlap():
    """Find papers using the same datasets."""
    _library_cache = _get_cache()
    dataset_papers = defaultdict(list)
    datasets_kw = ["ANES", "WVS", "V-Dem", "Afrobarometer", "ESS", "CPS", "GSS",
                    "CCES", "World Bank", "UN", "LAPOP", "Eurobarometer", "Gallup"]
    for d in _library_cache:
        title = (d.get("title", "") + " " + d.get("academic_topic", "")).lower()
        for ds in datasets_kw:
            if ds.lower() in title:
                dataset_papers[ds].append({
                    "sha256": d.get("sha256", ""),
                    "title": d.get("title", ""),
                    "author": d.get("author", ""),
                })
    return {"datasets": {k: v for k, v in dataset_papers.items() if len(v) > 0}}


@router.get("/api/library/author-network")
async def author_network():
    """Map co-authorship across library."""
    _library_cache = _get_cache()
    coauthorship = defaultdict(set)
    author_papers = defaultdict(list)

    for d in _library_cache:
        raw = d.get("author", "")
        if not raw:
            continue
        authors = [a.strip() for a in re.split(r",|&|\band\b", raw) if a.strip() and len(a.strip()) > 2]
        for a in authors:
            author_papers[a].append(d.get("title", ""))
        for i, a1 in enumerate(authors):
            for a2 in authors[i+1:]:
                coauthorship[a1].add(a2)
                coauthorship[a2].add(a1)

    nodes = [{"id": a, "papers": len(p), "titles": p[:3]} for a, p in author_papers.items()]
    edges = []
    seen = set()
    for a, coauthors in coauthorship.items():
        for b in coauthors:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                edges.append({"source": a, "target": b})

    return {"nodes": nodes[:100], "edges": edges[:200], "total_authors": len(author_papers)}


@router.get("/api/library/temporal")
async def temporal_coverage():
    """Year-spread heatmap per topic."""
    _library_cache = _get_cache()
    topic_years = defaultdict(lambda: defaultdict(int))
    for d in _library_cache:
        t = d.get("academic_topic", "General")
        y = d.get("year", "")
        if y:
            try:
                topic_years[t][int(str(y)[:4])] += 1
            except (ValueError, TypeError):
                pass
    heatmap = []
    for topic, years in sorted(topic_years.items()):
        heatmap.append({
            "topic": topic,
            "years": dict(years),
            "range": [min(years.keys()), max(years.keys())] if years else [0, 0],
            "total": sum(years.values()),
        })
    return {"heatmap": heatmap}


@router.get("/api/library/cite")
async def citation_clipboard(sha256: str = ""):
    """One-click copy formatted citation."""
    _library_cache = _get_cache()
    doc = None
    for d in _library_cache:
        if d.get("sha256") == sha256:
            doc = d
            break
    if not doc:
        return {"citation": "", "error": "Not found"}

    author = doc.get("author", "Unknown Author")
    year = doc.get("year", "n.d.")
    title = doc.get("title", "Untitled")
    apa = f"{author} ({year}). {title}."
    chicago = f'{author}. "{title}." {year}.'
    first_author = author.split(",")[0].split(" ")[-1].lower() if author else "unknown"
    bibtex = f"@article{{{first_author}{year},\n  author = {{{author}}},\n  title = {{{title}}},\n  year = {{{year}}}\n}}"
    return {"apa": apa, "chicago": chicago, "bibtex": bibtex}


@router.post("/api/library/notes")
async def paper_notes_endpoint(request: Request):
    """Free-text notes per paper (stored server-side)."""
    import os
    body = await request.json()
    sha256 = body.get("sha256", "")
    note = body.get("note")
    action = body.get("action", "get")

    data_root = os.environ.get("EDITH_DATA_ROOT", ".")
    notes_file = Path(data_root) / ".edith_paper_notes.json"
    try:
        notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
    except Exception:
        notes = {}

    if action == "set" and note is not None:
        notes[sha256] = {"note": note, "updated": _time.time()}
        notes_file.write_text(json.dumps(notes, indent=2))
        return {"ok": True, "sha256": sha256}
    else:
        return {"sha256": sha256, "note": notes.get(sha256, {}).get("note", ""), "updated": notes.get(sha256, {}).get("updated")}


@router.get("/api/library/mission-history")
async def mission_history():
    """Return mission launch history from audit log."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", ".")
    history = []
    audit_file = Path(data_root) / ".edith_audit.jsonl"
    if audit_file.exists():
        try:
            for line in audit_file.read_text().strip().split("\n")[-50:]:
                entry = json.loads(line)
                if entry.get("event") == "activity" and entry.get("type") == "mission_launched":
                    history.append({"detail": entry.get("detail", ""), "ts": entry.get("ts", "")})
        except Exception:
            pass
    return {"history": history[-20:]}


@router.get("/api/suggestions")
async def smart_suggestions_endpoint(sha256: str = "", title: str = "", limit: int = 5):
    """Smart suggestions — given a paper, find related papers by topic, method, and author overlap."""
    _library_cache = _get_cache()
    if not sha256 and not title:
        return {"suggestions": [], "reason": "No paper specified"}

    target = None
    for doc in _library_cache:
        if sha256 and doc.get("sha256") == sha256:
            target = doc
            break
        if title and title.lower() in (doc.get("title") or "").lower():
            target = doc
            break

    if not target:
        return {"suggestions": [], "reason": "Paper not found in library"}

    target_topic = (target.get("academic_topic") or "").lower()
    target_method = (target.get("method") or "").lower()
    target_author = (target.get("author") or "").lower()
    target_class = (target.get("class") or "").lower()

    scored = []
    for doc in _library_cache:
        if doc.get("sha256") == target.get("sha256"):
            continue

        score = 0
        reasons = []

        doc_topic = (doc.get("academic_topic") or "").lower()
        if target_topic and doc_topic and target_topic in doc_topic or doc_topic in target_topic:
            score += 3
            reasons.append("same topic")

        doc_method = (doc.get("method") or "").lower()
        if target_method and doc_method and target_method in doc_method:
            score += 2
            reasons.append("same method")

        doc_class = (doc.get("class") or "").lower()
        if target_class and doc_class and target_class == doc_class:
            score += 1
            reasons.append("same subfield")

        doc_author = (doc.get("author") or "").lower()
        if target_author and doc_author:
            target_names = set(target_author.replace(",", " ").split())
            doc_names = set(doc_author.replace(",", " ").split())
            overlap = target_names & doc_names - {"and", "et", "al", "al."}
            if len(overlap) >= 1:
                score += 2
                reasons.append("shared author")

        if score > 0:
            scored.append({
                "sha256": doc.get("sha256", ""),
                "title": doc.get("title", "Untitled"),
                "author": doc.get("author", ""),
                "year": doc.get("year", ""),
                "score": score,
                "reasons": reasons,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"suggestions": scored[:limit], "target": target.get("title", "")}

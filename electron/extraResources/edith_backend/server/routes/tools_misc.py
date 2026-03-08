"""
§TOOLS: Miscellaneous Tool Routes
Extracted from main.py — flashcards, method decode, reclassify, activity log,
annotations search, chat followups/debate, export (bibtex, csv, overleaf), vault health,
metrics/routes
"""
import csv
import io
import json
import logging
import os
import time as _time
from pathlib import Path
from fastapi import APIRouter, Request
from starlette.responses import Response

router = APIRouter(tags=["Tools"])
log = logging.getLogger("edith.tools_misc")

# §FIX #3: Simple per-endpoint rate limiter for LLM calls
_llm_call_counts: dict[str, list[float]] = {}  # endpoint -> list of timestamps
_LLM_RATE_LIMIT = 10  # max calls per minute per endpoint
_LLM_RATE_WINDOW = 60  # seconds

def _check_llm_rate(endpoint: str) -> bool:
    """Returns True if under rate limit, False if over."""
    now = _time.time()
    if endpoint not in _llm_call_counts:
        _llm_call_counts[endpoint] = []
    # Prune old timestamps
    _llm_call_counts[endpoint] = [t for t in _llm_call_counts[endpoint] if now - t < _LLM_RATE_WINDOW]
    if len(_llm_call_counts[endpoint]) >= _LLM_RATE_LIMIT:
        return False
    _llm_call_counts[endpoint].append(now)
    return True


def _get_cache():
    from server.server_state import library_cache as _library_cache, library_lock as _library_lock
    with _library_lock:
        return list(_library_cache)


def _get_cache_state():
    from server.server_state import library_cache as _library_cache, library_lock as _library_lock, library_cache_ts as _library_cache_ts, library_building as _library_building
    with _library_lock:
        return list(_library_cache), _library_cache_ts, _library_building


# ── Metrics ──────────────────────────────────────────────────────────

@router.get("/api/metrics/routes")
async def route_metrics():
    """Live route call counts for the Dashboard."""
    from server.server_state import route_call_counts as _route_call_counts, route_call_start as _route_call_start
    from server.main import app as _app
    total_calls = sum(_route_call_counts.values())
    total_routes = len([r for r in _app.routes if hasattr(r, 'path')])
    sorted_routes = sorted(_route_call_counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "total_routes": total_routes,
        "total_calls": total_calls,
        "uptime_s": round(_time.time() - _route_call_start),
        "busiest": [
            {"path": path, "calls": count}
            for path, count in sorted_routes[:20]
        ],
        "connectors_wired": 16,
        "frontend_coverage": "100%",
    }


# ── Method Decode ────────────────────────────────────────────────────

@router.post("/api/method/decode")
async def method_decode_endpoint(request: Request):
    """Generic method decoder — works for ANY research methodology."""
    body = await request.json()
    paper_text = body.get("text", "")[:15000]
    method_hint = body.get("method", "")

    if not paper_text:
        return {"error": "No text provided"}
    if not _check_llm_rate("method_decode"):
        return {"error": "Rate limit exceeded — max 10 calls/minute", "analysis": {}}

    ANALYSIS_PROMPTS = {
        "conjoint": "Extract: (1) Attributes and Levels, (2) AMCE estimates if present, (3) Sample size, (4) Randomization checks, (5) Power analysis assessment",
        "rct": "Extract: (1) Treatment arms, (2) Control group, (3) Randomization method, (4) Sample size per arm, (5) ITT vs LATE analysis, (6) Attrition rates",
        "did": "Extract: (1) Treatment and control groups, (2) Pre/post periods, (3) Parallel trends evidence, (4) Treatment timing, (5) Staggered adoption if present",
        "rdd": "Extract: (1) Running variable, (2) Cutoff threshold, (3) Bandwidth selection, (4) McCrary test results, (5) Local polynomial order",
        "iv": "Extract: (1) Instrument(s), (2) First-stage F-statistic, (3) Exclusion restriction argument, (4) Endogenous variable(s)",
        "survey": "Extract: (1) Survey design, (2) Sampling strategy, (3) Response rate, (4) Weighting scheme, (5) Mode of administration",
        "matching": "Extract: (1) Matching method (PSM, CEM, etc.), (2) Covariates used, (3) Balance diagnostics, (4) Common support, (5) ATT/ATE estimates",
        "qualitative": "Extract: (1) Approach (case study, ethnography, etc.), (2) Case selection logic, (3) Data sources, (4) Coding scheme, (5) Triangulation methods",
        "theory": "Extract: (1) Core framework, (2) Key assumptions, (3) Causal mechanisms, (4) Scope conditions, (5) Testable hypotheses",
        "democratic_theory": "Extract: (1) Model of democracy, (2) Key normative claims, (3) Institutional implications, (4) Relationship to empirical findings",
        "rational_choice": "Extract: (1) Actors and preferences, (2) Strategic interaction, (3) Equilibrium concept, (4) Information assumptions, (5) Predictions vs evidence",
        "institutional": "Extract: (1) Type of institutionalism, (2) Key institutions, (3) Path dependence, (4) Change mechanisms",
        "dataset": "Extract: (1) Dataset name/source, (2) Unit of analysis, (3) Time coverage, (4) Geographic coverage, (5) Key variables, (6) Measurement validity, (7) Missing data",
        "coding": "Extract: (1) Coding scheme, (2) Inter-coder reliability, (3) Category definitions, (4) Decision rules",
        "writing": "Assess: (1) Argument structure, (2) Literature integration, (3) Evidence-claim alignment, (4) Prose quality, (5) Section organization, (6) Citation completeness",
        "introduction": "Assess: (1) Hook/motivation, (2) Research question clarity, (3) Contribution statement, (4) Roadmap, (5) Gap identification",
        "geographic": "Extract: (1) Countries/regions, (2) Subnational variation, (3) Cross-national logic, (4) Case selection, (5) Generalizability",
    }

    default_prompt = "Extract: (1) Research design, (2) Theoretical framework, (3) Key assumptions, (4) Sample/data, (5) Identification strategy, (6) Potential weaknesses"
    extraction_prompt = ANALYSIS_PROMPTS.get(method_hint.lower(), default_prompt)

    try:
        prompt = f"""Analyze this research paper excerpt and {extraction_prompt}.

Also assess:
- Statistical power: Is the sample size sufficient?
- Internal validity: Are the identification assumptions credible?
- External validity: How generalizable are the findings?

Paper text:
{paper_text[:10000]}

Respond as structured JSON with keys: method_type, identification_strategy, sample_info, power_assessment, validity_notes, key_findings, limitations."""

        import google.generativeai as genai
        _model = genai.GenerativeModel("gemini-2.0-flash")
        _resp = _model.generate_content(prompt)
        result = _resp.text
        try:
            parsed = json.loads(result.strip().strip("```json").strip("```"))
        except Exception:
            parsed = {"raw_analysis": result, "method_type": method_hint or "unknown"}

        return {"analysis": parsed, "method_hint": method_hint}
    except Exception as e:
        return {"error": str(e), "analysis": {"method_type": method_hint or "unknown"}}


# ── Reclassify ───────────────────────────────────────────────────────

@router.post("/api/reclassify")
async def reclassify_endpoint(request: Request):
    """Re-classify a paper and UPDATE its metadata in ChromaDB."""
    body = await request.json()
    sha256 = body.get("sha256", "")
    if not sha256:
        return {"ok": False, "error": "sha256 required"}

    _library_cache = _get_cache()
    doc = None
    for d in _library_cache:
        if d.get("sha256") == sha256:
            doc = d
            break
    if not doc:
        return {"ok": False, "error": "Paper not found"}
    if not _check_llm_rate("reclassify"):
        return {"ok": False, "error": "Rate limit exceeded — max 10 calls/minute"}

    try:
        from server.chroma_backend import _get_client
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if not chroma_dir:
            return {"ok": False, "error": "CHROMA_DIR not configured"}

        client = _get_client(chroma_dir)
        collection = client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})
        results = collection.get(where={"sha256": sha256}, include=["documents", "metadatas"], limit=5)

        if not results["documents"]:
            return {"ok": False, "error": "No chunks found for this paper"}

        text_sample = " ".join(results["documents"][:3])[:2500]

        import google.generativeai as genai
        prompt = f"""Re-classify this academic paper. Return ONLY a JSON object with:
- "topic": Primary academic topic/subfield
- "method": Primary research method ("Conjoint", "RCT", "DiD", "Survey", "Qualitative", etc. or "")
- "country": Primary country/region studied (or "")
- "theory": Main theoretical framework (or "")
- "doc_type": Document type ("empirical_paper", "theoretical_paper", "methods_paper", "review_article", etc.)

Title: {doc.get('title', 'Unknown')}
Author: {doc.get('author', 'Unknown')}

Text:
{text_sample}

Return ONLY valid JSON."""

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        new_class = json.loads(result_text.strip())

        chunk_ids = results["ids"]
        old_metas = results["metadatas"]
        updated_metas = []
        for meta in old_metas:
            m = dict(meta)
            for k in ("topic", "method", "country", "theory", "doc_type"):
                if new_class.get(k):
                    mkey = "academic_topic" if k == "topic" else k
                    m[mkey] = new_class[k]
            updated_metas.append(m)

        collection.update(ids=chunk_ids, metadatas=updated_metas)

        # Invalidate library cache
        from server.server_state import library_cache_ts
        import server.server_state as _ss
        _ss.library_cache_ts = 0

        log.info(f"§RECLASSIFY: Updated {len(chunk_ids)} chunks for {doc.get('title', sha256)}: {new_class}")
        return {
            "ok": True, "sha256": sha256, "title": doc.get("title", ""),
            "old": {"topic": doc.get("academic_topic", ""), "method": doc.get("method", ""), "country": doc.get("country", "")},
            "new": new_class, "chunks_updated": len(chunk_ids),
        }
    except ImportError as e:
        return {"ok": False, "error": f"Missing dependency: {e}"}
    except Exception as e:
        log.error(f"§RECLASSIFY: Error: {e}")
        return {"ok": False, "error": str(e)}


# ── Activity Log ─────────────────────────────────────────────────────

@router.post("/api/activity/log")
async def activity_log_endpoint(request: Request):
    """Log a research activity event from the frontend."""
    from server.security import audit
    body = await request.json()
    audit("activity", type=body.get("type", ""), detail=body.get("detail", ""), doc_sha=body.get("doc_sha", ""))
    return {"ok": True}


# ── Flashcards ───────────────────────────────────────────────────────

@router.get("/api/tools/flashcard/stats")
async def flashcard_stats_endpoint():
    """Return flashcard stats for the Research Pulse."""
    vault_root = os.environ.get("EDITH_VAULT_ROOT", "")
    flashcard_dir = Path(vault_root) / "Corpus" / "Vault" / "flashcards" if vault_root else None
    total = 0
    due = 0
    now = _time.time()

    if flashcard_dir and flashcard_dir.is_dir():
        for card_file in flashcard_dir.glob("*.json"):
            try:
                card = json.loads(card_file.read_text())
                total += 1
                created = card.get("created", "")
                interval = card.get("interval", 1)
                if created:
                    try:
                        from datetime import datetime as _dt
                        created_ts = _dt.fromisoformat(created).timestamp()
                        if now - created_ts > interval * 86400:
                            due += 1
                    except Exception:
                        due += 1
            except Exception:
                pass

    return {"total": total, "due": due}


@router.post("/api/tools/flashcard")
async def flashcard_endpoint(request: Request):
    """Auto-generate spaced repetition cards from Q&A pairs."""
    body = await request.json()
    question = body.get("question", "")
    answer = body.get("answer", "")
    source = body.get("source", "")

    if not question or not answer:
        return {"created": False, "reason": "Missing question or answer"}

    vault_root = os.environ.get("EDITH_VAULT_ROOT", "")
    flashcard_dir = Path(vault_root) / "Corpus" / "Vault" / "flashcards" if vault_root else Path(".") / "flashcards"
    flashcard_dir.mkdir(parents=True, exist_ok=True)

    import uuid
    card = {
        "id": f"card-{uuid.uuid4().hex[:12]}",
        "front": question[:500],
        "back": answer[:500],
        "source": source[:200],
        "created": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interval": 1,
        "ease": 2.5,
        "reviews": 0,
    }

    card_file = flashcard_dir / f"{card['id']}.json"
    card_file.write_text(json.dumps(card, indent=2))

    # Append to training data for Winnie
    try:
        train_file = Path(vault_root) / "Brain" / "edith_master_train.jsonl" if vault_root else None
        if train_file:
            with open(train_file, "a") as f:
                f.write(json.dumps({"type": "flashcard", "q": question[:500], "a": answer[:500], "source": source}) + "\n")
    except Exception:
        pass

    return {"created": True, "card_id": card["id"]}


# ── Annotations Search ───────────────────────────────────────────────

@router.get("/api/annotations/search")
async def annotation_search(q: str = ""):
    """Search across all annotations — stub for potential server-side search."""
    return {"results": [], "query": q, "note": "Annotations are stored in localStorage. Pass query to frontend search."}


# ── Chat Extras ──────────────────────────────────────────────────────

@router.post("/api/chat/followups")
async def chat_followups(request: Request):
    """Generate 3 follow-up question suggestions based on conversation context."""
    body = await request.json()
    last_response = body.get("response", "")[:500]
    query = body.get("query", "")[:200]
    try:
        if not _check_llm_rate("chat_followups"):
            return {"followups": []}
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-2.0-flash")
        r = model.generate_content(
            f"Based on this Q&A, suggest 3 concise follow-up questions (one per line, no numbering).\n"
            f"Q: {query}\nA: {last_response[:300]}"
        )
        lines = [l.strip().lstrip("0123456789.-) ") for l in r.text.strip().split("\n") if l.strip()]
        return {"followups": lines[:3]}
    except Exception:
        return {"followups": []}


@router.post("/api/chat/debate")
async def debate_detect(request: Request):
    """Flag when sources disagree with each other."""
    body = await request.json()
    sources = body.get("sources", [])
    disagreements = []
    markers = ["however", "contrary", "on the other hand", "disagrees", "mixed results",
               "no effect", "fails to find", "does not support", "inconsistent"]
    for i, s in enumerate(sources):
        text = (s.get("text", "") + s.get("content", "")).lower()
        if any(m in text for m in markers):
            disagreements.append({
                "source_idx": i,
                "title": s.get("title", f"Source {i+1}"),
                "flag": "potential_disagreement",
            })
    return {"has_debate": len(disagreements) > 0, "disagreements": disagreements}


# ── Export ────────────────────────────────────────────────────────────

@router.post("/api/export/overleaf")
async def overleaf_export_endpoint(request: Request):
    """Export session notes as clean LaTeX for Overleaf import."""
    body = await request.json()
    content = body.get("content", "")
    title = body.get("title", "E.D.I.T.H. Export")

    # Escape LaTeX special characters in user-provided title
    _latex_special = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                      '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                      '^': r'\textasciicircum{}'}
    safe_title = title
    for ch, esc in _latex_special.items():
        safe_title = safe_title.replace(ch, esc)

    latex_doc = f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath,amssymb,natbib,geometry,hyperref}}
\\geometry{{margin=1in}}
\\title{{{safe_title}}}
\\author{{Generated by E.D.I.T.H.}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

{content}

\\end{{document}}
"""
    return {"latex": latex_doc, "filename": f"{title[:30].replace(' ', '_')}.tex"}


@router.post("/api/export/bibtex")
async def bibtex_export_endpoint(request: Request):
    """Export documents as BibTeX format."""
    body = await request.json()
    docs = body.get("docs", [])
    if not docs:
        return {"bibtex": "% No documents to export\n"}

    entries = []
    for i, doc in enumerate(docs):
        title = doc.get("title", "Untitled")
        author = doc.get("author", "Unknown")
        year = doc.get("year", "n.d.")
        doc_type = doc.get("doc_type", "article")
        sha = doc.get("sha256", f"doc{i}")[:8]
        first_author = author.split(",")[0].split(" ")[-1] if author else "Unknown"
        key = f"{first_author}{year}_{sha}"

        bib_type = "article"
        if doc_type and "book" in doc_type.lower():
            bib_type = "book"
        elif doc_type and "report" in doc_type.lower():
            bib_type = "techreport"
        elif doc_type and "thesis" in doc_type.lower():
            bib_type = "phdthesis"

        entry = f"@{bib_type}{{{key},\n  title = {{{title}}},\n  author = {{{author}}},\n  year = {{{year}}}\n}}"
        entries.append(entry)

    return {"bibtex": "\n\n".join(entries), "count": len(entries)}


@router.get("/api/export/csv")
async def export_csv():
    """Export library metadata as CSV."""
    _library_cache = _get_cache()
    output = io.StringIO()
    fields = ["title", "author", "year", "academic_topic", "method", "country",
              "theory", "doc_type", "project", "sha256"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for d in _library_cache:
        writer.writerow({k: d.get(k, "") for k in fields})

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=edith_library.csv"},
    )


# ── Vault Health ─────────────────────────────────────────────────────

@router.get("/api/vault/health")
async def vault_health():
    """Vault health — index freshness + disk usage."""
    _library_cache, _library_cache_ts, _library_building = _get_cache_state()
    vault_root = os.environ.get("EDITH_VAULT_ROOT", "")

    result = {
        "vault_root": vault_root,
        "vault_exists": os.path.isdir(vault_root) if vault_root else False,
        "library_docs": len(_library_cache),
        "cache_fresh": (_time.time() - _library_cache_ts < 300) if _library_cache_ts else False,
        "building": _library_building,
    }

    if vault_root and os.path.isdir(vault_root):
        import asyncio

        def _scan_vault():
            file_count = 0
            total_size = 0
            ext_counts = {}
            for dp, _, fns in os.walk(vault_root):
                for f in fns:
                    fp = os.path.join(dp, f)
                    ext = os.path.splitext(f)[1].lower()
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass
                    file_count += 1
            return file_count, total_size, ext_counts

        file_count, total_size, ext_counts = await asyncio.to_thread(_scan_vault)
        result["vault_files"] = file_count
        result["vault_size_mb"] = round(total_size / (1024 * 1024), 1)
        result["file_types"] = ext_counts
        result["indexed_pct"] = round(len(_library_cache) / max(file_count, 1) * 100, 1)

    return result

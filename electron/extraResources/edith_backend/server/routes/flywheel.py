"""
Flywheel Enhancement Routes
=============================
Endpoints for the 10 additional Flywheel capabilities:
  #1 Oracle auto-pull, #2 Stata→LaTeX, #3 Focus propagation,
  #4 Training dashboard, #5 Auto-bibliography from notes,
  #6 Reading progress, #7 Citation chain walking,
  #9 Dataset reconciliation, #10 Confidence calibration
"""

import logging
import os
import re
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.routes.flywheel")
router = APIRouter()


def _data_root():
    return os.environ.get("DATA_ROOT", os.environ.get("EDITH_DATA_ROOT", ""))


# ═══════════════════════════════════════════════════════════════════
# #1 — Oracle → Library Auto-Pull
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/oracle/auto-pull", tags=["Oracle"])
async def oracle_auto_pull(request: Request):
    """Pull a paper detected by Oracle directly into the Library.

    Takes a paper URL or DOI from an Oracle alert, downloads/fetches it,
    and triggers speculative indexing.
    """
    body = await request.json()
    doi = body.get("doi", "")
    url = body.get("url", "")
    title = body.get("title", "")
    alert_id = body.get("alert_id", "")

    if not doi and not url:
        return JSONResponse(status_code=400,
                            content={"error": "Provide 'doi' or 'url' to auto-pull"})

    results = {"steps": [], "status": "ok"}
    sha = ""  # Initialize before conditional blocks to avoid NameError

    # Step 1: Try to fetch paper metadata from OpenAlex or CrossRef via DOI
    paper_text = ""
    if doi:
        try:
            import httpx
            # Try OpenAlex first
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"https://api.openalex.org/works/doi:{doi}")
                if resp.status_code == 200:
                    meta = resp.json()
                    title = title or meta.get("title", "")
                    abstract = meta.get("abstract_inverted_index", {})
                    if abstract and isinstance(abstract, dict):
                        # Reconstruct abstract from inverted index
                        word_positions = []
                        for word, positions in abstract.items():
                            for pos in positions:
                                word_positions.append((pos, word))
                        word_positions.sort()
                        paper_text = " ".join(w for _, w in word_positions)
                    results["steps"].append({"step": "openalex_fetch", "status": "ok", "title": title})
        except Exception as e:
            results["steps"].append({"step": "openalex_fetch", "status": "failed", "error": str(e)[:100]})

    # Step 2: Index the paper text if available
    if paper_text or title:
        try:
            chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
            if chroma_dir:
                from server.chroma_backend import _get_client
                client = _get_client(chroma_dir)
                collection = client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})
                import hashlib
                sha = hashlib.sha256((paper_text or title).encode()).hexdigest()
                collection.add(
                    ids=[sha[:16]],
                    documents=[paper_text or title],
                    metadatas=[{
                        "title": title, "doi": doi, "source": "oracle_auto_pull",
                        "sha256": sha, "indexed_at": time.time(),
                        "url": url,
                    }],
                )
                results["steps"].append({"step": "chroma_index", "status": "ok", "sha256": sha})
            else:
                results["steps"].append({"step": "chroma_index", "status": "skipped", "reason": "no EDITH_CHROMA_DIR"})
        except Exception as e:
            results["steps"].append({"step": "chroma_index", "status": "failed", "error": str(e)[:100]})

    # Step 3: Add to library cache
    try:
        from server.server_state import library_cache as _library_cache, library_lock as _library_lock
        with _library_lock:
            _library_cache.append({
                "title": title, "doi": doi, "sha256": sha if paper_text else "",
                "source": "oracle_auto_pull", "indexed_at": time.time(),
            })
        results["steps"].append({"step": "library_cache", "status": "ok"})
    except Exception as e:
        results["steps"].append({"step": "library_cache", "status": "failed", "error": str(e)[:100]})

    results["title"] = title
    results["doi"] = doi
    results["alert_id"] = alert_id
    return results


# ═══════════════════════════════════════════════════════════════════
# #2 — Stata → LaTeX Table Pipeline
# ═══════════════════════════════════════════════════════════════════

def _parse_stata_table(log_text: str) -> list[dict]:
    """Parse Stata regression output (.log/.smcl) into structured table data.

    Handles common Stata commands: regress, xtreg, logit, probit, ivregress.
    Returns list of tables, each with columns, rows, and stats.
    """
    tables = []

    # Clean SMCL tags if present
    log_text = re.sub(r'\{[^}]*\}', '', log_text)

    # Find regression output blocks
    reg_patterns = [
        r'(?:Linear regression|Logistic regression|IV.*regression|Probit regression|'
        r'Fixed-effects.*regression|Random-effects.*regression)'
        r'.*?(?=\n\n\n|\Z)',
    ]

    # Split by common separator lines
    blocks = re.split(r'-{40,}', log_text)

    current_table: dict = {}
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Detect variable coefficient lines: varname | coef stderr t p [ci]
        coef_lines = re.findall(
            r'^\s*(\S+)\s+\|\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)',
            block, re.MULTILINE
        )

        if coef_lines:
            if current_table and current_table.get("rows"):
                tables.append(current_table)

            rows = []
            for match in coef_lines:
                var, coef, se, t_stat, p_val = match
                if var.startswith('_cons') or var.startswith('_'):
                    var = 'Constant' if '_cons' in var else var
                rows.append({
                    "variable": var,
                    "coefficient": float(coef),
                    "std_error": float(se),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "stars": "***" if float(p_val) < 0.01 else "**" if float(p_val) < 0.05 else "*" if float(p_val) < 0.1 else "",
                })

            # Extract R-squared, N, F-stat
            r2_match = re.search(r'R-squared\s*=\s*([\d.]+)', block)
            n_match = re.search(r'(?:Number of obs|Obs)\s*=\s*([\d,]+)', block)
            f_match = re.search(r'F\(.*?\)\s*=\s*([\d.]+)', block)

            current_table = {
                "rows": rows,
                "n_obs": int(n_match.group(1).replace(',', '')) if n_match else None,
                "r_squared": float(r2_match.group(1)) if r2_match else None,
                "f_stat": float(f_match.group(1)) if f_match else None,
            }

    if current_table and current_table.get("rows"):
        tables.append(current_table)

    return tables


def _tables_to_latex(tables: list[dict], title: str = "Regression Results") -> str:
    """Convert parsed Stata tables to publication-quality LaTeX."""
    if not tables:
        return "% No regression tables found in Stata output\n"

    latex_parts = []
    for i, table in enumerate(tables):
        n_cols = len(tables) if len(tables) <= 5 else 1
        col_label = f"({i+1})" if n_cols > 1 else ""

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{title}}}",
            r"\begin{tabular}{l" + "c" * max(1, n_cols) + "}",
            r"\hline\hline",
            r"& " + col_label + r" \\",
            r"\hline",
        ]

        for row in table["rows"]:
            coef_str = f"{row['coefficient']:.4f}{row['stars']}"
            se_str = f"({row['std_error']:.4f})"
            lines.append(f"{row['variable']} & {coef_str} \\\\")
            lines.append(f" & {se_str} \\\\")

        lines.append(r"\hline")
        if table.get("n_obs"):
            lines.append(f"Observations & {table['n_obs']:,} \\\\")
        if table.get("r_squared"):
            lines.append(f"R-squared & {table['r_squared']:.4f} \\\\")
        if table.get("f_stat"):
            lines.append(f"F-statistic & {table['f_stat']:.2f} \\\\")
        lines.append(r"\hline\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex_parts.append("\n".join(lines))

    return "\n\n".join(latex_parts)


@router.post("/api/stata/to-latex", tags=["Stata"])
async def stata_to_latex(request: Request):
    """Convert Stata .log/.smcl output to publication-quality LaTeX tables.

    Parses regression output, extracts coefficients/SE/stars,
    and formats as \\begin{tabular} with proper alignment.
    """
    body = await request.json()
    log_text = body.get("log_text", "")
    log_path = body.get("log_path", "")
    title = body.get("title", "Regression Results")

    if log_path and not log_text:
        # §SECURITY: Validate log_path to prevent path traversal
        log_path = os.path.abspath(log_path)
        data_root = _data_root()
        allowed_dirs = [data_root, os.path.expanduser("~")] if data_root else [os.path.expanduser("~")]
        if not any(log_path.startswith(os.path.abspath(d)) for d in allowed_dirs if d):
            return JSONResponse(status_code=403,
                                content={"error": "log_path must be within your home or data directory"})
        try:
            with open(log_path, 'r', errors='ignore') as f:
                log_text = f.read()
        except Exception as e:
            return JSONResponse(status_code=400,
                                content={"error": f"Cannot read log file: {e}"})

    if not log_text:
        return JSONResponse(status_code=400,
                            content={"error": "Provide 'log_text' or 'log_path'"})

    tables = _parse_stata_table(log_text)
    latex = _tables_to_latex(tables, title)

    return {
        "latex": latex,
        "tables_found": len(tables),
        "rows_total": sum(len(t.get("rows", [])) for t in tables),
        "tables": tables,  # structured data for frontend preview
    }


# ═══════════════════════════════════════════════════════════════════
# #3 — Cross-Panel Focus Propagation
# ═══════════════════════════════════════════════════════════════════

# In-memory focus state (broadcast to all panels)
_current_focus = {
    "type": None,  # "paper" | "concept" | "chapter"
    "title": "",
    "sha256": "",
    "path": "",
    "author": "",
    "topic": "",
    "updated_at": 0,
}


@router.post("/api/focus/broadcast", tags=["Focus"])
async def focus_broadcast(request: Request):
    """Broadcast focus change to all panels.

    When a paper/concept is focused anywhere, ALL panels should react:
      - Library highlights the paper
      - Atlas zooms to the concept
      - Map pans to the geo-location
      - Forensic loads the paper
      - Socratic pre-loads relevant challenges
    """
    global _current_focus
    body = await request.json()
    _current_focus = {
        "type": body.get("type", "paper"),
        "title": body.get("title", ""),
        "sha256": body.get("sha256", ""),
        "path": body.get("path", ""),
        "author": body.get("author", ""),
        "topic": body.get("topic", ""),
        "method": body.get("method", ""),
        "updated_at": time.time(),
    }

    # Also update Bridge focus if available
    try:
        from server.citadel_bridge import citadel_bridge
        if citadel_bridge:
            citadel_bridge.focus_paper(
                title=_current_focus["title"],
                path=_current_focus["path"],
                author=_current_focus["author"],
            )
    except Exception:
        pass

    return {"status": "broadcasted", "focus": _current_focus}


@router.get("/api/focus/current", tags=["Focus"])
async def focus_current():
    """Get current focus state — polled by all panels."""
    return _current_focus


# ═══════════════════════════════════════════════════════════════════
# #4 — Sharpening Loop Dashboard
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/training/dashboard", tags=["Training"])
async def training_dashboard():
    """Training data visibility — shows what Winnie has learned.

    Scans edith_master_train.jsonl for stats: total entries, DPO pairs,
    topic distribution, and accuracy trends.
    """
    data_root = os.environ.get("DATA_ROOT", os.environ.get("EDITH_DATA_ROOT", ""))
    train_files = [
        os.path.join(data_root, "edith_master_train.jsonl"),
        os.path.join(data_root, "training", "edith_master_train.jsonl"),
        "edith_master_train.jsonl",
    ]

    train_path = None
    for f in train_files:
        if f and os.path.isfile(f):
            train_path = f
            break

    stats = {
        "total_entries": 0,
        "dpo_pairs": 0,
        "topics": {},
        "sources": {},
        "recent_entries": [],
        "file_path": train_path,
        "file_size_mb": 0,
    }

    if not train_path:
        stats["status"] = "no_training_file"
        return stats

    try:
        stats["file_size_mb"] = round(os.path.getsize(train_path) / (1024 * 1024), 2)
        with open(train_path, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["total_entries"] += 1

                # Count DPO pairs
                if entry.get("chosen") and entry.get("rejected"):
                    stats["dpo_pairs"] += 1

                # Topic distribution
                topic = entry.get("topic", entry.get("academic_topic", "general"))
                stats["topics"][topic] = stats["topics"].get(topic, 0) + 1

                # Source distribution
                source = entry.get("source", "unknown")
                stats["sources"][source] = stats["sources"].get(source, 0) + 1

                # Keep last 5 entries for preview
                if stats["total_entries"] <= 5:
                    stats["recent_entries"].append({
                        "prompt": (entry.get("prompt", entry.get("messages", [{}])[0].get("content", "")) or "")[:100],
                        "source": source,
                        "topic": topic,
                        "timestamp": entry.get("timestamp", ""),
                    })

        stats["status"] = "ok"
    except Exception as e:
        stats["status"] = f"error: {str(e)[:100]}"

    return stats


# ═══════════════════════════════════════════════════════════════════
# #5 — Auto-Bibliography from Notes
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/bibliography/from-notes", tags=["Export"])
async def bibliography_from_notes(request: Request):
    """Scan notes text for @Author2024-style citations and auto-generate .bib.

    Searches ChromaDB for matching papers and builds BibTeX entries.
    """
    body = await request.json()
    notes_text = body.get("text", "")
    if not notes_text:
        return JSONResponse(status_code=400, content={"error": "Provide 'text' to scan"})

    # Extract citation-like patterns
    patterns = [
        r'@(\w+\d{4})',                    # @Author2024
        r'(\w+)\s+\((\d{4})\)',            # Author (2024)
        r'(\w+)\s+et\s+al\.\s*\((\d{4})\)',  # Author et al. (2024)
        r'(\w+)\s+and\s+(\w+)\s*\((\d{4})\)', # Author and Author (2024)
    ]

    found_refs = []
    for pattern in patterns:
        for match in re.finditer(pattern, notes_text):
            groups = match.groups()
            if len(groups) == 1:
                # @Author2024 format
                ref = groups[0]
                author = re.sub(r'\d+', '', ref)
                year = re.search(r'\d{4}', ref)
                found_refs.append({"author": author, "year": year.group() if year else ""})
            elif len(groups) == 2:
                found_refs.append({"author": groups[0], "year": groups[1]})
            elif len(groups) == 3:
                found_refs.append({"author": f"{groups[0]} and {groups[1]}", "year": groups[2]})

    # Deduplicate
    seen = set()
    unique_refs = []
    for ref in found_refs:
        key = f"{ref['author'].lower()}_{ref['year']}"
        if key not in seen:
            seen.add(key)
            unique_refs.append(ref)

    # Search ChromaDB for matching papers
    bibtex_entries = []
    matched = 0
    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")

    if chroma_dir and unique_refs:
        try:
            from server.chroma_backend import _get_client
            client = _get_client(chroma_dir)
            collection = client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})

            for ref in unique_refs:
                query = f"{ref['author']} {ref['year']}"
                results = collection.query(query_texts=[query], n_results=1, include=["metadatas"])
                if results["metadatas"] and results["metadatas"][0]:
                    meta = results["metadatas"][0][0]
                    title = meta.get("title", "")
                    author = meta.get("author", ref["author"])
                    year = meta.get("year", ref["year"])
                    first_author = author.split(",")[0].split(" ")[-1] if author else ref["author"]
                    key = f"{first_author}{year}"
                    entry = f"@article{{{key},\n  title = {{{title}}},\n  author = {{{author}}},\n  year = {{{year}}}\n}}"
                    bibtex_entries.append(entry)
                    matched += 1
                    ref["matched"] = True
                    ref["title"] = title
                else:
                    # Generate stub entry
                    key = f"{ref['author']}{ref['year']}"
                    entry = f"@article{{{key},\n  author = {{{ref['author']}}},\n  year = {{{ref['year']}}},\n  note = {{Auto-generated stub — paper not found in vault}}\n}}"
                    bibtex_entries.append(entry)
                    ref["matched"] = False
        except Exception as e:
            log.warning(f"ChromaDB search failed: {e}")

    return {
        "bibtex": "\n\n".join(bibtex_entries),
        "references_found": len(unique_refs),
        "matched_in_vault": matched,
        "unmatched": len(unique_refs) - matched,
        "references": unique_refs,
    }


# ═══════════════════════════════════════════════════════════════════
# #6 — Reading Progress State
# ═══════════════════════════════════════════════════════════════════

# In-memory reading progress (persisted to disk on update)
_reading_progress: dict[str, dict] = {}
_reading_progress_file = ""


def _load_reading_progress():
    global _reading_progress, _reading_progress_file
    data_root = os.environ.get("DATA_ROOT", os.environ.get("EDITH_DATA_ROOT", ""))
    _reading_progress_file = os.path.join(data_root, "reading_progress.json") if data_root else ""
    if _reading_progress_file and os.path.isfile(_reading_progress_file):
        try:
            with open(_reading_progress_file, 'r') as f:
                _reading_progress = json.load(f)
        except Exception:
            pass


def _save_reading_progress():
    if _reading_progress_file and os.path.dirname(_reading_progress_file):
        try:
            os.makedirs(os.path.dirname(_reading_progress_file), exist_ok=True)
            with open(_reading_progress_file, 'w') as f:
                json.dump(_reading_progress, f, indent=2)
        except Exception:
            pass


_load_reading_progress()


@router.post("/api/library/reading-progress", tags=["Library"])
async def set_reading_progress(request: Request):
    """Set reading progress for a paper: unread/reading/done."""
    body = await request.json()
    sha256 = body.get("sha256", "")
    status = body.get("status", "unread")  # unread | reading | done
    if not sha256:
        return JSONResponse(status_code=400, content={"error": "sha256 required"})
    if status not in ("unread", "reading", "done"):
        return JSONResponse(status_code=400, content={"error": "status must be unread/reading/done"})

    _reading_progress[sha256] = {
        "status": status,
        "updated_at": time.time(),
        "notes_count": body.get("notes_count", 0),
    }
    _save_reading_progress()
    return {"status": "ok", "sha256": sha256, "reading_status": status}


@router.get("/api/library/reading-progress", tags=["Library"])
async def get_reading_progress():
    """Get reading progress for all papers."""
    return {
        "progress": _reading_progress,
        "stats": {
            "unread": sum(1 for p in _reading_progress.values() if p.get("status") == "unread"),
            "reading": sum(1 for p in _reading_progress.values() if p.get("status") == "reading"),
            "done": sum(1 for p in _reading_progress.values() if p.get("status") == "done"),
            "total": len(_reading_progress),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# #7 — Citation Chain Walking
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/citation-graph/walk", tags=["Search"])
async def citation_chain_walk(request: Request):
    """Walk the citation chain N hops backward from a paper."""
    body = await request.json()
    seed_sha = str(body.get("sha256", "") or "").strip()
    max_hops = min(int(body.get("hops", 3) or 3), 5)
    direction = body.get("direction", "backward")

    if not seed_sha:
        return JSONResponse(status_code=400, content={"error": "sha256 required"})

    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if not chroma_dir:
        return JSONResponse(status_code=500, content={"error": "EDITH_CHROMA_DIR not configured"})

    try:
        from server.chroma_backend import _get_client
        client = _get_client(chroma_dir)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

    def _list_collections_with_counts() -> list[tuple[str, int]]:
        pairs: list[tuple[str, int]] = []
        try:
            listed = client.list_collections()
        except Exception:
            listed = []
        for item in listed or []:
            name = getattr(item, "name", None)
            if not name and isinstance(item, dict):
                name = item.get("name")
            if not name:
                continue
            try:
                count = int(client.get_collection(name).count())
            except Exception:
                count = 0
            pairs.append((name, count))
        return pairs

    def _resolve_collection() -> tuple[object, str]:
        preferred = []
        try:
            import server.main as _m
            preferred.append(str(getattr(_m, "CHROMA_COLLECTION", "") or ""))
        except Exception:
            pass
        preferred.extend([
            str(os.environ.get("EDITH_CHROMA_COLLECTION", "") or ""),
            "edith_docs_pdf",
            "edith_docs_v2_metadata",
            "edith_docs_v2",
            "edith_corpus",
        ])

        seen = set()
        collection_counts = dict(_list_collections_with_counts())

        for name in preferred:
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                coll = client.get_collection(name)
                if int(collection_counts.get(name, coll.count())) > 0:
                    return coll, name
            except Exception:
                continue

        if collection_counts:
            best_name = max(collection_counts.items(), key=lambda kv: kv[1])[0]
            try:
                return client.get_collection(best_name), best_name
            except Exception:
                pass

        raise RuntimeError("No usable Chroma collection found")

    try:
        collection, collection_name = _resolve_collection()
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

    def _flatten_first(items):
        if not items:
            return []
        if isinstance(items, list) and items and isinstance(items[0], list):
            return items[0]
        return items

    def _paper_lookup(sha: str) -> tuple[list[str], list[dict]]:
        for key in ("sha256", "doc_sha256", "source_sha256"):
            try:
                res = collection.get(where={key: sha}, include=["documents", "metadatas"], limit=4)
            except Exception:
                continue
            docs = _flatten_first(res.get("documents") or [])
            metas = _flatten_first(res.get("metadatas") or [])
            if docs:
                return docs, metas

        try:
            res = collection.get(ids=[sha], include=["documents", "metadatas"])
            docs = _flatten_first(res.get("documents") or [])
            metas = _flatten_first(res.get("metadatas") or [])
            if docs:
                return docs, metas
        except Exception:
            pass

        return [], []

    chain = []
    visited = set()
    current_shas = [seed_sha]

    for hop in range(max_hops):
        next_shas = []
        for sha in current_shas:
            if sha in visited:
                continue
            visited.add(sha)

            docs, metas = _paper_lookup(sha)
            if not docs:
                continue

            text = " ".join((docs or [])[:2])[:4000]
            meta = (metas[0] if metas else {}) or {}

            node = {
                "sha256": sha,
                "title": meta.get("title", meta.get("file_name", "")),
                "author": meta.get("author", ""),
                "year": meta.get("year", ""),
                "hop": hop,
            }

            cited_authors = re.findall(r'(\w+)\s+(?:et\s+al\.\s*)?\((\d{4})\)', text)
            references = []
            for author, year in cited_authors[:10]:
                try:
                    ref_results = collection.query(
                        query_texts=[f"{author} {year}"],
                        n_results=1,
                        include=["metadatas"],
                    )
                except Exception:
                    continue
                ref_metas = _flatten_first(ref_results.get("metadatas") or [])
                if not ref_metas:
                    continue
                ref_meta = (ref_metas[0] if ref_metas else {}) or {}
                ref_sha = (
                    ref_meta.get("sha256")
                    or ref_meta.get("doc_sha256")
                    or ref_meta.get("source_sha256")
                    or ""
                )
                if ref_sha and ref_sha not in visited:
                    next_shas.append(ref_sha)
                    references.append({
                        "sha256": ref_sha,
                        "title": ref_meta.get("title", ref_meta.get("file_name", "")),
                        "author": author,
                        "year": year,
                    })

            node["cites"] = references
            chain.append(node)

        current_shas = next_shas
        if not current_shas:
            break

    if not chain:
        return JSONResponse(
            status_code=404,
            content={
                "error": "seed_not_found_in_collection",
                "seed": seed_sha,
                "collection": collection_name,
                "direction": direction,
            },
        )

    return {
        "chain": chain,
        "seed": seed_sha,
        "hops_done": len(set(n["hop"] for n in chain)),
        "nodes_visited": len(chain),
        "direction": direction,
        "collection": collection_name,
    }


# ═══════════════════════════════════════════════════════════════════
# #9 — Cross-Paper Dataset Reconciliation
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/datasets/reconcile", tags=["Forensic"])
async def dataset_reconciliation(request: Request):
    """Find papers that use the same dataset and show how they differ.

    Groups papers by dataset name, then compares variables, years, methods.
    """
    body = await request.json()
    dataset_name = body.get("dataset", "")

    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if not chroma_dir:
        return JSONResponse(status_code=500, content={"error": "EDITH_CHROMA_DIR not configured"})

    try:
        from server.chroma_backend import _get_client
        client = _get_client(chroma_dir)
        collection = client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

    # Search for papers mentioning this dataset
    if dataset_name:
        results = collection.query(
            query_texts=[dataset_name],
            n_results=20,
            include=["documents", "metadatas"],
        )
    else:
        # Get all papers and group by detected datasets
        results = collection.get(limit=100, include=["documents", "metadatas"])

    papers = []
    dataset_groups: dict[str, list] = {}

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    # Handle nested lists from query vs get
    if docs and isinstance(docs[0], list):
        docs = docs[0]
        metas = metas[0] if metas else []

    for i, (doc, meta) in enumerate(zip(docs or [], metas or [])):
        if not doc:
            continue
        meta = meta or {}
        text_lower = doc.lower()[:2000]

        # Detect common datasets
        common_datasets = [
            "American Community Survey", "Census", "Current Population Survey",
            "ANES", "GSS", "World Values Survey", "Eurobarometer",
            "ICPSR", "FRED", "BLS", "BEA", "WHO", "World Bank",
            "Panel Study of Income Dynamics", "PSID", "NLSY",
            "Survey of Consumer Finances", "Medical Expenditure Panel",
        ]

        detected = []
        for ds in common_datasets:
            if ds.lower() in text_lower:
                detected.append(ds)

        if dataset_name and dataset_name.lower() not in text_lower and not detected:
            continue

        paper = {
            "title": meta.get("title", "")[:100],
            "author": meta.get("author", ""),
            "year": meta.get("year", ""),
            "method": meta.get("method", ""),
            "sha256": meta.get("sha256", ""),
            "datasets_detected": detected or [dataset_name] if dataset_name else [],
        }
        papers.append(paper)

        for ds in paper["datasets_detected"]:
            if ds not in dataset_groups:
                dataset_groups[ds] = []
            dataset_groups[ds].append(paper)

    # Build reconciliation report
    reconciliation = {}
    for ds, group in dataset_groups.items():
        if len(group) < 2:
            continue
        methods_used = list(set(p.get("method", "") for p in group if p.get("method")))
        years_covered = list(set(p.get("year", "") for p in group if p.get("year")))
        reconciliation[ds] = {
            "papers_count": len(group),
            "papers": [{"title": p["title"], "author": p["author"], "year": p["year"], "method": p["method"]} for p in group],
            "methods_used": methods_used,
            "years_covered": sorted(years_covered),
            "method_diversity": len(methods_used) > 1,
        }

    return {
        "dataset_query": dataset_name,
        "total_papers_found": len(papers),
        "datasets_with_multiple_papers": len(reconciliation),
        "reconciliation": reconciliation,
    }


# ═══════════════════════════════════════════════════════════════════
# #10 — Winnie Confidence Calibration
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/chat/calibrate", tags=["Chat"])
async def confidence_calibration(request: Request):
    """Add confidence calibration to a chat response.

    Takes a response + query, checks how many indexed papers support it,
    and returns a calibrated confidence score with evidence breakdown.
    """
    body = await request.json()
    response_text = body.get("response", "")
    query = body.get("query", "")

    if not response_text:
        return JSONResponse(status_code=400, content={"error": "response text required"})

    calibration = {
        "confidence": 0.5,
        "level": "medium",
        "evidence_sources": 0,
        "retrieval_grounded": False,
        "breakdown": {},
    }

    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if chroma_dir and query:
        try:
            from server.chroma_backend import _get_client
            client = _get_client(chroma_dir)
            collection = client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})

            # Check how many indexed papers are relevant
            results = collection.query(query_texts=[query], n_results=5, include=["distances", "metadatas"])
            distances = results.get("distances", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            relevant = sum(1 for d in distances if d < 0.5)
            calibration["evidence_sources"] = relevant
            calibration["retrieval_grounded"] = relevant > 0

            # Confidence formula:
            # Base 0.3 (model knowledge) + 0.15 per relevant source, capped at 0.95
            confidence = min(0.95, 0.3 + relevant * 0.15)
            calibration["confidence"] = round(confidence, 2)
            calibration["level"] = (
                "high" if confidence >= 0.8 else
                "medium" if confidence >= 0.5 else
                "low"
            )

            calibration["breakdown"] = {
                "base_knowledge": 0.3,
                "retrieval_boost": round(relevant * 0.15, 2),
                "sources_checked": len(distances),
                "sources_relevant": relevant,
                "closest_papers": [
                    {"title": m.get("title", "")[:80], "distance": round(d, 3)}
                    for d, m in zip(distances[:3], metas[:3])
                ],
            }
        except Exception as e:
            calibration["breakdown"]["error"] = str(e)[:100]

    return calibration

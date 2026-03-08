"""
Library routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
import threading
import time as _time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Body, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from server.server_state import state as _server_state

log = logging.getLogger("edith")
router = APIRouter(tags=["Library"])

# Module-level state needed by library cache system
_library_cache = []
_library_cache_ts = 0
_library_building = False
_library_build_progress = {}
_library_lock = threading.Lock()
_sources_cache: dict = {}
_chroma_collection = None
_UPLOAD_ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".tex", ".ipynb", ".rmd"}
_UPLOAD_MAX_MB = 50

# Library should show readable source files, not GIS index internals/binaries.
# User intent: prioritize published research articles only.
_LIBRARY_ALLOWED_EXT = {".pdf"}
_LIBRARY_EXCLUDED_EXT = {
    ".cfs", ".atx", ".sbn", ".sbx", ".spx", ".horizon", ".freelist",
    ".gdbtable", ".gdbtablx", ".gdbindexes", ".shp", ".shx", ".dbf",
    ".prj", ".cpg", ".gph", ".gpkg", ".tif", ".tiff", ".gif", ".png",
    ".jpg", ".jpeg", ".bmp", ".sqlite", ".sqlite3", ".db", ".lock",
    ".zip", ".7z", ".rar", ".tar", ".gz",
}
_LIBRARY_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "backups",
    "chroma", "ChromaDB", ".Spotlight-V100", ".fseventsd", ".TemporaryItems",
}
_LIBRARY_INDEX_MARKERS = ("/index/", "_index/")
_LIBRARY_TIER_FOLDERS = {"canon", "past", "inbox", "projects"}
_LIBRARY_GENERIC_DIRS = {
    "library", "edithdata", "vault", "corpus", "courses", "course", "class", "classes",
    "courses and projects", "methods",
    "research", "datasets", "dataset", "data", "readings", "reading", "papers", "paper",
    "articles", "article", "docs", "documents", "notes", "code", "scripts", "figures",
    "figure", "tables", "table", "appendix", "appendices", "slides", "exports",
    "archive", "archives", "tmp", "temp", "drafts", "final", "finals",
}
_LIBRARY_TERM_MARKERS = {"active", "past", "current", "spring", "summer", "fall", "winter"}


def _candidate_data_roots() -> list[Path]:
    """Resolve viable data roots in priority order."""
    roots: list[Path] = []

    def _add(raw: Any) -> None:
        if not raw:
            return
        try:
            p = Path(str(raw)).expanduser().resolve()
        except Exception:
            p = Path(str(raw)).expanduser()
        if p.exists() and p.is_dir() and p not in roots:
            roots.append(p)

    _add(_var("DATA_ROOT"))
    _add(os.environ.get("EDITH_DATA_ROOT"))

    chroma_raw = _var("CHROMA_DIR") or os.environ.get("EDITH_CHROMA_DIR")
    if chroma_raw:
        try:
            chroma_path = Path(str(chroma_raw)).expanduser().resolve()
        except Exception:
            chroma_path = Path(str(chroma_raw)).expanduser()
        if chroma_path.name.lower() in {"chroma", "chromadb"}:
            _add(chroma_path.parent)

    _add("/Volumes/Edith Bolt")
    try:
        _root_dir = _var("ROOT_DIR")
        if _root_dir:
            _add(Path(str(_root_dir)).expanduser() / "data")
    except Exception:
        pass

    return roots


def _resolved_data_root() -> Optional[Path]:
    roots = _candidate_data_roots()
    return roots[0] if roots else None


def _is_generic_library_dir(name: str) -> bool:
    raw = (name or "").strip()
    if not raw:
        return True
    lower = raw.lower()
    compact = lower.replace("_", " ").strip()
    if compact in _LIBRARY_GENERIC_DIRS:
        return True
    if compact in _LIBRARY_TIER_FOLDERS:
        return True
    if compact in _LIBRARY_TERM_MARKERS:
        return True
    if compact.startswith("week ") or compact.startswith("module ") or compact.startswith("unit "):
        return True
    if compact.startswith("chapter ") or compact.startswith("lecture "):
        return True
    return False


def _derive_library_context(rel_path: str, file_name: str = "") -> dict:
    """Derive class/topic/project/tier from deep nested folder structures."""
    rel_norm = str(rel_path or "").replace("\\", "/").strip("/")
    parts = [p.strip() for p in rel_norm.split("/") if p.strip()]
    if not parts and file_name:
        parts = [file_name]
    lower = [p.lower() for p in parts]

    file_index = len(parts) - 1 if (parts and (Path(parts[-1]).suffix or file_name)) else len(parts)
    dir_parts = parts[:file_index]
    dir_lower = lower[:file_index]

    tier = ""
    class_name = ""
    class_idx = -1

    # Legacy /canon|past|inbox|projects/<Class>/... structure
    for i, pl in enumerate(dir_lower):
        if pl in _LIBRARY_TIER_FOLDERS and i + 1 < len(dir_parts):
            tier = dir_parts[i]
            class_name = dir_parts[i + 1].strip()
            class_idx = i + 1
            break

    # /Library/Courses/<term>/<Class>/...
    if not class_name:
        for i, pl in enumerate(dir_lower):
            if pl == "courses" and i + 2 < len(dir_parts):
                tier = tier or dir_parts[i + 1].strip()
                class_name = dir_parts[i + 2].strip()
                class_idx = i + 2
                break

    # /Library/Courses and Projects/<Class>/...
    if not class_name:
        for i, pl in enumerate(dir_lower):
            if pl == "courses and projects" and i + 1 < len(dir_parts):
                class_name = dir_parts[i + 1].strip()
                class_idx = i + 1
                break

    # /Library/Methods/<Class>/...
    if not class_name:
        for i, pl in enumerate(dir_lower):
            if pl == "methods" and i + 1 < len(dir_parts):
                class_name = dir_parts[i + 1].strip()
                class_idx = i + 1
                break

    # /Library/Research/<Project>/...
    if not class_name:
        for i, pl in enumerate(dir_lower):
            if pl == "research" and i + 1 < len(dir_parts):
                tier = tier or dir_parts[i]
                class_name = dir_parts[i + 1].strip()
                class_idx = i + 1
                break

    # Generic fallback: first non-generic folder after root-like prefixes
    if not class_name:
        start = 0
        for i, pl in enumerate(dir_lower):
            if pl in {"library", "vault", "corpus", "edithdata"}:
                start = i + 1
                break
        for i in range(start, len(dir_parts)):
            cand = dir_parts[i].strip()
            if not _is_generic_library_dir(cand):
                class_name = cand
                class_idx = i
                break

    class_name = class_name.strip() if class_name else ""

    # Topic: first non-generic subfolder under class, else class itself.
    topic = class_name
    if class_idx >= 0:
        for i in range(class_idx + 1, len(dir_parts)):
            cand = dir_parts[i].strip()
            if not _is_generic_library_dir(cand):
                topic = cand
                break

    return {
        "class_name": class_name or None,
        "project": class_name or None,
        "topic": (topic or class_name or "General"),
        "tier": tier or None,
    }


def __getattr__(name):
    """Module-level fallback: resolve any missing name from server.main.
    This allows extracted handlers to reference CHROMA_DIR, DATA_ROOT, etc."""
    try:
        import server.main as m
        return getattr(m, name)
    except (ImportError, AttributeError):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_main():
    import server.main as m
    return m


def _var(name):
    """Resolve a variable from this module or server.main.
    Use inside functions where bare name lookup skips module __getattr__."""
    mod = __import__(__name__)
    for part in __name__.split(".")[1:]:
        mod = getattr(mod, part)
    val = getattr(mod, name, None)
    if val is not None:
        return val
    return getattr(_get_main(), name)


def _library_ext(file_name: str, rel_path: str) -> str:
    # Prefer explicit file_name extension, but fall back to rel_path when
    # metadata file_name is truncated/missing extension.
    ext = Path(file_name or "").suffix.lower()
    if ext:
        return ext
    return Path(rel_path or "").suffix.lower()


def _is_library_artifact(rel_path: str, file_name: str) -> bool:
    rel_norm = str(rel_path or "").replace("\\", "/").lower()
    ext = _library_ext(file_name, rel_path)
    if ext in _LIBRARY_EXCLUDED_EXT:
        return True
    # In launch mode, only known allowed extensions (PDF) are valid.
    # Missing extensions usually come from index artifacts/noisy metadata.
    if not ext:
        return True
    if ext not in _LIBRARY_ALLOWED_EXT:
        return True
    if any(marker in rel_norm for marker in _LIBRARY_INDEX_MARKERS) and ext in _LIBRARY_EXCLUDED_EXT:
        return True
    stem = Path(file_name or Path(rel_path or "").name).stem.strip("._-")
    if stem.isdigit() and ext in {".json", ".jsonl"} and any(marker in rel_norm for marker in _LIBRARY_INDEX_MARKERS):
        return True
    return False


def _filesystem_library_docs(limit: int = 5000) -> list[dict]:
    data_root = _resolved_data_root()
    if not data_root:
        return []

    docs: list[dict] = []
    seen: set[str] = set()
    for root, dirs, files in os.walk(data_root):
        dirs[:] = [d for d in dirs if d not in _LIBRARY_SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            ext = Path(fn).suffix.lower()
            if ext not in _LIBRARY_ALLOWED_EXT:
                continue
            abs_path = Path(root) / fn
            try:
                rel = str(abs_path.relative_to(data_root))
            except Exception:
                rel = str(abs_path)
            if _is_library_artifact(rel, fn):
                continue
            if rel in seen:
                continue
            seen.add(rel)

            title = Path(fn).stem.replace("_", " ").strip() or "Untitled"
            if title.isdigit():
                title = f"Document {title}"
            ctx = _derive_library_context(rel, fn)
            row_sha = hashlib.sha256(rel.encode("utf-8")).hexdigest()
            docs.append({
                "sha256": row_sha,
                "title": title,
                "author": None,
                "year": None,
                "doc_type": "paper" if ext == ".pdf" else ext.lstrip("."),
                "tier": ctx.get("tier"),
                "academic_topic": ctx.get("topic"),
                "version_stage": None,
                "rel_path": rel,
                "file_name": fn,
                "project": ctx.get("project"),
                "tag": None,
                "class_name": ctx.get("class_name"),
                "chunk_count": 0,
                "indexed": False,
            })
            if len(docs) >= limit:
                return docs
    return docs


def _score_collection_for_library(collection) -> tuple[int, int]:
    """Return (usable_docs, total_chunks) score for collection selection."""
    try:
        total = collection.count()
        if total <= 0:
            return (0, 0)
        sample_n = min(total, 3000)
        sample = collection.get(limit=sample_n, offset=0, include=["metadatas"])
        usable = 0
        seen_sha: set[str] = set()
        for meta in sample.get("metadatas") or []:
            if not meta:
                continue
            rel = meta.get("rel_path") or meta.get("path") or ""
            file_name = meta.get("file_name") or (Path(rel).name if rel else "")
            if _is_library_artifact(rel, file_name):
                continue
            sha = str(meta.get("sha256") or rel or file_name).strip()
            if not sha or sha in seen_sha:
                continue
            seen_sha.add(sha)
            usable += 1
        return (usable, total)
    except Exception:
        return (0, 0)



async def upload_file(file: UploadFile = File(...)):
    """Upload a file for immediate use in chat or future indexing.
    
    Saves to DATA_ROOT/uploads/ and returns extracted text preview.
    Supported: PDF, DOCX, TXT, MD, CSV, TEX, IPYNB, RMD (max 50MB).
    """
    import sys
    try:
        _data_root = _var("DATA_ROOT") or os.environ.get("EDITH_DATA_ROOT", "")
    except Exception:
        _data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not _data_root:
        raise HTTPException(status_code=500, detail="DATA_ROOT not configured")
    
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _UPLOAD_ALLOWED_EXT:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(_UPLOAD_ALLOWED_EXT))}"
        )
    
    # Read file content
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > _UPLOAD_MAX_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f}MB > {_UPLOAD_MAX_MB}MB)")
    
    # Save to uploads directory
    upload_dir = Path(_data_root) / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    # Unique filename to avoid collisions — sanitize against path traversal
    import hashlib
    import re as _re_upload
    name_hash = hashlib.sha256(content[:4096]).hexdigest()[:8]
    # Strip any path components and sanitize the stem
    raw_stem = Path(file.filename or "file").name  # basename only
    raw_stem = Path(raw_stem).stem  # remove extension
    raw_stem = _re_upload.sub(r'[^\w\-. ]', '_', raw_stem)  # only safe chars
    raw_stem = raw_stem.strip('._')[:100] or "upload"  # limit length, no leading dots
    safe_name = f"{raw_stem}_{name_hash}{ext}"
    dest = upload_dir / safe_name
    # Final guard: ensure dest is inside upload_dir
    if not dest.resolve().is_relative_to(upload_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    dest.write_bytes(content)
    
    # Extract text preview (first 500 chars)
    preview = ""
    try:
        if ext in {".txt", ".md", ".csv", ".tex"}:
            preview = content.decode("utf-8", errors="replace")[:500]
        elif ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=content, filetype="pdf")
                preview = "\n".join(page.get_text() for page in doc[:3])[:500]
                doc.close()
            except Exception:
                preview = "(PDF text extraction unavailable)"
    except Exception:
        preview = "(preview unavailable)"
    
    try:
        audit("file_upload", filename=file.filename, size_mb=round(size_mb, 2))
    except Exception:
        log.debug("audit() unavailable — skipping upload audit log")

    # §FIX Robustness 1: Track subprocess PID for cleanup on server shutdown
    indexing_status = "pending"
    try:
        import subprocess as _up_sp
        try:
            _root_dir = _var("ROOT_DIR") or os.environ.get("EDITH_DATA_ROOT", ".")
        except Exception:
            _root_dir = os.environ.get("EDITH_DATA_ROOT", ".")
        _idx_proc = _up_sp.Popen(
            [sys.executable, "scripts/chroma_index.py",
             "--single-file", str(dest)],
            cwd=str(_root_dir),
            stdout=_up_sp.DEVNULL,
            stderr=_up_sp.DEVNULL,
        )
        # Track PID for cleanup (app state for shutdown hook)
        try:
            if not hasattr(app, "_reindex_pids"):
                app._reindex_pids = []
            app._reindex_pids.append(_idx_proc.pid)
        except Exception:
            pass  # app reference not available in this context
        indexing_status = "auto_indexing"
        log.info(f"§INDEX: Auto-indexing uploaded file: {safe_name} (pid={_idx_proc.pid})")

        # §FIX GAP 1: Copy to Vault/inbox/ so paper is in the canonical location
        try:
            vault_inbox = Path(DATA_ROOT) / "Corpus" / "Vault" / "inbox"
            vault_inbox.mkdir(parents=True, exist_ok=True)
            vault_dest = vault_inbox / safe_name
            if not vault_dest.exists():
                import shutil
                shutil.copy2(str(dest), str(vault_dest))
                log.info(f"§VAULT: Copied upload to Vault inbox: {safe_name}")
        except Exception as _vault_err:
            log.warning(f"§VAULT: Could not copy to Vault inbox: {_vault_err}")

        # §FIX GAP 2: Force library cache refresh after indexing finishes
        def _wait_and_refresh():
            try:
                _idx_proc.wait(timeout=300)  # Wait for indexing to finish
                if _idx_proc.returncode == 0:
                    global _library_cache_ts
                    _library_cache_ts = 0  # Force cache rebuild on next request
                    log.info(f"§CACHE: Invalidated library cache after indexing {safe_name}")
            except Exception:
                pass
        _refresh_t = threading.Thread(target=_wait_and_refresh, daemon=True)
        _refresh_t.start()

    except Exception as _idx_err:
        log.warning(f"Auto-index trigger failed: {_idx_err}")
        indexing_status = "manual_index_needed"

    return {
        "filename": safe_name,
        "original_name": file.filename,
        "size_mb": round(size_mb, 2),
        "path": f"uploads/{safe_name}",
        "preview": preview[:500],
        "indexing": indexing_status,
        "message": f"Uploaded '{file.filename}' ({size_mb:.1f}MB). {'Auto-indexing in background.' if indexing_status == 'auto_indexing' else 'Run indexing to include in search.'}",
    }


def _build_library_cache_sync():
    """Scan ChromaDB for unique documents using batched metadata retrieval (runs in bg thread)."""
    global _library_cache, _library_cache_ts, _library_building, _library_build_progress
    import time as _time
    _library_building = True
    _library_build_progress = {"batch": 0, "total_batches": 0, "chunks_scanned": 0, "total_chunks": 0, "docs_found": 0, "elapsed_s": 0, "eta_s": 0, "stage": "connecting"}

    def _hydrate_from_filesystem(stage: str, detail: str) -> None:
        global _library_cache, _library_cache_ts, _library_build_progress
        fs_docs = _filesystem_library_docs()
        with _library_lock:
            _library_cache = sorted(fs_docs, key=lambda d: (d.get("title") or "").lower())
            _library_cache_ts = _time.time()
            _library_build_progress = {
                "stage": stage,
                "detail": detail,
                "docs_found": len(_library_cache),
                "source": "filesystem",
            }
        log.warning(f"Library build fallback ({stage}): {detail} — {len(_library_cache)} docs from filesystem")

    try:
        if not _var('chroma_runtime_available')():
            _hydrate_from_filesystem("fallback", "ChromaDB not available")
            return

        from server.chroma_backend import _get_client
        _library_build_progress["stage"] = "connecting"
        client = _get_client(_var('CHROMA_DIR'))
        # Scan collections and pick the one with the best usable library docs,
        # not just the highest raw chunk count (which can be GIS metadata noise).
        collection = None
        _coll_name = _var('CHROMA_COLLECTION')
        best_count = 0
        best_usable = -1
        try:
            all_colls = [
                c for c in client.list_collections()
                if not str(getattr(c, "name", "")).lower().endswith("_metadata")
            ]
            for cand in all_colls:
                try:
                    usable, cnt = _score_collection_for_library(cand)
                    log.info(
                        f"Library build: candidate '{cand.name}' has {cnt} chunks, {usable} usable docs"
                    )
                    if usable > best_usable or (usable == best_usable and cnt > best_count):
                        best_usable = usable
                        best_count = cnt
                        collection = cand
                except Exception:
                    continue
        except Exception:
            # Fallback: try the configured collection name directly
            try:
                collection = client.get_collection(name=_coll_name)
                best_count = collection.count()
            except Exception:
                pass
        if collection and best_count > 0:
            log.info(
                f"Library build: using '{collection.name}' with {best_count} chunks "
                f"({max(best_usable, 0)} usable docs)"
            )
        if not collection or collection.count() == 0:
            _hydrate_from_filesystem("fallback", "No populated collection")
            return
        total_count = collection.count()
        log.info(f"Library build: scanning {total_count} chunks from {_var('CHROMA_DIR')}")

        seen: dict = {}
        batch_size = 1000
        num_batches = (total_count + batch_size - 1) // batch_size
        _library_build_progress = {"batch": 0, "total_batches": num_batches, "chunks_scanned": 0, "total_chunks": total_count, "docs_found": 0, "elapsed_s": 0, "eta_s": 0, "stage": "scanning"}
        start_t = _time.time()
        for batch_idx, offset in enumerate(range(0, total_count, batch_size)):
            try:
                batch = collection.get(
                    include=["metadatas"],
                    offset=offset,
                    limit=batch_size,
                )
                metas = batch.get("metadatas") or []
            except Exception as e:
                log.error(f"Library build: batch {batch_idx} failed at offset {offset}: {e}")
                break

            for m in metas:
                sha = str(m.get("sha256") or "").strip()
                if not sha:
                    continue
                if sha in seen:
                    seen[sha]["chunk_count"] += 1
                    continue
                rel_path = m.get("rel_path") or m.get("path") or ""
                file_name = m.get("file_name") or (Path(rel_path).name if rel_path else "")
                if _is_library_artifact(rel_path, file_name):
                    continue
                # Skip dataset files -- they belong in the Datasets panel
                _DATASET_EXTS = {".dta", ".shp", ".csv", ".xlsx", ".xls", ".dbf",
                                 ".sav", ".rds", ".gph", ".dbase", ".tsv", ".parquet",
                                 ".feather", ".rdata", ".por", ".mat", ".png", ".jpg",
                                 ".jpeg", ".gif", ".tif", ".tiff", ".bmp"}
                _fn_ext = ("." + file_name.rsplit(".", 1)[-1]).lower() if "." in file_name else ""
                if _fn_ext in _DATASET_EXTS:
                    continue
                # Fix title: use file_name if title is empty or just an extension
                raw_title = m.get("title") or m.get("title_guess") or ""
                if not raw_title or raw_title.startswith(".") or len(raw_title.strip()) < 3:
                    raw_title = Path(file_name).stem if file_name else "Untitled"
                # Clean up title: strip whitespace, replace underscores
                raw_title = raw_title.strip().replace("_", " ")
                # For numeric-only titles, make them more readable
                if raw_title.isdigit():
                    section = m.get("section_heading") or m.get("author", "") or ""
                    raw_title = section.strip() if section.strip() else f"Document {raw_title}"

                ctx = _derive_library_context(rel_path, file_name)

                # §FIX: Coerce year to consistent string type
                raw_year = m.get("year")
                if raw_year is None:
                    year_str = None
                elif isinstance(raw_year, (int, float)):
                    year_str = str(int(raw_year)) if raw_year else None
                else:
                    year_str = str(raw_year).strip() or None

                # Use path-derived class/topic fallback — Chroma metadata is often sparse.
                derived_topic = m.get("academic_topic") or ctx.get("topic")

                seen[sha] = {
                    "sha256": sha,
                    "title": raw_title,
                    "author": m.get("author") or None,
                    "year": year_str,
                    "doc_type": m.get("doc_type") or None,
                    "tier": m.get("tier") or ctx.get("tier"),
                    "academic_topic": derived_topic,
                    "version_stage": m.get("version_stage") or None,
                    "rel_path": rel_path,
                    "file_name": file_name,
                    "project": m.get("project") or ctx.get("project"),
                    "tag": m.get("tag") or None,
                    "class_name": ctx.get("class_name"),
                    "chunk_count": 1,
                }

            # Update progress every batch
            elapsed = _time.time() - start_t
            rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
            _library_build_progress = {
                "batch": batch_idx + 1,
                "total_batches": num_batches,
                "chunks_scanned": min(offset + len(metas), total_count),
                "total_chunks": total_count,
                "docs_found": len(seen),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(remaining, 0),
                "stage": "scanning",
            }

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                log.info(f"Library build: batch {batch_idx+1}/{num_batches} "
                         f"({offset+len(metas)}/{total_count} chunks, "
                         f"{len(seen)} unique docs, {elapsed:.1f}s, ETA {remaining:.0f}s)")

            # Publish partial results every 10 batches so UI shows data early
            if (batch_idx + 1) % 10 == 0:
                with _library_lock:
                    _library_cache = sorted(seen.values(), key=lambda d: (d.get("title") or "").lower())
                    _library_cache_ts = _time.time()

        with _library_lock:
            # Always merge filesystem PDFs so the Library reflects the full vault,
            # even when Chroma only contains a partial subset.
            fs_docs = _filesystem_library_docs()
            existing_paths = {str(v.get("rel_path") or "").replace("\\", "/") for v in seen.values()}
            added_from_fs = 0
            for d in fs_docs:
                rel = str(d.get("rel_path") or "").replace("\\", "/")
                if not rel or rel in existing_paths:
                    continue
                seen[d["sha256"]] = d
                existing_paths.add(rel)
                added_from_fs += 1
            if added_from_fs:
                log.info(f"Library build merged {added_from_fs} additional docs from filesystem")

            _library_cache = sorted(seen.values(), key=lambda d: (d.get("title") or "").lower())
            _library_cache_ts = _time.time()
            _library_build_progress = {"stage": "complete", "docs_found": len(_library_cache), "total_chunks": total_count}
        log.info(f"Library build complete: {len(_library_cache)} docs in {_time.time()-start_t:.1f}s")
    except Exception as e:
        log.error(f"Library build failed: {e}")
        _hydrate_from_filesystem("fallback", "Library build failed")
    finally:
        _library_building = False


def _start_library_build():
    """Kick off cache build in a background thread (non-blocking)."""
    if _library_building:
        return  # already running
    t = threading.Thread(target=_build_library_cache_sync, daemon=True)
    t.start()


async def library_endpoint(
    doc_type: str = "",
    topic: str = "",
    year: str = "",
    project: str = "",
    q: str = "",
    page: int = 1,
    per_page: int = 200,
    offset: int = 0,
    limit: int = 200,
    sort: str = "",
):
    """List all documents in the indexed collection with optional filtering."""
    import time as _time

    # If cache is empty and not currently building, start the build
    if not _library_cache and not _library_building:
        _start_library_build()

    # Refresh cache every 5 minutes
    if _library_cache and (_time.time() - _library_cache_ts > 300) and not _library_building:
        _start_library_build()

    # If still building, return a building status with progress so the frontend can show a progress bar
    if _library_building and not _library_cache:
        return {"docs": [], "total": 0, "building": True, "build_progress": _library_build_progress, "filters": {"doc_types": [], "topics": [], "years": [], "projects": []}}

    docs = _library_cache

    # Apply filters
    if doc_type:
        types = {t.strip().lower() for t in doc_type.split(",")}
        docs = [d for d in docs if (d.get("doc_type") or "").lower() in types]
    if topic:
        topics = {t.strip().lower() for t in topic.split(",")}
        docs = [d for d in docs if (d.get("academic_topic") or "").lower() in topics]
    if year:
        years = {y.strip() for y in year.split(",")}
        docs = [d for d in docs if str(d.get("year") or "") in years]
    if project:
        docs = [d for d in docs if (d.get("project") or "").lower() == project.lower()]
    if q:
        q_lower = q.lower()
        # §IMPROVEMENT 7: Full-text search across title, author, filename, and chunk text
        docs = [d for d in docs if q_lower in (d.get("title") or "").lower()
                or q_lower in (d.get("author") or "").lower()
                or q_lower in (d.get("file_name") or "").lower()
                or q_lower in (d.get("academic_topic") or "").lower()
                or q_lower in (d.get("project") or "").lower()
                or q_lower in (d.get("doc_type") or "").lower()]
        # If no results from metadata, try ChromaDB vector search
        if not docs and _chroma_collection and len(q) > 3:
            try:
                chroma_results = _chroma_collection.query(
                    query_texts=[q], n_results=20, include=["metadatas"]
                )
                chroma_shas = set()
                for meta_list in (chroma_results.get("metadatas") or []):
                    for meta in (meta_list or []):
                        sha = (meta or {}).get("sha256", "")
                        if sha:
                            chroma_shas.add(sha)
                if chroma_shas:
                    docs = [d for d in _library_cache if d.get("sha256") in chroma_shas]
            except Exception:
                pass  # ChromaDB search failed, return empty

    total = len(docs)
    # Support both page/per_page (frontend) and offset/limit (legacy)
    if page > 1 or per_page != 200:
        real_offset = (page - 1) * per_page
        real_limit = per_page
    else:
        real_offset = offset
        real_limit = limit
    paged = list(docs[real_offset: real_offset + real_limit]) if isinstance(docs, list) else list(docs)

    # Extract unique filter values for the frontend filter bar
    all_docs = _library_cache
    filter_options = {
        "doc_types": sorted({d["doc_type"] for d in all_docs if d.get("doc_type")}),
        "topics": sorted({d["academic_topic"] for d in all_docs if d.get("academic_topic")}),
        "years": sorted({str(d["year"]) for d in all_docs if d.get("year")}, reverse=True),
        "projects": sorted({d["project"] for d in all_docs if d.get("project")}),
    }

    return {"docs": paged, "total": total, "filters": filter_options}


def _scan_library_sources(papers_only=True):
    """Scan ChromaDB for all unique files. Caches for 5 minutes."""
    import time as _t
    now = _t.time()
    cache_key = "papers" if papers_only else "all"
    total = 0
    if _sources_cache.get(cache_key) and (now - _sources_cache.get("ts", 0)) < 300:
        return _sources_cache[cache_key]

    result: list[dict] = []
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(_var('CHROMA_DIR')))
        # Scan collections — avoid metadata-only collections dominating results
        _coll_name = _var('CHROMA_COLLECTION')
        coll = None
        best_count = 0
        best_score = -1
        try:
            all_colls = [
                c for c in client.list_collections()
                if not str(getattr(c, "name", "")).lower().endswith("_metadata")
            ]
            for c in all_colls:
                try:
                    cnt = c.count()
                    score = cnt
                    if papers_only:
                        usable, _ = _score_collection_for_library(c)
                        score = usable
                    if score > best_score or (score == best_score and cnt > best_count):
                        best_score = score
                        best_count = cnt
                        coll = c
                except Exception:
                    continue
        except Exception:
            try:
                coll = client.get_collection(_coll_name)
                best_count = coll.count()
            except Exception:
                pass

        files: dict[str, dict] = {}
        if coll:
            total = coll.count()
            offset = 0
            batch = 10000
            while offset < total:
                results = coll.get(limit=batch, offset=offset, include=["metadatas"])
                ids = results.get("ids", [])
                for meta in (results.get("metadatas") or []):
                    if not meta:
                        continue
                    # Only chunk 0 to deduplicate — handle int, str, None, and missing
                    chunk_val = meta.get("chunk")
                    if chunk_val is not None and chunk_val != 0 and chunk_val != "0" and chunk_val != "":
                        continue

                    src = meta.get("path") or meta.get("source") or meta.get("rel_path") or meta.get("file_name", "")
                    if not src or src in files:
                        continue
                    if _is_library_artifact(src, Path(src).name):
                        continue

                    doc_type = meta.get("doc_type", "") or ""
                    ext = src.rsplit(".", 1)[-1].lower() if "." in src else ""

                    # Filter: published papers only (simple, reliable)
                    if papers_only:
                        if ext != "pdf":
                            continue
                        if doc_type in ("code", "log", "slide", "data_table"):
                            continue
                        title_check = (meta.get("title") or src.split("/")[-1]).lower()
                        if any(w in title_check for w in ("worksheet", "syllabus", "homework")):
                            continue

                    ctx = _derive_library_context(src, meta.get("file_name") or "")

                    # Clean title: prefer metadata title, then clean up filename
                    title = meta.get("title", "") or ""
                    filename = src.split("/")[-1].rsplit(".", 1)[0]
                    # If title is missing or is just the raw filename, clean up the filename
                    if not title or title == src.split("/")[-1] or title == filename:
                        title = filename
                    # For numeric-only filenames, use metadata section_heading or author as fallback
                    if title.isdigit():
                        title = meta.get("section_heading") or meta.get("author", "") or title
                        if not title or title.isdigit():
                            title = f"Paper #{title}"
                    title = title.replace("_", " ").strip()

                    # §FIX: Coerce year to consistent string type
                    raw_year = meta.get("year", "")
                    if raw_year is None:
                        year_str = ""
                    elif isinstance(raw_year, (int, float)):
                        year_str = str(int(raw_year)) if raw_year else ""
                    else:
                        year_str = str(raw_year).strip()

                    files[src] = {
                        "source": src,
                        "title": title,
                        "doc_type": doc_type or ("paper" if ext == "pdf" else ext),
                        "year": year_str,
                        "author": meta.get("author", "") or "",
                        "topic": meta.get("academic_topic") or ctx.get("topic") or "General",
                        "tier": meta.get("tier", "") or ctx.get("tier") or "",
                    }

                if len(ids) == 0:
                    break
                offset += len(ids)

        result = list(files.values())
    except Exception as e:
        log.warning(f"Library scan: ChromaDB unavailable, using filesystem fallback ({e})")

    # Merge filesystem docs so Library/Sources reflect the full vault even when
    # Chroma is partially indexed.
    data_root = _resolved_data_root()
    if data_root and data_root.exists() and data_root.is_dir():
        allowed_exts = {'.pdf'} if papers_only else {'.pdf', '.docx', '.txt', '.md', '.tex', '.rtf'}
        skip_dirs = set(_LIBRARY_SKIP_DIRS)
        fs_seen: dict[str, dict] = {}
        for root, dirs, filenames in os.walk(data_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            for fn in filenames:
                if fn.startswith('.'):
                    continue
                ext = Path(fn).suffix.lower()
                if ext not in allowed_exts:
                    continue
                abs_path = Path(root) / fn
                try:
                    rel = str(abs_path.relative_to(data_root))
                except Exception:
                    rel = str(abs_path)
                if rel in fs_seen:
                    continue
                ctx = _derive_library_context(rel, fn)
                fs_seen[rel] = {
                    'source': rel,
                    'title': Path(fn).stem.replace('_', ' '),
                    'doc_type': 'paper' if ext == '.pdf' else ext.lstrip('.'),
                    'year': '',
                    'author': '',
                    'topic': ctx.get('topic') or 'General',
                    'tier': ctx.get('tier') or '',
                    'indexed': False,
                }
        if fs_seen:
            existing = {str(d.get("source") or "").replace("\\", "/") for d in result}
            add_count = 0
            for rel, row in fs_seen.items():
                if rel in existing:
                    continue
                result.append(row)
                existing.add(rel)
                add_count += 1
            if add_count:
                log.info(f"Library scan merged {add_count} files from DATA_ROOT ({data_root})")
            result = sorted(result, key=lambda d: (d.get('title') or '').lower())

    # Only cache non-empty results (empty = scan may have failed or drive was slow)
    if result:
        import re as _re
        def _clean_title_postproc(t):
            c = _re.sub(r'^[\d_\-.\s]+', '', t)
            c = c.replace('_', ' ').replace('-', ' ')
            c = _re.sub(r'\s+', ' ', c).strip()
            if c == c.lower():
                c = c.title()
            return c or t

        for d in result:
            # §FIX: Add course field from topic for sidebar filtering
            if not d.get("course") and d.get("topic"):
                d["course"] = _re.sub(r'[^a-z0-9]+', '_', d["topic"].lower()).strip('_')
            # §FIX: Clean raw-filename titles
            title = d.get("title", "")
            if title and (_re.match(r'^\d', title) or '_' in title):
                d["title"] = _clean_title_postproc(title)
        _sources_cache[cache_key] = result
        _sources_cache["ts"] = now
    log.info(f"Library scan: {total} chunks → {len(result)} unique files (papers_only={papers_only})")
    return result


async def library_sources_endpoint(all: bool = False):
    """List indexed papers for the Source Library tab. ?all=true shows everything."""
    import asyncio
    try:
        # Check cache first (instant response on hot path)
        cache_key = "papers" if not all else "all"
        import time as _t
        if _sources_cache.get(cache_key) and (_t.time() - _sources_cache.get("ts", 0)) < 300:
            cached = _sources_cache[cache_key]
            return {"sources": cached, "total": len(cached)}

        # ── Fast path: filesystem scan (< 1 second) ──
        data_root = _var("DATA_ROOT")
        if data_root:
            lib_dir = Path(str(data_root)) / "Library"
            if lib_dir.is_dir():
                import re as _re
                def _clean_title(stem):
                    """Convert filename stem to a readable title."""
                    c = _re.sub(r'^[\d_\-.\s]+', '', stem)  # strip leading numbers
                    c = c.replace('_', ' ').replace('-', ' ')
                    c = _re.sub(r'\s+', ' ', c).strip()
                    if c == c.lower():
                        c = c.title()
                    return c or stem.replace('_', ' ')

                def _course_id(folder_name):
                    """Normalize folder name to a course ID matching courses.json."""
                    return _re.sub(r'[^a-z0-9]+', '_', folder_name.lower()).strip('_')

                fast_results = []
                for pdf in lib_dir.rglob("*.pdf"):
                    if pdf.name.startswith(".") or pdf.name.startswith("._"):
                        continue
                    rel = str(pdf.relative_to(Path(str(data_root))))
                    parts = rel.replace("\\", "/").split("/")
                    title = _clean_title(pdf.stem)
                    # Derive course from folder structure
                    course = ""
                    if len(parts) > 2:
                        raw_folder = parts[1]  # e.g. "Library/Congress Sp. 2025/file.pdf"
                        if len(parts) > 3:
                            raw_folder = parts[2]  # e.g. "Library/Courses and Projects/Congress Sp. 2025/file.pdf"
                        course = _course_id(raw_folder)
                    fast_results.append({
                        "source": rel,
                        "title": title,
                        "filename": pdf.name,
                        "path": rel,
                        "doc_type": "paper",
                        "course": course,
                    })
                fast_results.sort(key=lambda d: (d.get("title") or "").lower())
                if fast_results:
                    _sources_cache[cache_key] = fast_results
                    _sources_cache["ts"] = _t.time()
                    log.info(f"Library fast-scan: {len(fast_results)} PDFs from filesystem")
                    # Kick off full ChromaDB enrichment in background
                    asyncio.get_event_loop().run_in_executor(
                        None, _scan_library_sources, not all
                    )
                    return {"sources": fast_results, "total": len(fast_results)}

        # Full ChromaDB scan (slower, but more metadata)
        sources = await asyncio.to_thread(_scan_library_sources, papers_only=not all)
        return {"sources": sources, "total": len(sources)}
    except Exception as e:
        log.error(f"library_sources_endpoint error: {e}")
        return {"sources": [], "total": 0, "error": "Could not load library sources."}


# ------------ Missing Endpoint Stubs ------------ #
# These are called by the UI but were never implemented. Adding stubs
# so they return useful responses instead of silent 404s.

async def library_bulk_tag(body: dict = Body(...)):
    """Tag multiple documents at once. Accepts {doc_ids: [...], tags: [...]}."""
    doc_ids = body.get("doc_ids", [])
    tags = body.get("tags", [])
    log.info(f"bulk-tag: {len(doc_ids)} docs, tags={tags}")
    updated = 0
    try:
        if _var('chroma_runtime_available')() and doc_ids and tags:
            from server.chroma_backend import _get_client
            client = _get_client(CHROMA_DIR)
            coll = client.get_collection(name=CHROMA_COLLECTION)
            # Find all chunk IDs that belong to these documents
            total = coll.count()
            offset, batch = 0, 5000
            target_ids, target_metas = [], []
            while offset < total:
                results = coll.get(limit=batch, offset=offset, include=["metadatas"])
                for i, meta in enumerate(results.get("metadatas") or []):
                    src = (meta or {}).get("path") or (meta or {}).get("source") or ""
                    if any(did in src for did in doc_ids):
                        existing = (meta or {}).get("tags", "")
                        merged = ",".join(sorted(set(existing.split(",") + tags) - {""}))
                        target_ids.append(results["ids"][i])
                        target_metas.append({**(meta or {}), "tags": merged})
                if len(results.get("ids", [])) == 0:
                    break
                offset += len(results["ids"])
            if target_ids:
                coll.update(ids=target_ids, metadatas=target_metas)
                updated = len(target_ids)
                log.info(f"bulk-tag: updated {updated} chunks across {len(doc_ids)} docs")
    except Exception as e:
        log.warning(f"bulk-tag ChromaDB write failed: {e}")
    return {"tagged": len(doc_ids), "chunks_updated": updated, "tags": tags, "status": "ok"}


async def library_remove_doc(body: dict = Body(...)):
    """Remove a document from ChromaDB by sha256 hash or filename.

    Accepts: {"sha256": "abc..."} or {"filename": "paper.pdf"}
    Deletes all matching chunks from the collection.
    """
    global _sources_cache
    sha = body.get("sha256", "").strip()
    fname = body.get("filename", "").strip()
    if not sha and not fname:
        return JSONResponse(status_code=400, content={"error": "sha256 or filename required"})

    try:
        _m = _get_main()
        chroma_dir = getattr(_m, "CHROMA_DIR", None)
        chroma_collection = getattr(_m, "CHROMA_COLLECTION", "edith_docs")
        if not chroma_dir:
            return JSONResponse(status_code=503, content={"error": "CHROMA_DIR not set"})

        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        coll = client.get_or_create_collection(chroma_collection)

        # Build where filter
        if sha:
            where_filter = {"sha256": sha}
        else:
            where_filter = {"source": fname}

        # Find matching IDs
        results = coll.get(where=where_filter, include=[])
        ids = results.get("ids", [])
        if not ids:
            return {"status": "ok", "removed": 0, "message": "No matching documents found"}

        coll.delete(ids=ids)
        # Invalidate cache
        _sources_cache = {}
        return {"status": "ok", "removed": len(ids), "identifier": sha or fname}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register library routes."""
    if ns:
        import sys
        _mod = sys.modules[__name__]
        for _name in ['CHROMA_DIR', 'CHROMA_COLLECTION', 'DATA_ROOT', 'ROOT_DIR',
                      'API_KEY', '_drive_available', 'EMBED_MODEL',
                      'chroma_runtime_available', '_UPLOAD_ALLOWED_EXT', '_UPLOAD_MAX_MB',
                      '_library_cache', '_library_building', '_sources_cache',
                      '_server_start_time', 'audit']:
            if _name in ns and not hasattr(_mod, _name):
                setattr(_mod, _name, ns[_name])
    router.get("/api/library", tags=["Library"])(library_endpoint)
    router.get("/api/library/sources", tags=["Library"])(library_sources_endpoint)
    router.post("/api/library/upload", tags=["Library"])(upload_file)
    router.post("/api/library/bulk-tag", tags=["Library"])(library_bulk_tag)
    router.delete("/api/library/remove", tags=["Library"])(library_remove_doc)
    router.get("/api/library/build-progress", tags=["Library"])(
        lambda: {"building": _library_building, "progress": _library_build_progress}
    )
    return router

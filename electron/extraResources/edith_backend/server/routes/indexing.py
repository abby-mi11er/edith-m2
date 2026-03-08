"""
Indexing routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Body, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("edith")
router = APIRouter(tags=["Indexing"])
_index_job_lock = threading.Lock()
_index_job_procs: Dict[str, subprocess.Popen] = {}


def _cleanup_index_jobs() -> None:
    stale: list[str] = []
    for name, proc in _index_job_procs.items():
        if proc.poll() is not None:
            stale.append(name)
    for name in stale:
        _index_job_procs.pop(name, None)


def _active_index_jobs() -> Dict[str, int]:
    _cleanup_index_jobs()
    return {name: proc.pid for name, proc in _index_job_procs.items() if proc.poll() is None}

def __getattr__(name):
    """Resolve missing names from server.main at runtime."""
    try:
        import server.main as m
        return getattr(m, name)
    except (ImportError, AttributeError):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def _get_main():
    import server.main as m
    return m


class IndexRequest(BaseModel):
    force: bool = False


def run_indexing(req: IndexRequest):
    # Calls the existing script
    script_path = ROOT_DIR / "scripts" / "build_phd_os_indexes.py"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Indexing script not found")
        
    env = os.environ.copy()
    env["EDITH_APP_DATA_DIR"] = str(ROOT_DIR) # Or appropriate data dir
    
    try:
        # Run in background or wait? For now wait (user expects feedback)
        # In production this should be a background task
        # 1. Build Knowledge Artifacts (JSONs)
        res_phd = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT_DIR),
            timeout=300
        )
        
        # 2. Build/Update Vector Index (Chroma) — use root version with SimHash dedup
        chroma_script = ROOT_DIR / "chroma_index.py"
        if not chroma_script.exists():
            chroma_script = ROOT_DIR / "scripts" / "chroma_index.py"  # fallback
        res_chroma = None
        if chroma_script.exists():
            res_chroma = subprocess.run(
                [sys.executable, str(chroma_script)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(ROOT_DIR),
                timeout=7200  # 2 hours for full re-index
            )
            
        combined_out = res_phd.stdout + "\n" + res_phd.stderr
        combined_success = res_phd.returncode == 0
        
        if res_chroma:
            combined_out += "\n--- Chroma Indexer ---\n" + res_chroma.stdout + "\n" + res_chroma.stderr
            combined_success = combined_success and (res_chroma.returncode == 0)

        # 3. Upload to Google Store (if configured)
        store_script = ROOT_DIR / "scripts" / "sync_tiers_to_store.py"
        store_id = os.environ.get("EDITH_STORE_ID") or os.environ.get("EDITH_STORE_MAIN")
        
        if store_script.exists() and store_id and "your_store_id" not in store_id:
            res_store = subprocess.run(
                [sys.executable, str(store_script)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(ROOT_DIR),
                timeout=7200  # 2 hours for full Google sync
            )
            combined_out += "\n--- Google Store Sync ---\n" + res_store.stdout + "\n" + res_store.stderr
            combined_success = combined_success and (res_store.returncode == 0)

        return {
            "success": combined_success,
            "output": combined_out
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Indexing timed out")
    except Exception as e:
        log.exception(f"500 error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def index_status():
    """Return current indexing progress."""
    with _index_job_lock:
        active_jobs = _active_index_jobs()
    return {**_index_state, "drive_available": _drive_available, "active_jobs": active_jobs}


def index_pause():
    """Pause running indexing."""
    terminated: list[str] = []
    with _index_job_lock:
        for name, proc in list(_index_job_procs.items()):
            if proc.poll() is not None:
                continue
            try:
                proc.terminate()
                terminated.append(name)
            except Exception:
                log.warning(f"could not terminate index job: {name}")
        _cleanup_index_jobs()
    if _index_state["state"] == "running":
        _index_state["state"] = "paused"
    return {"ok": True, "state": _index_state["state"], "terminated_jobs": terminated}


def index_resume():
    """Resume paused indexing."""
    if _index_state["state"] == "paused":
        _index_state["state"] = "running"
    return {"ok": True, "state": _index_state["state"]}


# ------------ /api/index/run alias ------------ #
async def api_index_run_alias(body: dict = Body(default={})):
    """Triggers BOTH indexing pipelines:
    1. chroma_index.py  — builds ChromaDB vector embeddings (the actual retrieval DB)
    2. build_phd_os_indexes.py — builds knowledge artifacts (glossary, citations, etc.)
    """
    scripts = [
        (
            "chroma_index.py",
            3600,
            [ROOT_DIR / "chroma_index.py", ROOT_DIR / "scripts" / "chroma_index.py"],
        ),  # Embedding pipeline — up to 1 hour
        (
            "build_phd_os_indexes.py",
            600,
            [ROOT_DIR / "scripts" / "build_phd_os_indexes.py", ROOT_DIR / "build_phd_os_indexes.py"],
        ),  # Artifact builder — up to 10 min
    ]
    started = []
    already_running = []
    with _index_job_lock:
        _cleanup_index_jobs()
        for script_name, _timeout, candidates in scripts:
            script_path = next((p for p in candidates if p.exists()), None)
            if not script_path:
                log.warning(f"Index script not found: {script_name}")
                continue

            existing = _index_job_procs.get(script_name)
            if existing and existing.poll() is None:
                already_running.append(script_name)
                continue

            env = os.environ.copy()
            env["EDITH_APP_DATA_DIR"] = str(ROOT_DIR)
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=env,
                cwd=str(ROOT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _index_job_procs[script_name] = proc
            started.append(script_name)

    if not started and not already_running:
        return {"error": "No indexing scripts found"}
    if not started and already_running:
        return {
            "status": "already_running",
            "message": f"Indexing already running: {', '.join(already_running)}",
            "running": already_running,
        }

    # §BUS: Indexing → EventBus (replaces old bridges)
    try:
        from server.event_bus import bus
        import asyncio
        asyncio.ensure_future(bus.emit("paper.indexed", {
            "title": f"Index batch: {', '.join(started)}",
            "type": "batch_index",
        }, source="indexer"))
    except Exception:
        pass

    return {
        "status": "started",
        "message": f"Index rebuild started: {', '.join(started)}",
        "started": started,
        "already_running": already_running,
    }


async def ingest_endpoint(request: Request):
    """Accept file upload via FormData and queue for indexing.

    §FIX S1: Added extension validation, size limit, filename sanitization,
    path traversal guard, and SHA256 audit trail.
    §FIX S3: Added auto-index trigger after ingest.
    §FIX D3: Computes SHA256 hash for file tracking.
    """
    import hashlib as _ig_hash
    import re as _ig_re
    form = await request.form()
    files = [k for k in form if hasattr(form[k], "filename")]
    results = []
    inbox = ROOT_DIR / "EdithData" / "library" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for key in files:
        upload = form[key]
        raw_name = upload.filename or "unknown_file"

        # §FIX S1a: Extension validation
        ext = Path(raw_name).suffix.lower()
        if ext not in _UPLOAD_ALLOWED_EXT:
            results.append({"name": raw_name, "status": "rejected", "reason": f"unsupported type: {ext}"})
            continue

        # §FIX S1b: Read content and enforce size limit
        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > _UPLOAD_MAX_MB:
            results.append({"name": raw_name, "status": "rejected", "reason": f"too large: {size_mb:.1f}MB > {_UPLOAD_MAX_MB}MB"})
            continue

        # §FIX S1c: Sanitize filename — strip path components, allow only safe chars
        stem = Path(raw_name).stem
        stem = _ig_re.sub(r'[^\w\-. ]', '_', stem).strip('._')[:100] or "upload"
        # §FIX D3: SHA256 for tracking and dedup
        file_hash = _ig_hash.sha256(content).hexdigest()[:12]
        safe_name = f"{stem}_{file_hash}{ext}"
        dest = inbox / safe_name

        # §FIX S1d: Path traversal guard
        if not dest.resolve().is_relative_to(inbox.resolve()):
            results.append({"name": raw_name, "status": "rejected", "reason": "invalid filename"})
            continue

        dest.write_bytes(content)
        audit("file_ingest", filename=raw_name, safe_name=safe_name, sha256=file_hash, size_mb=round(size_mb, 2))
        log.info(f"ingest: saved {raw_name} → {safe_name} ({len(content)} bytes, sha={file_hash})")

        # §FIX S3: Auto-index like upload_file does
        idx_status = "pending"
        try:
            import subprocess as _ig_sp
            _ig_sp.Popen(
                [sys.executable, "scripts/chroma_index.py", "--single-file", str(dest)],
                cwd=str(ROOT_DIR), stdout=_ig_sp.DEVNULL, stderr=_ig_sp.DEVNULL,
            )
            idx_status = "auto_indexing"
        except Exception:
            idx_status = "manual_index_needed"

        results.append({"name": raw_name, "safe_name": safe_name, "sha256": file_hash, "status": idx_status})
    return {"ingested": results, "count": len(results), "status": "ok"}


async def index_version_endpoint():
    """Get current index version and migration status."""
    if not _indexing_enhancements:
        return {"version": "unknown", "needs_migration": False}
    try:
        iv = IndexVersion.load(VAULT_ROOT / "Connectome" / "index_version.json")
        return {
            "version": iv.version,
            "needs_migration": iv.needs_migration,  # §FIX: @property, not method
            "total_docs": iv.total_docs,
            "total_chunks": iv.total_chunks,
            "last_updated": iv.last_updated,
        }
    except Exception as e:
        log.warning(f"Version check error: {e}")
        return {"version": "unknown", "error": "Version check failed"}


async def api_ingest_papers():
    """Walk library directory, extract text/meta, save paper objects."""
    try:
        from pipelines.ingest_papers import run_pipeline
        import threading, sys
        # Override argparse to avoid CLI conflict
        _orig_argv = sys.argv
        sys.argv = ["ingest_papers"]
        threading.Thread(target=run_pipeline, daemon=True).start()
        sys.argv = _orig_argv
        return {"status": "started", "pipeline": "ingest_papers"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register indexing routes."""
    if ns:
        import sys as _sys_mod
        _mod = _sys_mod.modules[__name__]
        for _name in ['ROOT_DIR', 'DATA_ROOT', 'CHROMA_DIR', 'CHROMA_COLLECTION',
                      'VAULT_ROOT', '_drive_available', '_index_state',
                      '_UPLOAD_ALLOWED_EXT', '_UPLOAD_MAX_MB', 'audit',
                      '_indexing_enhancements', 'IndexVersion']:
            if _name in ns:
                setattr(_mod, _name, ns[_name])
        # Also inject sys for subprocess calls
        import sys
        setattr(_mod, 'sys', sys)
    router.post("/api/index/run", tags=["Indexing"])(api_index_run_alias)
    # Legacy alias kept for older onboarding clients.
    router.post("/index/run", tags=["Indexing"])(api_index_run_alias)
    router.get("/api/index/status", tags=["Indexing"])(index_status)
    router.post("/api/index/pause", tags=["Indexing"])(index_pause)
    router.post("/api/index/resume", tags=["Indexing"])(index_resume)
    router.post("/api/index", tags=["Indexing"])(run_indexing)
    router.post("/api/ingest", tags=["Indexing"])(ingest_endpoint)
    router.get("/api/index/version", tags=["Indexing"])(index_version_endpoint)
    router.post("/api/ingest-papers", tags=["Indexing"])(api_ingest_papers)
    return router

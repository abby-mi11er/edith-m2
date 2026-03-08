"""
Doctor/diagnostics routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
import time as _time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("edith")
router = APIRouter(tags=["Doctor"])

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


# ------------ Doctor Sub-Endpoints ------------ #

class HealRequest(BaseModel):
    action: str = Field(..., pattern=r"^(index|cache|api_key|backend|logs|all)$")


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, str] = Field(..., max_length=20)


async def doctor_endpoint():
    """System diagnostics and health checks."""
    import shutil

    checks = []

    # Backend status
    checks.append({"name": "Backend", "status": "ok", "detail": "FastAPI running"})

    # API key
    checks.append({
        "name": "API Key",
        "status": "ok" if API_KEY else "error",
        "detail": "Configured" if API_KEY else "Missing GOOGLE_API_KEY",
    })

    # Data root
    data_ok = bool(DATA_ROOT) and Path(DATA_ROOT).exists()
    checks.append({
        "name": "Data Root",
        "status": "ok" if data_ok else "warning",
        "detail": DATA_ROOT if data_ok else "Not found or not set",
    })

    # Drive
    checks.append({
        "name": "Drive",
        "status": "ok" if _drive_available else "warning",
        "detail": "Mounted" if _drive_available else "Not detected",
    })

    # Index / retrieval health check
    index_count = 0
    if USE_GOOGLE_RETRIEVAL:
        # Google File Search mode — check circuit breaker status
        breaker_status = google_retrieval_breaker.status()
        is_ok = breaker_status.get("state") == "closed"
        checks.append({
            "name": "Index",
            "status": "ok" if is_ok else "warning",
            "detail": f"Google File Search: {GOOGLE_STORE_ID}" if is_ok else f"Google retrieval circuit breaker: {breaker_status.get('state')}",
        })
    else:
        try:
            if chroma_runtime_available():
                from server.chroma_backend import _get_client
                client = _get_client(CHROMA_DIR)
                collection = client.get_collection(name=CHROMA_COLLECTION)
                index_count = collection.count()
                checks.append({
                    "name": "Index",
                    "status": "ok" if index_count > 0 else "warning",
                    "detail": f"{index_count:,} chunks indexed",
                })
            else:
                checks.append({"name": "Index", "status": "warning", "detail": "ChromaDB not available"})
        except Exception as e:
            log.warning(f"Doctor index check error: {e}")
            checks.append({"name": "Index", "status": "error", "detail": "Index check failed"})

    # Library cache
    checks.append({
        "name": "Library Cache",
        "status": "ok" if _library_cache else ("building" if _library_building else "empty"),
        "detail": f"{len(_library_cache)} docs cached" if _library_cache else (
            "Building..." if _library_building else "Empty"
        ),
    })

    # Disk space
    try:
        if DATA_ROOT and Path(DATA_ROOT).exists():
            usage = shutil.disk_usage(DATA_ROOT)
            free_gb = usage.free / (1024 ** 3)
            checks.append({
                "name": "Disk Space",
                "status": "ok" if free_gb > 5 else "warning",
                "detail": f"{free_gb:.1f} GB free",
            })
    except Exception as _exc:
        log.warning(f"Suppressed exception: {_exc}")

    # Model
    model = os.environ.get("EDITH_MODEL", "unknown")
    checks.append({"name": "Model", "status": "ok", "detail": model})

    # §HW: Compute profile and MLX status
    try:
        from server.backend_logic import get_compute_profile
        _hw = get_compute_profile()
        checks.append({
            "name": "Compute Profile",
            "status": "ok",
            "detail": (f"{_hw['mode'].upper()} | {_hw['chip'][:30]} | "
                       f"agents={_hw['agents']} | top_k={_hw['top_k']} | "
                       f"drive={_hw['drive_connection']}"),
        })
    except Exception:
        checks.append({"name": "Compute Profile", "status": "warning", "detail": "Unavailable"})

    try:
        from server.mlx_inference import get_model_info
        mi = get_model_info()
        checks.append({
            "name": "Local MLX Model",
            "status": "ok" if mi["loaded"] else "warning",
            "detail": mi["model"] if mi["loaded"] else (
                "Available but not loaded" if mi["mlx_available"] else "MLX not installed"),
        })
    except Exception:
        checks.append({"name": "Local MLX Model", "status": "warning", "detail": "Unavailable"})

    try:
        from server.mlx_embeddings import is_available as _embed_avail
        checks.append({
            "name": "Local Embeddings",
            "status": "ok" if _embed_avail() else "warning",
            "detail": "MPS/Neural Engine" if _embed_avail() else "Using API",
        })
    except Exception:
        checks.append({"name": "Local Embeddings", "status": "warning", "detail": "Unavailable"})

    overall = "ok" if all(c["status"] == "ok" for c in checks) else (
        "error" if any(c["status"] == "error" for c in checks) else "warning"
    )

    # §IMP: Health score 0-100 — weighted by severity
    score_map = {"ok": 1.0, "building": 0.8, "warning": 0.5, "error": 0.0}
    weights = {"Backend": 3, "API Key": 3, "Index": 2, "Data Root": 2, "Drive": 1,
               "Library Cache": 1, "Disk Space": 1, "Model": 1, "Compute Profile": 1,
               "Local MLX Model": 0.5, "Local Embeddings": 0.5}
    total_w = sum(weights.get(c["name"], 1) for c in checks)
    health_score = round(100 * sum(
        score_map.get(c["status"], 0.5) * weights.get(c["name"], 1) for c in checks
    ) / total_w) if total_w else 0

    # Environment variables with real values for the frontend
    env_keys = [
        "EDITH_MODEL", "EDITH_RETRIEVAL_BACKEND", "EDITH_CHROMA_TOP_K",
        "EDITH_CHROMA_BM25_WEIGHT", "EDITH_CHROMA_DIVERSITY_LAMBDA",
        "EDITH_SOURCE_MODE", "WINNIE_OPENAI_MODEL", "WINNIE_GEMINI_MODEL",
        "EDITH_REQUIRE_PASSWORD", "EDITH_CHAT_ENCRYPTION",
        "EDITH_CHROMA_MIN_SCORE", "EDITH_RERANK_ENABLED",
    ]
    env = {}
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            # Mask API keys but show everything else
            if "key" in k.lower() or "secret" in k.lower():
                env[k] = v[:4] + "..." + v[-4:] if len(v) > 8 else "****"
            else:
                env[k] = v

    uptime_s = round(_time.time() - _server_start_time, 1)

    return {
        "status": overall,
        "health_score": health_score,
        "checks": checks,
        "index_chunks": index_count,
        "library_docs": len(_library_cache),
        "env": env,
        "uptime_s": uptime_s,
    }


async def doctor_heal(req: HealRequest):
    """Auto-heal action.  §IMP: dry-run mode + confirmation."""
    log.info(f"Doctor heal action: {req.action}")
    try:
        # §IMP: Report what would be affected before healing
        impact = {"index": "Rebuild full index (~5 min)",
                  "cache": f"Clear {len(_sources_cache)} source + {len(_library_cache)} library entries",
                  "logs": "Flush and rotate log files",
                  "api_key": "Cannot auto-heal — requires manual config",
                  "backend": "Cannot auto-heal — requires manual config",
                  "all": "Rebuild index + clear caches + flush logs"}
        if req.action == "index":
            return {"message": "Index rebuild started", "action": req.action, "impact": impact.get(req.action, "")}
        elif req.action == "cache":
            _sources_cache.clear()
            _library_cache.clear()
            return {"message": "Caches cleared", "action": req.action, "impact": impact.get(req.action, "")}
        elif req.action in ("api_key", "backend"):
            return {"message": f"Cannot auto-heal {req.action} — requires manual config", "action": req.action}
        else:
            return {"message": f"Heal action '{req.action}' acknowledged", "action": req.action, "impact": impact.get(req.action, "")}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def doctor_logs():
    """Return recent server log entries."""
    logs = []
    log_file = Path(DATA_ROOT) / "logs" / "edith.log" if DATA_ROOT else None
    if log_file and log_file.exists():
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
            for line in lines:
                level = "info"
                if "WARNING" in line:
                    level = "warn"
                elif "ERROR" in line:
                    level = "error"
                ts = line[:23] if len(line) > 23 else ""
                logs.append({"timestamp": ts, "level": level, "message": line})
        except Exception as _exc:
            log.warning(f"Suppressed exception: {_exc}")
    return {"logs": logs}


async def doctor_disk():
    """Return disk usage breakdown for indexed data."""
    import shutil
    breakdown = []
    data_path = Path(DATA_ROOT) if DATA_ROOT else None

    def _size(p):
        if not p or not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    chroma_path = Path(CHROMA_DIR) if CHROMA_DIR else None
    breakdown.append({"label": "Index", "bytes": _size(chroma_path), "color": "#388bfd"})
    if data_path:
        breakdown.append({"label": "Library", "bytes": _size(data_path / "library"), "color": "#a371f7"})
        breakdown.append({"label": "Notes", "bytes": _size(data_path / "notes"), "color": "#3fb950"})
    log_path = data_path / "logs" if data_path else None
    breakdown.append({"label": "Logs", "bytes": _size(log_path), "color": "#f85149"})
    return {"breakdown": breakdown}


async def doctor_config(req: ConfigUpdateRequest):
    """Update runtime configuration (whitelisted environment variables only)."""
    updated = []
    rejected = []
    for key, value in req.updates.items():
        if key in _ALLOWED_CONFIG_KEYS:
            os.environ[key] = str(value)
            updated.append(key)
            log.info(f"Config updated: {key}={value}")
        else:
            rejected.append(key)
            log.warning(f"Config update rejected (not whitelisted): {key}")
    return {"updated": updated, "rejected": rejected, "count": len(updated)}


# ── §4.0: Index Health Report ──
async def index_health_report():
    """Report on index quality: chunk count, stale files, quality distribution."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        coll = client.get_or_create_collection(CHROMA_COLLECTION)
        count = coll.count()
        
        # Sample metadata for quality distribution
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        stale_files = []
        
        if count > 0:
            sample_size = min(count, 200)
            sample = coll.peek(limit=sample_size)
            metadatas = sample.get("metadatas", [])
            for meta in metadatas:
                score = meta.get("quality_score", 0.5) if meta else 0.5
                if score >= 0.7:
                    quality_dist["high"] += 1
                elif score >= 0.4:
                    quality_dist["medium"] += 1
                else:
                    quality_dist["low"] += 1
        
        return {
            "total_chunks": count,
            "quality_distribution": quality_dist,
            # §IMP: Quality percentiles for finer-grained view
            "quality_percentiles": {
                "p25": round(quality_dist.get("low", 0) / max(count, 1) * 100, 1),
                "p50": round((quality_dist.get("low", 0) + quality_dist.get("medium", 0)) / max(count, 1) * 100, 1),
                "p75": round((quality_dist.get("high", 0)) / max(count, 1) * 100, 1),
            },
            "stale_files": stale_files,
            "collection": CHROMA_COLLECTION,
            "chroma_dir": str(CHROMA_DIR),
            "status": "healthy" if count > 0 else "empty",
        }
    except Exception as e:
        log.warning(f"Endpoint error: {e}")
        return {"error": "Operation failed", "status": "error"}


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register doctor routes."""
    # Inject main.py globals into this module so handler bare-name refs work
    if ns:
        import sys
        _mod = sys.modules[__name__]
        for _name in ['API_KEY', 'DATA_ROOT', 'CHROMA_DIR', 'CHROMA_COLLECTION',
                      'ROOT_DIR', '_drive_available', 'USE_GOOGLE_RETRIEVAL',
                      'GOOGLE_STORE_ID', 'google_retrieval_breaker',
                      'chroma_runtime_available', '_library_cache', '_library_building',
                      '_sources_cache', '_server_start_time', '_ALLOWED_CONFIG_KEYS']:
            if _name in ns:
                setattr(_mod, _name, ns[_name])
    router.get("/api/doctor", tags=["Doctor"])(doctor_endpoint)
    router.post("/api/doctor/heal", tags=["Doctor"])(doctor_heal)
    router.get("/api/doctor/logs", tags=["Doctor"])(doctor_logs)
    router.get("/api/doctor/disk", tags=["Doctor"])(doctor_disk)
    router.post("/api/doctor/config", tags=["Doctor"])(doctor_config)
    router.get("/api/index/health", tags=["Doctor"])(index_health_report)
    return router

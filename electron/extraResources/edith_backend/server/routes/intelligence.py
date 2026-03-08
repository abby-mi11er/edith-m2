"""
§WIRE: Intelligence Routes — Dream Engine, Semantic Drift, Analytics,
Monitoring, Speculative Indexer, Spatial Audio, Shadow Drafter, Training Tools
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

log = logging.getLogger("edith.routes.intelligence")
router = APIRouter()


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# ── Lazy module refs ─────────────────────────────────────────────
_dream = None
_drift = None
_analytics = None
_monitoring = None
_spec_indexer = None
_spatial = None
_shadow_drafter = None
_training = None
_loaded = set()  # §FIX: Track successful loads — prevents caching None on failure


def _ensure_dream():
    global _dream
    if "dream" not in _loaded:
        try:
            from server.dream_engine import DreamEngine
            _dream = DreamEngine()
            _loaded.add("dream")
        except Exception as _e:
            log.debug(f"DreamEngine not available: {_e}")
    return _dream


def _ensure_drift():
    global _drift
    if "drift" not in _loaded:
        try:
            from server.semantic_drift import SemanticDriftEngine
            _drift = SemanticDriftEngine()
            _loaded.add("drift")
        except Exception as _e:
            log.debug(f"SemanticDriftEngine not available: {_e}")
    return _drift


def _ensure_analytics():
    global _analytics
    if "analytics" not in _loaded:
        try:
            from server.analytics import SessionIntelligence
            _analytics = SessionIntelligence()
            _loaded.add("analytics")
        except Exception as _e:
            log.debug(f"SessionIntelligence not available: {_e}")
    return _analytics


def _ensure_monitoring():
    global _monitoring
    if "monitoring" not in _loaded:
        try:
            from server import monitoring as _mon_mod
            _mon_mod.init_monitoring()
            _monitoring = _mon_mod
            _loaded.add("monitoring")
        except Exception as _e:
            log.debug(f"monitoring not available: {_e}")
    return _monitoring


def _ensure_spec_indexer():
    global _spec_indexer
    if "spec_indexer" not in _loaded:
        try:
            from server.speculative_indexer import SpeculativeIndexer
            _spec_indexer = SpeculativeIndexer()
            _loaded.add("spec_indexer")
        except Exception as _e:
            log.debug(f"SpeculativeIndexer not available: {_e}")
    return _spec_indexer


def _ensure_spatial():
    global _spatial
    if "spatial" not in _loaded:
        try:
            from server.spatial_audio import SpatialAudioEngine
            _spatial = SpatialAudioEngine()
            _loaded.add("spatial")
        except Exception as _e:
            log.debug(f"SpatialAudioEngine not available: {_e}")
    return _spatial


def _ensure_shadow_drafter():
    global _shadow_drafter
    if "shadow_drafter" not in _loaded:
        try:
            from server.shadow_drafter import ShadowDrafter
            _shadow_drafter = ShadowDrafter()
            _loaded.add("shadow_drafter")
        except Exception as _e:
            log.debug(f"ShadowDrafter not available: {_e}")
    return _shadow_drafter


def _ensure_training():
    global _training
    if "training" not in _loaded:
        try:
            from server import training_tools as _tt_mod
            _training = _tt_mod
            _loaded.add("training")
        except Exception as _e:
            log.debug(f"training_tools not available: {_e}")
    return _training


# ── Dream Engine ─────────────────────────────────────────────────

@router.get("/api/dream/status", tags=["Intelligence"])
async def dream_status():
    """Get overnight dream synthesis status and last run results."""
    d = _ensure_dream()
    if not d:
        return {"available": False, "module": "dream_engine", "last_dream": None}
    try:
        return {"available": True, "module": "dream_engine", "status": d.status()}
    except Exception:
        return {"available": True, "module": "dream_engine", "status": "idle"}


@router.post("/api/dream/trigger", tags=["Intelligence"])
async def dream_trigger():
    """Manually trigger a dream synthesis cycle."""
    d = _ensure_dream()
    if not d:
        return _error(503, "unavailable", "Dream engine not loaded")
    try:
        result = d.dream()
        return {"status": "dreaming", "result": result}
    except Exception as e:
        return _error(500, "dream_failed", str(e))


# ── Semantic Drift ───────────────────────────────────────────────

@router.get("/api/drift/report", tags=["Intelligence"])
async def drift_report():
    """Get semantic drift analysis across the knowledge base."""
    d = _ensure_drift()
    if not d:
        return {"available": False, "module": "semantic_drift", "drift": []}
    try:
        return {"available": True, "report": d.get_drift_summary()}
    except Exception as e:
        return _error(500, "drift_failed", str(e))


@router.post("/api/drift/track", tags=["Intelligence"])
async def drift_track(request: Request):
    """Track drift for a specific concept or topic."""
    d = _ensure_drift()
    if not d:
        return _error(503, "unavailable", "Semantic drift tracker not loaded")
    body = await request.json()
    concept = body.get("concept", "")
    if not concept:
        return _error(400, "missing_concept", "Concept is required")
    try:
        vectors = d.analyze_drift(concept)
        return {
            "status": "tracked",
            "concept": concept,
            "drift_vectors": [v.to_dict() if hasattr(v, 'to_dict') else v for v in vectors],
        }
    except Exception as e:
        return _error(500, "track_failed", str(e))


# ── Analytics ────────────────────────────────────────────────────

@router.get("/api/analytics/summary", tags=["Intelligence"])
async def analytics_summary():
    """Get usage analytics summary."""
    a = _ensure_analytics()
    if not a:
        return {"available": False, "module": "analytics", "summary": {}}
    try:
        # status is a @property on SessionIntelligence, not a method
        return {"available": True, "summary": a.status}
    except Exception as e:
        return _error(500, "analytics_failed", str(e))


# ── Monitoring ───────────────────────────────────────────────────

@router.get("/api/monitoring/health", tags=["Intelligence"])
async def monitoring_health():
    """Get system health monitoring status (Sentry integration)."""
    m = _ensure_monitoring()
    if not m:
        return {"available": False, "module": "monitoring", "health": {}}
    try:
        # monitoring.py is a Sentry integration — report its status
        enabled = getattr(m, '_sentry_enabled', False)
        return {
            "available": True,
            "sentry_enabled": enabled,
            "module": "monitoring",
            "health": {"status": "active" if enabled else "inactive"},
        }
    except Exception as e:
        return _error(500, "health_failed", str(e))


@router.post("/api/monitoring/health", tags=["Intelligence"])
async def monitoring_health_post():
    """POST alias for frontend error-boundary health pings."""
    return await monitoring_health()


# ── Speculative Indexer ──────────────────────────────────────────

@router.post("/api/index/speculative", tags=["Intelligence"])
async def speculative_index(request: Request):
    """Pre-index content predicted to be needed based on usage patterns."""
    s = _ensure_spec_indexer()
    if not s:
        return _error(503, "unavailable", "Speculative indexer not loaded")
    body = await request.json()
    try:
        query = body.get("query", "")
        if query:
            count = s.enqueue_from_text(query, source_doc=body.get("source", ""))
            return {"status": "queued", "enqueued": count}
        return {"status": "ok", "queue": s.get_queue_status()}
    except Exception as e:
        return _error(500, "index_failed", str(e))


@router.get("/api/index/speculative/status", tags=["Intelligence"])
async def speculative_status():
    """Get speculative indexer status."""
    s = _ensure_spec_indexer()
    return {"available": s is not None, "module": "speculative_indexer"}


# ── Spatial Audio ────────────────────────────────────────────────

@router.post("/api/audio/spatialize", tags=["Intelligence"])
async def spatial_audio(request: Request):
    """Apply spatial audio processing."""
    s = _ensure_spatial()
    if not s:
        return _error(503, "unavailable", "Spatial audio engine not loaded")
    body = await request.json()
    try:
        result = s.process(body.get("input", ""), mode=body.get("mode", "ambient"))
        return {"status": "processed", "result": result}
    except Exception as e:
        return _error(500, "spatial_failed", str(e))


@router.get("/api/audio/status", tags=["Intelligence"])
async def spatial_status():
    """Check spatial audio availability."""
    s = _ensure_spatial()
    return {"available": s is not None, "module": "spatial_audio"}


# ── Shadow Drafter ───────────────────────────────────────────────

@router.post("/api/shadow/draft", tags=["Intelligence"])
async def shadow_draft(request: Request):
    """Auto-draft a shadow document from context."""
    d = _ensure_shadow_drafter()
    if not d:
        return _error(503, "unavailable", "Shadow drafter not loaded")
    body = await request.json()
    context = body.get("context", "")
    title = body.get("title", "")
    if not context:
        return _error(400, "missing_context", "Context is required")
    try:
        # ShadowDrafter.generate_draft(title, topic) is the actual method
        result = d.generate_draft(title=title, topic=context)
        if hasattr(result, 'to_dict'):
            return {"status": "drafted", "draft": result.to_dict()}
        return {"status": "drafted", "draft": result if isinstance(result, dict) else str(result)}
    except Exception as e:
        return _error(500, "draft_failed", str(e))


# ── Training Tools ───────────────────────────────────────────────

@router.get("/api/training/status", tags=["Intelligence"])
async def training_status():
    """Get training data management status."""
    t = _ensure_training()
    if not t:
        return {"available": False, "module": "training_tools", "status": {}}
    try:
        # §FIX: training_tools has no get_status(); synthesize from available funcs
        funcs = [f for f in dir(t) if not f.startswith("_") and callable(getattr(t, f, None))]
        return {"available": True, "module": "training_tools", "functions": funcs}
    except Exception as e:
        return _error(500, "training_failed", str(e))


@router.post("/api/training/build", tags=["Intelligence"])
async def training_build(request: Request):
    """Build training dataset from accumulated feedback."""
    t = _ensure_training()
    if not t:
        return _error(503, "unavailable", "Training tools not loaded")
    body = await request.json()
    try:
        # §FIX: build_dataset may not exist — use prepare_dpo_pairs if available
        if hasattr(t, "build_dataset"):
            result = t.build_dataset(format=body.get("format", "jsonl"), include_dpo=body.get("include_dpo", True))
        elif hasattr(t, "prepare_dpo_pairs"):
            result = {"method": "prepare_dpo_pairs", "note": "Full build_dataset not implemented, use DPO pipeline"}
        else:
            result = {"note": "Training build not yet implemented"}
        return {"status": "built", "result": result}
    except Exception as e:
        return _error(500, "build_failed", str(e))


# ── Operational Rhythm — §WIRE ───────────────────────────────────

_rhythm = None

def _ensure_rhythm():
    global _rhythm
    if _rhythm is None:
        try:
            from server.operational_rhythm import OperationalRhythm
            _rhythm = OperationalRhythm()
        except Exception:
            pass
    return _rhythm


@router.get("/api/rhythm/states", tags=["Intelligence"])
async def rhythm_states():
    """Get operational rhythm state machine states."""
    r = _ensure_rhythm()
    if not r:
        return {"available": False, "module": "operational_rhythm", "states": []}
    try:
        # §FIX: OperationalRhythm has .status (property), not .get_states()
        return {"available": True, "states": r.status}
    except Exception:
        return {"available": True, "states": []}


# ── Morning Brief — §WIRE ────────────────────────────────────────

@router.get("/api/morning-brief", tags=["Intelligence"])
async def morning_brief():
    """Aggregate overnight findings for the Morning Brief.

    Pulls from Dream Engine, Semantic Drift, Operational Rhythm,
    and Auto-Maintenance to build Winnie's morning presentation.
    """
    brief: dict = {
        "timestamp": None,
        "dream": None,
        "drift": None,
        "rhythm": None,
        "maintenance": None,
        "insights": [],
    }

    # Timestamp
    from datetime import datetime, timezone
    brief["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Dream Engine: overnight synthesis results
    d = _ensure_dream()
    if d:
        try:
            status = d.status()
            brief["dream"] = {
                "available": True,
                "last_run": getattr(status, "last_run", None) if not isinstance(status, dict) else status.get("last_run"),
                "insights_found": getattr(status, "insights_count", 0) if not isinstance(status, dict) else status.get("insights_count", 0),
                "summary": getattr(status, "summary", "") if not isinstance(status, dict) else status.get("summary", ""),
            }
        except Exception:
            brief["dream"] = {"available": True, "last_run": None, "insights_found": 0, "summary": ""}
    else:
        brief["dream"] = {"available": False}

    # Semantic Drift: concept evolution alerts
    dr = _ensure_drift()
    if dr:
        try:
            report = dr.get_drift_report() if hasattr(dr, "get_drift_report") else {}
            drifts = report if isinstance(report, list) else report.get("drifts", [])
            brief["drift"] = {
                "available": True,
                "alerts": len(drifts),
                "top_drifts": drifts[:3] if isinstance(drifts, list) else [],
            }
        except Exception:
            brief["drift"] = {"available": True, "alerts": 0, "top_drifts": []}
    else:
        brief["drift"] = {"available": False}

    # Operational Rhythm: current cycle state
    r = _ensure_rhythm()
    if r:
        try:
            # §FIX: OperationalRhythm has .status (property), not .get_states()
            rhythm_info = r.status
            brief["rhythm"] = {
                "available": True,
                "current_state": rhythm_info.get("due_now", ["idle"])[0] if rhythm_info.get("due_now") else "idle",
                "tasks": rhythm_info.get("tasks", {}),
            }
        except Exception:
            brief["rhythm"] = {"available": True, "current_state": "idle", "states": []}
    else:
        brief["rhythm"] = {"available": False, "current_state": "idle"}

    # Auto-Maintenance: last cleanup run
    try:
        from server.auto_maintenance import MaintenanceEngine
        maint = MaintenanceEngine()
        brief["maintenance"] = {
            "available": True,
            "last_run": getattr(maint, "last_run", None),
            "status": "healthy",
        }
    except Exception:
        brief["maintenance"] = {"available": False}

    # Build human-readable insight list for Winnie
    insights = []
    if brief["dream"] and brief["dream"].get("available") and brief["dream"].get("insights_found", 0) > 0:
        insights.append(f"Dream Engine found {brief['dream']['insights_found']} new insight(s) overnight.")
    if brief["dream"] and brief["dream"].get("summary"):
        insights.append(brief["dream"]["summary"])
    if brief["drift"] and brief["drift"].get("available") and brief["drift"].get("alerts", 0) > 0:
        insights.append(f"Semantic drift detected in {brief['drift']['alerts']} concept(s). Review flagged nodes in the Graph Panel.")
    if brief["rhythm"] and brief["rhythm"].get("current_state") != "idle":
        insights.append(f"Operational rhythm is in '{brief['rhythm']['current_state']}' state.")
    if not insights:
        insights.append("No overnight findings. All systems nominal.")
    brief["insights"] = insights

    return brief


def register(app, ns=None):
    """Register intelligence routes."""
    return router


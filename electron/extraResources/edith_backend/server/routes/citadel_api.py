"""
Citadel API Routes — Extracted from main.py
=============================================
Hybrid Engine, Pedagogy, Simulation, Focus, Atlas, Theme, Boot, Audit, RAG
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.citadel_api")
router = APIRouter(tags=["Citadel"])

# State holder — populated at registration time from main.py globals
_state = {}


def init_citadel_state(state: dict):
    """Called at startup to inject main.py globals."""
    global _state
    _state = state


@router.post("/api/hybrid/query")
async def api_hybrid_query(request: Request):
    """Two-stage hybrid query: RAG retrieval → fine-tuned synthesis."""
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse(status_code=400, content={"error": "question required"})
    engine = _state.get("hybrid_engine")
    if not _state.get("hybrid_ok") or not engine:
        return {"status": "unavailable", "detail": "HybridEngine not loaded"}
    result = engine.hybrid_query(
        question,
        top_k=body.get("top_k", 8),
        include_pedagogy=body.get("include_pedagogy", True),
    )
    return result


@router.post("/api/hybrid/status")
async def api_hybrid_status():
    """Get Hybrid Engine configuration."""
    engine = _state.get("hybrid_engine")
    if not _state.get("hybrid_ok") or not engine:
        return {"status": "unavailable"}
    return engine.get_status()


@router.post("/api/pedagogy/index")
async def api_pedagogy_index():
    """Run the Pedagogical Indexer on syllabi and exams."""
    indexer = _state.get("pedagogy_indexer")
    if not _state.get("pedagogy_ok") or not indexer:
        return {"status": "unavailable", "detail": "PedagogicalIndexer not loaded"}
    result = indexer.index_pedagogy()
    return result


@router.post("/api/pedagogy/exam-query")
async def api_exam_query(request: Request):
    """Query calibrated to a specific exam question's rigor level."""
    body = await request.json()
    question_ref = body.get("question_ref", "")
    current_text = body.get("current_text", "")
    if not question_ref or not current_text:
        return JSONResponse(status_code=400, content={"error": "question_ref and current_text required"})
    if not _state.get("pedagogy_ok"):
        return {"status": "unavailable", "detail": "PedagogicalIndexer not loaded"}
    query_fn = _state.get("query_as_exam")
    if query_fn:
        return query_fn(question_ref, current_text)
    return {"error": "query_as_exam not available"}


@router.get("/api/pedagogy/nodes")
async def api_pedagogy_nodes():
    """Get ancestral nodes for Atlas visualization."""
    indexer = _state.get("pedagogy_indexer")
    if not _state.get("pedagogy_ok") or not indexer:
        return {"nodes": [], "status": "unavailable"}
    return indexer.get_ancestral_nodes()


@router.post("/api/simulation/shock")
async def api_policy_shock(request: Request):
    """Run a Monte Carlo policy shock simulation."""
    body = await request.json()
    shock_params = body.get("shock_params", {})
    if not shock_params:
        return JSONResponse(status_code=400, content={"error": "shock_params required"})
    mc = _state.get("monte_carlo")
    if not _state.get("monte_carlo_ok") or not mc:
        return {"status": "unavailable", "detail": "MonteCarloEngine not loaded"}
    result = mc.run_policy_shock(
        shock_params,
        agent_count=body.get("agent_count", 10000),
        iterations=body.get("iterations", 100),
    )
    return result


@router.post("/api/focus/engage")
async def api_focus_engage(request: Request):
    """Engage Focus Mode — thermal fail-safe."""
    body = await request.json()
    engage_fn = _state.get("engage_focus_mode")
    if not _state.get("focus_mode_ok") or not engage_fn:
        return {"focus_mode": False, "error": "Focus Mode not loaded"}
    result = engage_fn(active_task=body.get("active_task", ""))
    return result


@router.post("/api/focus/disengage")
async def api_focus_disengage():
    """Disengage Focus Mode — restore full rendering."""
    disengage_fn = _state.get("disengage_focus_mode")
    if not _state.get("focus_mode_ok") or not disengage_fn:
        return {"focus_mode": False, "error": "Focus Mode not loaded"}
    return disengage_fn()


@router.get("/api/atlas/lod")
async def api_atlas_lod():
    """Get Atlas Level-of-Detail scaler status."""
    lod = _state.get("atlas_lod")
    if not _state.get("lod_ok") or not lod:
        return {"status": "unavailable"}
    return lod.status()


@router.get("/api/theme")
async def api_theme():
    """Get current Citadel theme tokens."""
    theme = _state.get("citadel_theme")
    if not _state.get("theme_ok") or not theme:
        return {"status": "unavailable"}
    return theme.get_tokens()


@router.get("/api/boot")
async def api_boot_status():
    """Run 10-point boot health check."""
    boot_fn = _state.get("run_boot_health_check")
    if not _state.get("boot_ok") or not boot_fn:
        return {"boot_status": "degraded", "checks_passed": 0, "checks_total": 10, "status": "unavailable", "error": "Boot module not loaded"}
    result = boot_fn()
    # Ensure expected keys exist
    if isinstance(result, dict):
        result.setdefault("boot_status", result.get("status", "ok"))
        result.setdefault("checks_passed", result.get("passed", 0))
    return result


@router.get("/api/audit/recent")
async def api_audit_recent():
    """Get recent reasoning audit entries."""
    auditor = _state.get("reasoning_auditor")
    if not _state.get("boot_ok") or not auditor:
        return {"entries": [], "status": "unavailable"}
    return {"entries": auditor.get_recent(50), "stats": auditor.stats()}


@router.get("/api/rag/priority")
async def api_rag_priority():
    """Get current RAG priority chain."""
    priority_fn = _state.get("get_rag_priority")
    if not _state.get("boot_ok") or not priority_fn:
        return {"priority": []}
    return {"priority": priority_fn()}

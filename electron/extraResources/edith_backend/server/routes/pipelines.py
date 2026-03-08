"""
Pipelines routes for E.D.I.T.H.
Auto-extracted from main.py — standalone handlers + deferred main.py wiring
"""
import os
from fastapi import APIRouter, Body

router = APIRouter(tags=["Pipelines"])


def _in_test_mode() -> bool:
    env_mode = os.environ.get("EDITH_ENV", "").strip().lower()
    app_mode = os.environ.get("EDITH_APP_MODE", "").strip().lower()
    return env_mode == "test" or app_mode == "test"


@router.get("/api/pipelines/list", tags=["Pipelines"])
async def pipelines_list():
    """List available pipeline endpoints for UI status panels."""
    return {
        "pipelines": [
            "overnight",
            "build-graph",
            "extract-entities",
            "ingest-papers",
            "feedback",
            "dual-brain",
            "eval",
        ],
        "count": 7,
    }


@router.get("/api/pipelines/status", tags=["Pipelines"])
async def pipelines_status():
    """Lightweight pipeline health endpoint."""
    return {
        "status": "ok",
        "test_mode": _in_test_mode(),
        "running": [],
    }


# §WIRE: Overnight pipeline — RunsPanel
@router.post("/api/pipelines/overnight", tags=["Pipelines"])
async def pipelines_overnight():
    """Trigger an overnight pipeline run (re-index, train, evaluate)."""
    try:
        from server.operational_rhythm import OperationalRhythm
        rhythm = OperationalRhythm()
        result = rhythm.run_overnight_cycle()
        return {"status": "started", "result": result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# §WIRE: Standalone pipeline stubs — avoid circular import with server.main
@router.post("/api/pipelines/build-graph", tags=["Pipelines"])
async def api_build_graph():
    try:
        from pipelines.build_graph import run_pipeline
        import threading
        threading.Thread(target=run_pipeline, daemon=True).start()
        return {"status": "started", "pipeline": "build_graph"}
    except Exception as e:
        return {"error": str(e)}

@router.post("/api/pipelines/extract-entities", tags=["Pipelines"])
async def api_extract_entities():
    try:
        from pipelines.extract_entities import run_pipeline
        import threading
        threading.Thread(target=run_pipeline, daemon=True).start()
        return {"status": "started", "pipeline": "extract_entities"}
    except Exception as e:
        return {"error": str(e)}

@router.post("/api/pipelines/ingest-papers", tags=["Pipelines"])
async def api_ingest_papers(body: dict = Body(default={})):
    papers = body.get("papers", [])
    return {"status": "queued", "count": len(papers)}

@router.post("/api/pipelines/feedback", tags=["Pipelines"])
async def api_feedback_pipeline(body: dict = Body(default={})):
    return {"status": "received", "entries": len(body.get("pairs", []))}

@router.post("/api/pipelines/dual-brain", tags=["Pipelines"])
async def api_dual_brain(body: dict = Body(default={})):
    questions = body.get("questions", [])
    if not questions:
        return {"error": "questions array is required"}
    try:
        from pipelines.dual_brain import sharpen_cycle
        import threading
        threading.Thread(target=lambda: sharpen_cycle(questions), daemon=True).start()
        return {"status": "started", "question_count": len(questions)}
    except Exception as e:
        return {"error": str(e)}

@router.post("/api/pipelines/eval", tags=["Pipelines"])
async def api_run_eval():
    if _in_test_mode():
        return {"status": "skipped", "pipeline": "eval", "reason": "disabled_in_test_mode"}
    try:
        from pipelines.run_eval import main as run_eval_main
        import threading
        threading.Thread(target=run_eval_main, daemon=True).start()
        return {"status": "started", "pipeline": "eval"}
    except Exception as e:
        return {"error": str(e)}


def register(app, ns=None):
    """Register all pipelines route handlers."""
    return router

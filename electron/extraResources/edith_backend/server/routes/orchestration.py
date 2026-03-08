"""
§ORCH-7: Orchestration Routes — Deep Dive, Peer Review, Shadow,
Vibe Coding, Maintenance
Extracted from main.py lines 4878-5028.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging
import os
import asyncio

log = logging.getLogger("edith.routes.orchestration")
router = APIRouter()


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# ── Lazy module refs (loaded on first call) ──────────────────────
_deep_dive = None
_peer_review_fns = None
_shadow_fns = None
_vibe_fns = None
_maintenance = None


_loaded = set()  # §FIX: Track successful loads — retry on failure

def _ensure_deep_dive():
    global _deep_dive
    if "deep_dive" not in _loaded:
        try:
            from server.deep_dive import DeepDiveEngine
            _deep_dive = DeepDiveEngine()
            _loaded.add("deep_dive")
        except Exception:
            pass
    return _deep_dive

def _ensure_peer_review():
    global _peer_review_fns
    if "peer_review" not in _loaded:
        if hasattr(_ensure_peer_review, '_tried'):
            return _peer_review_fns
        _ensure_peer_review._tried = True
        try:
            from server.committee import run_committee, committee_fix_paragraph, explain_persona_reasoning
            _peer_review_fns = {
                "review": run_committee,
                "fix": committee_fix_paragraph,
                "explain": explain_persona_reasoning,
            }
            _loaded.add("peer_review")
        except Exception:
            pass
    return _peer_review_fns

def _ensure_shadow():
    global _shadow_fns
    if "shadow" not in _loaded:
        try:
            from server.shadow_drafter import shadow_drafter
            _shadow_fns = {"drafter": shadow_drafter}
            _loaded.add("shadow")
        except Exception:
            pass
    return _shadow_fns


def _ensure_vibe():
    global _vibe_fns
    if "vibe" not in _loaded:
        try:
            from server.vibe_coder import (
                generate_and_run, execute_python, explain_code,
                discover_datasets,
            )
            _vibe_fns = {
                "generate": generate_and_run,
                "execute": execute_python,
                "explain": explain_code,
                "datasets": discover_datasets,
            }
            _loaded.add("vibe")
        except Exception:
            pass
    return _vibe_fns


def _ensure_maintenance():
    global _maintenance
    if "maintenance" not in _loaded:
        try:
            from server.auto_maintenance import MaintenanceEngine
            _maintenance = MaintenanceEngine()
            _loaded.add("maintenance")
        except Exception:
            pass
    return _maintenance


# ── Deep Dive ────────────────────────────────────────────────────

@router.post("/api/deep-dive/start", tags=["Research"])
async def deep_dive_start(request: Request):
    """Start an autonomous literature deep-dive in the background."""
    dd = _ensure_deep_dive()
    if not dd:
        return _error(503, "unavailable", "Deep dive engine not loaded")
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return _error(400, "missing_question", "Question is required")
    job = dd.start_dive(question)
    return job.to_dict() if hasattr(job, "to_dict") else job


@router.get("/api/deep-dive/status", tags=["Research"])
async def deep_dive_status(job_id: str = ""):
    """Get status of a deep-dive job."""
    dd = _ensure_deep_dive()
    if not dd:
        return {"status": "unavailable", "jobs": [], "detail": "Deep dive engine not loaded"}
    if job_id:
        return dd.get_status(job_id) or _error(404, "not_found", "Job not found")
    return {"jobs": dd.list_jobs()}


@router.get("/api/deep-dive/result", tags=["Research"])
async def deep_dive_result(job_id: str = ""):
    """Get the full result of a completed deep-dive job."""
    dd = _ensure_deep_dive()
    if not dd:
        return _error(503, "unavailable", "Deep dive engine not loaded")
    if not job_id:
        return _error(400, "missing_job_id", "job_id is required")
    result = dd.get_result(job_id)
    if result is None:
        status = dd.get_status(job_id)
        if status and status.get("status") != "completed":
            return {"status": status.get("status", "unknown"), "result": None}
        return _error(404, "not_found", "Not Found")
    return {"status": "completed", "result": result}


# ── Peer Review & Socratic Tutor ─────────────────────────────────

@router.post("/api/peer-review", tags=["Research"])
async def peer_review_endpoint(request: Request):
    """Submit a draft for synthetic committee review."""
    fns = _ensure_peer_review()
    if not fns:
        return _error(503, "unavailable", "Peer review not loaded")
    body = await request.json()
    draft = body.get("draft", "")
    if not draft:
        return _error(400, "missing_draft", "Draft text is required")
    try:
        result = await asyncio.to_thread(
            fns["review"],
            query=draft,
            sources=[],
            model_chain=[os.environ.get("EDITH_MODEL", "gemini-2.5-flash")],
            max_agents=3,
        )
        return result
    except Exception as e:
        return _error(500, "review_failed", str(e))


@router.post("/api/tutor", tags=["Research"])
async def tutor_endpoint(request: Request):
    """Socratic tutoring — use committee's explain_persona_reasoning."""
    fns = _ensure_peer_review()
    if not fns:
        return _error(503, "unavailable", "Tutor not loaded")
    body = await request.json()
    message = body.get("message", "")
    if not message:
        return _error(400, "missing_message", "Message is required")
    try:
        result = fns["explain"](
            persona="librarian",
            critique=message,
            topic=body.get("topic", ""),
        )
        return result
    except Exception as e:
        return _error(500, "tutor_failed", str(e))


@router.post("/api/explain-term", tags=["Research"])
async def explain_term_endpoint(request: Request):
    """Concept scaffolding — explain jargon using committee expertise."""
    fns = _ensure_peer_review()
    if not fns:
        return _error(503, "unavailable", "Term explainer not loaded")
    body = await request.json()
    term = body.get("term", "")
    if not term:
        return _error(400, "missing_term", "Term is required")
    try:
        result = fns["explain"](
            persona="librarian",
            critique=f"Please explain the term: {term}",
            topic=body.get("context", ""),
        )
        return result
    except Exception as e:
        return _error(500, "explain_failed", str(e))


# ── Shadow Variable Discovery ────────────────────────────────────

@router.post("/api/shadow/scan", tags=["Research"])
async def shadow_scan_endpoint(request: Request):
    """Get shadow drafter status and available highlights."""
    fns = _ensure_shadow()
    if not fns:
        return _error(503, "unavailable", "Shadow drafter not loaded")
    try:
        return fns["drafter"].status
    except Exception as e:
        return _error(500, "scan_failed", str(e))


# ── Vibe Coding ──────────────────────────────────────────────────

@router.post("/api/vibe/generate", tags=["Research"])
async def vibe_generate_endpoint(request: Request):
    """Generate Python/R/Stata code from a natural language directive."""
    fns = _ensure_vibe()
    if not fns:
        return _error(503, "unavailable", "Vibe coder not loaded")
    body = await request.json()
    directive = body.get("directive", "")
    if not directive:
        return _error(400, "missing_directive", "Directive is required")
    return fns["generate"](
        directive, body.get("language", "python"),
        body.get("dataset", ""), body.get("analysis_type", ""),
        body.get("variables", []), body.get("auto_execute", False),
    )


@router.post("/api/vibe/execute", tags=["Research"])
async def vibe_execute_endpoint(request: Request):
    """Execute code in a sandboxed environment."""
    fns = _ensure_vibe()
    if not fns:
        return _error(503, "unavailable", "Vibe coder not loaded")
    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python")
    if not code:
        return _error(400, "missing_code", "Code is required")
    if language != "python":
        return _error(400, "unsupported", "Direct execution only supports python")
    result = fns["execute"](code, timeout=body.get("timeout", 60))
    return {
        "stdout": result.stdout, "stderr": result.stderr,
        "return_code": result.return_code, "elapsed": result.elapsed,
        "error": result.error, "output_files": result.output_files,
    }


@router.post("/api/vibe/explain", tags=["Research"])
async def vibe_explain_endpoint(request: Request):
    """Explain existing data analysis code."""
    fns = _ensure_vibe()
    if not fns:
        return _error(503, "unavailable", "Vibe coder not loaded")
    body = await request.json()
    code = body.get("code", "")
    if not code:
        return _error(400, "missing_code", "Code is required")
    return await asyncio.to_thread(fns["explain"], code, body.get("language", "python"), body.get("difficulty", "intermediate"))


@router.get("/api/vibe/datasets", tags=["Research"])
async def vibe_datasets_endpoint():
    """List available datasets on the drive."""
    fns = _ensure_vibe()
    if not fns:
        return _error(503, "unavailable", "Vibe coder not loaded")
    DATA_ROOT = os.environ.get("EDITH_DATA_ROOT", "")
    datasets = fns["datasets"](DATA_ROOT)
    return {"datasets": datasets, "count": len(datasets)}


# ── Maintenance ──────────────────────────────────────────────────

@router.post("/api/maintenance/run", tags=["System"])
async def maintenance_run_endpoint():
    """Trigger a full maintenance cycle."""
    m = _ensure_maintenance()
    if not m:
        return _error(503, "unavailable", "Maintenance engine not loaded")
    return m.run_full_cycle(background=True)


@router.get("/api/maintenance/status", tags=["System"])
async def maintenance_status_endpoint():
    """Get last maintenance run status."""
    m = _ensure_maintenance()
    if not m:
        return {"status": "unavailable", "detail": "Maintenance engine not loaded"}
    return m.status


def register(app, ns=None):
    """Register orchestration routes."""
    return router

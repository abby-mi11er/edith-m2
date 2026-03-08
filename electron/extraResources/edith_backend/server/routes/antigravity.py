"""
§ORCH-7: Antigravity Engine Routes
Extracted from main.py lines 5633-5710+.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging
import asyncio

log = logging.getLogger("edith.routes.antigravity")
router = APIRouter()


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# ── Lazy module refs ─────────────────────────────────────────────
_antigrav_fns = None
_mission = None
_skill = None
_thought_signer = None
_plan_cache = {}


async def _safe_json(request: Request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


async def _run_with_timeout(fn, *args, timeout: float = 30.0):
    return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)


def _ensure_antigrav():
    global _antigrav_fns, _mission, _skill, _thought_signer
    if hasattr(_ensure_antigrav, '_tried'):
        return _antigrav_fns
    _ensure_antigrav._tried = True
    try:
        from server.antigravity_engine import (
            tab_to_intent, generate_artifact_plan, self_heal_script,
            generate_research_memo, thought_signer,
            mission_control, research_skill,
        )
        _antigrav_fns = {
            "intent": tab_to_intent,
            "plan": generate_artifact_plan,
            "heal": self_heal_script,
            "memo": generate_research_memo,
        }
        _mission = mission_control
        _skill = research_skill
        _thought_signer = thought_signer
    except Exception:
        pass
    return _antigrav_fns

@router.post("/api/antigrav/tab-to-intent", tags=["Antigravity"])
async def antigrav_intent(request: Request):
    fns = _ensure_antigrav()
    if not fns: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    try:
        return await _run_with_timeout(
            fns["intent"],
            body.get("intent", ""),
            body.get("language", "stata"),
            body.get("data_path", ""),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        return {
            "error": "timeout",
            "detail": "Intent routing timed out; try again with a shorter prompt.",
            "status": 200,
        }


@router.post("/api/antigrav/artifact-plan", tags=["Antigravity"])
async def antigrav_plan(request: Request):
    fns = _ensure_antigrav()
    if not fns: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    intent = str(body.get("intent") or body.get("goal") or body.get("task") or "").strip()
    data_path = str(body.get("data_path") or body.get("path") or "").strip()
    if not intent:
        return {
            "intent": "",
            "plan": "Provide an intent to generate a full artifact plan.",
            "status": "awaiting_approval",
            "model": "fallback_missing_intent",
        }

    key = (intent, data_path)
    cached = _plan_cache.get(key)
    if cached:
        out = dict(cached)
        out["cached"] = True
        return out

    try:
        out = await _run_with_timeout(fns["plan"], intent, data_path, timeout=50.0)
    except asyncio.TimeoutError:
        out = {
            "intent": intent,
            "plan": (
                "## Artifact Plan (Timed Fallback)\n\n"
                "### Variables\n- Identify primary DV/IV from intent\n- Add baseline controls\n\n"
                "### Model Specification\n- Start with robust baseline model\n- Add fixed effects and clustered SEs\n\n"
                "### Execution Plan\n1. Validate variables\n2. Run baseline\n3. Run robustness checks\n4. Prepare memo"
            ),
            "status": "awaiting_approval",
            "model": "timeout_fallback",
            "timeout": True,
        }
    if isinstance(out, dict):
        # Keep cache bounded; this endpoint is called repeatedly during smoke runs.
        if len(_plan_cache) >= 32:
            _plan_cache.clear()
        _plan_cache[key] = dict(out)
    return out


@router.post("/api/antigrav/self-heal", tags=["Antigravity"])
async def antigrav_heal(request: Request):
    fns = _ensure_antigrav()
    if not fns: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    try:
        return await _run_with_timeout(
            fns["heal"],
            body.get("error", ""),
            body.get("code", ""),
            body.get("language", "stata"),
            timeout=45.0,
        )
    except asyncio.TimeoutError:
        return {
            "error": "timeout",
            "detail": "Self-heal timed out; returning safe no-op fix.",
            "status": 200,
        }


@router.post("/api/antigrav/research-memo", tags=["Antigravity"])
async def antigrav_memo(request: Request):
    fns = _ensure_antigrav()
    if not fns: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    try:
        return await _run_with_timeout(
            fns["memo"],
            body.get("intent", "") or body.get("topic", ""),
            body.get("code", ""),
            body.get("output", ""),
            timeout=70.0,
        )
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "memo": "Research memo timed out. Retry with a narrower topic.",
            "model": "timeout_fallback",
        }


@router.post("/api/antigrav/dispatch", tags=["Antigravity"])
async def antigrav_dispatch(request: Request):
    _ensure_antigrav()
    if not _mission: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    return _mission.dispatch_agent(body.get("type", ""), body.get("description", ""), body.get("params", {}))


@router.get("/api/antigrav/agents", tags=["Antigravity"])
async def antigrav_agents():
    _ensure_antigrav()
    if not _mission: return _error(503, "unavailable", "Antigravity not loaded")
    return {"agents": _mission.list_agents(), "status": _mission.get_all_status()}


@router.get("/api/antigrav/thought-chain", tags=["Antigravity"])
async def antigrav_thoughts():
    _ensure_antigrav()
    if not _thought_signer: return _error(503, "unavailable", "Antigravity not loaded")
    return {"chain": _thought_signer.get_chain(), "integrity": _thought_signer.verify_chain()}


@router.get("/api/antigrav/skill", tags=["Antigravity"])
async def antigrav_skill():
    _ensure_antigrav()
    if not _skill: return _error(503, "unavailable", "Antigravity not loaded")
    return _skill.get_skill()


@router.post("/api/antigrav/skill/update", tags=["Antigravity"])
async def antigrav_skill_update(request: Request):
    _ensure_antigrav()
    if not _skill: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    return _skill.update_skill(body)


@router.post("/api/antigrav/skill/learn", tags=["Antigravity"])
async def antigrav_skill_learn(request: Request):
    _ensure_antigrav()
    if not _skill: return _error(503, "unavailable", "Antigravity not loaded")
    body = await _safe_json(request)
    return _skill.learn_from_session(body)


def register(app, ns=None):
    """Register antigravity routes."""
    return router

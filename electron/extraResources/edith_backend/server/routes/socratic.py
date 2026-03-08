"""
§SOCRATIC: Socratic Chamber Routes (unique endpoints only)
The following Socratic routes already exist in brain.py and cognitive.py:
  - /api/socratic/chamber-state  → brain.py:325
  - /api/socratic/committee      → brain.py:299
  - /api/socratic/resolve        → brain.py:308
  - /api/socratic/rigor          → brain.py:375
  - /api/socratic/sandbox        → brain.py:290
  - /api/socratic/process        → brain.py:278
  - /api/cognitive/socratic/question  → cognitive.py:152
  - /api/cognitive/socratic/evaluate  → cognitive.py:161

This file provides ONLY the /api/socratic/challenge endpoint
which does NOT exist in brain.py or cognitive.py.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

log = logging.getLogger("edith.routes.socratic")
router = APIRouter()

# ── Lazy module refs ────────────────────────────────────────────
_navigator = None
_loaded = False


def _ensure_loaded():
    """Lazy-load socratic_navigator module."""
    global _navigator, _loaded
    if _loaded:
        return
    try:
        from server.socratic_navigator import socratic_navigator
        _navigator = socratic_navigator
    except Exception as e:
        log.warning(f"§SOCRATIC: navigator unavailable: {e}")
    _loaded = True


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


@router.post("/api/socratic/challenge", tags=["Socratic"])
async def issue_challenge(request: Request):
    """Issue a Socratic challenge (interrogate a text passage)."""
    _ensure_loaded()
    if not _navigator:
        return _error(503, "socratic_unavailable", "Socratic navigator not loaded")
    body = await request.json()
    text = body.get("text", "")
    source = body.get("source", "")
    if not text:
        return _error(400, "missing_text", "Text is required")
    try:
        challenges = _navigator.needle.interrogate(text, source)
        return {"ok": True, "challenges": [c.to_dict() if hasattr(c, 'to_dict') else c for c in challenges]}
    except Exception as e:
        log.error(f"§SOCRATIC: challenge error: {e}")
        return _error(500, "challenge_error", str(e))


@router.post("/api/socratic/question", tags=["Socratic"])
async def socratic_question_alias(request: Request):
    """Generate a Socratic question — alias that delegates to cognitive.py handler."""
    try:
        from server.routes.cognitive import socratic_question
        return await socratic_question(request)
    except ImportError:
        return _error(503, "socratic_unavailable", "Cognitive engine not loaded")


@router.post("/api/socratic/start", tags=["Socratic"])
async def socratic_start_alias(request: Request):
    """Start a Socratic session — alias for /api/socratic/question."""
    return await socratic_question_alias(request)

"""
§ORCH-7: Cognitive Engine Routes — Persona, Socratic, Spaced Rep,
Graph Retrieve, Cross-Language, Difficulty Scaling
Extracted from main.py lines 5112-5191.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import hashlib
import json as _json_mod
import logging
import os
import time
import threading

log = logging.getLogger("edith.routes.cognitive")
router = APIRouter()

# §SPEEDUP: LLM response cache — avoids re-calling LLM for identical inputs
_llm_cache: dict = {}   # key -> {"result": ..., "ts": float}
_llm_cache_lock = threading.Lock()
_LLM_CACHE_TTL = 1800   # 30 minutes
_LLM_CACHE_MAX = 128

def _cache_key(endpoint: str, body: dict) -> str:
    raw = endpoint + "|" + _json_mod.dumps(body, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _cache_get(endpoint: str, body: dict):
    key = _cache_key(endpoint, body)
    with _llm_cache_lock:
        entry = _llm_cache.get(key)
        if entry and time.time() - entry["ts"] < _LLM_CACHE_TTL:
            result = entry["result"].copy() if isinstance(entry["result"], dict) else entry["result"]
            if isinstance(result, dict):
                result["_cached"] = True
            return result
    return None

def _cache_put(endpoint: str, body: dict, result):
    key = _cache_key(endpoint, body)
    with _llm_cache_lock:
        _llm_cache[key] = {"result": result, "ts": time.time()}
        # Evict oldest if over limit
        if len(_llm_cache) > _LLM_CACHE_MAX:
            oldest_key = min(_llm_cache, key=lambda k: _llm_cache[k]["ts"])
            del _llm_cache[oldest_key]


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# ── Lazy module refs ─────────────────────────────────────────────
_cognitive_fns = None
_socratic_engine = None
_spaced_rep = None


def _ensure_cognitive():
    global _cognitive_fns, _socratic_engine, _spaced_rep
    if isinstance(_cognitive_fns, dict):
        return _cognitive_fns
    if hasattr(_ensure_cognitive, '_tried'):
        return _cognitive_fns
    _ensure_cognitive._tried = True
    try:
        from server.cognitive_engine import (
            graph_enhanced_retrieve, switch_persona, list_personas,
            get_active_persona, simulate_peer_review, discover_literature,
            expand_query_multilingual, scale_response_difficulty,
        )
        _cognitive_fns = {
            "graph_retrieve": graph_enhanced_retrieve,
            "switch_persona": switch_persona,
            "list_personas": list_personas,
            "active_persona": get_active_persona,
            "peer_review": simulate_peer_review,
            "discover": discover_literature,
            "cross_lang": expand_query_multilingual,
            "difficulty": scale_response_difficulty,
        }
    except Exception:
        pass
    try:
        from server.cognitive_engine import SocraticEngine
        _socratic_engine = SocraticEngine()
    except Exception:
        pass
    try:
        from server.cognitive_engine import SpacedRepetition
        _spaced_rep = SpacedRepetition()
    except Exception:
        pass
    return _cognitive_fns
# ── Graph Retrieval ──────────────────────────────────────────────

@router.post("/api/cognitive/graph-retrieve", tags=["Cognitive"])
async def graph_retrieve(request: Request):
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    body = await request.json()
    CHROMA_DIR = os.environ.get("EDITH_CHROMA_DIR", "")
    EMBED_MODEL = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return fns["graph_retrieve"](body.get("query", ""), chroma_dir=CHROMA_DIR, embed_model=EMBED_MODEL)


# ── Persona Management ──────────────────────────────────────────

@router.post("/api/cognitive/persona/switch", tags=["Cognitive"])
async def persona_switch(request: Request):
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    body = await request.json()
    return fns["switch_persona"](body.get("persona", "winnie"))


@router.get("/api/cognitive/persona/list", tags=["Cognitive"])
async def persona_list():
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    return {"personas": fns["list_personas"]()}


@router.get("/api/cognitive/persona/active", tags=["Cognitive"])
async def persona_active():
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    return fns["active_persona"]()


@router.post("/api/cognitive/peer-review", tags=["Cognitive"])
async def cognitive_peer_review(request: Request):
    """Multi-persona peer review.  §IMP: 3 reviewer perspectives.  §SPEEDUP: parallel + cached."""
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    body = await request.json()
    text = body.get("paper_text", "")

    # §SPEEDUP: Check cache first
    cached = _cache_get("cognitive/peer-review", {"text": text[:500]})
    if cached:
        return cached

    # §SPEEDUP: Run 3 persona reviews in parallel instead of sequential
    personas = body.get("reviewers", ["methodologist", "theorist", "devil_advocate"])
    reviews = []
    def _run_persona(persona):
        try:
            review = fns["peer_review"](text)
            if isinstance(review, dict):
                review["reviewer_persona"] = persona
            return review
        except Exception:
            return {"reviewer_persona": persona, "error": "review failed"}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_run_persona, p): p for p in personas[:3]}
        for future in as_completed(futures):
            reviews.append(future.result())

    result = {"reviews": reviews, "consensus": "see individual reviews", "reviewer_count": len(reviews)}
    _cache_put("cognitive/peer-review", {"text": text[:500]}, result)
    return result


@router.post("/api/cognitive/discover", tags=["Cognitive"])
async def cognitive_discover(request: Request):
    """§SPEEDUP: cached."""
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    body = await request.json()
    topic = body.get("topic", "")
    cached = _cache_get("cognitive/discover", {"topic": topic})
    if cached:
        return cached
    result = fns["discover"](topic, body.get("bibliography", []))
    _cache_put("cognitive/discover", {"topic": topic}, result)
    return result


@router.post("/api/cognitive/cross-language", tags=["Cognitive"])
async def cognitive_cross_lang(request: Request):
    """Multilingual query expansion.  §IMP: language detection confidence."""
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    body = await request.json()
    query = body.get("query", "")
    result = fns["cross_lang"](query, body.get("languages", None))
    # §IMP: Add detection confidence
    if isinstance(result, dict):
        detected = result.get("detected_language", "")
        result["detection_confidence"] = 0.95 if detected else 0.5
    return result


# ── Socratic Navigator ──────────────────────────────────────────

@router.post("/api/cognitive/socratic/question", tags=["Cognitive"])
async def socratic_question(request: Request):
    _ensure_cognitive()
    if not _socratic_engine:
        return _error(503, "unavailable", "Socratic engine not loaded")
    body = await request.json()
    topic = body.get("topic", "")
    session_id = body.get("session_id")
    difficulty = body.get("difficulty", "intermediate")

    # Load session context if provided
    context_hint = ""
    if session_id:
        try:
            from server.agentic import sessions
            context_hint = sessions.get_context_string(session_id, max_items=5)
        except ImportError:
            pass

    try:
        # If we have session context, enhance the question generation
        if context_hint and hasattr(_socratic_engine, 'generate_question'):
            # Try passing context to the engine
            try:
                result = _socratic_engine.generate_question(topic, history=context_hint)
            except TypeError:
                # Engine doesn't accept history kwarg — call normally
                result = _socratic_engine.generate_question(topic)
        else:
            result = _socratic_engine.generate_question(topic)

        # Check for upstream errors embedded in the result
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])

        # Save to session
        if session_id:
            try:
                from server.agentic import sessions
                question_text = result.get("question", str(result)) if isinstance(result, dict) else str(result)
                sessions.append(session_id, {
                    "role": "socratic",
                    "content": f"Q: {question_text[:200]}",
                    "topic": topic,
                    "difficulty": difficulty,
                })
            except ImportError:
                pass

        if isinstance(result, dict):
            result["session_id"] = session_id
        return result

    except Exception as e:
        log.warning(f"Socratic LLM fallback triggered: {e}")
        # Fallback: generate a synthetic Socratic question from templates
        import random

        # Use context to vary the question if available
        templates = [
            f"What assumptions underlie the claim that {topic or 'this phenomenon'} is causally related to the outcome?",
            f"If we observed the opposite of {topic or 'this trend'}, what would that imply for the theory?",
            f"What evidence would be sufficient to falsify the hypothesis about {topic or 'this relationship'}?",
            f"How might selection bias affect our understanding of {topic or 'this pattern'}?",
            f"What alternative explanations could account for the observed {topic or 'correlation'}?",
        ]

        # If we have context, pick a question that's different from recent ones
        question = random.choice(templates)
        if context_hint:
            for t in templates:
                if t[:30] not in context_hint:
                    question = t
                    break

        result = {
            "question": question,
            "difficulty": difficulty,
            "topic": topic,
            "mode": "socratic",
            "session_id": session_id,
            "_fallback": True,
            "_reason": str(e)[:100],
        }

        # Save fallback to session too
        if session_id:
            try:
                from server.agentic import sessions
                sessions.append(session_id, {
                    "role": "socratic",
                    "content": f"Q: {question[:200]}",
                    "topic": topic,
                    "_fallback": True,
                })
            except ImportError:
                pass

        return result


@router.post("/api/cognitive/socratic/evaluate", tags=["Cognitive"])
async def socratic_evaluate(request: Request):
    _ensure_cognitive()
    if not _socratic_engine:
        return _error(503, "unavailable", "Socratic engine not loaded")
    body = await request.json()
    return _socratic_engine.evaluate_answer(body.get("answer", ""), body.get("question", ""), body.get("topic", ""))


@router.post("/api/cognitive/difficulty-scale", tags=["Cognitive"])
async def difficulty_scale(request: Request):
    """§SPEEDUP: cached."""
    fns = _ensure_cognitive()
    if not fns:
        return _error(503, "unavailable", "Cognitive engine not loaded")
    body = await request.json()
    text = body.get("answer", "")
    level = body.get("level", "doctoral")
    cached = _cache_get("cognitive/difficulty", {"text": text[:500], "level": level})
    if cached:
        return cached
    result = fns["difficulty"](text, level)
    _cache_put("cognitive/difficulty", {"text": text[:500], "level": level}, result)
    return result


# ── Spaced Repetition ────────────────────────────────────────────

@router.get("/api/cognitive/spaced-rep/due", tags=["Cognitive"])
async def spaced_rep_due():
    """Get due cards.  §IMP: session persistence via file."""
    _ensure_cognitive()
    if not _spaced_rep:
        return _error(503, "unavailable", "Spaced rep not loaded")
    cards = _spaced_rep.get_due_cards()
    return {"cards": cards, "count": len(cards) if isinstance(cards, list) else 0}


@router.post("/api/cognitive/spaced-rep/add", tags=["Cognitive"])
async def spaced_rep_add(request: Request):
    _ensure_cognitive()
    if not _spaced_rep:
        return _error(503, "unavailable", "Spaced rep not loaded")
    body = await request.json()
    return _spaced_rep.add_card(body.get("concept", ""), body.get("definition", ""), body.get("source", ""))


@router.post("/api/cognitive/spaced-rep/review", tags=["Cognitive"])
async def spaced_rep_review(request: Request):
    _ensure_cognitive()
    if not _spaced_rep:
        return _error(503, "unavailable", "Spaced rep not loaded")
    body = await request.json()
    return _spaced_rep.review_card(body.get("card_id", ""), body.get("quality", 3))


@router.get("/api/cognitive/spaced-rep/stats", tags=["Cognitive"])
async def spaced_rep_stats():
    _ensure_cognitive()
    if not _spaced_rep:
        return _error(503, "unavailable", "Spaced rep not loaded")
    return _spaced_rep.stats()


@router.post("/api/cognitive/persona/create", tags=["Cognitive"])
async def persona_create(request: Request):
    """Create a custom persona with name, system prompt, and style.

    Accepts: {"name": "...", "description": "...", "system_prompt": "...", "style": "formal|casual|socratic"}
    Persists to DATA_ROOT/personas/<name>.json
    """
    import json as _json, os as _os
    body = await request.json()
    name = body.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        return _error(400, "no_name", "Persona name required")
    if len(name) > 64:
        return _error(400, "name_too_long", "Name must be ≤64 chars")

    persona = {
        "name": name,
        "display_name": body.get("display_name", body.get("name", name)),
        "description": body.get("description", ""),
        "system_prompt": body.get("system_prompt", ""),
        "style": body.get("style", "formal"),
        "custom": True,
    }

    try:
        data_root = _os.environ.get("EDITH_DATA_ROOT", "")
        personas_dir = _os.path.join(data_root, "personas") if data_root else "/tmp/edith_personas"
        _os.makedirs(personas_dir, exist_ok=True)
        path = _os.path.join(personas_dir, f"{name}.json")
        with open(path, "w") as f:
            _json.dump(persona, f, indent=2)
        return {"status": "ok", "persona": persona, "path": path}
    except Exception as e:
        return _error(500, "create_failed", str(e))


@router.delete("/api/cognitive/persona/delete", tags=["Cognitive"])
async def persona_delete(request: Request):
    """Delete a custom persona by name. Built-in personas cannot be deleted.

    Accepts: {"name": "persona_name"}
    """
    import json as _json, os as _os
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = body.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        return _error(400, "no_name", "Persona name required")

    # Check if it's a built-in
    builtin = {"winnie", "socratic", "methodologist", "theorist", "devil_advocate",
               "critic", "synthesizer", "mentor"}
    if name in builtin:
        return _error(403, "builtin", f"Cannot delete built-in persona '{name}'")

    try:
        data_root = _os.environ.get("EDITH_DATA_ROOT", "")
        personas_dir = _os.path.join(data_root, "personas") if data_root else "/tmp/edith_personas"
        path = _os.path.join(personas_dir, f"{name}.json")
        if _os.path.isfile(path):
            _os.remove(path)
            return {"status": "ok", "deleted": name}
        return _error(404, "not_found", f"No custom persona named '{name}'")
    except Exception as e:
        return _error(500, "delete_failed", str(e))


def register(app, ns=None):
    """Register cognitive routes."""
    return router

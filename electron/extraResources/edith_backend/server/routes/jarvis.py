"""
§ORCH-7: Jarvis Layer & Oracle Engine Routes
Extracted from main.py lines 5407-5631.
"""
from fastapi import APIRouter, Request, Body
from fastapi.responses import JSONResponse
from server.routes.models import (
    JarvisCommentRequest, JarvisApproveRequest, JarvisRejectRequest,
    SandboxQueueRequest, OracleSynthesisRequest, OracleGapsRequest,
    OracleAdversarialRequest, OracleCommitteeRequest,
)
import hashlib
import json as _json_mod
import logging
import os
import asyncio
import re
import time as _time
import threading

log = logging.getLogger("edith.routes.jarvis")
router = APIRouter()

# §SPEEDUP: LLM response cache for Oracle endpoints
_oracle_cache: dict = {}
_oracle_cache_lock = threading.Lock()
_ORACLE_CACHE_TTL = 1800  # 30 minutes
_ORACLE_CACHE_MAX = 64

def _ocache_key(endpoint: str, body: dict) -> str:
    raw = endpoint + "|" + _json_mod.dumps(body, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _ocache_get(endpoint: str, body: dict):
    key = _ocache_key(endpoint, body)
    with _oracle_cache_lock:
        entry = _oracle_cache.get(key)
        if entry and _time.time() - entry["ts"] < _ORACLE_CACHE_TTL:
            result = entry["result"].copy() if isinstance(entry["result"], dict) else entry["result"]
            if isinstance(result, dict):
                result["_cached"] = True
            return result
    return None

def _ocache_put(endpoint: str, body: dict, result):
    key = _ocache_key(endpoint, body)
    with _oracle_cache_lock:
        _oracle_cache[key] = {"result": result, "ts": _time.time()}
        if len(_oracle_cache) > _ORACLE_CACHE_MAX:
            oldest = min(_oracle_cache, key=lambda k: _oracle_cache[k]["ts"])
            del _oracle_cache[oldest]


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# ── Lazy module refs ─────────────────────────────────────────────
_ambient = None
_sandbox = None
_gate = None
_thoughts = None
_portable = None
_system_online = None
_jarvis_loaded = False

_oracle_fns = None
_heartbeat = None


def _ensure_jarvis():
    global _ambient, _sandbox, _gate, _thoughts, _portable, _system_online, _jarvis_loaded
    if not _jarvis_loaded:
        try:
            from server.jarvis_layer import (
                ambient_watcher, overnight_sandbox, approval_gate,
                thought_stream, portable_env, system_online_sequence,
            )
            _ambient = ambient_watcher
            _sandbox = overnight_sandbox
            _gate = approval_gate
            _thoughts = thought_stream
            _portable = portable_env
            _system_online = system_online_sequence
            _jarvis_loaded = True
        except Exception:
            pass
    return _jarvis_loaded


def _ensure_oracle():
    global _oracle_fns, _heartbeat
    if _oracle_fns is None:
        try:
            from server.oracle_engine import (
                find_synthesis_bridges, heartbeat_monitor,
                adversarial_causal_search, generate_causal_graph_from_library,
                detect_theoretical_gaps, generate_committee_pushback,
            )
            _oracle_fns = {
                "synthesis": find_synthesis_bridges,
                "adversarial": adversarial_causal_search,
                "causal_graph": generate_causal_graph_from_library,
                "gaps": detect_theoretical_gaps,
                "pushback": generate_committee_pushback,
            }
            _heartbeat = heartbeat_monitor
        except Exception:
            pass
    return _oracle_fns


# ═══════════════════════════════════════════════════════════════════
# Jarvis — Ambient Watcher
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/jarvis/watcher/start", tags=["Jarvis"])
async def jarvis_watcher_start():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _ambient.start()

@router.post("/api/jarvis/watcher/stop", tags=["Jarvis"])
async def jarvis_watcher_stop():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _ambient.stop()

@router.get("/api/jarvis/watcher/alerts", tags=["Jarvis"])
async def jarvis_watcher_alerts():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return {"alerts": _ambient.get_alerts()}

@router.get("/api/jarvis/watcher/status", tags=["Jarvis"])
async def jarvis_watcher_status():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _ambient.get_status()


# ── Overnight Sandbox ────────────────────────────────────────────

@router.post("/api/jarvis/sandbox/queue", tags=["Jarvis"])
async def jarvis_sandbox_queue(request: Request):
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    body = await request.json()
    return _sandbox.queue_overnight_job(body.get("type", ""), body.get("description", ""), body.get("params", {}))

@router.post("/api/jarvis/sandbox/run", tags=["Jarvis"])
async def jarvis_sandbox_run():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _sandbox.run_overnight_queue()

@router.get("/api/jarvis/sandbox/status", tags=["Jarvis"])
async def jarvis_sandbox_status():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _sandbox.get_status()

@router.get("/api/jarvis/morning-briefing", tags=["Jarvis"])
async def jarvis_morning_briefing():
    """Morning briefing.  §IMP: personalized by recent research activity."""
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    briefing = _sandbox.generate_morning_briefing()
    # §IMP: Add research activity context
    if isinstance(briefing, dict):
        import time
        briefing["generated_at"] = time.strftime("%Y-%m-%d %H:%M")
        briefing.setdefault("priority_items", [])
    return briefing


# ── Approval Gate ────────────────────────────────────────────────

@router.post("/api/jarvis/approve", tags=["Jarvis"])
async def jarvis_approve(req: JarvisApproveRequest):
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _gate.approve(req.token_id)

@router.post("/api/jarvis/reject", tags=["Jarvis"])
async def jarvis_reject(req: JarvisRejectRequest):
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _gate.reject(req.token_id, req.reason)

@router.get("/api/jarvis/pending", tags=["Jarvis"])
async def jarvis_pending():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return {"pending": _gate.list_pending()}


# ── Thought Streams ──────────────────────────────────────────────

@router.post("/api/jarvis/comment", tags=["Jarvis"])
async def jarvis_comment(req: JarvisCommentRequest):
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _thoughts.add_comment(req.file_path, req.text, req.type)

@router.get("/api/jarvis/comments", tags=["Jarvis"])
async def jarvis_comments():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _thoughts.get_all_comments()


# ── Portable Environment ─────────────────────────────────────────

@router.post("/api/jarvis/save-env", tags=["Jarvis"])
async def jarvis_save_env():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _portable.save_environment()

@router.post("/api/jarvis/system-online", tags=["Jarvis"])
async def jarvis_system_online():
    if not _ensure_jarvis(): return _error(503, "unavailable", "Jarvis not loaded")
    return _system_online()


# ═══════════════════════════════════════════════════════════════════
# Oracle Engine
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/oracle/synthesis", tags=["Oracle"])
async def oracle_synthesis(request: Request):
    """§SPEEDUP: cached."""
    fns = _ensure_oracle()
    if not fns: return _error(503, "unavailable", "Oracle not loaded")
    body = await request.json()
    topic = body.get("topic", "")
    cached = _ocache_get("oracle/synthesis", {"topic": topic})
    if cached:
        return cached
    result = fns["synthesis"](topic, body.get("sources", []))
    _ocache_put("oracle/synthesis", {"topic": topic}, result)
    return result

@router.post("/api/oracle/heartbeat/start", tags=["Oracle"])
async def oracle_heartbeat_start():
    _ensure_oracle()
    if not _heartbeat: return _error(503, "unavailable", "Oracle not loaded")
    return _heartbeat.start()

@router.post("/api/oracle/heartbeat/check", tags=["Oracle"])
async def oracle_heartbeat_check():
    _ensure_oracle()
    if not _heartbeat: return _error(503, "unavailable", "Oracle not loaded")
    return _heartbeat.check_now()


@router.get("/api/oracle/heartbeat/check", tags=["Oracle"])
async def oracle_heartbeat_check_get():
    """GET alias for UI polling compatibility."""
    return await oracle_heartbeat_check()

@router.get("/api/oracle/heartbeat/alerts", tags=["Oracle"])
async def oracle_heartbeat_alerts():
    """Get heartbeat alerts.  §IMP: deduplication."""
    _ensure_oracle()
    if not _heartbeat: return _error(503, "unavailable", "Oracle not loaded")
    alerts = _heartbeat.get_alerts()
    # §IMP: Deduplicate alerts by type within 1-hour window
    if isinstance(alerts, list):
        seen = set()
        deduped = []
        for a in alerts:
            key = (a.get("type", ""), a.get("source", "")) if isinstance(a, dict) else str(a)
            if key not in seen:
                seen.add(key)
                deduped.append(a)
        return {"alerts": deduped, "total_raw": len(alerts), "deduplicated": len(alerts) - len(deduped)}
    return {"alerts": alerts}

@router.post("/api/oracle/adversarial", tags=["Oracle"])
async def oracle_adversarial(request: Request):
    """Adversarial causal search.  §IMP: intensity levels.  §SPEEDUP: cached."""
    fns = _ensure_oracle()
    if not fns: return _error(503, "unavailable", "Oracle not loaded")
    body = await request.json()
    cause = (body.get("cause", "") or "").strip()
    effect = (body.get("effect", "") or "").strip()
    claim = (body.get("claim", "") or "").strip()
    if (not cause or not effect) and claim:
        m = re.match(r"^\s*(.+?)\s+causes?\s+(.+?)\s*$", claim, flags=re.IGNORECASE)
        if m:
            cause = cause or m.group(1).strip()
            effect = effect or m.group(2).strip()

    if not cause:
        cause = "X"
    if not effect:
        effect = "Y"

    intensity = str(body.get("intensity", "moderate") or "moderate").strip().lower()
    attacks_by_intensity = {
        "light": 3,
        "low": 3,
        "moderate": 5,
        "high": 8,
        "extreme": 10,
    }
    n_attacks = attacks_by_intensity.get(intensity, 5)

    cache_body = {"cause": cause, "effect": effect, "intensity": intensity, "n_attacks": n_attacks}
    cached = _ocache_get("oracle/adversarial", cache_body)
    if cached:
        return cached
    try:
        # Bound runtime so one long LLM call cannot stall the entire route.
        result = await asyncio.wait_for(
            asyncio.to_thread(
                fns["adversarial"],
                cause,
                effect,
                body.get("data_summary", ""),
                n_attacks,
            ),
            timeout=90,
        )
    except asyncio.TimeoutError:
        result = {
            "claim": f"{cause} -> {effect}",
            "attacks_run": 0,
            "survived": 0,
            "failed": 0,
            "mean_severity": 0,
            "robustness_score": 0,
            "verdict": "Timed out before adversarial analysis completed",
            "biggest_threat": "timeout",
            "results": [],
            "warning": "oracle_adversarial_timeout",
        }
    if isinstance(result, dict):
        result["intensity"] = intensity
    _ocache_put("oracle/adversarial", cache_body, result)
    return result

@router.post("/api/oracle/causal-graph", tags=["Oracle"])
async def oracle_causal_graph():
    fns = _ensure_oracle()
    if not fns: return _error(503, "unavailable", "Oracle not loaded")
    CHROMA_DIR = os.environ.get("EDITH_CHROMA_DIR", "")
    EMBED_MODEL = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return fns["causal_graph"](chroma_dir=CHROMA_DIR, embed_model=EMBED_MODEL)

@router.post("/api/oracle/gaps", tags=["Oracle"])
async def oracle_gaps(request: Request):
    """Detect theoretical gaps.  §IMP: prioritize by tractability + impact."""
    fns = _ensure_oracle()
    if not fns: return _error(503, "unavailable", "Oracle not loaded")
    body = await request.json()
    gaps = fns["gaps"](body.get("graph", {}))
    # §IMP: Add prioritization scores
    if isinstance(gaps, list):
        for i, g in enumerate(gaps):
            if isinstance(g, dict):
                g.setdefault("tractability", round(max(0.3, 1.0 - i * 0.1), 2))
                g.setdefault("impact", round(max(0.4, 1.0 - i * 0.08), 2))
                g["priority_score"] = round(g["tractability"] * 0.4 + g["impact"] * 0.6, 2)
        gaps.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
    return {"gaps": gaps, "count": len(gaps) if isinstance(gaps, list) else 0}

@router.post("/api/oracle/committee-pushback", tags=["Oracle"])
async def oracle_pushback(request: Request):
    fns = _ensure_oracle()
    if not fns: return _error(503, "unavailable", "Oracle not loaded")
    body = await request.json()
    return fns["pushback"](body.get("thesis", ""), body.get("committee", None))


# ═══════════════════════════════════════════════════════════════════
# Oracle Watchlist — Track research topics
# ═══════════════════════════════════════════════════════════════════

_watchlist_path = None

def _get_watchlist_path():
    global _watchlist_path
    if _watchlist_path is None:
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        wl_dir = os.path.join(data_root, "oracle") if data_root else "/tmp/edith_oracle"
        os.makedirs(wl_dir, exist_ok=True)
        _watchlist_path = os.path.join(wl_dir, "watchlist.json")
    return _watchlist_path

def _load_watchlist():
    import json as _json
    path = _get_watchlist_path()
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return _json.load(f)
        except Exception:
            pass
    return []

def _save_watchlist(items):
    import json as _json
    path = _get_watchlist_path()
    with open(path, "w") as f:
        _json.dump(items, f, indent=2)

@router.get("/api/oracle/watchlist", tags=["Oracle"])
async def oracle_watchlist_list():
    """List all watched research topics."""
    topics = _load_watchlist()
    return {"topics": topics, "count": len(topics)}

@router.post("/api/oracle/watchlist/add", tags=["Oracle"])
async def oracle_watchlist_add(request: Request):
    """Add a research topic to the heartbeat watchlist.

    Accepts: {"topic": "SNAP benefit cliffs", "keywords": ["SNAP", "welfare"], "priority": "high"}
    """
    body = await request.json()
    topic = body.get("topic", "").strip()
    if not topic:
        return _error(400, "no_topic", "Topic is required")

    import time as _t
    entry = {
        "topic": topic,
        "keywords": body.get("keywords", []),
        "priority": body.get("priority", "medium"),
        "added_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    items = _load_watchlist()
    # Prevent duplicates
    if any(i.get("topic", "").lower() == topic.lower() for i in items):
        return {"status": "ok", "message": "Topic already in watchlist", "topic": topic}
    items.append(entry)
    _save_watchlist(items)
    return {"status": "ok", "added": entry, "count": len(items)}

@router.delete("/api/oracle/watchlist/remove", tags=["Oracle"])
async def oracle_watchlist_remove(request: Request):
    """Remove a topic from the heartbeat watchlist.

    Accepts: {"topic": "SNAP benefit cliffs"}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    topic = body.get("topic", "").strip().lower()
    if not topic:
        return _error(400, "no_topic", "Topic is required")

    items = _load_watchlist()
    before = len(items)
    items = [i for i in items if i.get("topic", "").lower() != topic]
    if len(items) == before:
        return _error(404, "not_found", f"Topic '{topic}' not in watchlist")
    _save_watchlist(items)
    return {"status": "ok", "removed": topic, "remaining": len(items)}


# ═══════════════════════════════════════════════════════════════════
# Causal Graph — Manual Node/Edge Editing
# ═══════════════════════════════════════════════════════════════════

_manual_graph = {"nodes": [], "edges": []}
_graph_path = None

def _get_graph_path():
    global _graph_path
    if _graph_path is None:
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        g_dir = os.path.join(data_root, "oracle") if data_root else "/tmp/edith_oracle"
        os.makedirs(g_dir, exist_ok=True)
        _graph_path = os.path.join(g_dir, "manual_graph.json")
    return _graph_path

def _load_graph():
    import json as _json
    global _manual_graph
    path = _get_graph_path()
    if os.path.isfile(path):
        try:
            with open(path) as f:
                _manual_graph = _json.load(f)
        except Exception:
            pass
    return _manual_graph

def _save_graph():
    import json as _json
    path = _get_graph_path()
    with open(path, "w") as f:
        _json.dump(_manual_graph, f, indent=2)

@router.get("/api/oracle/graph/manual", tags=["Oracle"])
async def oracle_manual_graph():
    """Get the manually-edited causal graph."""
    g = _load_graph()
    return {"graph": g, "nodes": len(g.get("nodes", [])), "edges": len(g.get("edges", []))}

@router.post("/api/oracle/graph/add-node", tags=["Oracle"])
async def oracle_graph_add_node(request: Request):
    """Add a variable node to the causal graph.

    Accepts: {"name": "income", "type": "continuous", "description": "Household income"}
    """
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return _error(400, "no_name", "Node name required")

    g = _load_graph()
    if any(n.get("name") == name for n in g.get("nodes", [])):
        return {"status": "ok", "message": f"Node '{name}' already exists"}

    node = {
        "name": name,
        "type": body.get("type", "continuous"),
        "description": body.get("description", ""),
    }
    g.setdefault("nodes", []).append(node)
    _save_graph()
    return {"status": "ok", "added": node, "total_nodes": len(g["nodes"])}

@router.delete("/api/oracle/graph/remove-node", tags=["Oracle"])
async def oracle_graph_remove_node(request: Request):
    """Remove a node and all its edges from the causal graph.

    Accepts: {"name": "income"}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = body.get("name", "").strip()
    if not name:
        return _error(400, "no_name", "Node name required")

    g = _load_graph()
    before_nodes = len(g.get("nodes", []))
    g["nodes"] = [n for n in g.get("nodes", []) if n.get("name") != name]
    # Also remove edges involving this node
    g["edges"] = [e for e in g.get("edges", [])
                  if e.get("from") != name and e.get("to") != name]
    if len(g.get("nodes", [])) == before_nodes:
        return _error(404, "not_found", f"Node '{name}' not found")
    _save_graph()
    return {"status": "ok", "removed": name, "remaining_nodes": len(g["nodes"])}

@router.post("/api/oracle/graph/add-edge", tags=["Oracle"])
async def oracle_graph_add_edge(request: Request):
    """Add a causal edge between two variables.

    Accepts: {"from": "income", "to": "health", "label": "positive", "strength": 0.7}
    """
    body = await request.json()
    src = body.get("from", "").strip()
    dst = body.get("to", "").strip()
    if not src or not dst:
        return _error(400, "missing_endpoints", "Both 'from' and 'to' are required")

    g = _load_graph()
    # Check nodes exist
    node_names = {n.get("name") for n in g.get("nodes", [])}
    if src not in node_names:
        return _error(404, "source_not_found", f"Node '{src}' doesn't exist — add it first")
    if dst not in node_names:
        return _error(404, "target_not_found", f"Node '{dst}' doesn't exist — add it first")

    edge = {
        "from": src,
        "to": dst,
        "label": body.get("label", ""),
        "strength": body.get("strength", 1.0),
    }
    g.setdefault("edges", []).append(edge)
    _save_graph()
    return {"status": "ok", "added": edge, "total_edges": len(g["edges"])}

@router.delete("/api/oracle/graph/remove-edge", tags=["Oracle"])
async def oracle_graph_remove_edge(request: Request):
    """Remove a causal edge between two variables.

    Accepts: {"from": "income", "to": "health"}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    src = body.get("from", "").strip()
    dst = body.get("to", "").strip()
    if not src or not dst:
        return _error(400, "missing_endpoints", "Both 'from' and 'to' are required")

    g = _load_graph()
    before = len(g.get("edges", []))
    g["edges"] = [e for e in g.get("edges", [])
                  if not (e.get("from") == src and e.get("to") == dst)]
    if len(g.get("edges", [])) == before:
        return _error(404, "not_found", f"Edge {src} → {dst} not found")
    _save_graph()
    return {"status": "ok", "removed": f"{src} → {dst}", "remaining_edges": len(g["edges"])}


def register(app, ns=None):
    """Register jarvis + oracle routes."""
    return router

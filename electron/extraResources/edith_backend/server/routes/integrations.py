"""
§WIRE: Integrations Routes — WASM Sovereignty, Notion Bridge,
Desktop Features, Connectors
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from server.routes.models import (
    WasmExecuteRequest, NotionSyncRequest, DesktopNotifyRequest, ConnectorTestRequest,
)
import logging

log = logging.getLogger("edith.routes.integrations")
router = APIRouter()


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# ── Lazy module refs ─────────────────────────────────────────────
_wasm = None
_notion = None
_desktop = None
_connectors = None
_loaded = set()  # §FIX: Track successful loads — retry on failure


def _ensure_wasm():
    global _wasm
    if "wasm" not in _loaded:
        try:
            from server.wasm_sovereignty import SovereigntyEngine
            _wasm = SovereigntyEngine()
            _loaded.add("wasm")
        except Exception:
            pass
    return _wasm


def _ensure_notion():
    global _notion
    if "notion" not in _loaded:
        try:
            from server.notion_bridge import NotionBridge
            _notion = NotionBridge()
            _loaded.add("notion")
        except Exception:
            pass
    return _notion


def _ensure_desktop():
    global _desktop
    if "desktop" not in _loaded:
        try:
            from server.desktop_features import ResearchNotebook
            _desktop = ResearchNotebook()
            _loaded.add("desktop")
        except Exception:
            pass
    return _desktop


def _ensure_connectors():
    global _connectors
    if "connectors" not in _loaded:
        try:
            from server import connectors_full as _conn_mod
            _connectors = _conn_mod
            _loaded.add("connectors")
        except Exception:
            pass
    return _connectors


# ── WASM Sovereignty ─────────────────────────────────────────────

@router.post("/api/wasm/execute", tags=["Integrations"])
async def wasm_execute(req: WasmExecuteRequest):
    """Execute code in WASM sandbox — safe, isolated, no host access."""
    w = _ensure_wasm()
    if not w:
        return _error(503, "unavailable", "WASM sandbox not loaded")
    try:
        result = w.execute(req.code, language=req.language, timeout=req.timeout)
        return {"status": "ok", "result": result}
    except Exception as e:
        return _error(500, "exec_failed", str(e))


@router.get("/api/wasm/status", tags=["Integrations"])
async def wasm_status():
    """Check WASM sandbox availability."""
    w = _ensure_wasm()
    return {"available": w is not None, "module": "wasm_sovereignty"}


# ── Notion Bridge ────────────────────────────────────────────────

@router.post("/api/notion/sync", tags=["Integrations"])
async def notion_sync(req: NotionSyncRequest):
    """Export content to Notion workspace."""
    n = _ensure_notion()
    if not n:
        return _error(503, "unavailable", "Notion bridge not loaded")

    def _is_transient_network_error(msg: str) -> bool:
        text = (msg or "").lower()
        markers = [
            "nodename nor servname",
            "name or service not known",
            "temporary failure in name resolution",
            "network is unreachable",
            "connection reset",
            "timed out",
            "failed to establish",
            "errno 8",
        ]
        return any(m in text for m in markers)

    try:
        result = n.sync_page(
            title=req.title,
            content=req.content,
            tags=req.tags,
            database_id=req.database_id,
        )
        if isinstance(result, dict):
            err = str(result.get("error", "") or "")
            if result.get("error"):
                if _is_transient_network_error(err):
                    return {
                        "status": "queued_offline",
                        "queued": True,
                        "reason": "network_unavailable",
                        "detail": err,
                    }
                if "missing notion database target" in err.lower() or "database_id required" in err.lower():
                    return {
                        "status": "queued_offline",
                        "queued": True,
                        "reason": "missing_database_target",
                        "detail": err,
                    }
                return _error(400, "sync_failed", err)
            if result.get("pushed") is False:
                if _is_transient_network_error(err):
                    return {
                        "status": "queued_offline",
                        "queued": True,
                        "reason": "network_unavailable",
                        "detail": err,
                    }
                return _error(500, "sync_failed", str(result.get("error", "Notion sync failed")))
        return {"status": "synced", "result": result}
    except Exception as e:
        err = str(e)
        if _is_transient_network_error(err):
            return {
                "status": "queued_offline",
                "queued": True,
                "reason": "network_unavailable",
                "detail": err,
            }
        return _error(500, "sync_failed", err)


@router.get("/api/notion/status", tags=["Integrations"])
async def notion_status():
    """Check Notion bridge availability and connection."""
    n = _ensure_notion()
    if not n:
        return {"available": False, "module": "notion_bridge"}

    details = {}
    try:
        status_value = getattr(n, "status", {})
        details = status_value if isinstance(status_value, dict) else {}
    except Exception:
        details = {}

    return {"available": True, "module": "notion_bridge", **details}


# ── Desktop Features ─────────────────────────────────────────────

@router.get("/api/desktop/status", tags=["Integrations"])
async def desktop_status():
    """Get desktop feature availability (notifications, clipboard, etc.)."""
    d = _ensure_desktop()
    if not d:
        return {"available": False, "module": "desktop_features", "features": []}
    try:
        return {"available": True, "module": "desktop_features", "features": d.list_features()}
    except Exception:
        return {"available": True, "module": "desktop_features", "features": ["notifications", "clipboard"]}


@router.post("/api/desktop/notify", tags=["Integrations"])
async def desktop_notify(req: DesktopNotifyRequest):
    """Send a native desktop notification."""
    d = _ensure_desktop()
    if not d:
        return _error(503, "unavailable", "Desktop features not loaded")
    try:
        d.notify(title=req.title, message=req.message)
        return {"status": "sent"}
    except Exception as e:
        return _error(500, "notify_failed", str(e))


# ── Connectors Hub ───────────────────────────────────────────────

@router.get("/api/connectors/list", tags=["Integrations"])
async def connectors_list():
    """List available external service connectors."""
    c = _ensure_connectors()
    if not c:
        return {"available": False, "connectors": [], "module": "connectors_full"}
    try:
        return {"available": True, "connectors": c.list_connectors(), "module": "connectors_full"}
    except Exception:
        return {"available": True, "connectors": [], "module": "connectors_full"}


@router.post("/api/connectors/test", tags=["Integrations"])
async def connectors_test(req: ConnectorTestRequest):
    """Test connectivity to a specific connector."""
    c = _ensure_connectors()
    if not c:
        return _error(503, "unavailable", "Connector hub not loaded")
    try:
        result = c.test_connection(req.connector)
        return {"status": "ok", "connector": req.connector, "result": result}
    except Exception as e:
        return _error(500, "test_failed", str(e))


def register(app, ns=None):
    """Register integrations routes."""
    return router

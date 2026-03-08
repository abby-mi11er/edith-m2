"""
Security routes for E.D.I.T.H.
Auto-extracted from main.py — 5 routes + heartbeat-sync + §WIRE security endpoints
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

log = logging.getLogger("edith.routes.security")
router = APIRouter(tags=["Security"])


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# Fix 4: Bolt heartbeat sync — receives Electron's drive poll state
@router.post("/api/bolt/heartbeat-sync", tags=["System"])
async def bolt_heartbeat_sync(request: Request):
    """Receive drive poll state from Electron and feed to BoltHeartbeat."""
    try:
        from server.security import update_bolt_from_electron
        body = await request.json()
        return update_bolt_from_electron(body)
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# §WIRE: Security Anomaly Detection — TrustCenterPanel
@router.get("/api/security/anomaly/alerts", tags=["Security"])
async def security_anomaly_alerts():
    """Get active anomaly alerts."""
    try:
        from server.anomaly import AnomalyDetector
        detector = AnomalyDetector()
        return {"alerts": detector.get_alerts()}
    except Exception:
        return {"alerts": []}


@router.get("/api/security/anomaly/stats", tags=["Security"])
async def security_anomaly_stats():
    """Get anomaly detection statistics."""
    try:
        from server.anomaly import AnomalyDetector
        detector = AnomalyDetector()
        return {"stats": detector.get_stats()}
    except Exception:
        return {"stats": {"total_alerts": 0, "active": 0, "resolved": 0}}


# §WIRE: Security Logs — TrustCenterPanel
@router.get("/api/security/logs/list", tags=["Security"])
async def security_logs_list():
    """List security event logs."""
    try:
        from server.security import IsolatedQueryLog
        logger = IsolatedQueryLog()
        return {"logs": logger.recent(limit=50)}
    except Exception:
        return {"logs": []}


# §WIRE: Soul Verification — TrustCenterPanel
@router.post("/api/security/soul/init", tags=["Security"])
async def security_soul_init(request: Request):
    """Initialize soul verification (identity attestation)."""
    try:
        from server.security import verify_physical_soul
        body = await request.json()
        result = verify_physical_soul(body.get("volumes_path", "/Volumes"))
        return result
    except Exception as e:
        return _error(500, "soul_init_failed", str(e))


@router.post("/api/security/soul/verify", tags=["Security"])
async def security_soul_verify(request: Request):
    """Verify soul attestation challenge (POST with body)."""
    try:
        from server.security import verify_physical_soul
        body = await request.json()
        result = verify_physical_soul(body.get("volumes_path", "/Volumes"))
        return {"verified": result.get("soul_detected", False), "detail": result}
    except Exception as e:
        return _error(500, "soul_verify_failed", str(e))


# §AUDIT-FIX: Frontend SoulGauge.tsx calls this as GET
@router.get("/api/security/soul/verify", tags=["Security"])
async def security_soul_verify_get():
    """Verify soul attestation (GET — default /Volumes)."""
    try:
        from server.security import verify_physical_soul
        result = verify_physical_soul("/Volumes")
        return {"verified": result.get("soul_detected", False), "detail": result}
    except Exception as e:
        return _error(500, "soul_verify_failed", str(e))


def register(app, ns=None):
    """Register security route handlers using the namespace dict from main.py."""
    if not ns:
        return router

    def _get(name):
        fn = ns.get(name)
        if fn is None:
            log.warning(f"§WIRE: handler '{name}' not found in main.py")
        return fn

    for path, name in [
        ("/api/lockdown", "lockdown_activate"),
        ("/api/auth/refresh", "auth_refresh"),
        ("/api/auth/revoke", "auth_revoke"),
        ("/api/manifest/verify", "verify_manifest"),
    ]:
        fn = _get(name)
        if fn:
            router.post(path, tags=["Security"])(fn)

    _deactivate = _get("lockdown_deactivate")
    if _deactivate:
        router.delete("/api/lockdown", tags=["Security"])(_deactivate)

    return router

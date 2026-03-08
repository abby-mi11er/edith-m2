"""
§MENDELEY: OAuth Handshake + Library Sync Routes
Extracted from main.py — 5 endpoints
"""
import logging
from fastapi import APIRouter
from starlette.responses import Response, RedirectResponse

router = APIRouter(tags=["Mendeley"])
log = logging.getLogger("edith.mendeley")


def register(app, ns=None):
    """Wire Mendeley routes — imports mendeley_bridge at registration time."""
    try:
        from server.mendeley_bridge import (
            get_auth_url, exchange_code, status as mendeley_status,
            full_sync as mendeley_full_sync, list_documents,
        )

        @router.get("/oauth/mendeley/start")
        async def mendeley_oauth_start():
            """Redirect to Mendeley OAuth authorization page."""
            url = get_auth_url()
            if not url:
                return {"error": "Mendeley Client ID not configured — set it in Settings → Connectors Hub"}
            return RedirectResponse(url)

        @router.get("/oauth/mendeley/callback")
        async def mendeley_oauth_callback(code: str = ""):
            """OAuth callback — exchange code for token."""
            if not code:
                return {"error": "No authorization code received"}
            result = await exchange_code(code)
            if result.get("error"):
                return result
            return Response(
                content="""
                <html><body style="background:#0f0f14;color:#e0e0f0;font-family:Inter,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh">
                <div style="text-align:center">
                    <h1 style="font-size:48px">✅</h1>
                    <h2>Mendeley Connected</h2>
                    <p style="opacity:0.6">Your token has been saved to the Secure Enclave.<br/>You can close this tab and return to E.D.I.T.H.</p>
                </div>
                </body></html>
                """,
                media_type="text/html",
            )

        @router.get("/api/mendeley/status")
        async def mendeley_status_route():
            return mendeley_status()

        @router.post("/api/mendeley/sync")
        async def mendeley_sync():
            """Trigger a full Mendeley library sync."""
            try:
                report = await mendeley_full_sync()
                return report
            except Exception as e:
                log.warning(f"§MENDELEY: Sync failed: {e}")
                return {"error": str(e)}

        @router.get("/api/mendeley/documents")
        async def mendeley_docs():
            """List Mendeley library documents."""
            try:
                docs = await list_documents(limit=50)
                return {"documents": docs, "count": len(docs)}
            except Exception as e:
                return {"error": str(e)}

        log.info("§MENDELEY: OAuth routes loaded — /oauth/mendeley/start, /api/mendeley/sync")
    except Exception as _e:
        log.warning(f"§MENDELEY: Could not load: {_e}")

    return router

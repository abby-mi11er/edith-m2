"""
Reasoning routes for E.D.I.T.H.
Extracted from main.py — 4 routes
"""
from fastapi import APIRouter

router = APIRouter(tags=["Reasoning"])

def _bind():
    from server.main import (
        confidence_endpoint, paragraph_confidence_endpoint,
        contradictions_endpoint, source_freshness_endpoint,
    )
    router.post("/api/confidence")(confidence_endpoint)
    router.post("/api/paragraph-confidence")(paragraph_confidence_endpoint)
    router.post("/api/contradictions")(contradictions_endpoint)
    router.post("/api/source-freshness")(source_freshness_endpoint)


def register(app, ns=None):
    """Register reasoning routes."""
    _bind()
    return router


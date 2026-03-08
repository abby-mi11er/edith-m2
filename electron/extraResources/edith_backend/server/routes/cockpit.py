"""
Cockpit Routes — Extracted from main.py
========================================
Atlas, Topology, Clusters, Cockpit Status
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.cockpit")
router = APIRouter(prefix="/api/cockpit", tags=["Cockpit"])

# State holder — populated at registration time from main.py globals
_state = {}


def init_cockpit_state(state: dict):
    """Called at startup to inject main.py globals."""
    global _state
    _state = state


def _error_response(status_code, error, detail):
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@router.post("/atlas")
async def cockpit_atlas(request: Request):
    if not _state.get("atlas_ok"):
        return _error_response(503, "unavailable", "Vector mapping not loaded")
    body = await request.json()
    build_atlas = _state.get("build_atlas_from_chroma")
    return build_atlas(
        chroma_dir=_state["CHROMA_DIR"],
        embed_model=_state["EMBED_MODEL"],
        sample_size=body.get("sample_size", 2000),
    )


@router.post("/topology")
async def cockpit_topology(request: Request):
    if not _state.get("atlas_ok"):
        return _error_response(503, "unavailable", "Vector mapping not loaded")
    body = await request.json()
    gen_topo = _state.get("generate_topological_summary")
    return gen_topo(body.get("cluster", "APE"))


@router.get("/clusters")
async def cockpit_clusters():
    if not _state.get("atlas_ok"):
        return _error_response(503, "unavailable", "Vector mapping not loaded")
    CLUSTER_CENTROIDS = _state.get("CLUSTER_CENTROIDS", {})
    CLUSTER_COLORS = _state.get("CLUSTER_COLORS", {})
    return {
        "clusters": {
            name: {"centroid": list(c), "color": CLUSTER_COLORS.get(name, "#94A3B8")}
            for name, c in CLUSTER_CENTROIDS.items()
        },
    }


@router.get("/status")
async def cockpit_status():
    return {
        "surfaces": ["atlas", "warroom", "committee", "cockpit"],
        "modules_loaded": {
            "atlas": _state.get("atlas_ok", False),
            "simulation": _state.get("sim_ok", False),
            "oracle": _state.get("oracle_ok", False),
            "antigravity": _state.get("antigrav_ok", False),
            "jarvis": _state.get("jarvis_ok", False),
            "causal": _state.get("causal_ok", False),
        },
        "endpoints": 358,
        "modules": 19,
    }

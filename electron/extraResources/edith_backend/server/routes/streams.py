"""
Subconscious Streams Routes — API endpoints for cross-domain handshakes
=========================================================================
Exposes the Subconscious Memory, Metabolic Balancer, Geospatial Loop,
Socratic Review, Speculative Horizon, Mirror Dissertation, and Sovereign Audit.
"""

import logging
from fastapi import APIRouter, Body

log = logging.getLogger("edith.routes.streams")

router = APIRouter(prefix="/api/streams", tags=["streams"])


# ── 1. Subconscious Shared Memory ──

@router.get("/subconscious/links")
async def get_subconscious_links(doc_id: str = "", min_strength: float = 0.5):
    """Get hidden connections discovered between papers."""
    from server.subconscious_streams import subconscious_memory
    links = subconscious_memory.get_links_for(doc_id or None, min_strength)
    return {"links": links, "total": len(links), "status": subconscious_memory.status}


@router.post("/subconscious/signal")
async def signal_paper_indexed(body: dict = Body(default={})):
    """Signal that a new paper was indexed — triggers subconscious link discovery."""
    from server.subconscious_streams import subconscious_memory
    doc_metadata = body.get("metadata", body)
    chunks = body.get("chunks", [])
    discovered = await subconscious_memory.on_paper_indexed(doc_metadata, chunks)
    return {
        "discovered_links": [l.to_dict() for l in (discovered or [])],
        "total_links": len(subconscious_memory._links),
    }


# ── 2. Metabolic Resource Balancer ──

@router.get("/metabolic/vitals")
async def get_vitals():
    """Read current system thermal and memory state."""
    from server.subconscious_streams import metabolic_balancer
    return metabolic_balancer.read_system_vitals()


@router.get("/metabolic/throttle")
async def check_throttle():
    """Should the system throttle? Returns model override and UI hint."""
    from server.subconscious_streams import metabolic_balancer
    return metabolic_balancer.should_throttle()


@router.post("/metabolic/heavy-task")
async def register_heavy_task(body: dict = Body(default={})):
    """Register or unregister a heavy task for metabolic awareness."""
    from server.subconscious_streams import metabolic_balancer
    task_name = body.get("task", "")
    action = body.get("action", "register")
    if action == "unregister":
        metabolic_balancer.unregister_heavy_task(task_name)
    else:
        metabolic_balancer.register_heavy_task(task_name)
    return {"status": metabolic_balancer.status}


# ── 3. Geospatial Causal Loop ──

@router.post("/geo-causal/analyze")
async def geo_causal_analyze(body: dict = Body(default={})):
    """
    Map-to-Stata pipeline: send coordinates, get causal dashboard overlay.
    
    As you move from Lubbock to Warsaw, coefficients update in real-time.
    """
    from server.subconscious_streams import geospatial_loop
    from server.main import app
    lat = body.get("lat", 0)
    lng = body.get("lng", 0)
    radius_km = body.get("radius_km", 50)
    variable = body.get("variable", "administrative_burden")
    result = await geospatial_loop.coordinates_to_causal(lat, lng, radius_km, variable, app)
    return result


# ── 4. Socratic Peer Review ──

@router.post("/socratic-review/briefing")
async def pre_defense_briefing(body: dict = Body(default={})):
    """
    Pre-Defense Briefing: three personas fire the Sniper at your draft.
    
    Returns severity-ranked findings and suggested citations.
    """
    from server.subconscious_streams import socratic_review
    from server.main import app
    draft = body.get("text", body.get("draft", ""))
    if not draft:
        return {"error": "Provide 'text' or 'draft' with your manuscript content"}
    return await socratic_review.pre_defense_briefing(draft, app)


# ── 5. Speculative Horizon (Dream Engine) ──

@router.get("/horizon/sparks")
async def get_sparks():
    """Get Morning Spark memos from speculative brainstorming."""
    from server.subconscious_streams import speculative_horizon
    return {
        "sparks": speculative_horizon.get_sparks(),
        "idle_minutes": round(speculative_horizon.idle_minutes, 1),
    }


@router.post("/horizon/dream")
async def trigger_dream():
    """Manually trigger speculative brainstorming (normally idle-triggered)."""
    from server.subconscious_streams import speculative_horizon
    from server.main import app
    spark = await speculative_horizon.check_and_dream(app)
    return {"spark": spark, "status": "dreamed" if spark else "not_idle_enough"}


# ── 6. Mirror Dissertation ──

@router.post("/mirror/draft-paragraph")
async def draft_paragraph(body: dict = Body(default={})):
    """
    Auto-generate a lit review paragraph for a paper in your voice.
    
    Send paper metadata → get APSR-formatted paragraph back.
    """
    from server.subconscious_streams import mirror_dissertation
    from server.main import app
    paper = body.get("paper", body)
    style = body.get("style", "APSR")
    notes = body.get("existing_notes", "")
    return await mirror_dissertation.draft_lit_review_paragraph(paper, notes, style, app)


# ── 7. Sovereign Audit ──

@router.post("/sovereign/certify")
async def generate_sovereign_proof(body: dict = Body(default={})):
    """
    Generate a Zero-Knowledge Proof certificate for a research result.
    
    Proves statistical significance without revealing the data.
    """
    from server.subconscious_streams import sovereign_audit
    result = body.get("result", {})
    data_hash = body.get("data_hash", "")
    paper_title = body.get("paper_title", "")
    if not result:
        return {"error": "Provide 'result' with coefficient, std_error, p_value, n, method"}
    return sovereign_audit.generate_certificate(result, data_hash, paper_title)


# ── 8. Method Lab → Vibe Coder ──

@router.post("/bridge/method-to-code")
async def method_to_code(body: dict = Body(default={})):
    """Auto-generate code for the winning method comparison."""
    try:
        from server.subconscious_streams import bridge_method_lab_to_vibe_coder
        from server.main import app
        winning_method = body.get("method", {})
        if not winning_method:
            return {"status": "ok", "code": "", "message": "No method specified"}
        result = await bridge_method_lab_to_vibe_coder(winning_method, app=app)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e), "code": ""}


# ── 9. Spaced Rep → Deep Dive ──

@router.post("/bridge/weak-cards")
async def weak_cards_deep_dive(body: dict = Body(default={})):
    """When flashcard accuracy drops, auto-queue deep dives on weak topics."""
    from server.subconscious_streams import bridge_spaced_rep_to_deep_dive
    from server.main import app
    weak_cards = body.get("cards", [])
    return await bridge_spaced_rep_to_deep_dive(weak_cards, app=app)


# ── Status / Health ──

@router.get("/status")
async def streams_status():
    """Status of all subconscious streams and cross-domain handshakes."""
    from server.subconscious_streams import (
        subconscious_memory, metabolic_balancer, speculative_horizon
    )
    return {
        "subconscious_memory": subconscious_memory.status,
        "metabolic_balancer": metabolic_balancer.status,
        "speculative_horizon": {
            "idle_minutes": round(speculative_horizon.idle_minutes, 1),
            "sparks_count": len(speculative_horizon._sparks),
        },
        "streams_active": 9,
        "handshakes": [
            "Vector↔Graph (Subconscious Memory)",
            "NPU↔API (Metabolic Balancer)",
            "Map↔Stata (Geospatial Causal Loop)",
            "Committee↔Sniper (Socratic Peer Review)",
            "Idle↔Dream (Speculative Horizon)",
            "Mendeley↔Drafter (Mirror Dissertation)",
            "Enclave↔Proof (Sovereign Audit)",
            "MethodLab↔VibeCoder",
            "SpacedRep↔DeepDive",
        ],
    }

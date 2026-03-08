from __future__ import annotations
"""
Brain routes for E.D.I.T.H. Citadel
=====================================
Extracted from main.py -- 25 routes across 6 tag groups:
  Bridge (7), Monitor (8), Socratic (4), Connectome (2), HUD (1), Master (6)

All imports are guarded: if a module isn't available, the endpoint
returns 503 instead of crashing the server.
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.routes.brain")

router = APIRouter()

# ── Helper ────────────────────────────────────────────────────────

def _error_response(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": code, "detail": detail},
    )


# ── Guarded imports ──────────────────────────────────────────────

_bridge = None
_monitor = None
_socratic = None
_connectome_logic = None
_neural_hud_inst = None
_master_ignite = None
_master_metabolize = None
_master_deep_click = None
_master_dream = None
_master_collapse = None
_master_hud = None

try:
    from server.citadel_bridge import citadel_bridge as _bridge
    log.info("CitadelBridge loaded (brain routes)")
except Exception as _e:
    log.warning(f"citadel_bridge not available: {_e}")

try:
    from server.metabolic_monitor import metabolic_monitor as _monitor
    log.info("MetabolicMonitor loaded (brain routes)")
except Exception as _e:
    log.warning(f"metabolic_monitor not available: {_e}")

try:
    from server.socratic_navigator import socratic_navigator as _socratic
    log.info("SocraticNavigator loaded (brain routes)")
except Exception as _e:
    log.warning(f"socratic_navigator not available: {_e}")

try:
    from server.connectome import connectome as _connectome_logic
    log.info("Connectome loaded (brain routes)")
except Exception as _e:
    log.warning(f"connectome not available: {_e}")

try:
    from server.neural_health_hud import neural_hud as _neural_hud_inst
    log.info("NeuralHealthHUD loaded (brain routes)")
except Exception as _e:
    log.warning(f"neural_health_hud not available: {_e}")

try:
    from server.citadel_connectome_master import (
        ignite as _master_ignite, metabolize as _master_metabolize,
        deep_click as _master_deep_click, dream as _master_dream,
        collapse as _master_collapse, hud as _master_hud,
    )
    log.info("CitadelConnectomeMaster loaded (brain routes)")
except Exception as _e:
    log.warning(f"citadel_connectome_master not available: {_e}")

_paper_deconstructor = None
_method_lab = None
_causal_raytracing = None

try:
    from server.paper_deconstructor import PaperDeconstructor
    _paper_deconstructor = PaperDeconstructor()
    log.info("PaperDeconstructor loaded (brain routes)")
except Exception as _e:
    log.warning(f"paper_deconstructor not available: {_e}")

try:
    from server.method_lab import MethodLab
    _method_lab = MethodLab()
    log.info("MethodLab loaded (brain routes)")
except Exception as _e:
    log.warning(f"method_lab not available: {_e}")

try:
    from server.causal_raytracing import CausalRayTracer
    _causal_raytracing = CausalRayTracer()
    log.info("CausalRaytracer loaded (brain routes)")
except Exception as _e:
    log.warning(f"causal_raytracing not available: {_e}")


# ═══════════════════════════════════════════════════════════════════
# Bridge: Cockpit CLI + Global Focus Variable
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/bridge/run", tags=["Bridge"])
async def bridge_run_command(request: Request):
    """Execute a Cockpit command (e.g. '// Plot residuals for Potter County')."""
    if not _bridge:
        return _error_response(503, "unavailable", "CitadelBridge not loaded")
    body = await request.json()
    command = body.get("command", "")
    if not command:
        return _error_response(400, "missing_command", "Provide a 'command' string")
    return _bridge.run(command)


@router.post("/api/bridge/focus/paper", tags=["Bridge"])
async def bridge_focus_paper(request: Request):
    """Set focus to a paper — broadcasts to all subscribed modules."""
    if not _bridge:
        return _error_response(503, "unavailable", "CitadelBridge not loaded")
    body = await request.json()
    result = _bridge.focus_paper(
        title=body.get("title", ""),
        path=body.get("path", ""),
        author=body.get("author", ""),
        methodology=body.get("methodology", ""),
    )
    # §AUTOPILOT: Emit paper.focused for reading progress tracking
    try:
        from server.event_bus import bus
        await bus.emit("paper.focused", {
            "sha256": body.get("sha256", ""), "title": body.get("title", ""),
            "source_panel": "bridge",
        }, source="bridge_focus")
    except Exception:
        pass
    return result


@router.post("/api/bridge/focus/concept", tags=["Bridge"])
async def bridge_focus_concept(request: Request):
    """Set focus to a concept — broadcasts to all modules."""
    if not _bridge:
        return _error_response(503, "unavailable", "CitadelBridge not loaded")
    body = await request.json()
    return _bridge.focus_concept(body.get("concept", ""))


@router.post("/api/bridge/focus/chapter", tags=["Bridge"])
async def bridge_focus_chapter(request: Request):
    """Set focus to a dissertation chapter."""
    if not _bridge:
        return _error_response(503, "unavailable", "CitadelBridge not loaded")
    body = await request.json()
    return _bridge.focus_chapter(
        chapter=body.get("chapter", 1),
        title=body.get("title", ""),
    )


@router.get("/api/bridge/status", tags=["Bridge"])
async def bridge_status():
    """Get current focus, cockpit history, and pipeline status."""
    if not _bridge:
        return {"status": "unavailable"}
    return _bridge.status


@router.post("/api/bridge/stage", tags=["Bridge"])
async def bridge_stage_dataset(request: Request):
    """Stage a dataset into the Live Swap for instant Cockpit access."""
    if not _bridge:
        return _error_response(503, "unavailable", "CitadelBridge not loaded")
    body = await request.json()
    # §SEC: Validate path is under allowed directories to prevent traversal
    path = body.get("path", "")
    if path:
        import os
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        resolved = os.path.realpath(path)
        if data_root and not resolved.startswith(os.path.realpath(data_root)):
            return _error_response(403, "path_blocked",
                                   "Path must be within the data root")
    return _bridge.stage_dataset(
        name=body.get("name", ""),
        path=path,
    )


@router.post("/api/bridge/pipe", tags=["Bridge"])
async def bridge_pipe_stata(request: Request):
    """Pipe Stata output → Auto-Annotator → Notion in one call."""
    if not _bridge:
        return _error_response(503, "unavailable", "CitadelBridge not loaded")
    body = await request.json()
    return _bridge.pipe(
        stata_log=body.get("log_path", ""),
        notion_page=body.get("notion_page", ""),
    )


# ═══════════════════════════════════════════════════════════════════
# Monitor: Vitals, Self-Healing, Ghost Variables, Paradigm Toggle
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/monitor/vitals", tags=["Monitor"])
async def monitor_vitals():
    """Real-time vitals: CPU, memory, Bolt SSD, thermal state."""
    if not _monitor:
        return {"status": "unavailable"}
    return _monitor.check_vitals()


@router.post("/api/monitor/heal", tags=["Monitor"])
async def monitor_heal(request: Request):
    """Attempt to self-heal a broken file link."""
    if not _monitor:
        return _error_response(503, "unavailable", "MetabolicMonitor not loaded")
    body = await request.json()
    return _monitor.heal(body.get("path", ""))


@router.get("/api/monitor/integrity", tags=["Monitor"])
async def monitor_integrity():
    """Full vault integrity scan with auto-healing."""
    if not _monitor:
        return {"status": "unavailable"}
    return _monitor.scan_integrity()


@router.post("/api/monitor/ghost-variables", tags=["Monitor"])
async def monitor_ghost_variables(request: Request):
    """Find omitted variables in a regression model."""
    if not _monitor:
        return _error_response(503, "unavailable", "MetabolicMonitor not loaded")
    body = await request.json()
    depvar = body.get("depvar", "")
    controls = body.get("controls", [])
    return {"ghosts": _monitor.find_ghost_variables(depvar, controls)}


@router.get("/api/monitor/overnight", tags=["Monitor"])
async def monitor_overnight_scan():
    """Run the full overnight metacognitive scan."""
    if not _monitor:
        return {"status": "unavailable"}
    return _monitor.overnight_scan()


@router.post("/api/monitor/paradigm", tags=["Monitor"])
async def monitor_toggle_paradigm(request: Request):
    """Toggle the theoretical paradigm (Institutionalist, Behavioralist, etc.)."""
    if not _monitor:
        return _error_response(503, "unavailable", "MetabolicMonitor not loaded")
    body = await request.json()
    return _monitor.toggle_paradigm(body.get("paradigm", "institutionalist"))


@router.get("/api/monitor/paradigms", tags=["Monitor"])
async def monitor_list_paradigms():
    """List available paradigm profiles."""
    if not _monitor:
        return {"paradigms": []}
    return {"paradigms": _monitor.paradigm.available_paradigms}


@router.get("/api/monitor/report", tags=["Monitor"])
async def monitor_metacognitive_report():
    """Full metacognitive status: healer, ghosts, paradigm, vitals."""
    if not _monitor:
        return {"status": "unavailable"}
    return _monitor.metacognitive_report()


# ═══════════════════════════════════════════════════════════════════
# Socratic Navigator: Adversarial Training
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/socratic/process", tags=["Socratic"])
async def socratic_process_reading(request: Request):
    """Process a reading through the Socratic engines."""
    if not _socratic:
        return _error_response(503, "unavailable", "SocraticNavigator not loaded")
    body = await request.json()
    return _socratic.process_reading(
        text=body.get("text", ""),
        source=body.get("source", ""),
    )


@router.post("/api/socratic/sandbox", tags=["Socratic"])
async def socratic_launch_sandbox(request: Request):
    """Launch a methodology sandbox (DiD, RDD, IV)."""
    if not _socratic:
        return _error_response(503, "unavailable", "SocraticNavigator not loaded")
    body = await request.json()
    return _socratic.launch_sandbox(body.get("method", "did"))


@router.post("/api/socratic/committee", tags=["Socratic"])
async def socratic_convene_committee(request: Request):
    """Convene the Committee of Sages on a claim."""
    if not _socratic:
        return _error_response(503, "unavailable", "SocraticNavigator not loaded")
    body = await request.json()
    return _socratic.convene_committee(body.get("claim", ""))


@router.post("/api/socratic/resolve", tags=["Socratic"])
async def socratic_resolve(request: Request):
    """Submit a defense to resolve a Socratic challenge."""
    if not _socratic:
        return _error_response(503, "unavailable", "SocraticNavigator not loaded")
    body = await request.json()
    challenge_id = body.get("challenge_id", body.get("debate_id", ""))
    defense = body.get("defense", body.get("response", ""))
    try:
        return _socratic.resolve(
            challenge_id=challenge_id,
            response=defense,
        )
    except Exception as e:
        return _error_response(500, "resolve_failed", str(e))


@router.get("/api/socratic/chamber-state", tags=["Socratic"])
async def socratic_chamber_state():
    """Full chamber state for the frontend Socratic Chamber panel.

    Returns sage avatars, needle feed transcript, active challenge, and rigor settings.
    """
    state: dict = {
        "sages": [
            {"id": "mettler", "name": "Mettler", "field": "Local Governance & Nonprofits", "color": "#00e5ff", "speaking": False},
            {"id": "aldrich", "name": "Aldrich", "field": "Political Institutions", "color": "#b388ff", "speaking": False},
            {"id": "kim", "name": "Kim", "field": "Causal Inference & Methods", "color": "#ffab40", "speaking": False},
        ],
        "needle_feed": [],
        "rigor": {"adversarial_level": 50, "evidence_threshold": 50, "methodology_rigor": 50},
    }

    if _socratic:
        try:
            # Pull challenge history for the needle feed
            if hasattr(_socratic, "_challenge_history"):
                state["needle_feed"] = [
                    {
                        "id": f"needle-{i}",
                        "sage": entry.get("sage", "Committee"),
                        "text": entry.get("text", ""),
                        "type": entry.get("type", "probe"),
                    }
                    for i, entry in enumerate(getattr(_socratic, "_challenge_history", [])[-20:])
                ]

            # Active challenge
            if hasattr(_socratic, "_active_challenge") and _socratic._active_challenge:
                state["active_challenge"] = _socratic._active_challenge

            # Rigor settings
            if hasattr(_socratic, "_rigor"):
                state["rigor"] = _socratic._rigor

            # Sage speaking state (if a challenge is active)
            if state.get("active_challenge"):
                active_sage = state["active_challenge"].get("current_sage", "")
                for sage in state["sages"]:
                    if sage["id"] == active_sage:
                        sage["speaking"] = True
        except Exception as e:
            log.warning(f"Chamber state error: {e}")

    return state


@router.post("/api/socratic/rigor", tags=["Socratic"])
async def socratic_set_rigor(request: Request):
    """Update adversarial rigor settings."""
    body = await request.json()
    if _socratic and hasattr(_socratic, "_rigor"):
        for key in ("adversarial_level", "evidence_threshold", "methodology_rigor"):
            if key in body:
                _socratic._rigor[key] = max(0, min(100, int(body[key])))
        return {"status": "updated", "rigor": _socratic._rigor}
    return {"status": "unavailable"}

@router.post("/api/connectome/audit", tags=["Connectome"])
async def connectome_audit_claim(request: Request):
    """Run a logic audit on a theoretical claim."""
    if not _connectome_logic:
        return _error_response(503, "unavailable", "Connectome not loaded")
    body = await request.json()
    return _connectome_logic.audit(body.get("claim", ""))


@router.post("/api/connectome/causal-proof", tags=["Connectome"])
async def connectome_causal_proof(request: Request):
    """Find causal evidence for a claim in Stata output and vault."""
    if not _connectome_logic:
        return _error_response(503, "unavailable", "Connectome not loaded")
    body = await request.json()
    return _connectome_logic.find_causal_proof(body.get("claim", ""))


# ═══════════════════════════════════════════════════════════════════
# Neural Health HUD
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/hud/html", tags=["HUD"])
async def hud_get_html():
    """Serve the Neural Health HUD as self-contained HTML."""
    if not _neural_hud_inst:
        return _error_response(503, "unavailable", "NeuralHealthHUD not loaded")
    html = _neural_hud_inst.render_html()
    return JSONResponse(content={"html": html})


@router.get("/api/hud/snapshot", tags=["HUD"])
async def hud_snapshot():
    """Full brain state as JSON — polled by the frontend Neural HUD panel.

    Merges data from:
      - neural_health_hud.snapshot() (brain, synapse, connectome, dreams, graph, bolt, weak_claims)
      - metabolic_monitor (vitals, paradigm, ghost variables, throughput)
      - connectome (last audit state for Logic Pulse)
    """
    snapshot: dict = {}

    # Neural HUD snapshot (brain, synapse, connectome, dreams, graph, bolt, weak_claims)
    if _neural_hud_inst:
        try:
            snapshot = _neural_hud_inst.snapshot()
        except Exception as e:
            log.warning(f"HUD snapshot error: {e}")
            snapshot = {"error": str(e)}

    # Merge metabolic monitor data (vitals, paradigm, ghosts)
    if _monitor:
        try:
            snapshot["vitals"] = _monitor.check_vitals()
        except Exception:
            pass
        try:
            snapshot["paradigm"] = {
                "active": _monitor.paradigm.active,
                "description": _monitor.paradigm.profile.get("description", ""),
            }
        except Exception:
            pass
        # Ghost variables from the latest overnight scan
        try:
            ghost_det = _monitor.ghost_detector
            if hasattr(ghost_det, "_last_scan_results") and ghost_det._last_scan_results:
                snapshot["ghosts"] = [
                    g.to_dict() if hasattr(g, "to_dict") else g
                    for g in ghost_det._last_scan_results[:5]
                ]
        except Exception:
            pass
        # Metabolic stream — Bolt throughput estimate
        try:
            vitals = snapshot.get("vitals", {})
            bolt = snapshot.get("bolt", {})
            snapshot["metabolic_stream"] = {
                "throughput_mbps": vitals.get("disk_read_mbps", 0),
                "max_throughput_mbps": 3100,  # Bolt USB4 theoretical max
                "prefetching": vitals.get("prefetching", False),
                "prefetch_target": vitals.get("prefetch_target", ""),
                "ram_loaded_gb": vitals.get("memory_used_gb", 0),
                "ram_total_gb": vitals.get("memory_total_gb", 0),
            }
        except Exception:
            pass

    # Last connectome audit for Logic Pulse
    if _connectome_logic:
        try:
            history = getattr(_connectome_logic, "_audit_history", [])
            if history:
                last = history[-1]
                snapshot["audit"] = {
                    "last_claim": last.get("claim", "")[:200],
                    "hud_action": last.get("hud_action", "none"),
                    "confidence": last.get("confidence", 1.0),
                    "issues": last.get("issues", []),
                    "contradictions": last.get("contradictions", []),
                }
        except Exception:
            pass

    return snapshot


# ═══════════════════════════════════════════════════════════════════
# Connectome Master: Lifecycle
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/master/ignite", tags=["Master"])
async def master_ignite():
    """Run the 10-point ignition sequence."""
    if not _master_ignite:
        return _error_response(503, "unavailable", "ConnectomeMaster not loaded")
    return _master_ignite()


@router.post("/api/master/metabolize", tags=["Master"])
async def master_metabolize_ep(request: Request):
    """Process a thought through the full metabolic loop."""
    if not _master_metabolize:
        return _error_response(503, "unavailable", "ConnectomeMaster not loaded")
    body = await request.json()
    thought = body.get("thought", "")
    if not thought:
        return _error_response(400, "missing_thought", "Provide a 'thought' string")
    try:
        result = _master_metabolize(thought)
        return result if isinstance(result, dict) else {"status": "metabolized", "result": str(result)}
    except Exception as e:
        return _error_response(500, "metabolize_failed", str(e))


@router.post("/api/master/deep-click", tags=["Master"])
async def master_deep_click_ep(request: Request):
    """Recursive drill-down on a concept."""
    if not _master_deep_click:
        return _error_response(503, "unavailable", "ConnectomeMaster not loaded")
    body = await request.json()
    concept = body.get("concept", "")
    if not concept:
        return _error_response(400, "missing_concept", "Provide a 'concept' string")
    try:
        result = _master_deep_click(concept, body.get("week", 0))
        return result if isinstance(result, dict) else {"status": "clicked", "result": str(result)}
    except Exception as e:
        return _error_response(500, "deep_click_failed", str(e))


@router.post("/api/master/dream", tags=["Master"])
async def master_dream_ep():
    """Trigger overnight speculative synthesis."""
    if not _master_dream:
        return _error_response(503, "unavailable", "ConnectomeMaster not loaded")
    return _master_dream()


@router.post("/api/master/collapse", tags=["Master"])
async def master_collapse_ep():
    """Sovereign Collapse — graceful shutdown."""
    if not _master_collapse:
        return _error_response(503, "unavailable", "ConnectomeMaster not loaded")
    return _master_collapse()


@router.get("/api/master/hud", tags=["Master"])
async def master_hud_ep():
    """Get the full HUD state from the Connectome Master."""
    if not _master_hud:
        return {"status": "unavailable"}
    return _master_hud()


# ═══════════════════════════════════════════════════════════════════
# Forensic Workbench: Paper Deconstructor + Method Lab + Raytracing
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/forensic/state", tags=["Forensic"])
async def forensic_state():
    """Full forensic workbench state for the frontend panel.

    Merges data from paper_deconstructor, method_lab, and causal_raytracing.
    """
    state: dict = {"paper_title": "", "citations": [], "datasets": [],
                   "estimators": [], "claims": [], "sandbox": None}

    if _paper_deconstructor:
        try:
            dec: dict = {}
            last_result = getattr(_paper_deconstructor, "last_result", None)
            if isinstance(last_result, dict):
                dec = last_result
            elif hasattr(_paper_deconstructor, "_audit_cache"):
                cache = getattr(_paper_deconstructor, "_audit_cache", {})
                if isinstance(cache, dict) and cache:
                    last_key = next(reversed(cache))
                    cached = cache.get(last_key)
                    if isinstance(cached, dict):
                        dec = cached

            if dec:
                paper = dec.get("paper", {})
                if isinstance(paper, dict):
                    state["paper_title"] = paper.get("title", "")
                else:
                    state["paper_title"] = dec.get("title", "")

                citations = dec.get("citations", [])
                if isinstance(citations, dict):
                    state["citations"] = citations.get("items", []) or []
                elif isinstance(citations, list):
                    state["citations"] = citations

                datasets = dec.get("datasets", [])
                if isinstance(datasets, dict):
                    state["datasets"] = datasets.get("items", []) or []
                elif isinstance(datasets, list):
                    state["datasets"] = datasets

                estimators = dec.get("estimators", [])
                if isinstance(estimators, dict):
                    all_estimators = estimators.get("all", [])
                    primary = estimators.get("primary")
                    if isinstance(all_estimators, list) and all_estimators:
                        state["estimators"] = all_estimators
                    elif isinstance(primary, dict):
                        state["estimators"] = [primary]
                elif isinstance(estimators, list):
                    state["estimators"] = estimators
        except Exception as e:
            log.warning(f"forensic state error (deconstructor): {e}")

    if _causal_raytracing:
        try:
            get_claims = getattr(_causal_raytracing, "get_claims", None)
            if callable(get_claims):
                state["claims"] = get_claims() or []
            else:
                get_analysis = getattr(_causal_raytracing, "get_argument_analysis", None)
                if callable(get_analysis):
                    analysis = get_analysis() or {}
                    contradictions = analysis.get("contradictions", []) if isinstance(analysis, dict) else []
                    if isinstance(contradictions, list):
                        state["claims"] = [
                            {
                                "claim": item.get("title", ""),
                                "status": "contradiction",
                                "source": item.get("doc_id", ""),
                            }
                            for item in contradictions
                            if isinstance(item, dict) and item.get("title")
                        ]
        except Exception as e:
            log.warning(f"forensic state error (raytracing): {e}")

    if _method_lab:
        try:
            state["sandbox"] = _method_lab.get_sandbox_state() or None
        except Exception as e:
            log.warning(f"forensic state error (method_lab): {e}")

    return JSONResponse(content=state)


@router.post("/api/forensic/deconstruct", tags=["Forensic"])
async def forensic_deconstruct(request: Request):
    """Trigger molecular deconstruction of the currently focused paper.

    Accepts either a file path (reads the file) or raw text.
    Uses the Global Focus Variable as fallback.
    """
    if not _paper_deconstructor:
        return _error_response(503, "unavailable", "PaperDeconstructor not loaded")

    # Get focused paper from bridge
    path = ""
    if _bridge and hasattr(_bridge, 'focus') and _bridge.focus:
        path = getattr(_bridge.focus, 'path', '')

    try:
        body = await request.json()
    except Exception:
        body = {}

    paper_path = body.get("path", path)
    text = body.get("text", "")

    # If a path is given, try to read the file
    if paper_path and not text:
        import os
        if os.path.isfile(paper_path):
            try:
                with open(paper_path, 'r', errors='ignore') as f:
                    text = f.read()
            except Exception:
                pass

    if not text:
        return _error_response(400, "no_content",
            "Provide 'text' or a valid file 'path' to deconstruct")
    try:
        result = _paper_deconstructor.full_forensic_audit(
            text=text,
            title=body.get("title", ""),
            author=body.get("author", ""),
        )
        # §AUTOPILOT: Emit paper.deconstructed for reading progress + concept tracking
        try:
            from server.event_bus import bus
            await bus.emit("paper.deconstructed", {
                "sha256": body.get("sha256", ""), "title": body.get("title", ""),
                "concepts": result.get("concepts", []) if isinstance(result, dict) else [],
            }, source="forensic_deconstruct")
        except Exception:
            pass
        return JSONResponse(content=result if isinstance(result, dict) else {"status": "done"})
    except Exception as e:
        return _error_response(500, "deconstruct_failed", str(e))


@router.post("/api/forensic/sandbox", tags=["Forensic"])
async def forensic_sandbox(request: Request):
    """Adjust sandbox coefficients and re-compute significance."""
    if not _method_lab:
        return _error_response(503, "unavailable", "MethodLab not loaded")
    body = await request.json()
    coefficients = body.get("coefficients", {})
    try:
        result = _method_lab.adjust(coefficients)
        return JSONResponse(content=result if isinstance(result, dict) else {"status": "adjusted"})
    except Exception as e:
        return _error_response(500, "sandbox_error", str(e))


@router.post("/api/forensic/compare", tags=["Forensic"])
async def forensic_compare(request: Request):
    """Side-by-side comparison of 2-3 papers.

    Accepts sha256 list, retrieves papers from ChromaDB, deconstructs each,
    and highlights disagreements in claims, methods, and conclusions.
    """
    body = await request.json()
    sha_list = body.get("papers", [])  # list of sha256 strings
    if len(sha_list) < 2:
        return _error_response(400, "need_papers", "Provide at least 2 paper sha256 values")
    if len(sha_list) > 3:
        sha_list = sha_list[:3]

    import os, json
    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if not chroma_dir:
        return _error_response(500, "no_chroma", "EDITH_CHROMA_DIR not configured")

    try:
        from server.chroma_backend import _get_client
        client = _get_client(chroma_dir)
        collection = client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})
    except Exception as e:
        return _error_response(503, "chroma_error", str(e))

    papers = []
    for sha in sha_list:
        results = collection.get(where={"sha256": sha}, include=["documents", "metadatas"], limit=10)
        if not results["documents"]:
            papers.append({"sha256": sha, "title": "Not found", "text": "", "meta": {}})
            continue
        text = " ".join(results["documents"][:5])[:8000]
        meta = results["metadatas"][0] if results["metadatas"] else {}
        papers.append({
            "sha256": sha,
            "title": meta.get("title", meta.get("source_file", "Unknown")),
            "author": meta.get("author", ""),
            "year": meta.get("year", ""),
            "method": meta.get("method", ""),
            "academic_topic": meta.get("academic_topic", ""),
            "text": text,
            "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
        })

    # Deconstruct each paper if deconstructor available
    deconstructions = []
    for p in papers:
        if _paper_deconstructor and p["text"]:
            try:
                dec = _paper_deconstructor.full_forensic_audit(
                    text=p["text"], title=p["title"], author=p.get("author", ""))
                deconstructions.append(dec if isinstance(dec, dict) else {})
            except Exception:
                deconstructions.append({})
        else:
            deconstructions.append({})

    # Find disagreements
    disagreements = []
    if len(papers) >= 2:
        # Method disagreements
        methods = [p.get("method", "") for p in papers]
        if len(set(m for m in methods if m)) > 1:
            disagreements.append({
                "type": "method",
                "finding": f"Papers use different methods: {', '.join(m or 'unspecified' for m in methods)}",
                "severity": "info",
            })

        # Conclusion/topic disagreements
        topics = [p.get("academic_topic", "") for p in papers]
        if len(set(t for t in topics if t)) > 1:
            disagreements.append({
                "type": "topic",
                "finding": f"Papers cover different topics: {', '.join(t or 'unspecified' for t in topics)}",
                "severity": "info",
            })

        # Check for contradiction markers in text
        contra_markers = ["however", "contrary", "on the other hand", "disagrees",
                          "no effect", "fails to find", "does not support", "inconsistent",
                          "contradicts", "refute", "challenge"]
        for i, p in enumerate(papers):
            text_lower = p["text"].lower()
            for j, other in enumerate(papers):
                if i >= j:
                    continue
                other_author = other.get("author", "").split(",")[0].strip()
                if other_author and other_author.lower() in text_lower:
                    # Paper i mentions Paper j's author — check context
                    for marker in contra_markers:
                        idx = text_lower.find(marker)
                        if idx > -1:
                            author_idx = text_lower.find(other_author.lower())
                            if abs(idx - author_idx) < 200:  # within 200 chars
                                snippet = p["text"][max(0, idx-50):idx+100].strip()
                                disagreements.append({
                                    "type": "contradiction",
                                    "finding": f"{p['title']} may contradict {other['title']}",
                                    "snippet": snippet,
                                    "severity": "warning",
                                })
                                break

    # Build comparison response
    comparison = []
    for i, p in enumerate(papers):
        dec = deconstructions[i] if i < len(deconstructions) else {}
        comparison.append({
            "sha256": p["sha256"],
            "title": p["title"],
            "author": p.get("author", ""),
            "year": p.get("year", ""),
            "method": p.get("method", ""),
            "topic": p.get("academic_topic", ""),
            "estimators": dec.get("estimators", []),
            "datasets": dec.get("datasets", []),
            "citations_count": len(dec.get("citations", [])),
            "claims": dec.get("claims", [])[:5],  # top 5 claims
        })

    return {
        "papers": comparison,
        "disagreements": disagreements,
        "count": len(comparison),
    }


# ═══════════════════════════════════════════════════════════════════
# Pipeline Improvements: Integration, PhD-OS, DAG, Training
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/pipelines/integration-improvements", tags=["Pipelines"])
async def pipeline_integration_improvements(request: Request):
    """Run integration improvements (OpenAlex, Zotero, ORCID, Semantic Scholar)."""
    try:
        from pipelines.integration_improvements import OpenAlexClient, ZoteroClient
        body = await request.json()
        query = body.get("query", "")
        action = body.get("action", "search")
        if action == "search" and query:
            client = OpenAlexClient()
            results = client.search_works(query, limit=5)
            return JSONResponse(content={"status": "ok", "results": results})
        return JSONResponse(content={"status": "ok", "capabilities": ["openalex_search", "zotero_sync", "orcid_lookup", "semantic_scholar"]})
    except Exception as e:
        return _error_response(500, "integration_error", str(e))


@router.post("/api/pipelines/phd-os-improvements", tags=["Pipelines"])
async def pipeline_phd_os_improvements(request: Request):
    """Run PhD-OS knowledge artifact improvements (linking, scoring, decay)."""
    try:
        from pipelines.phd_os_improvements import ConceptLinker, score_artifact_quality, apply_temporal_decay
        body = await request.json()
        action = body.get("action", "status")
        if action == "score" and body.get("entry"):
            result = score_artifact_quality(body["entry"])
            return JSONResponse(content={"status": "ok", "score": result})
        if action == "decay" and body.get("entries"):
            decayed = apply_temporal_decay(body["entries"], half_life_days=body.get("half_life", 180))
            return JSONResponse(content={"status": "ok", "entries": decayed})
        linker = ConceptLinker()
        return JSONResponse(content={"status": "ok", "capabilities": ["concept_linking", "quality_scoring", "temporal_decay", "contradiction_detection"]})
    except Exception as e:
        return _error_response(500, "phd_os_error", str(e))


@router.post("/api/pipelines/pipeline-improvements", tags=["Pipelines"])
async def pipeline_pipeline_improvements(request: Request):
    """Run pipeline DAG improvements (orchestration, idempotency, monitoring)."""
    try:
        from pipelines.pipeline_improvements import PipelineDAG, PipelineMonitor, validate_schema
        body = await request.json()
        action = body.get("action", "status")
        if action == "validate" and body.get("data") and body.get("schema"):
            result = validate_schema(body["data"], body["schema"])
            return JSONResponse(content={"status": "ok", "validation": result})
        monitor = PipelineMonitor()
        return JSONResponse(content={"status": "ok", "capabilities": ["dag_orchestration", "idempotency", "monitoring", "schema_validation"]})
    except Exception as e:
        return _error_response(500, "pipeline_dag_error", str(e))


@router.post("/api/pipelines/training-improvements", tags=["Pipelines"])
async def pipeline_training_improvements(request: Request):
    """Run training improvements (DPO, active learning, curriculum, ablation)."""
    try:
        from pipelines.training_improvements import extract_dpo_pairs, format_dpo_for_openai, TrainingDataFilter
        body = await request.json()
        action = body.get("action", "status")
        if action == "extract_dpo" and body.get("debate_log"):
            pairs = extract_dpo_pairs(body["debate_log"])
            return JSONResponse(content={"status": "ok", "dpo_pairs": pairs})
        if action == "format_dpo" and body.get("pairs"):
            formatted = format_dpo_for_openai(body["pairs"])
            return JSONResponse(content={"status": "ok", "formatted": formatted})
        filt = TrainingDataFilter()
        return JSONResponse(content={"status": "ok", "capabilities": ["dpo_extraction", "dpo_formatting", "active_learning", "curriculum_scheduling"]})
    except Exception as e:
        return _error_response(500, "training_imp_error", str(e))


@router.post("/api/pipelines/eval-improvements", tags=["Pipelines"])
async def pipeline_eval_improvements(request: Request):
    """Run evaluation improvements (MRR, NDCG, precision, auto-eval)."""
    try:
        from pipelines.eval_improvements import AutoEvalRunner, mean_reciprocal_rank, ndcg_at_k
        body = await request.json()
        action = body.get("action", "status")
        if action == "run_eval":
            runner = AutoEvalRunner()
            result = runner.run()
            return JSONResponse(content={"status": "ok", "eval": result})
        return JSONResponse(content={"status": "ok", "capabilities": ["auto_eval", "mrr", "ndcg", "precision_at_k"]})
    except Exception as e:
        return _error_response(500, "eval_imp_error", str(e))


@router.post("/api/pipelines/feedback-training", tags=["Pipelines"])
async def pipeline_feedback_training(request: Request):
    """Run feedback training loop — learn from user corrections."""
    try:
        from pipelines.feedback_trainer import FeedbackTrainer
        body = await request.json()
        trainer = FeedbackTrainer()
        if body.get("correction"):
            result = trainer.process_correction(body["correction"])
            return JSONResponse(content={"status": "ok", "result": result})
        return JSONResponse(content={"status": "ok", "capabilities": ["correction_capture", "dpo_pair_generation", "feedback_loop"]})
    except Exception as e:
        return _error_response(500, "feedback_error", str(e))

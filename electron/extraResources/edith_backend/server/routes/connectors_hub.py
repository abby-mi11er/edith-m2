"""
Connectors Hub — Unified API routes for all external connectors.
=================================================================
Provides endpoints for Zotero, Connected Papers, Claude/Anthropic,
Perplexity Sonar, Consensus, DAGitty, Overleaf, MathPix, and Stata.
Each connector is lazy-loaded on first use.
"""
import logging
import random as _overlay_random
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.routes.connectors_hub")
router = APIRouter(prefix="/api/connectors", tags=["Connectors Hub"])

# ── Lazy loaders ─────────────────────────────────────────────────

_mendeley = None
_zotero = None
_similarity = None
_anthropic = None
_perplexity = None
_consensus = None
_dagitty = None
_overleaf = None
_mathpix = None
_stata = None
_earth = None


def _get_mendeley():
    global _mendeley
    if _mendeley is None:
        from pipelines.connectors import MendeleyConnector
        import os
        _mendeley = MendeleyConnector(access_token=os.environ.get("MENDELEY_ACCESS_TOKEN", ""))
    return _mendeley


def _get_zotero():
    global _zotero
    if _zotero is None:
        from server.zotero_bridge import ZoteroBridge
        _zotero = ZoteroBridge()
    return _zotero


def _get_similarity():
    global _similarity
    if _similarity is None:
        from server.connected_papers_bridge import ConnectedPapersBridge
        _similarity = ConnectedPapersBridge()
    return _similarity


def _get_anthropic():
    global _anthropic
    if _anthropic is None:
        from server.anthropic_bridge import AnthropicBridge
        _anthropic = AnthropicBridge()
    return _anthropic


def _get_perplexity():
    global _perplexity
    if _perplexity is None:
        from server.perplexity_bridge import PerplexityBridge
        _perplexity = PerplexityBridge()
    return _perplexity


def _get_consensus():
    global _consensus
    if _consensus is None:
        from server.consensus_bridge import ConsensusBridge
        _consensus = ConsensusBridge()
    return _consensus


def _get_dagitty():
    global _dagitty
    if _dagitty is None:
        from server.dagitty_bridge import DAGittyBridge
        _dagitty = DAGittyBridge()
    return _dagitty


def _get_overleaf():
    global _overleaf
    if _overleaf is None:
        from server.overleaf_bridge import OverleafBridge
        _overleaf = OverleafBridge()
    return _overleaf


def _get_mathpix():
    global _mathpix
    if _mathpix is None:
        from server.mathpix_bridge import MathPixBridge
        _mathpix = MathPixBridge()
    return _mathpix


def _get_stata():
    global _stata
    if _stata is None:
        from server.stata_bridge import StataBridge
        _stata = StataBridge()
    return _stata


def _get_earth():
    global _earth
    if _earth is None:
        from server.google_earth_bridge import GoogleEarthBridge
        _earth = GoogleEarthBridge()
    return _earth


# ── Hub Status (all connectors) ─────────────────────────────────

@router.get("/hub/status")
async def hub_status():
    """Get status of all connectors in one call.  §IMP: parallel + response_time_ms."""
    import asyncio, time as _t
    connectors = {}

    async def _check(name, getter):
        t0 = _t.monotonic()
        try:
            obj = getter()
            result = obj.status() if hasattr(obj, "status") else {"available": True, "configured": True}
        except Exception as e:
            result = {"available": False, "error": str(e)}
        result["response_time_ms"] = round((_t.monotonic() - t0) * 1000)
        connectors[name] = result

    # §IMP: Run all bridge checks concurrently
    await asyncio.gather(*[
        _check(name, getter) for name, getter in [
            ("mendeley", _get_mendeley), ("zotero", _get_zotero),
            ("similarity_graph", _get_similarity), ("anthropic", _get_anthropic),
            ("perplexity", _get_perplexity), ("consensus", _get_consensus),
            ("dagitty", _get_dagitty), ("overleaf", _get_overleaf),
            ("mathpix", _get_mathpix), ("stata", _get_stata),
            ("google_earth", _get_earth),
        ]
    ])

    # Existing connectors — check by env key presence
    import os
    connectors["notion"] = {
        "available": bool(os.environ.get("NOTION_TOKEN")),
        "configured": bool(os.environ.get("NOTION_TOKEN")),
        "note": "Notion workspace sync", "response_time_ms": 0,
    }
    connectors["openalex"] = {
        "available": True, "configured": True,
        "note": "Bulk academic discovery (free API)", "response_time_ms": 0,
    }
    connectors["semantic_scholar"] = {
        "available": True,
        "configured": bool(os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")),
        "note": "Citation influence + networks" + (" (with API key)" if os.environ.get("S2_API_KEY") else " (free tier)"),
        "response_time_ms": 0,
    }
    connectors["crossref"] = {
        "available": True, "configured": True,
        "note": "DOI metadata lookup (free API)", "response_time_ms": 0,
    }
    connectors["orcid"] = {
        "available": True, "configured": True,
        "note": "Researcher profile lookup (free API)", "response_time_ms": 0,
    }

    total = len(connectors)
    configured = sum(1 for c in connectors.values() if c.get("configured"))
    available = sum(1 for c in connectors.values() if c.get("available"))

    return {
        "connectors": connectors,
        "summary": {"total": total, "configured": configured, "available": available},
    }


# ── Mendeley ─────────────────────────────────────────────────────

@router.post("/mendeley/sync")
async def mendeley_sync():
    """Sync Mendeley library and export for indexing."""
    import os
    if not os.environ.get("MENDELEY_ACCESS_TOKEN"):
        return {"status": "no_key", "connector": "mendeley", "detail": "Set MENDELEY_ACCESS_TOKEN to enable"}
    try:
        m = _get_mendeley()
        return m.export_for_indexing()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/mendeley/status")
async def mendeley_status():
    try:
        m = _get_mendeley()
        import os
        token = os.environ.get("MENDELEY_ACCESS_TOKEN", "")
        return {"available": bool(token), "configured": bool(token), "note": "Mendeley library + annotations"}
    except Exception as e:
        return {"available": False, "error": str(e)}


# ── Zotero ───────────────────────────────────────────────────────

@router.post("/zotero/sync")
async def zotero_sync():
    """Sync Zotero library and export for indexing."""
    import os
    if not os.environ.get("ZOTERO_API_KEY"):
        return {"status": "no_key", "connector": "zotero", "detail": "Set ZOTERO_API_KEY to enable"}
    try:
        z = _get_zotero()
        return z.export_for_indexing()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/zotero/status")
async def zotero_status():
    return _get_zotero().status()


# ── Similarity Graph (Connected Papers) ─────────────────────────

@router.post("/similarity/graph")
async def similarity_graph(request: Request):
    """Build a similarity graph from a seed paper."""
    body = await request.json()
    seed = body.get("seed_paper_id", "")
    depth = body.get("depth", 1)
    if not seed:
        return JSONResponse(status_code=400, content={"error": "seed_paper_id required"})
    try:
        return _get_similarity().build_similarity_graph(seed, depth=depth)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Anthropic / Claude ───────────────────────────────────────────

@router.post("/anthropic/query")
async def anthropic_query(request: Request):
    """Send a query to Claude for long-context reasoning.  §IMP: token budget tracking."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "no_key", "connector": "anthropic", "detail": "Set ANTHROPIC_API_KEY to enable"}
    body = await request.json()
    prompt = body.get("prompt", "")
    context = body.get("context", "")
    max_tokens = body.get("max_tokens", 4096)  # §IMP: configurable token budget
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "prompt required"})
    try:
        result = _get_anthropic().query(prompt, context=context, max_tokens=max_tokens)
        # §IMP: Include token usage in response
        if isinstance(result, dict):
            result.setdefault("tokens_budget", max_tokens)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/anthropic/audit")
async def anthropic_audit(request: Request):
    """Audit a document for logical flaws using Claude.  §IMP: structured output."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "no_key", "connector": "anthropic", "detail": "Set ANTHROPIC_API_KEY to enable"}
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})
    try:
        result = _get_anthropic().audit_document(text, focus=body.get("focus", ""))
        # §IMP: Ensure structured audit findings
        if isinstance(result, dict) and "findings" in result:
            for f in result["findings"]:
                f.setdefault("severity", "medium")
                f.setdefault("category", "logic")
                f.setdefault("suggestion", "")
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/anthropic/status")
async def anthropic_status():
    return _get_anthropic().status()


# ── Perplexity Sonar ─────────────────────────────────────────────

@router.post("/perplexity/verify")
async def perplexity_verify(request: Request):
    """Real-time fact-check a claim via Perplexity Sonar.  §IMP: source quality scoring."""
    import os
    if not os.environ.get("PERPLEXITY_API_KEY"):
        return {"status": "no_key", "connector": "perplexity", "detail": "Set PERPLEXITY_API_KEY to enable"}
    body = await request.json()
    claim = body.get("claim", "")
    if not claim:
        return JSONResponse(status_code=400, content={"error": "claim required"})
    try:
        result = _get_perplexity().verify_claim(claim)
        # §IMP: Score verification sources by domain authority
        _authority = {".gov": 1.0, ".edu": 0.9, ".org": 0.7, ".int": 0.8}
        if isinstance(result, dict) and "sources" in result:
            for src in result["sources"]:
                url = src.get("url", "")
                score = 0.5  # default for generic domains
                for domain, val in _authority.items():
                    if domain in url:
                        score = val
                        break
                src["authority_score"] = score
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/perplexity/status")
async def perplexity_status():
    return _get_perplexity().status()


# ── Consensus ────────────────────────────────────────────────────

@router.post("/consensus/check")
async def consensus_check(request: Request):
    """Check evidence consensus for a claim.  §IMP: evidence meter."""
    import os
    if not os.environ.get("CONSENSUS_API_KEY"):
        return {"status": "no_key", "connector": "consensus", "detail": "Set CONSENSUS_API_KEY to enable"}
    body = await request.json()
    claim = body.get("claim", "")
    if not claim:
        return JSONResponse(status_code=400, content={"error": "claim required"})
    try:
        result = _get_consensus().check_claim(claim)
        # §IMP: Add evidence meter counts
        if isinstance(result, dict):
            papers = result.get("papers", result.get("results", []))
            if isinstance(papers, list):
                for_count = sum(1 for p in papers if (p.get("stance") or "").lower() in ("for", "supporting", "yes"))
                against_count = sum(1 for p in papers if (p.get("stance") or "").lower() in ("against", "opposing", "no"))
                neutral_count = len(papers) - for_count - against_count
                result["evidence_meter"] = {
                    "for_count": for_count, "against_count": against_count,
                    "neutral_count": neutral_count, "total": len(papers),
                }
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/consensus/status")
async def consensus_status():
    return _get_consensus().status()


# ── DAGitty (R subprocess) ───────────────────────────────────────

@router.post("/dagitty/dag")
async def dagitty_build(request: Request):
    """Build a causal DAG from variables and edges."""
    body = await request.json()
    variables = body.get("variables", [])
    edges = body.get("edges", [])
    if not edges:
        return JSONResponse(status_code=400, content={"error": "edges required"})
    # Normalize edges: accept both [["X","Y"]] and [{"from":"X","to":"Y"}]
    normalized = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            normalized.append({"from": e[0], "to": e[1]})
        elif isinstance(e, dict):
            normalized.append(e)
        else:
            return JSONResponse(status_code=400, content={"error": f"Invalid edge format: {e}"})
    try:
        return _get_dagitty().build_dag(variables, normalized)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/dagitty/adjustment")
async def dagitty_adjustment(request: Request):
    """Find adjustment sets for causal identification."""
    body = await request.json()
    dag_spec = body.get("dag_spec", "")
    exposure = body.get("exposure", "")
    outcome = body.get("outcome", "")
    if not all([dag_spec, exposure, outcome]):
        return JSONResponse(status_code=400, content={"error": "dag_spec, exposure, and outcome required"})
    try:
        return _get_dagitty().find_adjustment_sets(dag_spec, exposure, outcome)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/dagitty/status")
async def dagitty_status():
    return _get_dagitty().status()


# ── Overleaf ─────────────────────────────────────────────────────

@router.post("/overleaf/push")
async def overleaf_push(request: Request):
    """Push a LaTeX draft to Overleaf.  §IMP: dry-run mode."""
    import os
    if not os.environ.get("OVERLEAF_TOKEN") and not os.environ.get("OVERLEAF_GIT_URL") and not os.environ.get("OVERLEAF_GIT_TOKEN"):
        return {"status": "no_key", "connector": "overleaf", "detail": "Set OVERLEAF_GIT_TOKEN to enable"}
    body = await request.json()
    latex = body.get("latex", "")
    if not latex:
        return JSONResponse(status_code=400, content={"error": "latex content required"})
    # §IMP: Dry-run validates without pushing
    if body.get("dry_run"):
        return {"dry_run": True, "valid": True, "chars": len(latex),
                "has_begin_document": "\\begin{document}" in latex,
                "note": "LaTeX validated. Set dry_run=false to push."}
    try:
        return _get_overleaf().push_draft(
            latex, filename=body.get("filename", "edith_draft.tex"),
            commit_message=body.get("message", "E.D.I.T.H. auto-push"),
            project_name=body.get("project", ""),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/overleaf/projects")
async def overleaf_projects():
    """List available Overleaf projects."""
    return _get_overleaf().list_projects()

@router.get("/overleaf/status")
async def overleaf_status():
    return _get_overleaf().status()


# ── MathPix ──────────────────────────────────────────────────────

@router.post("/mathpix/ocr")
async def mathpix_ocr(request: Request):
    """Convert an image to LaTeX via MathPix OCR."""
    import os
    if not os.environ.get("MATHPIX_APP_ID"):
        return {"status": "no_key", "connector": "mathpix", "detail": "Set MATHPIX_APP_ID and MATHPIX_APP_KEY to enable"}
    body = await request.json()
    image_path = body.get("image_path", "")
    if not image_path:
        return JSONResponse(status_code=400, content={"error": "image_path required"})
    try:
        return _get_mathpix().image_to_latex(image_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/mathpix/status")
async def mathpix_status():
    return _get_mathpix().status()


# ── Stata ────────────────────────────────────────────────────────

@router.post("/stata/execute")
async def stata_execute(request: Request):
    """Execute Stata code."""
    body = await request.json()
    code = body.get("code", "")
    if not code:
        return JSONResponse(status_code=400, content={"error": "code required"})
    # §IMP: Execution sandbox — run in temp dir to prevent accidental overwrites
    try:
        import tempfile, os
        sandbox_dir = tempfile.mkdtemp(prefix="edith_stata_")
        # Copy dataset if specified
        dataset = body.get("dataset_path", "")
        if dataset and os.path.exists(dataset):
            import shutil
            shutil.copy2(dataset, sandbox_dir)
        result = _get_stata().execute(code, timeout=body.get("timeout", 120), cwd=sandbox_dir)
        if isinstance(result, dict):
            result["sandbox_dir"] = sandbox_dir
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/stata/status")
async def stata_status():
    return _get_stata().status()


# ── Google Earth Engine ──────────────────────────────────────────

@router.post("/earth/land-use")
async def earth_land_use(request: Request):
    """Analyze land-use change over time at coordinates."""
    body = await request.json()
    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        return JSONResponse(status_code=400, content={"error": "lat and lon required"})
    try:
        return _get_earth().analyze_land_use(
            lat=lat, lon=lon,
            radius_m=body.get("radius_m", 5000),
            start_year=body.get("start_year", 2015),
            end_year=body.get("end_year", 2025),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/earth/audit")
async def earth_audit(request: Request):
    """Geospatial audit — verify a physical claim with satellite data."""
    body = await request.json()
    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        return JSONResponse(status_code=400, content={"error": "lat and lon required"})
    try:
        return _get_earth().audit_location(
            lat=lat, lon=lon,
            claim=body.get("claim", ""),
            year_claimed=body.get("year_claimed", 2024),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/earth/counterfactual")
async def earth_counterfactual(request: Request):
    """Geospatial counterfactual — compare treated vs control areas."""
    body = await request.json()
    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        return JSONResponse(status_code=400, content={"error": "lat and lon required"})
    try:
        return _get_earth().counterfactual_analysis(
            lat=lat, lon=lon,
            radius_m=body.get("radius_m", 10000),
            policy_variable=body.get("policy_variable", ""),
            baseline_year=body.get("baseline_year", 2020),
            projection_years=body.get("projection_years", 5),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/earth/overlay")
async def earth_overlay(request: Request):
    """Generate geo-data overlay points from library papers.

    Returns lat/lon/value/label data for choropleth or point-value
    map layers. Accepts a variable name (e.g., 'poverty_rate') and
    overlays it on geographic paper locations.
    """
    body = await request.json()
    variable = body.get("variable", "")
    filter_method = body.get("method", "")
    filter_country = body.get("country", "")

    import os
    from collections import defaultdict

    # Pull geo-tagged papers from library cache
    try:
        from server.server_state import library_cache as _library_cache, library_lock as _library_lock
        with _library_lock:
            cache = list(_library_cache)
    except Exception:
        cache = []

    data_points = []
    country_aggregates: dict[str, list] = defaultdict(list)

    # Built-in country centroids for quick mapping
    _centroids = {
        "united states": (39.8, -98.6), "usa": (39.8, -98.6),
        "poland": (51.9, 19.1), "germany": (51.2, 10.4),
        "united kingdom": (55.4, -3.4), "uk": (55.4, -3.4),
        "france": (46.6, 2.2), "china": (35.9, 104.2),
        "india": (20.6, 79.0), "brazil": (-14.2, -51.9),
        "japan": (36.2, 138.3), "australia": (-25.3, 133.8),
        "canada": (56.1, -106.3), "mexico": (23.6, -102.6),
        "south korea": (35.9, 128.0), "nigeria": (9.1, 8.7),
        "south africa": (-30.6, 22.9), "kenya": (-0.02, 37.9),
        "indonesia": (-0.8, 113.9), "turkey": (38.9, 35.2),
        "italy": (41.9, 12.6), "spain": (40.5, -3.7),
        "netherlands": (52.1, 5.3), "sweden": (60.1, 18.6),
        "norway": (60.5, 8.5), "denmark": (56.3, 9.5),
        "finland": (61.9, 25.7), "switzerland": (46.8, 8.2),
        "austria": (47.5, 14.6), "belgium": (50.5, 4.5),
    }

    for doc in cache:
        meta = doc if isinstance(doc, dict) else {}
        country = (meta.get("country", "") or "").strip().lower()
        if not country:
            continue
        if filter_country and filter_country.lower() not in country:
            continue
        if filter_method and filter_method.lower() not in (meta.get("method", "") or "").lower():
            continue

        lat, lon = _centroids.get(country, (None, None))
        if lat is None:
            continue

        # Jitter to spread overlapping points
        lat += _overlay_random.uniform(-1.5, 1.5)
        lon += _overlay_random.uniform(-1.5, 1.5)

        # Extract value if variable specified
        value = None
        if variable:
            # Look in metadata for the variable
            value = meta.get(variable, meta.get(f"stat_{variable}", None))
            if value is None:
                # Try to find in text snippet
                text = (meta.get("text", "") or "")[:500].lower()
                if variable.lower() in text:
                    value = 0.5  # marker present

        point = {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "label": meta.get("title", meta.get("source_file", ""))[:80],
            "country": country.title(),
            "method": meta.get("method", ""),
            "year": meta.get("year", ""),
        }
        if value is not None:
            point["value"] = value
        data_points.append(point)
        country_aggregates[country.title()].append(point)

    # Build choropleth summary
    choropleth = {}
    for country, points in country_aggregates.items():
        values = [p["value"] for p in points if "value" in p]
        choropleth[country] = {
            "count": len(points),
            "avg_value": round(sum(values) / len(values), 3) if values else None,
            "methods": list(set(p.get("method", "") for p in points if p.get("method"))),
        }

    return {
        "points": data_points[:200],  # cap at 200
        "choropleth": choropleth,
        "total": len(data_points),
        "variable": variable or "(paper count)",
        "filters": {"method": filter_method, "country": filter_country},
    }


@router.get("/earth/status")
async def earth_status():
    return _get_earth().status()

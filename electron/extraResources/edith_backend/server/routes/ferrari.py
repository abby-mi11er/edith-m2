"""
Unified Routers — Ferrari Consolidation Layer
==============================================
Smart aggregation routers that unify scattered endpoints into clean domains.
Old paths remain as aliases — zero capabilities lost.

Domains:
  /api/research/*  — unified search across all sources
  /api/system/*    — unified system dashboard
"""

import asyncio
import logging
import time
from fastapi import APIRouter, Request, Query

log = logging.getLogger("edith.ferrari")

router = APIRouter(tags=["ferrari"])


# ═══════════════════════════════════════════════════════════════════
# UNIFIED RESEARCH — /api/research/search
# Consolidates: openalex, scholar, crossref, orcid, /api/search
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/research/search")
async def unified_search(
    q: str = Query("", description="Search query"),
    source: str = Query("all", description="Source: openalex, scholar, crossref, orcid, local, all"),
    per_page: int = Query(10, description="Results per page"),
    page: int = Query(1, description="Page number"),
):
    """
    Unified search across all research sources.
    
    Use `source=all` to search everywhere, or specify a source:
    - `openalex` — 250M+ academic works
    - `scholar` — Google Scholar
    - `crossref` — DOI metadata
    - `orcid` — Researcher profiles
    - `local` — Your indexed library (ChromaDB)
    """
    if not q:
        return {"error": "Query required", "hint": "?q=welfare+policy&source=all"}

    results = {}
    errors = {}

    async def search_openalex():
        try:
            from server import openalex as oamod
            r = oamod.search_openalex(q, per_page=per_page)
            results["openalex"] = r.get("results", r.get("works", []))
        except Exception as e:
            errors["openalex"] = str(e)

    async def search_scholar():
        try:
            from server.scholarly_repositories import search_scholar as _ss
            r = _ss(q, max_results=per_page)
            results["scholar"] = r if isinstance(r, list) else r.get("results", [])
        except Exception as e:
            errors["scholar"] = str(e)

    async def search_crossref():
        try:
            from server.scholarly_repositories import search_crossref as _sc
            r = _sc(q, max_results=per_page)
            results["crossref"] = r if isinstance(r, list) else r.get("results", [])
        except Exception as e:
            errors["crossref"] = str(e)

    async def search_orcid():
        try:
            from server.scholarly_repositories import search_orcid as _so
            r = _so(q)
            results["orcid"] = r if isinstance(r, list) else r.get("results", [])
        except Exception as e:
            errors["orcid"] = str(e)

    async def search_local():
        try:
            from server.chroma_backend import retrieve_local_sources
            r = retrieve_local_sources(q, top_k=per_page)
            results["local"] = r if isinstance(r, list) else []
        except Exception as e:
            errors["local"] = str(e)

    # Route based on source parameter
    source_map = {
        "openalex": [search_openalex],
        "scholar": [search_scholar],
        "crossref": [search_crossref],
        "orcid": [search_orcid],
        "local": [search_local],
        "all": [search_openalex, search_scholar, search_crossref, search_local],
    }

    tasks = source_map.get(source.lower(), source_map["all"])
    start = time.time()
    await asyncio.gather(*[t() for t in tasks], return_exceptions=True)
    elapsed = round((time.time() - start) * 1000)

    # Merge results
    total_count = sum(len(v) for v in results.values())

    # Emit discovery event for auto-flashcards
    if total_count > 0:
        try:
            from server.event_bus import bus
            all_papers = []
            for papers in results.values():
                if isinstance(papers, list):
                    all_papers.extend(papers[:5])
            if all_papers:
                asyncio.ensure_future(bus.emit("discovery.results", {
                    "papers": all_papers[:10], "query": q,
                }, source="unified_search"))
        except Exception:
            pass

    return {
        "query": q,
        "source": source,
        "total_results": total_count,
        "results": results,
        "errors": errors if errors else None,
        "elapsed_ms": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════
# UNIFIED SYSTEM DASHBOARD — /api/system/dashboard
# Consolidates: status, doctor, bus, streams, connectors, metrics
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/system/dashboard")
async def unified_dashboard():
    """
    One-call system dashboard — everything you need to see E.D.I.T.H.'s state.
    Replaces checking 6+ separate endpoints.
    """
    dashboard = {}

    # Core status
    try:
        from server.main import get_status
        status = await get_status()
        dashboard["status"] = {
            "overall": status.get("status", "unknown"),
            "uptime_s": status.get("uptime_s", 0),
            "subsystems": status.get("subsystems", {}),
        }
    except Exception as e:
        dashboard["status"] = {"error": str(e)}

    # EventBus
    try:
        from server.event_bus import bus
        bus_status = bus.status
        dashboard["event_bus"] = {
            "subscribers": bus_status["subscriber_count"],
            "total_events": bus_status["total_events"],
            "errors": bus_status["error_count"],
            "top_events": dict(list(bus_status.get("top_events", {}).items())[:5]),
        }
    except Exception as e:
        dashboard["event_bus"] = {"error": str(e)}

    # Subconscious Streams
    try:
        from server.subconscious_streams import (
            metabolic_balancer, subconscious_memory, speculative_horizon
        )
        dashboard["streams"] = {
            "metabolic": metabolic_balancer.should_throttle(),
            "subconscious_links": len(subconscious_memory.links),
            "idle_minutes": speculative_horizon.idle_minutes,
        }
    except Exception as e:
        dashboard["streams"] = {"error": str(e)}

    # Route metrics
    try:
        from server.server_state import route_call_counts as _route_call_counts, server_start_time as _server_start_time
        total_calls = sum(_route_call_counts.values())
        dashboard["metrics"] = {
            "total_routes": 355,
            "total_calls": total_calls,
            "uptime_s": round(time.time() - _server_start_time),
        }
    except Exception as e:
        dashboard["metrics"] = {"error": str(e)}

    # Doctor health
    try:
        from server.routes.doctor import doctor_endpoint
        doc = await doctor_endpoint()
        dashboard["health"] = {
            "score": doc.get("health_score", 0),
            "issues": len(doc.get("issues", [])),
        }
    except Exception as e:
        dashboard["health"] = {"error": str(e)}

    # Learning HUD
    try:
        import os, json
        from pathlib import Path
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        if data_root:
            hud_path = Path(data_root) / "pedagogy" / "mastery_hud.json"
            if hud_path.exists():
                mastery = json.loads(hud_path.read_text())
                dashboard["mastery"] = mastery
            else:
                dashboard["mastery"] = {}
        else:
            dashboard["mastery"] = {}
    except Exception:
        dashboard["mastery"] = {}

    return dashboard


# ═══════════════════════════════════════════════════════════════════
# UNIFIED EXPORT — /api/export/unified
# Consolidates: latex, word, ris, refworks, notion
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/export/unified")
async def unified_export(request: Request):
    """
    Single export endpoint supporting all formats.
    
    Body: { "format": "latex|word|ris|refworks|notion", "content": "...", ... }
    """
    body = await request.json()
    fmt = body.get("format", "latex").lower()
    content = body.get("content", "")
    title = body.get("title", "Untitled")

    if not content:
        return {"error": "Content required"}

    # Emit export event
    try:
        from server.event_bus import bus
        asyncio.ensure_future(bus.emit("export.started", {
            "content": content[:5000], "format": fmt, "title": title,
        }, source="unified_export"))
    except Exception:
        pass

    # Route to appropriate exporter
    if fmt == "latex":
        from server.routes.export import export_latex
        # Simulate request by calling the handler logic directly
        return await _dispatch_export(body, "latex")
    elif fmt == "word":
        return await _dispatch_export(body, "word")
    elif fmt == "ris":
        return await _dispatch_export(body, "ris")
    elif fmt == "refworks":
        return await _dispatch_export(body, "refworks")
    elif fmt == "notion":
        return await _dispatch_export(body, "notion")
    else:
        return {"error": f"Unknown format: {fmt}", "supported": ["latex", "word", "ris", "refworks", "notion"]}


async def _dispatch_export(body: dict, fmt: str) -> dict:
    """Dispatch to the correct export handler."""
    content = body.get("content", "")
    sources = body.get("sources", [])
    title = body.get("title", "Untitled")

    if fmt == "latex":
        try:
            from server.export_academic import to_latex
            latex = to_latex(content, sources, title=title)
            return {"format": "latex", "output": latex, "title": title}
        except Exception as e:
            return {"format": "latex", "output": content, "error": str(e)}

    elif fmt == "word":
        try:
            from server.export_academic import to_word_html
            html = to_word_html(content, sources, title=title)
            return {"format": "word", "output": html, "title": title}
        except Exception as e:
            return {"format": "word", "output": f"<html><body>{content}</body></html>", "error": str(e)}

    elif fmt == "ris":
        try:
            from server.export_academic import to_ris
            ris = to_ris(sources)
            return {"format": "ris", "output": ris, "count": len(sources)}
        except Exception as e:
            return {"format": "ris", "output": "", "error": str(e)}

    elif fmt == "refworks":
        try:
            from server.export_academic import to_refworks
            rw = to_refworks(sources)
            return {"format": "refworks", "output": rw, "count": len(sources)}
        except Exception as e:
            return {"format": "refworks", "output": "", "error": str(e)}

    elif fmt == "notion":
        try:
            from server.export_notes import export_to_notion
            result = export_to_notion(content, title=title)
            return {"format": "notion", "result": result}
        except Exception as e:
            return {"format": "notion", "error": str(e)}

    return {"error": f"Unsupported format: {fmt}"}

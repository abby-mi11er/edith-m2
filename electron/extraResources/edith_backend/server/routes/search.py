"""
Search routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("edith")
router = APIRouter(tags=["Search"])

# Module state for geo_papers
_library_building = False
_library_cache = []
_COUNTRY_CENTROIDS = {}


def _resolve_library_state() -> tuple[list, bool]:
    """Resolve live library docs/building state from the library routes module."""
    docs = _library_cache if isinstance(_library_cache, list) else []
    building = bool(_library_building)
    try:
        from server.routes import library as _lib_mod
        live_docs = getattr(_lib_mod, "_library_cache", None)
        if isinstance(live_docs, list):
            docs = live_docs
        live_building = getattr(_lib_mod, "_library_building", None)
        if isinstance(live_building, bool):
            building = live_building
        # If cache is cold, trigger a build so map can populate.
        if not docs and not building:
            starter = getattr(_lib_mod, "_start_library_build", None)
            if callable(starter):
                starter()
                building = bool(getattr(_lib_mod, "_library_building", False))
    except Exception:
        pass
    return docs, building


def __getattr__(name):
    """Resolve missing names from server.main at runtime."""
    try:
        import server.main as m
        return getattr(m, name)
    except (ImportError, AttributeError):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def _get_main():
    import server.main as m
    return m


# ---------------------------------------------------------------------------
# Discovery Endpoint — Search external sources
# ---------------------------------------------------------------------------
class DiscoveryRequest(BaseModel):
    topic: str
    max_results: int = Field(10, ge=1, le=50)


def _format_search_results(results: list[dict]) -> list[dict]:
    return [
        {
            "id": r.get("sha256", ""),
            "title": r.get("title", r.get("file_name", "Untitled")),
            "snippet": (r.get("snippet", r.get("text", "")) or "")[:200],
            "source": r.get("rel_path", r.get("source", "")),
            "score": r.get("score", 0),
        }
        for r in (results or [])
    ]


def _keyword_fallback_search(q: str, limit: int = 20) -> list[dict]:
    tokens = [t for t in q.lower().split() if len(t) > 2]
    if not tokens:
        return []

    docs, _ = _resolve_library_state()
    if not docs:
        try:
            from server.routes import library as _lib_mod
            scan = _lib_mod._scan_library_sources(papers_only=True)  # type: ignore[attr-defined]
            if isinstance(scan, list):
                docs = scan
        except Exception:
            docs = []

    scored: list[dict] = []
    for d in docs or []:
        if not isinstance(d, dict):
            continue
        title = str(d.get("title", "") or "")
        source = str(d.get("source", d.get("rel_path", "")) or "")
        author = str(d.get("author", "") or "")
        snippet = str(d.get("snippet", "") or "")
        hay = " ".join([title, source, author, snippet]).lower()
        score = sum(1 for t in tokens if t in hay)
        if score <= 0:
            continue
        scored.append({
            "id": d.get("sha256", ""),
            "title": title or "Untitled",
            "snippet": snippet[:200],
            "source": source,
            "score": float(score),
        })

    scored.sort(key=lambda r: r.get("score", 0), reverse=True)
    return scored[: max(1, int(limit))]


def search_endpoint(q: str = "", limit: int = 20):
    """Semantic search across indexed documents."""
    q = (q or "").strip()
    if not q:
        return {"results": []}

    env_mode = os.environ.get("EDITH_ENV", "").strip().lower()
    app_mode = os.environ.get("EDITH_APP_MODE", "").strip().lower()
    if env_mode == "test" or app_mode == "test":
        return {"results": []}

    # §FIX: Fast keyword search on library cache (instant, no embedding needed)
    try:
        from server.routes import library as _lib_mod
        _cached = getattr(_lib_mod, "_sources_cache", {})
        _cached_papers = _cached.get("papers") or []
        if _cached_papers:
            tokens = [t for t in q.lower().split() if len(t) > 2]
            if tokens:
                keyword_hits = []
                for d in _cached_papers:
                    hay = " ".join([
                        str(d.get("title", "")),
                        str(d.get("author", "")),
                        str(d.get("source", "")),
                        str(d.get("course", "")),
                    ]).lower()
                    score = sum(1 for t in tokens if t in hay)
                    if score > 0:
                        keyword_hits.append({
                            "id": d.get("sha256", ""),
                            "title": d.get("title", "Untitled"),
                            "snippet": d.get("source", ""),
                            "source": d.get("source", ""),
                            "score": float(score),
                        })
                keyword_hits.sort(key=lambda r: r["score"], reverse=True)
                if keyword_hits:
                    return {"results": keyword_hits[:limit], "fallback": "keyword_cache"}
    except Exception:
        pass

    candidates = [
        str(CHROMA_COLLECTION or "").strip(),
        str(os.environ.get("EDITH_CHROMA_COLLECTION", "")).strip(),
        "edith_docs_pdf",
        "edith_docs_v2_metadata",
        "edith_docs_v2",
        "edith_corpus",
    ]

    seen = set()
    retrieval_errors = []
    for coll in candidates:
        if not coll or coll in seen:
            continue
        seen.add(coll)
        try:
            results = retrieve_local_sources(
                queries=[q],
                chroma_dir=CHROMA_DIR,
                collection_name=coll,
                embed_model=EMBED_MODEL,
                top_k=limit,
            )
            if results:
                return {"results": _format_search_results(results), "collection": coll}
        except Exception as e:
            retrieval_errors.append(f"{coll}: {e}")

    fallback = _keyword_fallback_search(q, limit=limit)
    if fallback:
        return {"results": fallback, "fallback": "keyword"}

    if retrieval_errors:
        log.error("Search error after collection fallback: %s", " | ".join(retrieval_errors[:4]))
    return {"results": []}


def discover_endpoint(req: DiscoveryRequest):
    """Search Semantic Scholar + OpenAlex for papers not in your corpus."""
    try:
        from server.discovery_mode import get_discovery_engine
        engine = get_discovery_engine()
        results = engine.search(req.topic, max_results=req.max_results)
        return results
    except Exception as e:
        log.warning(f"Endpoint error: {e}")
        return {"error": "Operation failed", "papers": [], "topic": req.topic}


# ── §4.0: CrossRef API ──
async def crossref_doi_lookup(doi: str):
    """Look up a paper by DOI via CrossRef.  §IMP: formatted APA citation."""
    try:
        from server.connectors_full import CrossRefConnector
        cr = CrossRefConnector()
        result = cr.lookup_doi(doi)
        if not result:
            return {"error": "DOI not found"}
        # §IMP: Auto-generate APA citation
        if isinstance(result, dict):
            author = result.get("author", "Unknown")
            title = result.get("title", "Untitled")
            year = result.get("year", "n.d.")
            journal = result.get("journal", result.get("container-title", ""))
            result["apa_citation"] = f"{author} ({year}). {title}. {journal}.".strip()
        return result
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def crossref_search(q: str, limit: int = 10):
    """Search CrossRef for works."""
    try:
        from server.connectors_full import CrossRefConnector
        cr = CrossRefConnector()
        return {"results": cr.search(q, limit=limit)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def geo_papers_endpoint(topic: str = "", country: str = "", year: str = ""):
    """Return geo-located paper clusters from library cache."""
    docs, is_building = _resolve_library_state()
    if not docs:
        # Fast fallback: use source scan so Map panel is populated immediately
        # even before full /api/library cache build completes.
        try:
            from server.routes import library as _lib_mod
            scan = _lib_mod._scan_library_sources(papers_only=True)  # type: ignore[attr-defined]
            if isinstance(scan, list):
                docs = [
                    {
                        "title": row.get("title", ""),
                        "sha256": row.get("sha256", ""),
                        "doc_type": row.get("doc_type", "paper"),
                        "author": row.get("author", ""),
                        "year": row.get("year", ""),
                        "academic_topic": row.get("topic") or "General",
                        "project": row.get("topic") or "General",
                    }
                    for row in scan
                    if isinstance(row, dict)
                ]
        except Exception:
            pass
    if is_building and not docs:
        return {"points": [], "building": True}
    if not docs:
        return {"points": [], "categories": [], "countries": [], "topics": []}

    # Apply optional filters
    if topic:
        topics_filter = {t.strip().lower() for t in topic.split(",")}
        docs = [d for d in docs if (d.get("academic_topic") or "").lower() in topics_filter]
    if year:
        docs = [d for d in docs if str(d.get("year", "")) == year.strip()]

    # Try to geo-locate by project name, topic keywords, or title mentions
    geo_clusters: dict = {}
    for d in docs:
        # Check project, title, topic for country mentions
        text = f"{d.get('project', '')} {d.get('title', '')} {d.get('academic_topic', '')}".lower()
        matched_country = None
        for country_name, (lat, lng) in _COUNTRY_CENTROIDS.items():
            if country_name in text:
                matched_country = country_name
                break

        if matched_country:
            key = matched_country
            if key not in geo_clusters:
                lat, lng = _COUNTRY_CENTROIDS[key]
                geo_clusters[key] = {
                    "id": f"geo:{key}",
                    "label": key.title(),
                    "lat": lat,
                    "lng": lng,
                    "category": d.get("academic_topic") or "General",
                    "doc_count": 0,
                    "docs": [],
                    "docs_by_topic": {},
                }
            geo_clusters[key]["doc_count"] += 1
            doc_entry = {
                "title": d.get("title", "Untitled"),
                "sha256": d.get("sha256"),
                "doc_type": d.get("doc_type"),
                "author": d.get("author"),
                "year": d.get("year"),
                "academic_topic": d.get("academic_topic") or "General",
            }
            geo_clusters[key]["docs"].append(doc_entry)
            # Group by topic within country
            doc_topic = d.get("academic_topic") or "General"
            geo_clusters[key]["docs_by_topic"].setdefault(doc_topic, []).append(doc_entry)

    # Apply country filter if specified
    if country:
        country_lower = country.strip().lower()
        geo_clusters = {k: v for k, v in geo_clusters.items() if country_lower in k.lower()}

    points = sorted(geo_clusters.values(), key=lambda p: -p["doc_count"])

    # Assign colors by category
    cat_colors = {}
    palette = ["#388bfd", "#a371f7", "#39d2c0", "#d29922", "#3fb950", "#f85149"]
    for p in points:
        cat = p["category"]
        if cat not in cat_colors:
            cat_colors[cat] = palette[len(cat_colors) % len(palette)]
        p["color"] = cat_colors[cat]
        p["size"] = min(6 + p["doc_count"] * 2, 16)

    # Collect all unique countries and topics for filter dropdowns
    all_countries = sorted({p["label"] for p in points})
    all_topics = sorted({d.get("academic_topic") or "General" for d in docs if d.get("academic_topic")})

    return {"points": points, "categories": list(cat_colors.keys()), "countries": all_countries, "topics": all_topics}


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register search routes."""
    if ns:
        import sys
        _mod = sys.modules[__name__]
        for _name in ['CHROMA_DIR', 'CHROMA_COLLECTION', 'EMBED_MODEL', 'DATA_ROOT',
                      'ROOT_DIR', 'retrieve_local_sources', '_library_cache',
                      '_library_building', '_COUNTRY_CENTROIDS']:
            if _name in ns:
                setattr(_mod, _name, ns[_name])
    router.get("/api/search", tags=["Search"])(search_endpoint)
    router.post("/api/discover", tags=["Search"])(discover_endpoint)
    router.get("/api/crossref/doi/{doi:path}", tags=["Search"])(crossref_doi_lookup)
    router.get("/api/crossref/search", tags=["Search"])(crossref_search)
    router.get("/api/geo/papers", tags=["Search"])(geo_papers_endpoint)
    return router

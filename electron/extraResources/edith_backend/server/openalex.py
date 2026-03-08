"""
OpenAlex API wrapper for Edith — paper discovery, recommendations, and citation data.
Uses the polite pool (no API key required, lower rate limits).
See: https://docs.openalex.org/
"""

import httpx
import os
import time
import hashlib
import json
import logging
from typing import Optional
from pathlib import Path

log = logging.getLogger("edith.openalex")

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://api.openalex.org"
POLITE_EMAIL = os.environ.get("OPENALEX_POLITE_EMAIL", "amiller@uark.edu")  # polite pool identifier
API_KEY = os.environ.get("OPENALEX_API_KEY", "")  # set in .env for 100k credits/day
CACHE_TTL = 3600  # 1 hour
CACHE_DIR = Path(__file__).parent / ".openalex_cache"
CACHE_DIR.mkdir(exist_ok=True)

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        headers = {"User-Agent": f"Edith/1.0 (mailto:{POLITE_EMAIL})"}
        # Build default query params — api_key goes as query param per OpenAlex docs
        default_params = {"mailto": POLITE_EMAIL}
        if API_KEY:
            default_params["api_key"] = API_KEY
            log.info("OpenAlex: using authenticated API key (100k credits/day)")
        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30.0,
            headers=headers,
            params=default_params,
        )
    return _client


# ── Cache helpers ───────────────────────────────────────────────────────────
def _cache_key(endpoint: str, params: dict) -> str:
    raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("_ts", 0) < CACHE_TTL:
            return data.get("payload")
    except Exception:
        pass
    return None


def _cache_set(key: str, payload: dict) -> None:
    path = CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps({"_ts": time.time(), "payload": payload}))
    except Exception:
        pass


# ── Normalize OpenAlex work into a clean dict ───────────────────────────────
def _normalize_work(w: dict) -> dict:
    """Convert raw OpenAlex work object to a clean, frontend-friendly dict."""
    authorships = w.get("authorships") or []
    authors = []
    for a in authorships[:5]:
        author_obj = a.get("author") or {}
        name = author_obj.get("display_name", "")
        if name:
            authors.append(name)

    # Get country from first institution
    countries = set()
    for a in authorships:
        for inst in (a.get("institutions") or []):
            cc = inst.get("country_code")
            if cc:
                countries.add(cc)

    # Primary topic / concept
    primary_topic = None
    topics = w.get("topics") or []
    if topics:
        primary_topic = topics[0].get("display_name")
    elif w.get("concepts"):
        concepts = sorted(w["concepts"], key=lambda c: c.get("score", 0), reverse=True)
        if concepts:
            primary_topic = concepts[0].get("display_name")

    # Open access
    oa = w.get("open_access") or {}

    return {
        "id": w.get("id", ""),
        "doi": w.get("doi"),
        "title": w.get("title") or "Untitled",
        "authors": authors,
        "authors_display": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
        "year": w.get("publication_year"),
        "journal": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "citation_count": w.get("cited_by_count", 0),
        "is_oa": oa.get("is_oa", False),
        "oa_url": oa.get("oa_url"),
        "abstract": w.get("abstract_inverted_index"),  # inverted index format
        "primary_topic": primary_topic,
        "countries": sorted(countries),
        "type": w.get("type", "unknown"),
        "relevance_score": w.get("relevance_score"),
    }


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    positions = {}
    for word, indices in inverted_index.items():
        for idx in indices:
            positions[idx] = word
    if not positions:
        return None
    max_pos = max(positions.keys())
    words = [positions.get(i, "") for i in range(max_pos + 1)]
    return " ".join(words)


# ── Public API ──────────────────────────────────────────────────────────────

async def search_works(
    query: str,
    year: Optional[str] = None,
    work_type: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """
    Search OpenAlex /works endpoint with keyword search + filters.
    Returns { results: [...], total: int, page: int, per_page: int }
    """
    params = {
        "search": query,
        "page": str(page),
        "per_page": str(per_page),
        "sort": "relevance_score:desc",
        "mailto": POLITE_EMAIL,
    }

    filters = []
    if year:
        filters.append(f"publication_year:{year}")
    if work_type:
        filters.append(f"type:{work_type}")
    if filters:
        params["filter"] = ",".join(filters)

    cache_k = _cache_key("search", params)
    cached = _cache_get(cache_k)
    if cached:
        return cached

    client = _get_client()
    try:
        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"OpenAlex search failed: {e}")
        return {"results": [], "total": 0, "page": page, "per_page": per_page, "error": str(e)}

    results = []
    for w in (data.get("results") or []):
        work = _normalize_work(w)
        # Reconstruct abstract for display
        work["abstract_text"] = _reconstruct_abstract(work.pop("abstract", None))
        results.append(work)

    payload = {
        "results": results,
        "total": data.get("meta", {}).get("count", 0),
        "page": page,
        "per_page": per_page,
    }

    _cache_set(cache_k, payload)
    return payload


async def recommend_works(text: str, per_page: int = 15) -> dict:
    """
    Use OpenAlex /find/works for semantic paper recommendations.
    Accepts a question, abstract, or any free text.
    Returns { results: [...], total: int }
    """
    params = {
        "query": text[:5000],  # Max 5000 chars for GET
        "per_page": str(per_page),
        "mailto": POLITE_EMAIL,
    }

    cache_k = _cache_key("recommend", params)
    cached = _cache_get(cache_k)
    if cached:
        return cached

    client = _get_client()
    try:
        resp = await client.get("/find/works", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"OpenAlex recommend failed: {e}")
        return {"results": [], "total": 0, "error": str(e)}

    results = []
    for w in (data.get("results") or []):
        work = _normalize_work(w)
        work["abstract_text"] = _reconstruct_abstract(work.pop("abstract", None))
        results.append(work)

    payload = {
        "results": results,
        "total": len(results),
    }

    _cache_set(cache_k, payload)
    return payload


async def get_work(openalex_id: str) -> Optional[dict]:
    """
    Fetch a single work by OpenAlex ID.
    ID format: https://openalex.org/W1234567890  or  W1234567890
    """
    if not openalex_id.startswith("http"):
        openalex_id = f"https://openalex.org/{openalex_id}"

    params = {"mailto": POLITE_EMAIL}
    cache_k = _cache_key("work", {"id": openalex_id})
    cached = _cache_get(cache_k)
    if cached:
        return cached

    client = _get_client()
    try:
        # Extract the short ID for the URL path
        short_id = openalex_id.split("/")[-1]
        resp = await client.get(f"/works/{short_id}", params=params)
        resp.raise_for_status()
        w = resp.json()
    except Exception as e:
        log.error(f"OpenAlex get_work failed: {e}")
        return None

    work = _normalize_work(w)
    work["abstract_text"] = _reconstruct_abstract(work.pop("abstract", None))

    # Get referenced works for citation context
    work["referenced_works"] = w.get("referenced_works", [])[:20]
    work["cited_by_api_url"] = w.get("cited_by_api_url")
    work["related_works"] = w.get("related_works", [])[:10]

    _cache_set(cache_k, work)
    return work


async def get_citations(openalex_id: str, per_page: int = 20) -> dict:
    """
    Get works that cite a given paper.
    Returns { citing: [...], referenced: [...] }
    """
    work = await get_work(openalex_id)
    if not work:
        return {"citing": [], "referenced": [], "error": "Work not found"}

    # Fetch citing works
    citing = []
    cited_by_url = work.get("cited_by_api_url")
    if cited_by_url:
        client = _get_client()
        try:
            resp = await client.get(
                cited_by_url,
                params={"per_page": str(per_page), "mailto": POLITE_EMAIL},
            )
            resp.raise_for_status()
            data = resp.json()
            for w in (data.get("results") or []):
                citing.append(_normalize_work(w))
        except Exception as e:
            log.error(f"OpenAlex citations fetch failed: {e}")

    return {
        "citing": citing,
        "referenced_ids": work.get("referenced_works", []),
    }


async def resolve_works(openalex_ids: list[str], per_page: int = 50) -> dict:
    """
    Batch-resolve OpenAlex IDs into full paper metadata.
    Uses the OpenAlex filter API: /works?filter=openalex:ID1|ID2|...
    Returns { results: [...] }
    """
    if not openalex_ids:
        return {"results": []}

    # Extract short IDs (W1234567890) from full URLs
    short_ids = []
    for oid in openalex_ids[:per_page]:
        sid = str(oid).replace("https://openalex.org/", "")
        if sid:
            short_ids.append(sid)

    if not short_ids:
        return {"results": []}

    filter_str = "|".join(short_ids)
    params = {
        "filter": f"openalex:{filter_str}",
        "per_page": str(min(len(short_ids), per_page)),
        "mailto": POLITE_EMAIL,
    }

    cache_k = _cache_key("resolve", params)
    cached = _cache_get(cache_k)
    if cached:
        return cached

    client = _get_client()
    try:
        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"OpenAlex resolve_works failed: {e}")
        return {"results": [], "error": str(e)}

    results = []
    for w in (data.get("results") or []):
        work = _normalize_work(w)
        work["abstract_text"] = _reconstruct_abstract(work.pop("abstract", None))
        results.append(work)

    payload = {"results": results, "total": len(results)}
    _cache_set(cache_k, payload)
    return payload

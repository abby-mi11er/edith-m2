"""
NYT Bridge — New York Times API connector for E.D.I.T.H.
==========================================================
Article Search (1851–present), Archive, Most Popular, Top Stories.
Free tier: 500 requests/day, no credit card required.

Setup:
    1. Register at https://developer.nytimes.com/accounts/create
    2. Create App → Get API key
    3. Paste in Settings → Connectors → NYT
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("edith.nyt")

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://api.nytimes.com/svc"
API_KEY = os.environ.get("NYT_API_KEY", "")
CACHE_TTL = 1800  # 30 min
CACHE_DIR = Path(__file__).parent / ".nyt_cache"
CACHE_DIR.mkdir(exist_ok=True)

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=20.0,
            headers={"User-Agent": "Edith/1.0 (research assistant)"},
        )
    return _client


def _get_key() -> str:
    """Get API key, reloading from env if changed via save-key endpoint."""
    global API_KEY
    API_KEY = os.environ.get("NYT_API_KEY", API_KEY)
    return API_KEY


# ── Cache ───────────────────────────────────────────────────────────────────
def _cache_key(name: str, params: dict) -> str:
    raw = f"{name}:{json.dumps(params, sort_keys=True)}"
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


def _cache_set(key: str, payload: dict):
    path = CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps({"_ts": time.time(), "payload": payload}))
    except Exception:
        pass


# ── Article Search ──────────────────────────────────────────────────────────
async def search_articles(
    query: str,
    begin_date: str = "",
    end_date: str = "",
    sort: str = "best",
    page: int = 0,
    fq: str = "",
) -> dict:
    """
    Search NYT articles from 1851 to present.

    Args:
        query: Search terms (e.g., "SNAP benefits welfare")
        begin_date: YYYYMMDD format (e.g., "20200101")
        end_date: YYYYMMDD format
        sort: "relevance" | "newest" | "oldest"
        page: Pagination (0-indexed, 10 results per page)
        fq: Filter query (e.g., 'section_name:("U.S." "Politics")')

    Returns:
        dict with articles, hit count, and metadata
    """
    key = _get_key()
    if not key:
        return {"error": "NYT_API_KEY not configured", "articles": []}

    params = {"q": query, "api-key": key, "sort": sort, "page": str(page)}
    if begin_date:
        params["begin_date"] = begin_date
    if end_date:
        params["end_date"] = end_date
    if fq:
        params["fq"] = fq

    ck = _cache_key("search", params)
    cached = _cache_get(ck)
    if cached:
        return cached

    client = _get_client()
    try:
        resp = await client.get(f"{BASE_URL}/search/v2/articlesearch.json", params=params)
        resp.raise_for_status()
        data = resp.json()

        response = data.get("response", {})
        articles = []
        for doc in response.get("docs", []):
            articles.append(_normalize_article(doc))

        result = {
            "query": query,
            "total_hits": response.get("meta", {}).get("hits", 0),
            "page": page,
            "articles": articles,
        }
        _cache_set(ck, result)
        log.info(f"§NYT: Search '{query}' → {len(articles)} articles, {result['total_hits']} total")
        return result

    except httpx.HTTPStatusError as e:
        log.warning(f"§NYT: Search failed: {e.response.status_code}")
        return {"error": f"HTTP {e.response.status_code}", "articles": []}
    except Exception as e:
        log.warning(f"§NYT: Search error: {e}")
        return {"error": str(e), "articles": []}


def _normalize_article(doc: dict) -> dict:
    """Convert NYT article doc into clean E.D.I.T.H. format."""
    multimedia = doc.get("multimedia", [])
    thumbnail = ""
    if isinstance(multimedia, list):
        for m in multimedia:
            if isinstance(m, dict) and (m.get("subtype") == "thumbnail" or m.get("type") == "image"):
                thumbnail = f"https://www.nytimes.com/{m.get('url', '')}"
                break

    # headline can be a dict {"main": "..."} or a plain string
    headline = doc.get("headline", {})
    title = headline.get("main", "") if isinstance(headline, dict) else str(headline)

    # byline can be a dict {"original": "By ..."} or a plain string
    byline = doc.get("byline", {})
    byline_str = byline.get("original", "") if isinstance(byline, dict) else str(byline)

    # keywords can be list of dicts or list of strings
    raw_kw = doc.get("keywords", [])
    keywords = []
    if isinstance(raw_kw, list):
        for kw in raw_kw[:10]:
            keywords.append(kw.get("value", str(kw)) if isinstance(kw, dict) else str(kw))

    return {
        "title": title,
        "abstract": doc.get("abstract", ""),
        "snippet": doc.get("snippet", ""),
        "lead_paragraph": doc.get("lead_paragraph", ""),
        "url": doc.get("web_url", ""),
        "pub_date": doc.get("pub_date", ""),
        "section": doc.get("section_name", ""),
        "subsection": doc.get("subsection_name", ""),
        "byline": byline_str,
        "word_count": doc.get("word_count", 0),
        "type": doc.get("type_of_material", ""),
        "source": "New York Times",
        "thumbnail": thumbnail,
        "keywords": keywords,
    }


# ── Most Popular ────────────────────────────────────────────────────────────
async def most_popular(period: int = 7, metric: str = "viewed") -> dict:
    """
    Get most popular NYT articles.

    Args:
        period: 1, 7, or 30 days
        metric: "viewed" | "shared" | "emailed"
    """
    key = _get_key()
    if not key:
        return {"error": "NYT_API_KEY not configured", "articles": []}

    if period not in (1, 7, 30):
        period = 7

    ck = _cache_key("popular", {"period": period, "metric": metric})
    cached = _cache_get(ck)
    if cached:
        return cached

    client = _get_client()
    try:
        resp = await client.get(
            f"{BASE_URL}/mostpopular/v2/{metric}/{period}.json",
            params={"api-key": key},
        )
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("results", []):
            articles.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "url": item.get("url", ""),
                "pub_date": item.get("published_date", ""),
                "section": item.get("section", ""),
                "byline": item.get("byline", ""),
                "source": "New York Times",
                "type": item.get("type", ""),
            })

        result = {"metric": metric, "period": period, "articles": articles}
        _cache_set(ck, result)
        return result

    except Exception as e:
        return {"error": str(e), "articles": []}


# ── Top Stories ─────────────────────────────────────────────────────────────
async def top_stories(section: str = "home") -> dict:
    """
    Get current top stories for a section.

    Sections: home, arts, automobiles, books/review, business, fashion,
    food, health, insider, magazine, movies, nyregion, obituaries,
    opinion, politics, realestate, science, sports, sundayreview,
    technology, theater, t-magazine, travel, upshot, us, world
    """
    key = _get_key()
    if not key:
        return {"error": "NYT_API_KEY not configured", "articles": []}

    ck = _cache_key("top", {"section": section})
    cached = _cache_get(ck)
    if cached:
        return cached

    client = _get_client()
    try:
        resp = await client.get(
            f"{BASE_URL}/topstories/v2/{section}.json",
            params={"api-key": key},
        )
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("results", []):
            articles.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "url": item.get("url", ""),
                "pub_date": item.get("published_date", ""),
                "section": item.get("section", ""),
                "subsection": item.get("subsection", ""),
                "byline": item.get("byline", ""),
                "source": "New York Times",
            })

        result = {"section": section, "articles": articles}
        _cache_set(ck, result)
        return result

    except Exception as e:
        return {"error": str(e), "articles": []}


# ── Policy-Focused Helpers ──────────────────────────────────────────────────
async def search_policy(
    topic: str,
    years_back: int = 5,
) -> dict:
    """
    Search for policy-related NYT coverage on a topic.
    Pre-configured with policy-relevant section filters.
    Uses April 2025 API filter syntax (section.name, not section_name).

    Args:
        topic: Policy topic (e.g., "SNAP benefits", "immigration reform")
        years_back: How many years back to search (default 5)
    """
    from datetime import datetime, timedelta
    begin = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")

    # Use new April 2025 filter field names
    return await search_articles(
        query=topic,
        begin_date=begin,
        end_date=end,
        sort="relevance",
        fq='section.name:("U.S." "Politics" "Business" "Opinion" "Health")',
    )


# ── Status ──────────────────────────────────────────────────────────────────
def status() -> dict:
    """Check NYT bridge status."""
    key = _get_key()
    return {
        "configured": bool(key),
        "api_key_set": bool(key),
        "cache_dir": str(CACHE_DIR),
        "rate_limit": "500 requests/day (free tier)",
    }


# ── Aliases for consistent naming ───────────────────────────────────────────
get_top_stories = top_stories
get_most_popular = most_popular

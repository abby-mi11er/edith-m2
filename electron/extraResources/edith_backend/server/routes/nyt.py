"""
§NYT: New York Times API Routes
Extracted from main.py — 5 endpoints
"""
import logging
from fastapi import APIRouter

router = APIRouter(tags=["NYT"])
log = logging.getLogger("edith.nyt")


def register(app, ns=None):
    """Wire NYT routes — imports nyt_bridge at registration time."""
    try:
        from server.nyt_bridge import (
            search_articles as nyt_search, search_policy as nyt_policy,
            most_popular as nyt_popular, top_stories as nyt_top,
            status as nyt_status,
        )

        @router.get("/api/nyt/search")
        async def nyt_search_route(q: str = "", begin: str = "", end: str = "", page: int = 0):
            """Search NYT articles 1851–present."""
            if not q:
                return {"error": "query required", "articles": []}
            return await nyt_search(q, begin_date=begin, end_date=end, page=page)

        @router.get("/api/nyt/policy")
        async def nyt_policy_route(topic: str = "", years: int = 5):
            """Policy-focused NYT search with section filters."""
            if not topic:
                return {"error": "topic required", "articles": []}
            return await nyt_policy(topic, years_back=years)

        @router.get("/api/nyt/popular")
        async def nyt_popular_route(period: int = 7, metric: str = "viewed"):
            return await nyt_popular(period=period, metric=metric)

        @router.get("/api/nyt/top")
        async def nyt_top_route(section: str = "home"):
            return await nyt_top(section=section)

        @router.get("/api/nyt/status")
        async def nyt_status_route():
            return nyt_status()

        log.info("§NYT: Routes loaded — /api/nyt/search, /api/nyt/policy, /api/nyt/popular, /api/nyt/top")
    except Exception as _e:
        log.warning(f"§NYT: Could not load: {_e}")

    return router

"""
OpenAlex Router — Extracted from main.py (CH1)
================================================
Academic paper search, recommendations, and citations via OpenAlex API.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server import openalex

router = APIRouter(prefix="/api/openalex", tags=["Discovery"])


class OpenAlexRecommendRequest(BaseModel):
    text: str
    per_page: int = 15


@router.get("/search")
async def search(
    q: str = "", year: str = "", type: str = "",
    page: int = 1, per_page: int = 20,
):
    """Search OpenAlex for papers."""
    if not q.strip():
        return {"results": [], "total": 0, "page": 1, "per_page": per_page}
    result = await openalex.search_works(
        query=q.strip(),
        year=year or None,
        work_type=type or None,
        page=page,
        per_page=per_page,
    )

    # §BUS: Discovery results → EventBus (replaces old bridge)
    try:
        papers = result.get("results", result.get("works", []))
        if isinstance(papers, list) and len(papers) > 0:
            from server.event_bus import bus
            import asyncio
            asyncio.ensure_future(bus.emit("discovery.results", {
                "papers": papers, "query": q,
            }, source="openalex"))
    except Exception:
        pass

    return result


@router.post("/recommend")
async def recommend(req: OpenAlexRecommendRequest):
    """Get semantically similar papers from OpenAlex."""
    if not req.text.strip():
        return {"results": [], "total": 0}
    return await openalex.recommend_works(text=req.text.strip(), per_page=req.per_page)


@router.get("/work/{openalex_id:path}")
async def get_work(openalex_id: str):
    """Get a single paper's full metadata from OpenAlex."""
    work = await openalex.get_work(openalex_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.get("/citations/{openalex_id:path}")
async def citations(openalex_id: str, per_page: int = 20):
    """Get citation network for a paper."""
    return await openalex.get_citations(openalex_id, per_page=per_page)


@router.post("/resolve")
async def resolve(body: dict = {}):
    """Batch-resolve a list of OpenAlex IDs into full paper metadata."""
    ids = body.get("ids", [])
    if not ids or not isinstance(ids, list):
        return {"results": []}
    return await openalex.resolve_works(ids, per_page=min(len(ids), 50))

@router.post("/import")
async def import_work(body: dict = {}):
    """Import an OpenAlex paper into the local library."""
    openalex_id = body.get("openalex_id", "")
    if not openalex_id:
        raise HTTPException(status_code=400, detail="openalex_id is required")
    work = await openalex.get_work(openalex_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found on OpenAlex")
    # Return the work data for the frontend to handle local storage
    return {
        "status": "imported",
        "work": work,
        "message": f"Imported: {work.get('title', 'Unknown')}",
    }

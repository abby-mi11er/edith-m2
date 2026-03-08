"""
Research routes for E.D.I.T.H. — extracted from main.py
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
router = APIRouter(tags=["Research"])

def _get_main():
    import server.main as m
    return m


# ---------------------------------------------------------------------------
# Literature Review Generator
# ---------------------------------------------------------------------------
class LitReviewRequest(BaseModel):
    topic: str
    model: str = "gemini-2.5-flash"
    max_sources: int = 20


# ---------------------------------------------------------------------------
# Research Design Assistant
# ---------------------------------------------------------------------------
class ResearchDesignRequest(BaseModel):
    question: str = ""
    topic: str = ""
    context: str = ""
    model: str = "gemini-2.5-flash"

    def get_question(self) -> str:
        return self.question or self.topic or ""


# ---------------------------------------------------------------------------
# Batch Questions — F5
# ---------------------------------------------------------------------------
class BatchQuestionsRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=10)
    model: str = "gemini-2.5-flash"
    mode: str = Field("grounded", pattern=r"^(grounded|general)$")


def literature_review_endpoint(req: LitReviewRequest):
    """Generate a structured literature review on a given topic."""
    from server.prompts import LIT_REVIEW_PROMPT
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured")
    
    # Multi-pass agentic retrieval for comprehensive coverage
    agentic_result = {}
    try:
        agentic_result = agentic_retrieve(
            query=req.topic,
            chroma_dir=CHROMA_DIR,
            collection_name=CHROMA_COLLECTION,
            embed_model=EMBED_MODEL,
            top_k=req.max_sources,
            max_attempts=3,
            bm25_weight=0.35,
            diversity_lambda=0.65,
            rerank_model=os.environ.get("EDITH_CHROMA_RERANK_MODEL", ""),
            rerank_top_n=int(os.environ.get("EDITH_CHROMA_RERANK_TOP_N", "18")),
        )
        sources = agentic_result.get("sources", [])
    except Exception:
        sources = []
    
    if not sources:
        return {"error": "No sources found for this topic", "sources": []}
    
    # Build source blocks (already imported at top level via backend_logic)
    blocks = build_support_audit_source_blocks(sources, max_snippet_chars=2000)
    
    prompt = (
        f"{LIT_REVIEW_PROMPT}\n\n"
        f"TOPIC: {req.topic}\n\n"
        f"SOURCES ({len(sources)} documents):\n{blocks}"
    )
    
    model_chain = [req.model, DEFAULT_MODEL]
    answer, used_model = generate_text_via_chain(
        prompt, model_chain, temperature=0.2,
    )
    
    return {
        "review": answer,
        "used_model": used_model,
        "source_count": len(sources),
        "sources": sources,
        "retrieval_attempts": agentic_result.get("attempts", 1),
    }


def research_design_endpoint(req: ResearchDesignRequest):
    """Provide methodology guidance for a research question."""
    from server.prompts import RESEARCH_DESIGN_PROMPT
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured")
    
    # Retrieve methodology-related sources
    method_query = f"{req.get_question()} methodology research design identification strategy"
    try:
        agentic_result = agentic_retrieve(
            query=method_query,
            chroma_dir=CHROMA_DIR,
            collection_name=CHROMA_COLLECTION,
            embed_model=EMBED_MODEL,
            top_k=12,
            max_attempts=2,
            bm25_weight=0.35,
            diversity_lambda=0.65,
        )
        sources = agentic_result["sources"]
    except Exception:
        sources = []
    
    from server.model_utils import build_support_audit_source_blocks
    blocks = build_support_audit_source_blocks(sources, max_snippet_chars=1500) if sources else ""
    
    prompt = (
        f"{RESEARCH_DESIGN_PROMPT}\n\n"
        f"RESEARCH QUESTION: {req.get_question()}\n"
    )
    if req.context:
        prompt += f"\nADDITIONAL CONTEXT: {req.context}\n"
    if blocks:
        prompt += f"\nMETHODOLOGICAL SOURCES:\n{blocks}"
    
    model_chain = [req.model, DEFAULT_MODEL]
    answer, used_model = generate_text_via_chain(
        prompt, model_chain, temperature=0.2,
    )
    
    return {
        "design_guidance": answer,
        "used_model": used_model,
        "sources": sources,
    }


def batch_questions_endpoint(req: BatchQuestionsRequest):
    """Submit multiple questions and get answers for each.
    
    Useful for overnight research — submit up to 10 questions,
    get grounded answers for all of them in one call.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured")
    
    results = []
    for i, question in enumerate(req.questions):
        try:
            # Retrieve sources for this question
            try:
                agentic_result = agentic_retrieve(
                    query=question,
                    chroma_dir=CHROMA_DIR,
                    collection_name=CHROMA_COLLECTION,
                    embed_model=EMBED_MODEL,
                    top_k=8,
                    max_attempts=2,
                    bm25_weight=0.35,
                    diversity_lambda=0.5,
                    rerank_model=os.environ.get("EDITH_CHROMA_RERANK_MODEL", ""),
                    rerank_top_n=int(os.environ.get("EDITH_CHROMA_RERANK_TOP_N", "12")),
                )
                sources = agentic_result["sources"]
            except Exception:
                sources = []
            
            # Build prompt
            if req.mode == "grounded" and sources:
                from server.prompts import GROUNDED_PROMPT
                blocks = build_support_audit_source_blocks(sources, max_snippet_chars=1500)
                prompt = f"{GROUNDED_PROMPT}\n\nQUESTION: {question}\n\nSOURCES:\n{blocks}"
            else:
                from server.prompts import GENERAL_PROMPT
                prompt = f"{GENERAL_PROMPT}\n\nQUESTION: {question}"
            
            model_chain = build_model_chain(req.model)
            answer, used_model = generate_text_via_chain(
                prompt, model_chain, temperature=0.15,
            )
            
            results.append({
                "question": question,
                "answer": answer,
                "used_model": used_model,
                "source_count": len(sources),
                "sources": sources[:3],  # brief source list
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "question": question,
                "answer": "",
                "status": "error",
                "error": str(exc),
            })
    
    ok_count = sum(1 for r in results if r["status"] == "ok")
    return {
        "results": results,
        "total": len(req.questions),
        "succeeded": ok_count,
        "failed": len(req.questions) - ok_count,
    }


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register research routes."""
    if ns:
        import sys
        this_module = sys.modules[__name__]
        for key, val in ns.items():
            if not key.startswith("__") and not hasattr(this_module, key):
                setattr(this_module, key, val)

    router.post("/api/research/lit-review", tags=["Research"])(literature_review_endpoint)
    router.post("/api/research/design", tags=["Research"])(research_design_endpoint)
    router.post("/api/research/batch", tags=["Research"])(batch_questions_endpoint)
    # Alias routes at paths the UI expects
    router.post("/api/literature-review", tags=["Research"])(literature_review_endpoint)
    router.post("/api/research-design", tags=["Research"])(research_design_endpoint)
    return router

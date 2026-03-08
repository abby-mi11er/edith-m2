"""
Export routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Body, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("edith")
router = APIRouter(tags=["Export"])

def _get_main():
    import server.main as m
    return m


# ---------------------------------------------------------------------------
# Export Endpoints — LaTeX, BibTeX, Slides
# ---------------------------------------------------------------------------
class ExportRequest(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    format: str = Field("latex", pattern=r"^(latex|bibtex|slides)$")
    title: str = "Research Output"


def export_endpoint(req: ExportRequest):
    """Export an answer to LaTeX, BibTeX, or Beamer slides."""
    try:
        from server.export_academic import (
            answer_to_latex, sources_to_bibtex, answer_to_slides,
        )
        sources = req.sources or []
        if req.format == "latex":
            tex_content = answer_to_latex(req.answer, sources, title=req.title)
            bib_content = sources_to_bibtex(sources)
            return {"format": "latex", "content": tex_content, "bib": bib_content}
        elif req.format == "bibtex":
            bib = sources_to_bibtex(sources)
            return {"format": "bibtex", "content": bib}
        elif req.format == "slides":
            slides = answer_to_slides(req.answer, title=req.title)
            return {"format": "slides", "content": slides}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def export_latex(body: dict = Body(default={})):
    """Export an answer as LaTeX source."""
    content = body.get("content", "")
    title = body.get("title", "E.D.I.T.H. Export")
    sources = body.get("sources", [])
    if not content:
        return {"error": "content is required"}
    
    bibtex = []
    for i, s in enumerate(sources):
        key = f"source{i+1}"
        bibtex.append(f"@article{{{key},\n  author = {{{s.get('author','Unknown')}}},\n  title = {{{s.get('title','Untitled')}}},\n  year = {{{s.get('year','n.d.')}}}\n}}")
    
    safe = content.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    
    latex = f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{natbib}}
\\usepackage{{amsmath}}
\\title{{{title}}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
{safe}
\\bibliographystyle{{apalike}}
\\end{{document}}
"""
    result = {"latex": latex, "bibtex": bibtex, "format": "latex"}

    # §BUS: Export → EventBus (replaces old bridge)
    try:
        from server.event_bus import bus
        import asyncio
        asyncio.ensure_future(bus.emit("export.started", {
            "content": content[:5000], "format": "latex", "title": title,
        }, source="export"))
    except Exception:
        pass

    # §SNIPER: Adversarial Defender — auto-audit your own draft before publishing
    if body.get("push_to_overleaf") or body.get("self_audit"):
        try:
            from server.sniper_audit import full_sniper_audit
            defender = full_sniper_audit(paper_text=content[:15000], n_simulations=200)
            summary = defender.get("tactical_summary", {})
            result["self_audit"] = {
                "overall": summary.get("overall", "unknown"),
                "fails": summary.get("fails", 0),
                "warnings": summary.get("warnings", 0),
                "verdicts": summary.get("verdicts", []),
            }
            if summary.get("fails", 0) > 0:
                result["self_audit"]["note"] = (
                    "⚠ The Adversarial Defender found weaknesses in your draft. "
                    "Review the verdicts before submitting."
                )
        except Exception as _def_e:
            result["self_audit"] = {"error": str(_def_e), "note": "Defender audit skipped"}

    # §MCL-2026: Optional push to Overleaf
    if body.get("push_to_overleaf"):
        try:
            from server.overleaf_bridge import OverleafBridge
            ol = OverleafBridge()
            push = ol.push_draft(
                latex,
                filename=body.get("overleaf_filename", "edith_draft.tex"),
                commit_message=f"E.D.I.T.H. export: {title[:50]}",
            )
            result["overleaf"] = push
        except Exception as _e:
            result["overleaf"] = {"error": str(_e)}

    return result


async def export_ris(body: dict = Body(default={})):
    """Export sources as RIS format."""
    try:
        from server.connectors_full import format_ris_export
        sources = body.get("sources", [])
        text = format_ris_export(sources)
        from starlette.responses import Response
        return Response(content=text, media_type="application/x-research-info-systems",
                        headers={"Content-Disposition": "attachment; filename=references.ris"})
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ── §4.0: RefWorks / RIS Export ──
async def export_refworks(body: dict = Body(default={})):
    """Export sources as RefWorks tagged format."""
    try:
        from server.connectors_full import format_refworks_export
        sources = body.get("sources", [])
        text = format_refworks_export(sources)
        from starlette.responses import Response
        return Response(content=text, media_type="text/plain",
                        headers={"Content-Disposition": "attachment; filename=references.txt"})
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ── §4.0: Word/LaTeX Export ──
async def export_word(body: dict = Body(default={})):
    """Export an answer as Word-compatible HTML (can be pasted into Word)."""
    content = body.get("content", "")
    title = body.get("title", "E.D.I.T.H. Export")
    sources = body.get("sources", [])
    if not content:
        return {"error": "content is required"}
    
    refs = ""
    if sources:
        refs = "\n<h2>References</h2>\n<ol>\n"
        for s in sources:
            author = s.get("author", "Unknown")
            year = s.get("year", "n.d.")
            stitle = s.get("title", "Untitled")
            refs += f"<li>{author} ({year}). <em>{stitle}</em>.</li>\n"
        refs += "</ol>"
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body {{ font-family: 'Times New Roman', serif; font-size: 12pt; margin: 1in; }}
h1 {{ font-size: 14pt; }} h2 {{ font-size: 13pt; }} p {{ line-height: 2; }}</style>
</head><body>
<h1>{title}</h1>
{content}
{refs}
</body></html>"""
    return {"html": html, "format": "word_html", "title": title}


async def export_notion_endpoint(body: dict = Body(default={})):
    """Export a note or answer to Notion."""
    try:
        from server.export_notes import export_to_notion
    except Exception:
        return {"error": "export_notes not available — requires NOTION_API_KEY"}
    title = body.get("title", "E.D.I.T.H. Export")
    content = body.get("content", "")
    tags = body.get("tags", [])
    try:
        result = export_to_notion(title, content, tags)
        return result
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def training_export():
    """§SHARPEN: Export accumulated training data as JSONL for fine-tuning."""
    try:
        from pipelines.feedback_trainer import FeedbackTrainer
        ft = FeedbackTrainer()
        export_path = ft.export_for_finetuning()
        stats = ft.stats()
        
        # Also check fine-tune readiness
        from server.training_tools import check_finetune_trigger
        trigger = check_finetune_trigger(ft.training_path)
        
        return {
            "status": "exported",
            "path": str(export_path),
            "stats": stats,
            "finetune_ready": trigger.get("trigger", False),
            "new_pairs": trigger.get("new_pairs", 0),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register routes."""
    router.post("/api/export", tags=["Export"])(export_endpoint)
    router.post("/api/export/latex", tags=["Export"])(export_latex)
    router.post("/api/export/word", tags=["Export"])(export_word)
    router.post("/api/export/notion", tags=["Export"])(export_notion_endpoint)
    router.post("/api/export/ris", tags=["Export"])(export_ris)
    router.post("/api/export/refworks", tags=["Export"])(export_refworks)
    router.get("/api/training/export", tags=["Export"])(training_export)
    return router

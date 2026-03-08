"""
Compatibility bridge routes.

These endpoints keep older frontend calls functional while route extraction is
in progress. Handlers prefer existing backend primitives and degrade safely.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["Compat"])
log = logging.getLogger("edith.routes.compat")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def _error(msg: str) -> dict:
    return {"status": "error", "error": msg}


@router.get("/api/agent/suggestions")
async def agent_suggestions(q: str = ""):
    """Compatibility endpoint for proactive suggestion cards in ChatPanel."""
    try:
        from server.proactive_suggestions import suggest_fast

        raw = suggest_fast(q or "research workflow")
        suggestions = [
            s.get("text", "") for s in raw if isinstance(s, dict) and s.get("text")
        ]
    except Exception:
        suggestions = []

    if not suggestions:
        suggestions = [
            "Run a quick literature gap scan on your current topic.",
            "Generate a structured outline before drafting.",
            "Check robustness assumptions before finalizing results.",
        ]
    return {"suggestions": suggestions[:5]}


def _coerce_sources(body: dict) -> list[dict]:
    src = body.get("sources")
    if isinstance(src, list) and src:
        return src

    titles = body.get("titles") or body.get("notes")
    if isinstance(titles, list) and titles:
        return [{"title": str(t)} for t in titles if str(t).strip()]

    text = body.get("text")
    if isinstance(text, str) and text.strip():
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return [{"title": ln} for ln in lines[:200]]

    return []


@router.post("/api/bibliography")
async def bibliography(body: dict | None = None):
    """Compatibility endpoint used by several panels for bibliography export."""
    payload = body or {}
    sources = _coerce_sources(payload)
    style = payload.get("style") or payload.get("format") or "apa"

    try:
        from server.citation_formatter import generate_bibliography

        bib = generate_bibliography(sources, style=style)
        return {
            "status": "ok",
            "style": style,
            "count": len(sources),
            "bibliography": bib,
            "bibtex": bib,
        }
    except Exception as exc:
        return _error(f"bibliography_failed: {exc}")


@router.post("/api/infra/cache/invalidate")
async def infra_cache_invalidate(request: Request):
    body = await _json_body(request)
    pattern = str(body.get("pattern", "")).strip()
    try:
        from server.infrastructure import response_cache

        if pattern and pattern != "*":
            response_cache.invalidate(query=pattern)
        else:
            response_cache.invalidate()
        return {"status": "ok", "pattern": pattern or "*", "stats": response_cache.stats}
    except Exception as exc:
        return _error(f"cache_invalidate_failed: {exc}")


@router.post("/api/infra/parallel-search")
async def infra_parallel_search(request: Request):
    body = await _json_body(request)
    query = str(body.get("query", "")).strip()
    if not query:
        return {"status": "ok", "results": [], "total_found": 0, "note": "query is empty"}

    try:
        from server.infrastructure import parallel_retrieve

        return parallel_retrieve(
            query=query,
            collections=body.get("collections"),
            chroma_dir=body.get("chroma_dir", ""),
            embed_model=body.get("embed_model", ""),
            top_k=_as_int(body.get("top_k", 20), 20),
            max_workers=_as_int(body.get("max_workers", 4), 4),
        )
    except Exception as exc:
        return _error(f"parallel_search_failed: {exc}")


@router.post("/api/infra/query-plan")
async def infra_query_plan(request: Request):
    body = await _json_body(request)
    query = str(body.get("query", "")).strip()
    if not query:
        return _error("query is required")

    try:
        from server.infrastructure import optimize_query_plan

        return optimize_query_plan(
            query=query,
            available_agents=_as_int(body.get("available_agents", 1), 1),
        )
    except Exception as exc:
        return _error(f"query_plan_failed: {exc}")


@router.post("/api/infra/reindex/scan")
async def infra_reindex_scan(request: Request):
    body = await _json_body(request)
    try:
        from server.infrastructure import IncrementalIndexer

        indexer = IncrementalIndexer(data_root=os.environ.get("EDITH_DATA_ROOT", ""))
        return indexer.scan_for_changes(
            directory=body.get("directory", ""),
            extensions=body.get("extensions"),
        )
    except Exception as exc:
        return _error(f"reindex_scan_failed: {exc}")


@router.get("/api/security/dashboard")
async def security_dashboard():
    try:
        from server.security import build_security_dashboard

        return build_security_dashboard()
    except Exception as exc:
        return _error(f"security_dashboard_failed: {exc}")


@router.post("/api/security/logs/save")
async def security_logs_save(request: Request):
    body = await _json_body(request)
    session_id = str(body.get("session_id") or f"session-{int(time.time())}")
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = []

    try:
        from server.security import EncryptedChatLog

        logger = EncryptedChatLog()
        return logger.save_session(session_id=session_id, messages=messages)
    except Exception as exc:
        return _error(f"security_logs_save_failed: {exc}")


@router.post("/api/security/wipe")
async def security_wipe():
    try:
        from server.security import secure_wipe_ram

        return secure_wipe_ram()
    except Exception as exc:
        return _error(f"security_wipe_failed: {exc}")


@router.get("/api/session/snapshots")
async def session_snapshots():
    try:
        from server.routes.flywheel_advanced import list_session_snapshots

        return await list_session_snapshots()
    except Exception as exc:
        return _error(f"session_snapshots_failed: {exc}")


@router.post("/api/session/load")
async def session_load(request: Request):
    body = await _json_body(request)
    requested = str(body.get("snapshot_id") or body.get("id") or body.get("name") or "").strip()
    if not requested:
        return _error("snapshot_id or name is required")

    try:
        from server.routes import flywheel_advanced as _snap

        # Prefer direct ID.
        for item in getattr(_snap, "_snapshots", []):
            if item.get("id") == requested or item.get("name") == requested:
                return await _snap.load_session_snapshot(item.get("id", requested))

        # Fallback to existing handler path behavior.
        return await _snap.load_session_snapshot(requested)
    except Exception as exc:
        return _error(f"session_load_failed: {exc}")


@router.get("/api/tools/route-map")
async def tools_route_map():
    try:
        from server.completions import get_route_map

        return get_route_map()
    except Exception as exc:
        return _error(f"route_map_failed: {exc}")


@router.get("/api/tools/models")
@router.post("/api/tools/models")
async def tools_models():
    try:
        from server.completions import list_quantized_models

        return {"models": list_quantized_models()}
    except Exception as exc:
        return _error(f"models_failed: {exc}")


@router.get("/api/tools/npu/tune")
@router.post("/api/tools/npu/tune")
async def tools_npu_tune():
    try:
        from server.completions import tune_npu_batch_size

        return tune_npu_batch_size()
    except Exception as exc:
        return _error(f"npu_tune_failed: {exc}")


@router.post("/api/tools/quantize")
async def tools_quantize(request: Request):
    body = await _json_body(request)
    model_name = str(body.get("model_name") or body.get("model") or "").strip()

    try:
        from server.completions import quantize_model, list_quantized_models

        # Frontend quick-action may call this endpoint without a model name.
        # In that case, fall back to the first discovered local model.
        if not model_name:
            discovered = list_quantized_models() or []
            if discovered:
                model_name = str(discovered[0])
            else:
                return {"status": "error", "error": "model_name is required", "available_models": []}

        return quantize_model(
            model_name=model_name,
            output_dir=str(body.get("output_dir", "")),
            bits=_as_int(body.get("bits", 4), 4),
        )
    except Exception as exc:
        return _error(f"quantize_failed: {exc}")


@router.post("/api/tools/speculative")
async def tools_speculative(request: Request):
    body = await _json_body(request)
    prompt = str(body.get("prompt") or body.get("query") or body.get("text") or "").strip()
    if not prompt:
        return _error("prompt/query is required")

    try:
        from server.completions import speculative_generate

        return speculative_generate(
            prompt=prompt,
            draft_model=str(body.get("draft_model", "gemini-2.5-flash")),
            verify_model=str(body.get("verify_model", "")),
        )
    except Exception as exc:
        return _error(f"speculative_failed: {exc}")


@router.post("/api/tools/scrape-citations")
async def tools_scrape_citations(request: Request):
    body = await _json_body(request)
    query = str(body.get("query") or body.get("topic") or body.get("url") or "").strip()
    if not query:
        return _error("query/topic/url is required")

    try:
        from server.completions import scrape_citations_from_openalex

        return scrape_citations_from_openalex(
            query=query,
            max_results=_as_int(body.get("max_results", 20), 20),
        )
    except Exception as exc:
        return _error(f"scrape_citations_failed: {exc}")


@router.post("/api/tools/lit-map")
async def tools_lit_map(request: Request):
    body = await _json_body(request)
    sources = body.get("sources")
    if not isinstance(sources, list):
        topic = str(body.get("topic") or body.get("query") or "").strip()
        sources = [{"metadata": {"title": topic or "Untitled"}, "text": topic}] if topic else []

    try:
        from server.completions import build_literature_map

        return build_literature_map(sources=sources)
    except Exception as exc:
        return _error(f"lit_map_failed: {exc}")


@router.post("/api/tools/ocr")
async def tools_ocr(request: Request):
    body = await _json_body(request)
    image_path = str(body.get("image_path") or body.get("path") or "").strip()

    if not image_path:
        # Compatibility: Tools panel sometimes passes raw text instead of a path.
        fallback_text = str(body.get("text") or "").strip()
        if fallback_text:
            return {"status": "ok", "method": "passthrough", "text": fallback_text}
        return _error("image_path/path is required")

    try:
        import asyncio
        from server.completions import extract_text_from_image

        return await asyncio.to_thread(
            extract_text_from_image,
            image_path=image_path,
            method=str(body.get("method") or "auto"),
        )
    except Exception as exc:
        return _error(f"ocr_failed: {exc}")


@router.get("/api/tools/study/status")
async def tools_study_status():
    try:
        from server.completions import study_session

        return study_session.get_status()
    except Exception as exc:
        return _error(f"study_status_failed: {exc}")


@router.post("/api/tools/study/start")
async def tools_study_start(request: Request):
    body = await _json_body(request)
    try:
        from server.completions import study_session

        return study_session.start_session(
            topic=str(body.get("topic") or "general"),
            session_type=str(body.get("type") or body.get("session_type") or "focus"),
            duration_min=_as_int(body.get("duration_min", 50), 50),
            break_min=_as_int(body.get("break_min", 10), 10),
        )
    except Exception as exc:
        return _error(f"study_start_failed: {exc}")


@router.post("/api/tools/study/end")
async def tools_study_end():
    try:
        from server.completions import study_session

        return study_session.end_session()
    except Exception as exc:
        return _error(f"study_end_failed: {exc}")


@router.get("/api/rhythm/efficiency")
@router.post("/api/rhythm/efficiency")
async def rhythm_efficiency():
    try:
        from server.operational_rhythm import operational_rhythm

        status = operational_rhythm.status
        tasks = status.get("tasks", {})
        total = len(tasks)
        overdue = sum(1 for task in tasks.values() if task.get("overdue"))
        return {
            "status": "ok",
            "efficiency": round((total - overdue) / max(total, 1), 3),
            "tasks_total": total,
            "tasks_overdue": overdue,
            "due_now": status.get("due_now", []),
        }
    except Exception as exc:
        return _error(f"rhythm_efficiency_failed: {exc}")


@router.get("/api/rhythm/hardware")
async def rhythm_hardware():
    try:
        from server.hw_monitor import get_full_hardware_status

        return {"status": "ok", "hardware": get_full_hardware_status()}
    except Exception as exc:
        return _error(f"rhythm_hardware_failed: {exc}")


@router.post("/api/rhythm/theory-map")
async def rhythm_theory_map():
    try:
        from server.completions import get_route_map

        route_map = get_route_map()
        return {
            "status": "ok",
            "modules": list((route_map.get("modules") or {}).keys()),
            "map": route_map,
        }
    except Exception as exc:
        return _error(f"rhythm_theory_map_failed: {exc}")


@router.post("/api/rhythm/intel-report")
async def rhythm_intel_report():
    try:
        from server.operational_rhythm import operational_rhythm

        return {
            "status": "ok",
            "generated_at": time.time(),
            "rhythm": operational_rhythm.status,
        }
    except Exception as exc:
        return _error(f"rhythm_intel_report_failed: {exc}")


@router.post("/api/rhythm/save-state")
async def rhythm_save_state(request: Request):
    body = await _json_body(request)
    try:
        from server.citadel_boot import StateWelder

        welder = StateWelder()
        res = welder.save_weld()
        if isinstance(res, dict):
            res.setdefault("session_id", body.get("session_id", ""))
        return res
    except Exception as exc:
        return _error(f"rhythm_save_state_failed: {exc}")


@router.post("/api/rhythm/restore-state")
async def rhythm_restore_state(request: Request):
    _ = await _json_body(request)
    try:
        from server.citadel_boot import StateWelder

        welder = StateWelder()
        return welder.restore_weld()
    except Exception as exc:
        return _error(f"rhythm_restore_state_failed: {exc}")


@router.post("/api/rhythm/quarterly-merge")
async def rhythm_quarterly_merge():
    try:
        from server.auto_maintenance import MaintenanceEngine

        engine = MaintenanceEngine()
        result = engine.run_full_cycle(background=True)
        return {"status": "ok", "operation": "quarterly_merge", "result": result}
    except Exception as exc:
        return _error(f"rhythm_quarterly_merge_failed: {exc}")


@router.post("/api/rhythm/night-cycle")
async def rhythm_night_cycle():
    try:
        from server.auto_maintenance import MaintenanceEngine

        engine = MaintenanceEngine()
        result = engine.run_full_cycle(background=True)
        return {"status": "ok", "operation": "night_cycle", "result": result}
    except Exception as exc:
        return _error(f"rhythm_night_cycle_failed: {exc}")


@router.post("/api/training/trigger")
async def training_trigger():
    """Compatibility trigger used by Runs/CommandPalette."""
    root = Path(os.environ.get("EDITH_DATA_ROOT", "."))
    train_file = root / "training_data" / "edith_feedback_train.jsonl"
    pairs = 0
    if train_file.is_file():
        try:
            with train_file.open() as f:
                pairs = sum(1 for _ in f)
        except Exception:
            pairs = 0

    return {
        "status": "queued",
        "pairs": pairs,
        "note": "Compatibility endpoint; use /api/training/lora for detailed control.",
    }


@router.post("/api/training/dpo-prepare")
async def training_dpo_prepare_alias():
    try:
        from server.routes.training import training_dpo_prepare

        return await training_dpo_prepare()
    except Exception as exc:
        return _error(f"training_dpo_prepare_failed: {exc}")

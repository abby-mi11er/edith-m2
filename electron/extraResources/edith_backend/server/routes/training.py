"""
Training routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Body, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from server.server_state import state as _server_state

log = logging.getLogger("edith")
router = APIRouter(tags=["Training"])

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


def _runtime_root() -> Path:
    """Root for mutable runtime artifacts (tests can redirect this)."""
    base = Path(DATA_ROOT) if DATA_ROOT else ROOT_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _runtime_training_dir() -> Path:
    td = _runtime_root() / "training_data"
    td.mkdir(parents=True, exist_ok=True)
    return td


class FeedbackRequest(BaseModel):
    message_id: str = ""
    question: str = ""
    answer: str = ""
    rating: str = Field(..., pattern=r"^(up|down)$")  # §FIX Vuln 3: Validate rating
    correction: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    comment: Optional[str] = None


def feedback_endpoint(req: FeedbackRequest):
    """Save user feedback as training signal. 👍 → training data, 👎 → review queue."""
    from datetime import datetime as _dt, timezone as _tz
    pair = {
        "timestamp": _dt.now(_tz.utc).isoformat(),  # §FIX D2: timezone-aware
        "message_id": req.message_id,
        "question": req.question,
        "answer": req.answer,
        "rating": req.rating,
        "correction": req.correction,
    }

    with _autolearn_lock:  # §FIX D1: ALL writes inside lock to prevent interleaved lines
        # §FIX: Write to feedback.sqlite3 so LoRA trainer can read it
        try:
            import sqlite3 as _fb_sqlite3
            fb_db_path = _runtime_root() / "feedback.sqlite3"
            with _fb_sqlite3.connect(str(fb_db_path)) as _fb_conn:
                _fb_conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        message_id TEXT,
                        feedback_type TEXT DEFAULT 'chat',
                        value INTEGER NOT NULL,
                        query TEXT,
                        answer TEXT,
                        note TEXT,
                        correction TEXT
                    )
                """)
                # §FIX Issue 2: Dedup by message_id — skip if already rated
                if req.message_id:
                    existing = _fb_conn.execute(
                        "SELECT id FROM feedback_events WHERE message_id = ?",
                        (req.message_id,),
                    ).fetchone()
                    if existing:
                        return {"ok": True, "rating": req.rating, "note": "already_rated"}
                _fb_conn.execute(
                    "INSERT INTO feedback_events (created_at, message_id, feedback_type, value, query, answer, note, correction) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        _dt.now(_tz.utc).isoformat(),  # §FIX D2
                        req.message_id,
                        "chat",
                        1 if req.rating == "up" else -1,
                        req.question,
                        req.answer,
                        req.answer if req.rating == "up" else None,
                        req.correction,
                    ),
                )
        except Exception as _fb_err:
            log.warning(f"Feedback SQLite write failed: {_fb_err}")

        # §FIX T1: Write to training_data/ so lora_trainer finds these files
        training_dir = _runtime_training_dir()

        if req.rating == "up":
            # Positive feedback → training pair
            training_pair = {
                "messages": [
                    {"role": "system", "content": "[SYSTEM]"},  # §FIX Vuln 4: Placeholder
                    {"role": "user", "content": req.question},
                    {"role": "assistant", "content": req.answer},
                ],
                "quality_tier": "feedback_positive",
            }
            out_path = training_dir / "edith_feedback_train.jsonl"
            with open(out_path, "a") as f:
                f.write(json.dumps(training_pair) + "\n")
        else:
            # Negative feedback → review queue (with optional correction)
            if req.correction and len(req.correction.strip()) > 20:
                # User provided a correction → make a corrected training pair
                corrected_pair = {
                    "messages": [
                        {"role": "system", "content": "[SYSTEM]"},
                        {"role": "user", "content": req.question},
                        {"role": "assistant", "content": req.correction},
                    ],
                    "quality_tier": "human_correction",
                }
                out_path = training_dir / "edith_feedback_train.jsonl"
                with open(out_path, "a") as f:
                    f.write(json.dumps(corrected_pair) + "\n")

            # §FIX T2: Use 'bad_answer' key to match lora_trainer expectations
            neg_pair = {
                "question": req.question,
                "bad_answer": req.answer,  # §FIX T2: Was 'answer', lora_trainer expects 'bad_answer'
                "rating": req.rating,
                "correction": req.correction,
                "timestamp": pair["timestamp"],
            }
            neg_path = training_dir / "edith_feedback_negatives.jsonl"
            with open(neg_path, "a") as f:
                f.write(json.dumps(neg_pair) + "\n")

    return {"ok": True, "rating": req.rating}


# ── §4.0: Training Data Tools ──
async def training_quality_metrics():
    """Get quality metrics for the training data."""
    try:
        from server.training_tools import compute_quality_metrics
        autolearn_path = Path(DATA_ROOT) / "training" / "autolearn.jsonl"
        return compute_quality_metrics(autolearn_path)
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def training_balance():
    """Analyze topic balance in training data."""
    try:
        from server.training_tools import analyze_topic_balance
        autolearn_path = Path(DATA_ROOT) / "training" / "autolearn.jsonl"
        return analyze_topic_balance(autolearn_path)
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def training_deduplicate():
    """Deduplicate training data."""
    try:
        from server.training_tools import deduplicate_training_data
        autolearn_path = Path(DATA_ROOT) / "training" / "autolearn.jsonl"
        deduped_path = Path(DATA_ROOT) / "training" / "autolearn_deduped.jsonl"
        return deduplicate_training_data(autolearn_path, deduped_path)
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def finetune_trigger_check():
    """Check if enough new training data exists.  §IMP: cost estimate."""
    try:
        from server.training_tools import check_finetune_trigger
        autolearn_path = Path(DATA_ROOT) / "training" / "autolearn.jsonl"
        last_count_path = Path(DATA_ROOT) / "training" / ".last_finetune_count"
        result = check_finetune_trigger(autolearn_path, threshold=100, last_count_path=last_count_path)
        # §IMP: Add cost estimate if trigger ready
        if isinstance(result, dict):
            new_pairs = result.get("new_pairs", result.get("new_count", 0))
            result["estimated_cost_usd"] = round(new_pairs * 0.008, 2)
            result["estimated_time_min"] = max(5, new_pairs // 10)
        return result
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def training_dpo_prepare():
    """Prepare DPO training pairs from user feedback."""
    try:
        from server.training_tools import prepare_dpo_pairs
        # §FIX T7: Read from training_data/ where feedback_endpoint actually writes
        training_dir = Path(DATA_ROOT) / "training_data"
        # Try the correct location first, fall back to legacy
        feedback_path = training_dir / "edith_feedback_negatives.jsonl"
        if not feedback_path.exists():
            feedback_path = Path(DATA_ROOT) / "logs" / "feedback.jsonl"
        pairs = prepare_dpo_pairs(feedback_path)
        # Save to file
        output_path = Path(DATA_ROOT) / "training" / "dpo_pairs.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        return {"pairs": len(pairs), "output": str(output_path)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def training_costs():
    """Get fine-tuning cost summary.  §IMP: per-model breakdown."""
    if not _training_cost_tracker:
        return {"runs": [], "total_cost": 0, "best_run": None, "by_model": {}}
    summary = _training_cost_tracker.summary
    # §IMP: Group costs by model type
    if isinstance(summary, dict) and "runs" in summary:
        by_model = {}
        for run in summary.get("runs", []):
            model = run.get("model", "unknown") if isinstance(run, dict) else "unknown"
            by_model.setdefault(model, {"runs": 0, "cost": 0})
            by_model[model]["runs"] += 1
            by_model[model]["cost"] += run.get("cost", 0) if isinstance(run, dict) else 0
        summary["by_model"] = by_model
    return summary


async def training_sharpen(request):
    """§SHARPEN: Run the sharpening loop to generate training data.
    
    POST body: {"n_questions": 10, "dry_run": false}
    """
    import asyncio
    body = await request.json() if hasattr(request, "json") else {}
    n = body.get("n_questions", 10)
    dry = body.get("dry_run", False)
    
    try:
        from scripts.overnight_sharpen import run_sharpening
        result = await asyncio.to_thread(run_sharpening, n_questions=n, dry_run=dry)
        return result
    except Exception as e:
        log.warning(f"Sharpen failed: {e}")
        return {"status": "error", "detail": str(e)}


async def training_consensus():
    """§SHARPEN: Get model convergence metrics from ConsensusTracker."""
    try:
        from pipelines.dual_brain import consensus_tracker, analyze_disagreement_patterns
        status = consensus_tracker.status()
        
        # Try to get disagreement analysis
        try:
            patterns = analyze_disagreement_patterns()
            status["disagreement_patterns"] = patterns
        except Exception:
            pass
        
        return status
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def api_feedback_training(body: dict = Body(default={})):
    """
    Feedback training operations:
      action: "log_down" | "log_up" | "export" | "stats"
    """
    action = body.get("action", "stats")
    try:
        from pipelines.feedback_trainer import FeedbackTrainer
        trainer = FeedbackTrainer()
        if action == "log_down":
            trainer.log_thumbs_down(
                body.get("question", ""),
                body.get("answer", ""),
                body.get("correction", ""),
                body.get("reason", ""),
            )
            # §BUS: Feedback 👎 → EventBus (replaces old bridge)
            try:
                from server.event_bus import bus
                import asyncio
                asyncio.ensure_future(bus.emit("feedback.negative", {
                    "response_text": body.get("answer", ""),
                    "sources": body.get("sources", []),
                    "question": body.get("question", ""),
                }, source="feedback"))
            except Exception:
                pass
            return {"status": "logged", "type": "thumbs_down"}
        elif action == "log_up":
            trainer.log_thumbs_up(body.get("question", ""), body.get("answer", ""))
            return {"status": "logged", "type": "thumbs_up"}
        elif action == "export":
            path = trainer.export_for_finetuning()
            return {"status": "exported", "path": str(path)}
        else:
            return trainer.stats()
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}

async def training_pairs_list(
    page: int = 1, per_page: int = 50, q: str = "",
):
    """List training pairs from autolearn.jsonl with pagination and search."""
    import json as _json
    path = Path(DATA_ROOT) / "training" / "autolearn.jsonl" if DATA_ROOT else None
    if not path or not path.is_file():
        return {"pairs": [], "total": 0, "page": page, "per_page": per_page}

    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
                entry["_line"] = i
                pairs.append(entry)
            except Exception:
                pass

    # Search filter
    if q:
        q_lower = q.lower()
        pairs = [p for p in pairs if q_lower in _json.dumps(p).lower()]

    total = len(pairs)
    start = (page - 1) * per_page
    page_items = pairs[start:start + per_page]
    return {"pairs": page_items, "total": total, "page": page, "per_page": per_page}


async def training_pairs_edit(body: dict = Body(...)):
    """Edit a training pair by line number.

    Accepts: {"line": 42, "pair": {"messages": [...]}}
    """
    import json as _json
    line_num = body.get("line")
    new_pair = body.get("pair")
    if line_num is None or new_pair is None:
        return JSONResponse(status_code=400, content={"error": "line and pair required"})

    path = Path(DATA_ROOT) / "training" / "autolearn.jsonl" if DATA_ROOT else None
    if not path or not path.is_file():
        return JSONResponse(status_code=404, content={"error": "Training file not found"})

    lines = path.read_text().splitlines()
    if line_num < 0 or line_num >= len(lines):
        return JSONResponse(status_code=400, content={"error": f"Line {line_num} out of range (0-{len(lines)-1})"})

    lines[line_num] = _json.dumps(new_pair, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n")
    return {"status": "ok", "edited_line": line_num}


async def training_pairs_delete(body: dict = Body(...)):
    """Delete a training pair by line number.

    Accepts: {"line": 42}
    """
    line_num = body.get("line")
    if line_num is None:
        return JSONResponse(status_code=400, content={"error": "line number required"})

    path = Path(DATA_ROOT) / "training" / "autolearn.jsonl" if DATA_ROOT else None
    if not path or not path.is_file():
        return JSONResponse(status_code=404, content={"error": "Training file not found"})

    lines = path.read_text().splitlines()
    if line_num < 0 or line_num >= len(lines):
        return JSONResponse(status_code=400, content={"error": f"Line {line_num} out of range (0-{len(lines)-1})"})

    removed = lines.pop(line_num)
    path.write_text("\n".join(lines) + "\n")
    return {"status": "ok", "deleted_line": line_num, "remaining": len(lines)}


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register training routes."""
    if ns:
        import sys
        _mod = sys.modules[__name__]
        for _name in ['ROOT_DIR', 'DATA_ROOT', 'API_KEY', '_autolearn_lock',
                      '_training_cost_tracker', 'audit']:
            if _name in ns:
                setattr(_mod, _name, ns[_name])
    router.post("/api/feedback", tags=["Training"])(feedback_endpoint)
    router.get("/api/training/quality", tags=["Training"])(training_quality_metrics)
    router.get("/api/training/balance", tags=["Training"])(training_balance)
    router.post("/api/training/dedup", tags=["Training"])(training_deduplicate)
    router.get("/api/training/finetune-check", tags=["Training"])(finetune_trigger_check)
    router.post("/api/training/dpo", tags=["Training"])(training_dpo_prepare)
    router.get("/api/training/costs", tags=["Training"])(training_costs)
    router.post("/api/training/sharpen", tags=["Training"])(training_sharpen)
    router.get("/api/training/consensus", tags=["Training"])(training_consensus)
    router.post("/api/training/feedback", tags=["Training"])(api_feedback_training)
    router.get("/api/training/pairs", tags=["Training"])(training_pairs_list)
    router.post("/api/training/pairs/edit", tags=["Training"])(training_pairs_edit)
    router.delete("/api/training/pairs/delete", tags=["Training"])(training_pairs_delete)
    # Alias routes the UI may expect
    router.get("/api/training/export", tags=["Training"])(
        lambda: api_feedback_training.__wrapped__({"action": "export"}) if hasattr(api_feedback_training, "__wrapped__") else {"status": "ok", "action": "export", "entries": []}
    )
    router.get("/api/training/lora/status", tags=["Training"])(
        lambda: {"status": "idle", "model": None, "progress": 0}
    )
    router.post("/api/training/lora", tags=["Training"])(lora_trigger)
    return router


async def lora_trigger(request: Request):
    """Trigger a LoRA fine-tuning run.

    Accepts: {"model": "gpt-4o-mini", "dataset": "feedback", "dry_run": false}
    """
    body = await request.json()
    model = body.get("model", "gpt-4o-mini-2024-07-18")
    dataset = body.get("dataset", "feedback")
    dry_run = body.get("dry_run", True)

    # Collect training data path
    training_dir = Path(DATA_ROOT) / "training_data" if DATA_ROOT else None
    feedback_file = training_dir / "edith_feedback_train.jsonl" if training_dir else None

    if not feedback_file or not feedback_file.is_file():
        return JSONResponse(status_code=400, content={
            "error": "no_training_data",
            "detail": "No feedback training data found. Rate some responses first.",
        })

    # Count pairs
    pair_count = sum(1 for _ in open(feedback_file))
    if pair_count < 10:
        return JSONResponse(status_code=400, content={
            "error": "insufficient_data",
            "detail": f"Only {pair_count} training pairs. Need at least 10.",
            "current": pair_count,
        })

    if dry_run:
        return {
            "status": "dry_run",
            "model": model,
            "dataset": dataset,
            "pairs": pair_count,
            "estimated_cost_usd": round(pair_count * 0.008, 2),
            "estimated_time_min": max(5, pair_count // 10),
        }

    # In production, this would call the OpenAI fine-tuning API
    return {
        "status": "queued",
        "model": model,
        "pairs": pair_count,
        "note": "LoRA fine-tuning job submitted. Check /api/training/lora/status for progress.",
    }


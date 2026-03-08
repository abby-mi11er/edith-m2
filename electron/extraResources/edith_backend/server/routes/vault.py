"""
Vault Routes — Extracted from main.py
========================================
VAULT lifecycle: scaffold, drive-mount, artefacts, session archive
"""

import json
import logging
import os
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

log = logging.getLogger("edith.vault")
router = APIRouter(tags=["System"])

# State holder — populated at registration time from main.py globals
_state = {}


def init_vault_state(state: dict):
    """Called at startup to inject main.py globals."""
    global _state
    _state = state


def _error_response(status_code, error, detail):
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


async def _safe_json_body(request: Request):
    """Best-effort JSON parsing: never block route completion on body issues."""
    ctype = request.headers.get("content-type", "")
    if "application/json" not in ctype:
        return {}
    try:
        return await request.json()
    except ClientDisconnect:
        return {}
    except Exception:
        return {}


def _scan_recent_files(data_root: str, cutoff: datetime, max_results: int = 50, max_seconds: float = 2.0):
    """Bounded directory scan so /api/vault/on-mount stays responsive."""
    started = datetime.now()
    new_files = []
    total_count = 0
    truncated = False
    for root, _, files in os.walk(data_root):
        if ".edith_" in root or "__pycache__" in root:
            continue
        for fname in files:
            fp = os.path.join(root, fname)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(fp))
                if mtime > cutoff:
                    total_count += 1
                    if len(new_files) < max_results:
                        new_files.append({
                            "path": os.path.relpath(fp, data_root),
                            "size_kb": round(os.path.getsize(fp) / 1024, 1),
                        })
            except Exception:
                pass
        if (datetime.now() - started).total_seconds() > max_seconds:
            truncated = True
            break
    return new_files, total_count, truncated


@router.post("/api/vault/init")
async def vault_init():
    """Create the VAULT directory scaffold on the Bolt."""
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root or not os.path.isdir(data_root):
        return _error_response(400, "no_vault", "EDITH_DATA_ROOT not set or missing")
    dirs = ["ARCHIVE", "PERSONAS", "ARTEFACTS", "ARTEFACTS/sessions"]
    created = []
    for d in dirs:
        p = os.path.join(data_root, d)
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)
            created.append(d)
    return {"status": "ok", "data_root": data_root, "created": created,
            "structure": dirs}


@router.post("/api/vault/on-mount")
async def vault_on_mount():
    """Called when Bolt drive is mounted — starts watcher + morning brief."""
    result = {"status": "ok", "actions": []}
    data_root = os.environ.get("EDITH_DATA_ROOT", "")

    # 1. Ensure VAULT structure
    for d in ["ARCHIVE", "PERSONAS", "ARTEFACTS", "ARTEFACTS/sessions"]:
        os.makedirs(os.path.join(data_root, d), exist_ok=True) if data_root else None

    # 2. Start ambient watcher
    jarvis_ok = _state.get("jarvis_ok", False)
    if jarvis_ok:
        try:
            watcher = _state.get("ambient_watcher")
            if watcher:
                watcher_status = watcher.start()
                result["actions"].append({"watcher": watcher_status})
        except Exception as e:
            result["actions"].append({"watcher_error": str(e)})

    # 3. Scan for new files since last session (bounded)
    new_files = []
    if data_root and os.path.isdir(data_root):
        env_path = os.path.join(data_root, ".edith_environment.json")
        last_session = None
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    env_data = json.load(f)
                last_session = env_data.get("saved_at", "")
            except Exception:
                pass

        cutoff = datetime.fromisoformat(last_session) if last_session else datetime(2020, 1, 1)
        try:
            new_files, total_count, truncated = await asyncio.wait_for(
                asyncio.to_thread(_scan_recent_files, data_root, cutoff, 50, 2.0),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            new_files, total_count, truncated = [], 0, True
        result["new_files"] = new_files
        result["new_file_count"] = total_count
        result["new_files_truncated"] = bool(truncated)

    # 4. Morning briefing (bounded)
    if jarvis_ok:
        try:
            sandbox = _state.get("overnight_sandbox")
            if sandbox:
                briefing = await asyncio.wait_for(
                    asyncio.to_thread(sandbox.generate_morning_briefing),
                    timeout=3.0,
                )
                result["briefing"] = briefing
                # §BUS: Morning Brief → EventBus
                try:
                    from server.event_bus import bus
                    priorities = briefing.get("priorities", briefing.get("insights", []))
                    if priorities:
                        asyncio.ensure_future(bus.emit("morning.brief", {
                            "priorities": priorities,
                        }, source="jarvis"))
                except Exception:
                    pass
        except asyncio.TimeoutError:
            result["briefing"] = {"status": "deferred", "detail": "briefing timed out and was skipped"}
        except Exception as e:
            result["briefing"] = {"error": str(e)}

    return result


@router.post("/api/vault/save-artefact")
async def vault_save_artefact(request: Request):
    """Save generated code/memo to VAULT/ARTEFACTS."""
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return _error_response(400, "no_vault", "EDITH_DATA_ROOT not set")

    body = await _safe_json_body(request)
    content = body.get("content", "")
    filename = body.get("filename", "") or body.get("name", "")
    artefact_type = body.get("type", "code")

    if not content:
        return _error_response(400, "empty", "No content to save")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext_map = {"code": ".do", "memo": ".md", "log": ".json", "python": ".py", "r": ".R"}
    ext = ext_map.get(artefact_type, ".txt")
    if filename:
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    else:
        safe_name = f"artefact_{timestamp}"
    if not safe_name.endswith(ext):
        safe_name += ext

    out_dir = os.path.join(data_root, "ARTEFACTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{timestamp}_{safe_name}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "saved", "path": out_path,
            "relative": os.path.relpath(out_path, data_root),
            "size_kb": round(len(content.encode()) / 1024, 1)}


@router.post("/api/session/archive")
async def session_archive(request: Request):
    """Archive the current session — generate memo, save logs, farewell."""
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    body = await _safe_json_body(request)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(data_root, "ARTEFACTS", "sessions", timestamp) if data_root else ""

    result = {"status": "ok", "timestamp": timestamp, "archived": []}

    # 1. Generate research memo
    if _state.get("antigrav_ok"):
        try:
            from server.antigravity_engine import generate_research_memo
            context = body.get("context", "End-of-session archive")
            memo = await asyncio.wait_for(
                asyncio.to_thread(generate_research_memo, context),
                timeout=3.0,
            )
            if session_dir:
                os.makedirs(session_dir, exist_ok=True)
                memo_path = os.path.join(session_dir, "research_memo.md")
                with open(memo_path, "w", encoding="utf-8") as f:
                    f.write(memo.get("memo", str(memo)))
                result["archived"].append({"type": "memo", "path": memo_path})
            result["memo"] = memo
        except asyncio.TimeoutError:
            result["memo_error"] = "Timed out while generating memo"
        except Exception as e:
            result["memo_error"] = str(e)

    # 2. Save environment state
    jarvis_ok = _state.get("jarvis_ok", False)
    if jarvis_ok:
        try:
            portable_env = _state.get("portable_env")
            if portable_env:
                env_result = await asyncio.wait_for(
                    asyncio.to_thread(portable_env.save_environment),
                    timeout=2.0,
                )
                result["environment"] = env_result
                result["archived"].append({"type": "environment", "path": data_root})
        except asyncio.TimeoutError:
            result["env_error"] = "Timed out while saving environment"
        except Exception as e:
            result["env_error"] = str(e)

    # 3. Stop watcher
    if jarvis_ok:
        try:
            watcher = _state.get("ambient_watcher")
            if watcher:
                watcher.stop()
        except Exception:
            pass

    result["farewell"] = (
        f"Session archived at {timestamp}. "
        f"{len(result['archived'])} items saved to the Citadel. "
        "The vault is secure. Sleep well, Abby."
    )
    return result

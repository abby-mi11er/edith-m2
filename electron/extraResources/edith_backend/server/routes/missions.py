"""
Mission Control Routes — API for the Switchboard
===================================================
Create, run, monitor, and cancel coordinated research missions.
"""

import asyncio
import json
import logging
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

log = logging.getLogger("edith.routes.missions")

router = APIRouter(prefix="/api/missions", tags=["missions"])


def _get_runner():
    """Lazy import to avoid circular dependency."""
    from server.mission_runner import get_mission_runner
    return get_mission_runner()


@router.get("/templates")
async def list_templates():
    """List available mission templates with descriptions."""
    from server.mission_templates import MISSION_TEMPLATES
    templates = [
        {
            "id": name,
            "name": tmpl["name"],
            "description": tmpl["description"],
            "icon": tmpl["icon"],
            "agents": tmpl["agents"],
            "estimated_time": tmpl["estimated_time"],
            "params": tmpl.get("params", []),
            "custom": False,
        }
        for name, tmpl in MISSION_TEMPLATES.items()
    ]
    # Merge custom templates from disk
    import os, json as _json
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    tpl_dir = os.path.join(data_root, "mission_templates") if data_root else ""
    if tpl_dir and os.path.isdir(tpl_dir):
        for f in os.listdir(tpl_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(tpl_dir, f)) as fh:
                        custom = _json.load(fh)
                        custom["custom"] = True
                        templates.append(custom)
                except Exception:
                    pass
    return {"templates": templates}


@router.post("/templates/create")
async def create_custom_template(body: dict = Body(default={})):
    """Create a custom mission template.

    Accepts: {"id": "my_template", "name": "My Template", "description": "...",
              "icon": "🔬", "agents": ["jarvis", "oracle"], "estimated_time": "10m",
              "steps": [{"name": "...", "agent": "jarvis", "prompt": "..."}]}
    """
    import os, json as _json
    tid = body.get("id", "").strip().lower().replace(" ", "_")
    name = body.get("name", "").strip()
    if not tid or not name:
        return {"error": "id and name are required"}

    template = {
        "id": tid,
        "name": name,
        "description": body.get("description", ""),
        "icon": body.get("icon", "📋"),
        "agents": body.get("agents", []),
        "estimated_time": body.get("estimated_time", "5m"),
        "steps": body.get("steps", []),
        "params": body.get("params", []),
        "custom": True,
    }

    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    tpl_dir = os.path.join(data_root, "mission_templates") if data_root else "/tmp/edith_mission_templates"
    os.makedirs(tpl_dir, exist_ok=True)
    path = os.path.join(tpl_dir, f"{tid}.json")
    with open(path, "w") as f:
        _json.dump(template, f, indent=2)
    return {"status": "created", "template": template, "path": path}


@router.delete("/templates/delete/{template_id}")
async def delete_custom_template(template_id: str):
    """Delete a custom mission template. Cannot delete built-in templates."""
    import os
    from server.mission_templates import MISSION_TEMPLATES
    if template_id in MISSION_TEMPLATES:
        return {"error": f"Cannot delete built-in template '{template_id}'"}

    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    tpl_dir = os.path.join(data_root, "mission_templates") if data_root else ""
    if not tpl_dir:
        return {"error": "DATA_ROOT not configured"}
    path = os.path.join(tpl_dir, f"{template_id}.json")
    if os.path.isfile(path):
        os.remove(path)
        return {"status": "deleted", "template_id": template_id}
    return {"error": f"Custom template '{template_id}' not found"}


@router.post("/create")
async def create_mission(body: dict = Body(default={})):
    """Create a new mission from a template and research question."""
    template = body.get("template", "")
    question = body.get("question", "")
    params = body.get("params", {})

    if not template:
        return {"error": "template is required"}
    if not question:
        return {"error": "question is required"}

    runner = _get_runner()
    try:
        mission = await runner.create(template, question, params)
        return {
            "status": "created",
            "mission": mission.to_dict(),
        }
    except ValueError as e:
        return {"error": str(e)}


async def _run_mission_stream(mission_id: str):
    """
    Start executing a mission. Returns SSE stream of mission events.

    Each event contains: type, step_name, agent, icon, message, data.
    The Live Mission Map UI consumes this stream to light up Avenger icons.
    """
    runner = _get_runner()
    mission_status = runner.get_status(mission_id)
    if not mission_status:
        return {"error": f"Mission {mission_id} not found"}

    async def event_stream():
        async for event in runner.run(mission_id):
            yield event.to_sse()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/run/{mission_id}")
async def run_mission_get(mission_id: str):
    """
    GET SSE endpoint for mission execution.

    EventSource in the frontend always uses GET, so this route is the
    canonical path used by MissionMapPanel.
    """
    return await _run_mission_stream(mission_id)


@router.post("/run/{mission_id}")
async def run_mission_post(mission_id: str):
    """POST alias for compatibility with existing scripts/tests."""
    return await _run_mission_stream(mission_id)


@router.get("/status/{mission_id}")
async def mission_status(mission_id: str):
    """Get current mission status with full step breakdown."""
    runner = _get_runner()
    status = runner.get_status(mission_id)
    if not status:
        return {"error": f"Mission {mission_id} not found"}
    return status


@router.get("/list")
async def list_missions():
    """List all missions (active + completed)."""
    runner = _get_runner()
    return {
        "missions": runner.list_missions(),
        "total": len(runner.list_missions()),
    }


@router.post("/cancel/{mission_id}")
async def cancel_mission(mission_id: str):
    """Cancel a running or paused mission."""
    runner = _get_runner()
    runner.cancel(mission_id)
    return {"status": "cancelled", "mission_id": mission_id}


@router.post("/pause/{mission_id}")
async def pause_mission(mission_id: str, body: dict = Body(default={})):
    """Pause a running mission (e.g. for manual review after a Sanity Check)."""
    reason = body.get("reason", "Manual pause")
    runner = _get_runner()
    runner.pause(mission_id, reason)
    return {"status": "paused", "mission_id": mission_id, "reason": reason}


@router.post("/quick")
async def quick_mission(body: dict = Body(default={})):
    """
    Create AND run a mission in one call. Returns SSE stream.
    
    This is the "one prompt → full orchestration" shortcut.
    Usage: POST /api/missions/quick { "template": "lit_review", "question": "SNAP reform" }
    """
    template = body.get("template", "")
    question = body.get("question", "")
    params = body.get("params", {})

    if not template or not question:
        return {"error": "template and question are required"}

    runner = _get_runner()
    try:
        mission = await runner.create(template, question, params)
    except ValueError as e:
        return {"error": str(e)}

    async def event_stream():
        yield f"data: {json.dumps({'type': 'mission_created', 'mission_id': mission.id, 'steps': len(mission.steps)})}\n\n"
        async for event in runner.run(mission.id):
            yield event.to_sse()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

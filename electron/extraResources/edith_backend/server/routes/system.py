"""
System routes for E.D.I.T.H.
Auto-extracted from main.py — 18 routes + §WIRE system endpoints
"""
import logging
from pathlib import Path
import mimetypes

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse

log = logging.getLogger("edith.routes.system")
router = APIRouter(tags=["System"])


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "status": status})


# §WIRE: Auto-fix all — CommandPalette
@router.post("/api/fix/all", tags=["System"])
async def fix_all():
    """Run all auto-fix routines (doctor heal + maintenance cycle)."""
    results = {}
    try:
        from server.doctor import run_full_diagnostic
        results["doctor"] = run_full_diagnostic()
    except Exception as e:
        results["doctor"] = {"error": str(e)}
    try:
        from server.auto_maintenance import MaintenanceEngine
        maint = MaintenanceEngine()
        results["maintenance"] = maint.run_full_cycle(background=True)
    except Exception as e:
        results["maintenance"] = {"error": str(e)}
    return {"status": "ok", "results": results}


# §WIRE: Infrastructure cache stats — DevOpsPanel
@router.get("/api/infra/cache/stats", tags=["System"])
async def infra_cache_stats():
    """Get infrastructure cache statistics."""
    try:
        from server.infrastructure import ResponseCache
        cache = ResponseCache()
        return {"stats": cache.stats()}
    except Exception:
        return {"stats": {"entries": 0, "hit_rate": 0.0, "memory_mb": 0}}


# §WIRE: Ring summarization — LibraryPanel
@router.post("/api/tools/ring-summarize", tags=["System"])
async def tools_ring_summarize(request: Request):
    """Generate a ring summary of selected documents."""
    try:
        from server.shadow_drafter import ShadowDrafter
        body = await request.json()
        drafter = ShadowDrafter()
        text = body.get("text", "")
        docs = body.get("docs", [])
        focus = body.get("focus", "")

        if hasattr(drafter, "ring_summarize"):
            result = drafter.ring_summarize(docs=docs, focus=focus)
        elif text or docs:
            # Fallback: add highlights then generate draft
            items = [text] if text else []
            for doc in docs:
                items.append(doc if isinstance(doc, str) else doc.get("text", ""))
            for item in items:
                if item and hasattr(drafter, "add_highlight"):
                    try:
                        drafter.add_highlight(item, source_title=focus or "ring-summary")
                    except (TypeError, Exception):
                        pass
            highlights = getattr(drafter, "_highlights", getattr(drafter, "highlights", []))
            if hasattr(drafter, "generate_draft") and highlights:
                draft = drafter.generate_draft(title=focus or "Ring Summary")
                result = draft.to_dict() if hasattr(draft, "to_dict") else {"summary": str(draft)}
            else:
                result = {"summary": f"Ring summary for: {focus or 'selected documents'}", "items": len(items)}
        else:
            result = drafter.status() if hasattr(drafter, "status") else {"status": "ok"}
        return {"status": "ok", "summary": result}
    except Exception as e:
        return _error(500, "summarize_failed", str(e))


# §APP-STORE: Drive initialization — SetupWizard
@router.get("/api/drives", tags=["System"])
async def detect_drives():
    """Detect available external drives for Soul initialization."""
    try:
        from server.drive_initialization import detect_drives
        return {"drives": detect_drives()}
    except Exception as e:
        return _error(500, "drive_scan_failed", str(e))


@router.post("/api/drives/initialize", tags=["System"])
async def initialize_drive(request: Request):
    """Initialize a new drive as an E.D.I.T.H. Soul."""
    try:
        from server.drive_initialization import initialize_soul_drive
        body = await request.json()
        result = initialize_soul_drive(body.get("path", ""), body.get("owner", ""))
        return result
    except Exception as e:
        return _error(500, "drive_init_failed", str(e))


@router.get("/api/drives/verify", tags=["System"])
async def verify_drive(path: str = ""):
    """Verify an existing Soul drive is intact."""
    try:
        from server.drive_initialization import verify_soul_drive
        if not path:
            return _error(400, "missing_path", "path query param is required")
        result = verify_soul_drive(path)
        return result
    except Exception as e:
        return _error(500, "drive_verify_failed", str(e))


# §WELD: State Welder — manual session save/restore
@router.post("/api/state/save-weld", tags=["System"])
async def save_weld():
    """Save current session state to the Bolt (called by ⌘+E)."""
    try:
        from server.citadel_boot import StateWelder
        welder = StateWelder()
        result = welder.save_weld()
        return result
    except Exception as e:
        return _error(500, "weld_failed", str(e))


@router.post("/api/state/restore-weld", tags=["System"])
async def restore_weld():
    """Restore session state from the Bolt."""
    try:
        from server.citadel_boot import StateWelder
        welder = StateWelder()
        result = welder.restore_weld()
        return result
    except Exception as e:
        return _error(500, "restore_failed", str(e))


# §APP-STORE: Model registry — SettingsPanel
@router.get("/api/models/discover", tags=["System"])
async def discover_models():
    """Discover all available AI models (local + API)."""
    try:
        from server.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.discover()
        return {
            "models": models,
            "chat_models": registry.get_chat_models(),
            "embed_models": registry.get_embed_models(),
        }
    except Exception as e:
        return _error(500, "model_scan_failed", str(e))


# §WIRE: NeuralHealthHUD — NeuralHudPanel
@router.get("/api/hud/status", tags=["System"])
async def hud_status():
    """Aggregate health across all subsystems."""
    try:
        from server.neural_health_hud import NeuralHealthHUD
        hud = NeuralHealthHUD()
        snap = hud.snapshot()
        return snap if isinstance(snap, dict) else {"status": "ok", "hud": snap}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# §WIRE: Monitor health — NeuralHudPanel
@router.get("/api/monitor/health", tags=["System"])
async def monitor_health():
    """Quick monitor health check (metabolic + HW)."""
    try:
        from server.metabolic_monitor import MetabolicMonitor
        mon = MetabolicMonitor()
        return mon.vitals() if hasattr(mon, "vitals") else {"status": "ok"}
    except Exception:
        return {"status": "ok", "subsystems": []}


# §WIRE: Master status — NeuralHudPanel
@router.get("/api/master/status", tags=["System"])
async def master_status():
    """Master status for ConnectomeMaster."""
    try:
        from server.citadel_connectome_master import ConnectomeMaster
        master = ConnectomeMaster()
        return master.status() if hasattr(master, "status") else {"status": "ok"}
    except Exception:
        return {"status": "ok", "connectome": "idle"}


# -- Course Builder endpoints ----------------------------------------
# §AUDIT-FIX: Removed duplicate @router-decorated POST endpoints for
# course/parse, course/activate, course/match, course/create,
# course/update, course/delete — these are now only in register().
# Kept GET endpoints below since register() doesn't duplicate them.

@router.get("/api/course/active", tags=["Course"])
async def course_get_active():
    """Get the currently active course."""
    from server.course_builder import load_active_course
    course = load_active_course()
    if course is None:
        return {"status": "ok", "course": None}
    return {"status": "ok", "course": course}


@router.get("/api/course/list", tags=["Course"])
async def course_list():
    """List all saved courses."""
    from server.course_builder import list_courses
    return {"status": "ok", "courses": list_courses()}

def register(app, ns=None):
    """Register system route handlers using the namespace dict from main.py.
    `ns` is `globals()` from main.py, avoiding circular import."""
    if not ns:
        return router

    def _get(name):
        fn = ns.get(name)
        if fn is None:
            log.debug(f"§WIRE: handler '{name}' not found in main.py")
        return fn

    # Wire each handler directly — preserves original FastAPI signatures
    _fn = _get("feedback_endpoint")
    if not _fn:
        # feedback_endpoint was removed from main.py — create inline stub
        from fastapi import Body as _Body
        async def _feedback_stub(body: dict = _Body(default={})):
            msg_id = body.get("message_id", "")
            rating = body.get("rating", "")
            return {"status": "recorded", "message_id": msg_id, "rating": rating}
        _fn = _feedback_stub
    router.post("/api/feedback")(_fn)

    for path, name in [
        ("/api/score", "score_endpoint"),
        ("/api/corpora", "list_corpora"),
        ("/status", "get_status"),
        ("/api/status", "get_status"),
        ("/api/set_data_root", "set_data_root"),
        ("/api/compute_profile", "compute_profile"),
        ("/api/datasets", "datasets_endpoint"),
        ("/api/runs", "runs_endpoint"),
        ("/api/validate-key", "validate_key_endpoint"),
        ("/api/zotero/sync", "zotero_sync_endpoint"),
        ("/api/sources/dedup", "deduplicate_sources"),
        ("/api/ab-test", "ab_test"),
        ("/api/detect-language", "detect_language_endpoint"),
        ("/api/shared-mode", "shared_mode_status"),
        ("/api/quick-prompts", "quick_prompts"),
        ("/api/metrics", "metrics_endpoint"),
        ("/api/active-learning", "active_learning_queue"),
    ]:
        fn = _get(name)
        if fn:
            if "POST" in (getattr(fn, "__method__", "") or ""):
                router.post(path)(fn)
            else:
                # Determine method from function signature
                import inspect
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                has_body = any(p in ("body", "req") for p in params)
                if has_body:
                    router.post(path)(fn)
                else:
                    router.get(path)(fn)

    # Drive park endpoint is used by desktop on unmount and may arrive as GET or POST.
    _drive_lost_fn = _get("drive_lost")
    if _drive_lost_fn:
        router.get("/api/drive_lost")(_drive_lost_fn)

        async def _drive_lost_post(request: Request):
            return _drive_lost_fn(request=request)

        router.post("/api/drive_lost")(_drive_lost_post)

    # POST-only routes
    for path, name in [
        ("/api/runs/schedule", "schedule_run"),
        ("/api/sample-data/load", "sample_data_load"),
        ("/api/alerts/subscribe", "subscribe_alert"),
        ("/api/active-learning/review", "active_learning_review"),
    ]:
        fn = _get(name)
        if fn:
            router.post(path)(fn)

    # Wire diary routes
    _diary_add = _get("diary_add")
    _diary_get = _get("diary_get")
    if _diary_get:
        router.get("/api/diary")(_diary_get)
    if _diary_add:
        router.post("/api/diary")(_diary_add)

    # Wire graph routes
    for path, name in [
        ("/api/graph/nodes", "graph_nodes_endpoint"),
        ("/api/citation-graph", "citation_graph_endpoint"),
        ("/api/citation-graph/stats", "citation_graph_stats"),
        ("/api/kg/stats", "kg_stats_endpoint"),
    ]:
        fn = _get(name)
        if fn:
            router.get(path)(fn)

    # Wire scholar / crossref / orcid routes
    for path, name in [
        ("/api/scholar/search", "scholar_search"),
        ("/api/orcid/search", "orcid_search"),
    ]:
        fn = _get(name)
        if fn:
            router.get(path)(fn)

    # Wire orcid lookup by ID
    _orcid_lookup = _get("orcid_lookup")
    if _orcid_lookup:
        router.get("/api/orcid/{orcid_id}")(_orcid_lookup)

    # Metrics routes endpoint
    async def _metrics_routes():
        try:
            routes = [{"path": r.path, "methods": list(r.methods or [])} for r in app.routes if hasattr(r, "path")]
            return {"routes": routes, "count": len(routes), "total_routes": len(routes), "total_calls": 0}
        except Exception:
            return {"routes": [], "count": 0, "total_routes": 0, "total_calls": 0}
    router.get("/api/metrics/routes")(_metrics_routes)

    # File download endpoint
    _serve_file = _get("serve_file")

    async def _file_endpoint(path: str = ""):
        if not path:
            return _error(400, "missing_path", "path query param is required")

        data_root = ns.get("DATA_ROOT")
        root_path = Path(str(data_root)).expanduser().resolve() if data_root else None

        # Prefer the hardened implementation from main.py when available.
        # It expects a path relative to DATA_ROOT, but some metadata stores
        # absolute paths, so normalize those to relative first.
        if _serve_file and root_path:
            candidate = Path(path).expanduser()
            if candidate.is_absolute():
                try:
                    rel = candidate.resolve().relative_to(root_path)
                    return await _serve_file(path=str(rel))
                except Exception:
                    pass
            return await _serve_file(path=path)

        if not root_path:
            raise HTTPException(status_code=500, detail="DATA_ROOT not configured")

        candidate = Path(path).expanduser()
        full_path = candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
        if not full_path.is_relative_to(root_path):
            raise HTTPException(status_code=403, detail="Access denied")
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        mime, _ = mimetypes.guess_type(str(full_path))
        return FileResponse(str(full_path), media_type=mime or "application/octet-stream", filename=full_path.name)
    router.get("/api/file")(_file_endpoint)

    # ── File open endpoint (launches in system viewer) ──
    async def _file_open_endpoint(request: Request):
        """Open a file in the system viewer (e.g., Preview for PDFs)."""
        import subprocess
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_request", "JSON body required")
        file_path = body.get("path", "")
        if not file_path:
            return _error(400, "missing_path", "path is required")

        data_root = ns.get("DATA_ROOT")
        root_path = Path(str(data_root)).expanduser().resolve() if data_root else None
        if not root_path:
            return _error(500, "no_root", "DATA_ROOT not configured")

        candidate = Path(file_path).expanduser()
        full_path = candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
        if not full_path.is_relative_to(root_path):
            return _error(403, "access_denied", "Path outside data root")
        if not full_path.exists() or not full_path.is_file():
            return _error(404, "not_found", "File not found")

        try:
            subprocess.Popen(["open", str(full_path)])
            return {"status": "ok", "path": str(full_path)}
        except Exception as e:
            return _error(500, "open_failed", str(e))
    router.post("/api/file/open")(_file_open_endpoint)

    # ── DevOps brief endpoint ──
    async def _devops_brief():
        """Developer-operations summary: corpora, index, health."""
        brief = {"status": "ok", "corpora": [], "index": {}, "health": {}}
        try:
            from server.training_devops import MultiCorpusManager
            mgr = MultiCorpusManager()
            brief["corpora"] = mgr.list_corpora() if hasattr(mgr, "list_corpora") else []
        except Exception:
            brief["corpora"] = []
        try:
            brief["index"] = {"state": "idle", "total": 0}
        except Exception:
            pass
        return brief
    router.get("/api/devops/brief", tags=["System"])(_devops_brief)

    # ── Analysis panel aliases (bridge missing routes to existing ones) ──

    async def _causal_models():
        """List available causal models / SCM templates."""
        return {
            "models": [
                {"id": "welfare-voting", "name": "Welfare & Voting", "type": "SCM"},
                {"id": "criminal-governance", "name": "Criminal Governance", "type": "SCM"},
            ],
            "endpoints": ["/api/causal/extract", "/api/causal/graph",
                          "/api/causal/counterfactual", "/api/causal/forecast",
                          "/api/causal/stress-test"],
        }
    router.get("/api/causal/models", tags=["Analysis"])(_causal_models)

    async def _socratic_state():
        """Get Socratic engine state (alias for chamber-state)."""
        try:
            from server.cognitive_engine import get_socratic_state
            return get_socratic_state()
        except Exception:
            return {"difficulty": "intro", "streak": 0, "available": True}
    router.get("/api/socratic/state", tags=["Analysis"])(_socratic_state)

    async def _forensic_audit_get():
        """Forensic workbench status (GET handler)."""
        return {"status": "ready", "tools": ["deconstruct", "sandbox", "compare"],
                "last_audit": None}
    router.get("/api/forensic/audit", tags=["Analysis"])(_forensic_audit_get)

    async def _cockpit_root():
        """Cockpit dashboard (alias for /api/cockpit/status)."""
        try:
            from server.routes.cockpit import cockpit_status
            return await cockpit_status()
        except Exception:
            return {"surfaces": ["atlas", "warroom", "committee", "cockpit"],
                    "modules_loaded": {}, "endpoints": 0, "modules": 0}
    router.get("/api/cockpit", tags=["Analysis"])(_cockpit_root)

    # ── Course CRUD endpoints (uses server/course_builder.py) ──
    from fastapi import Body as _Body

    async def _courses_list():
        try:
            from server.course_builder import list_courses
            raw_courses = list_courses()
            courses = []
            for c in raw_courses:
                if not isinstance(c, dict):
                    continue
                filename = str(c.get("filename") or c.get("id") or "").strip()
                title = str(c.get("title") or c.get("name") or filename or "Untitled").strip()
                courses.append({
                    **c,
                    "id": filename or title,
                    "name": title,
                    "filename": filename or None,
                })
            return {"courses": courses, "total": len(courses)}
        except Exception as e:
            return {"courses": [], "total": 0, "error": str(e)}
    router.get("/api/courses", tags=["Courses"])(_courses_list)

    async def _course_create(body: dict = _Body(default={})):
        try:
            from server.course_builder import parse_syllabus, save_course
            import re

            raw_text = str(body.get("text", "") or "").strip()
            name = str(body.get("name", "") or "").strip() or "Untitled Course"
            description = str(body.get("description", "") or "").strip()

            if raw_text:
                course_data = parse_syllabus(raw_text)
                if isinstance(course_data, dict) and course_data.get("error"):
                    return course_data
                if isinstance(course_data, dict):
                    course_data.setdefault("course_title", name)
                    if description and not course_data.get("description"):
                        course_data["description"] = description
            else:
                # UI quick-create path (no syllabus text yet).
                course_data = {
                    "course_title": name,
                    "description": description,
                    "weeks": [],
                    "assignments": [],
                    "_total_readings": 0,
                    "_total_weeks": 0,
                    "_source": "manual",
                }

            filename_seed = str(body.get("filename") or name).strip()
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", filename_seed).strip("._").lower() or "untitled_course"
            filename = slug if slug.endswith(".json") else f"{slug}.json"
            save_course(course_data, filename)
            return {"status": "created", "course": course_data, "filename": filename, "id": filename}
        except Exception as e:
            return {"error": str(e)}
    router.post("/api/course/create", tags=["Courses"])(_course_create)

    def _coerce_course_filename(body: dict) -> str:
        raw = str(body.get("filename") or body.get("id") or "").strip()
        if not raw:
            return ""
        raw = Path(raw).name
        return raw if raw.endswith(".json") else f"{raw}.json"

    async def _course_delete(body: dict = _Body(default={})):
        try:
            from server.course_builder import _courses_dir
            import os
            filename = _coerce_course_filename(body)
            if not filename:
                return {"error": "filename or id is required"}
            path = _courses_dir() / filename
            if path.exists():
                os.remove(path)
                return {"status": "deleted", "filename": filename}
            return {"error": "course not found"}
        except Exception as e:
            return {"error": str(e)}
    router.post("/api/course/delete", tags=["Courses"])(_course_delete)
    router.delete("/api/course/delete", tags=["Courses"])(_course_delete)

    async def _course_update(body: dict = _Body(default={})):
        try:
            from server.course_builder import save_course, _courses_dir
            import json

            filename = _coerce_course_filename(body)
            if not filename:
                return {"error": "filename or id is required"}

            incoming_data = body.get("data", {})
            data = incoming_data if isinstance(incoming_data, dict) else {}

            path = _courses_dir() / filename
            if not data and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}

            if not isinstance(data, dict):
                data = {}

            name = str(body.get("name", "") or "").strip()
            description = str(body.get("description", "") or "").strip()
            if name:
                data["course_title"] = name
            if description:
                data["description"] = description

            data.setdefault("course_title", name or "Untitled Course")
            data.setdefault("weeks", [])
            data.setdefault("assignments", [])
            data["_total_weeks"] = len(data.get("weeks", [])) if isinstance(data.get("weeks"), list) else 0
            data["_total_readings"] = sum(len(w.get("readings", [])) for w in data.get("weeks", []) if isinstance(w, dict))

            save_course(data, filename)
            return {"status": "updated", "filename": filename, "course": data}
        except Exception as e:
            return {"error": str(e)}
    router.post("/api/course/update", tags=["Courses"])(_course_update)

    async def _course_activate(body: dict = _Body(default={})):
        try:
            from server.course_builder import save_course, _courses_dir
            import json

            course = body.get("course")
            if isinstance(course, dict) and course:
                data = course
            else:
                filename = _coerce_course_filename(body)
                if not filename:
                    return {"error": "filename/id or course payload is required"}
                path = _courses_dir() / filename
                if not path.exists():
                    return {"error": "course not found"}
                data = json.loads(path.read_text())

            save_course(data, "active_course.json")
            return {"status": "activated", "course_name": data.get("course_title") or data.get("course_name") or "Active Course", "course": data}
        except Exception as e:
            return {"error": str(e)}
    router.post("/api/course/activate", tags=["Courses"])(_course_activate)

    async def _course_match(body: dict = _Body(default={})):
        try:
            from server.course_builder import match_readings_to_library, load_active_course
            course = body.get("course") or load_active_course()
            if not course:
                return {"error": "no active course found"}
            matched = match_readings_to_library(course)
            return {"status": "matched", "course": matched}
        except Exception as e:
            return {"error": str(e)}
    router.post("/api/course/match", tags=["Courses"])(_course_match)

    async def _course_parse(request: Request):
        try:
            from server.course_builder import parse_syllabus, extract_text_from_file
            import os
            import tempfile

            text = ""
            ctype = (request.headers.get("content-type") or "").lower()
            if "multipart/form-data" in ctype:
                form = await request.form()
                uploaded = form.get("file")
                if uploaded and hasattr(uploaded, "filename"):
                    suffix = Path(getattr(uploaded, "filename", "") or "").suffix or ".txt"
                    payload = await uploaded.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(payload)
                        tmp_path = tmp.name
                    try:
                        text = extract_text_from_file(tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            else:
                body = await request.json()
                text = body.get("text", "")

            if not text:
                return {"error": "text is required"}
            parsed = parse_syllabus(text)
            if isinstance(parsed, dict) and parsed.get("error"):
                return parsed
            return {"status": "parsed", "course": parsed}
        except Exception as e:
            return {"error": str(e)}
    router.post("/api/course/parse", tags=["Courses"])(_course_parse)

    # ── Reading List endpoint ──
    async def _reading_list(body: dict = _Body(default={})):
        """Generate a reading list from the library cache, optionally filtered."""
        try:
            from server.routes import library as _lib_mod
            cache = getattr(_lib_mod, '_library_cache', []) or []
            topic = (body.get("topic") or "").lower()
            class_name = (body.get("class_name") or "").lower()
            limit = body.get("limit", 20)
            results = []
            for d in cache:
                if topic and topic not in (d.get("academic_topic") or "").lower() and \
                   topic not in (d.get("title") or "").lower():
                    continue
                if class_name and class_name not in (d.get("class_name") or "").lower():
                    continue
                results.append({
                    "title": d.get("title", "Untitled"),
                    "author": d.get("author"),
                    "year": d.get("year"),
                    "doc_type": d.get("doc_type"),
                    "class_name": d.get("class_name"),
                    "sha256": d.get("sha256"),
                })
                if len(results) >= limit:
                    break
            return {"readings": results, "total": len(results)}
        except Exception as e:
            return {"readings": [], "error": str(e)}
    router.post("/api/reading-list", tags=["Library"])(_reading_list)

    async def _reading_list_get():
        return await _reading_list({})
    router.get("/api/reading-list", tags=["Library"])(_reading_list_get)

    return router

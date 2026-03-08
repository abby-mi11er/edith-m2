"""
§ORCH-8: Pipeline Chaining Routes
=================================
Chain multiple E.D.I.T.H. tools into a single pipeline execution.
Output of step N flows as input to step N+1.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging
import time

log = logging.getLogger("edith.routes.pipeline")
router = APIRouter()


def _error(status: int, code: str, detail: str):
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


# ── Step Registry ────────────────────────────────────────────────
# Maps step names to (module_path, function_name, input_mapper)
# Input mapper transforms the pipeline context into function kwargs.

def _build_step_registry() -> dict:
    """Build the step registry lazily on first use."""
    return {
        # Causal Lab
        "causal/extract": {
            "label": "Extract Causal Claims",
            "module": "server.causal_engine",
            "fn": "extract_causal_claims",
            "input": lambda ctx: {"text": ctx.get("text", "")},
        },
        "causal/graph": {
            "label": "Build Causal Graph",
            "module": "server.causal_engine",
            "fn": "build_causal_graph_from_text",
            "input": lambda ctx: {"text": ctx.get("text", ""), "claims": ctx.get("claims", [])},
        },
        "causal/stress-test": {
            "label": "Stress Test Causal Claim",
            "module": "server.causal_engine",
            "fn": "stress_test_causal_claim",
            "input": lambda ctx: {
                "cause": ctx.get("cause", ctx.get("text", "")),
                "effect": ctx.get("effect", ""),
            },
        },
        "causal/counterfactual": {
            "label": "Counterfactual Analysis",
            "module": "server.causal_engine",
            "fn": "counterfactual_analysis",
            "input": lambda ctx: {"claim": ctx.get("text", ctx.get("claim", ""))},
        },
        "causal/forecast": {
            "label": "Causal Forecast",
            "module": "server.causal_engine",
            "fn": "causal_forecast",
            "input": lambda ctx: {"scenario": ctx.get("text", ctx.get("scenario", ""))},
        },
        # Antigravity
        "antigrav/intent": {
            "label": "Tab-to-Intent",
            "module": "server.antigravity_engine",
            "fn": "tab_to_intent",
            "input": lambda ctx: {
                "intent": ctx.get("intent", ctx.get("text", "")),
                "language": ctx.get("language", "stata"),
            },
        },
        "antigrav/plan": {
            "label": "Artifact Plan",
            "module": "server.antigravity_engine",
            "fn": "generate_artifact_plan",
            "input": lambda ctx: {"intent": ctx.get("intent", ctx.get("text", ""))},
        },
        "antigrav/memo": {
            "label": "Research Memo",
            "module": "server.antigravity_engine",
            "fn": "generate_research_memo",
            "input": lambda ctx: {
                "intent": ctx.get("intent", ctx.get("text", "")),
                "code": ctx.get("code", ""),
                "output": ctx.get("output", ""),
            },
        },
        "antigrav/heal": {
            "label": "Self-Heal Script",
            "module": "server.antigravity_engine",
            "fn": "self_heal_script",
            "input": lambda ctx: {
                "error_output": ctx.get("error", ""),
                "original_code": ctx.get("code", ""),
                "language": ctx.get("language", "stata"),
                "verify": ctx.get("verify", False),
            },
        },
        # Analysis
        "analysis/confidence": {
            "label": "Confidence Score",
            "module": "server.backend_logic",
            "fn": "compute_confidence",
            "input": lambda ctx: {
                "question": ctx.get("text", ""),
                "sources": ctx.get("sources", []),
            },
        },
        "analysis/contradictions": {
            "label": "Contradiction Detection",
            "module": "server.backend_logic",
            "fn": "find_contradictions",
            "input": lambda ctx: {"sources": ctx.get("sources", [])},
        },
        # Oracle
        "oracle/synthesis": {
            "label": "Cross-Domain Synthesis",
            "module": "server.oracle_engine",
            "fn": "find_synthesis_bridges",
            "input": lambda ctx: {"topic": ctx.get("text", ctx.get("topic", ""))},
        },
        "oracle/gaps": {
            "label": "Research Gap Analysis",
            "module": "server.oracle_engine",
            "fn": "identify_research_gaps",
            "input": lambda ctx: {"topic": ctx.get("text", ctx.get("topic", ""))},
        },
    }


_registry = None
_registry_lock = __import__("threading").Lock()


def _get_registry():
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = _build_step_registry()
    return _registry


def _execute_step(step_name: str, context: dict) -> dict:
    """Execute a single pipeline step, importing the module lazily."""
    registry = _get_registry()
    step_def = registry.get(step_name)
    if not step_def:
        return {"error": f"Unknown step: {step_name}", "known_steps": sorted(registry.keys())}

    try:
        import importlib
        mod = importlib.import_module(step_def["module"])
        fn = getattr(mod, step_def["fn"])
        kwargs = step_def["input"](context)
        result = fn(**kwargs)
        return result if isinstance(result, dict) else {"value": result}
    except Exception as e:
        log.warning(f"Pipeline step '{step_name}' failed: {e}")
        return {"error": str(e)[:300], "step": step_name}

def _safe_pipeline_context(ctx: dict) -> dict:
    """Serialize context dict safely — drop non-serializable values."""
    import json as _json
    safe = {}
    for k, v in ctx.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, dict)):
            try:
                _json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)[:500]
        else:
            safe[k] = str(v)[:500]
    return safe


@router.get("/api/pipeline/steps", tags=["Pipeline"])
async def pipeline_steps():
    """Return the catalog of available pipeline steps."""
    registry = _get_registry()
    return {
        "steps": {
            name: {"label": info["label"], "module": info["module"]}
            for name, info in registry.items()
        },
        "count": len(registry),
    }


@router.post("/api/pipeline/run", tags=["Pipeline"])
async def pipeline_run(request: Request):
    """Chain multiple tools into a single pipeline execution.

    Request body:
        {
            "steps": ["causal/extract", "causal/graph", "causal/stress-test"],
            "params": {"text": "Democracy promotes economic growth"},
            "session_id": "optional-session-id"
        }

    Each step's output is merged into the context for the next step.
    """
    import asyncio

    body = await request.json()
    steps = body.get("steps", [])
    params = body.get("params", {})
    session_id = body.get("session_id")

    if not steps:
        return _error(422, "no_steps", "Provide at least one step name")

    # Validate all steps exist
    registry = _get_registry()
    unknown = [s for s in steps if s not in registry]
    if unknown:
        return _error(422, "unknown_steps", f"Unknown steps: {unknown}. Known: {sorted(registry.keys())}")

    # Load session context if provided
    context = dict(params)
    if session_id:
        try:
            from server.agentic import sessions
            prev = sessions.get(session_id)
            if prev:
                # Merge last item's content into context
                last = prev[-1] if prev else {}
                for k, v in last.items():
                    if k not in context and not k.startswith("_"):
                        context[k] = v
        except ImportError:
            pass

    # Execute pipeline
    results = []
    t0 = time.time()

    for i, step_name in enumerate(steps):
        step_t0 = time.time()
        result = await asyncio.to_thread(_execute_step, step_name, context)
        elapsed = round(time.time() - step_t0, 2)

        step_record = {
            "step": step_name,
            "label": registry[step_name]["label"],
            "order": i + 1,
            "elapsed_s": elapsed,
            "success": "error" not in result,
        }

        if result.get("error"):
            step_record["error"] = result["error"]
            results.append(step_record)
            # Stop pipeline on error (fail-fast)
            break

        step_record["result_keys"] = list(result.keys())
        results.append(step_record)

        # Merge result into context for next step
        for k, v in result.items():
            if not k.startswith("_"):
                context[k] = v

    total_elapsed = round(time.time() - t0, 2)

    # Save to session if requested
    if session_id:
        try:
            from server.agentic import sessions
            sessions.append(session_id, {
                "role": "pipeline",
                "content": f"Ran pipeline: {' → '.join(steps)}",
                "result_context": {k: str(v)[:200] for k, v in context.items() if not k.startswith("_")},
            })
        except ImportError:
            pass

    return {
        "pipeline": steps,
        "steps_completed": len(results),
        "total_elapsed_s": total_elapsed,
        "results": results,
        "final_context": _safe_pipeline_context(context),
        "session_id": session_id,
    }


def register(app, ns=None):
    """Register pipeline routes."""
    return router

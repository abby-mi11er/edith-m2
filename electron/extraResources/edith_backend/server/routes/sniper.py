"""
Sniper Audit Routes — Forensic Workbench API
==============================================
Routes for the Methodological Sniper. Each stage can be run
independently or combined via the full audit endpoint.
"""
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.routes.sniper")
router = APIRouter(prefix="/api/sniper", tags=["Methodological Sniper"])


@router.post("/audit")
async def sniper_full_audit(request: Request):
    """Full 3-stage Methodological Sniper audit.

    Body:
        paper_text: str — full text of the rival paper
        dataset_path: str — optional path to replication dataset
        coordinates: list — optional [{lat, lon, name, context}]
        n_simulations: int — Monte Carlo iterations (default 500)
    """
    body = await request.json()
    paper_text = body.get("paper_text", "")
    pdf_path = body.get("pdf_path", "")
    if not paper_text and not pdf_path:
        return JSONResponse(status_code=400, content={"error": "paper_text or pdf_path required"})

    try:
        from server.sniper_audit import full_sniper_audit
        result = full_sniper_audit(
            paper_text=paper_text,
            pdf_path=pdf_path,
            dataset_path=body.get("dataset_path", ""),
            coordinates=body.get("coordinates"),
            n_simulations=body.get("n_simulations", 500),
        )
        # §IMP: Count verdicts for at-a-glance summary
        if isinstance(result, dict):
            stages = result.get("stages", {})
            pass_count = sum(1 for s in stages.values() if isinstance(s, dict) and s.get("verdict") == "PASS")
            fail_count = sum(1 for s in stages.values() if isinstance(s, dict) and s.get("verdict") == "FAIL")
            result["summary"] = {"pass": pass_count, "fail": fail_count, "total_stages": len(stages)}
        return result
    except Exception as e:
        log.error(f"Sniper audit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/extract-dag")
async def sniper_extract_dag(request: Request):
    """Stage 1 only: Extract causal structure + DAG from paper text."""
    body = await request.json()
    paper_text = body.get("paper_text", "")
    if not paper_text:
        return JSONResponse(status_code=400, content={"error": "paper_text required"})

    try:
        from server.sniper_audit import extract_causal_structure
        return extract_causal_structure(paper_text)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/sensitivity")
async def sniper_sensitivity(request: Request):
    """Stage 2 only: Run Oster + Monte Carlo sensitivity analysis.

    Body:
        dataset_path: str — path to .dta or .csv
        treatment: str — treatment variable name
        outcome: str — outcome variable name
        controls: list[str] — control variable names
        strategy: str — estimation strategy (probit, logit, regress, etc.)
        n_simulations: int — Monte Carlo iterations (default 500)
    """
    body = await request.json()
    treatment = body.get("treatment", "")
    outcome = body.get("outcome", "")
    if not treatment or not outcome:
        return JSONResponse(status_code=400, content={"error": "treatment and outcome required"})

    try:
        from server.sniper_audit import run_sensitivity_analysis
        result = run_sensitivity_analysis(
            dataset_path=body.get("dataset_path", ""),
            treatment=treatment,
            outcome=outcome,
            controls=body.get("controls", []),
            strategy=body.get("strategy", "probit"),
            n_simulations=body.get("n_simulations", 500),
        )
        # §IMP: Benchmarked thresholds by field
        if isinstance(result, dict):
            delta = result.get("oster_delta", result.get("delta"))
            field = body.get("field", "economics")
            thresholds = {"economics": 1.0, "health": 2.0, "education": 1.5, "psychology": 1.0}
            threshold = thresholds.get(field.lower(), 1.0)
            if delta is not None:
                result["benchmark"] = {
                    "field": field, "threshold": threshold,
                    "robust": abs(float(delta)) > threshold,
                    "note": f"|δ| = {abs(float(delta)):.2f} vs {field} threshold of {threshold}",
                }
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/integrity")
async def sniper_integrity(request: Request):
    """Stage 3 only: Geospatial data integrity audit.

    Body:
        paper_text: str — paper text to extract locations from
        coordinates: list — optional pre-extracted [{lat, lon, name, context}]
    """
    body = await request.json()
    paper_text = body.get("paper_text", "")
    if not paper_text and not body.get("coordinates"):
        return JSONResponse(status_code=400, content={"error": "paper_text or coordinates required"})

    try:
        from server.sniper_audit import audit_data_integrity
        return audit_data_integrity(
            paper_text=paper_text,
            coordinates=body.get("coordinates"),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/defend")
async def sniper_defend(request: Request):
    """The Adversarial Defender — run the Sniper on YOUR OWN paper.

    Same as full audit but framed as self-defense:
    finds weaknesses before reviewers do.

    Body:
        paper_text: str — your own paper/draft
        dataset_path: str — your replication dataset
        n_simulations: int — Monte Carlo iterations
    """
    body = await request.json()
    paper_text = body.get("paper_text", "")
    if not paper_text:
        return JSONResponse(status_code=400, content={"error": "paper_text required"})

    try:
        from server.sniper_audit import full_sniper_audit
        result = full_sniper_audit(
            paper_text=paper_text,
            dataset_path=body.get("dataset_path", ""),
            n_simulations=body.get("n_simulations", 1000),  # Higher for self-defense
        )
        result["mode"] = "adversarial_defender"
        result["note"] = (
            "Self-audit complete. Any FAIL verdicts should be addressed "
            "before submission to make your work un-debunkable."
        )
        # §IMP: Auto-fix suggestions per weakness
        fixes = []
        for stage_name, stage_data in result.get("stages", {}).items():
            if isinstance(stage_data, dict) and stage_data.get("verdict") == "FAIL":
                weakness = stage_data.get("reason", stage_data.get("detail", ""))
                fix = {"stage": stage_name, "weakness": weakness}
                if "omitted" in weakness.lower() or "variable" in weakness.lower():
                    fix["suggestion"] = "Add the identified omitted variable as a control in your specification."
                elif "sensitivity" in weakness.lower() or "delta" in weakness.lower():
                    fix["suggestion"] = "Run additional robustness checks (IV, bounds, alternative samples)."
                elif "geo" in stage_name.lower() or "satellite" in weakness.lower():
                    fix["suggestion"] = "Cross-reference claimed location data with satellite imagery dates."
                else:
                    fix["suggestion"] = "Review methodology for this stage and strengthen the evidential claim."
                fixes.append(fix)
        result["auto_fix_suggestions"] = fixes

        # §BUS: Sniper weakness → EventBus (replaces old bridge)
        try:
            from server.event_bus import bus
            for fix in fixes:
                asyncio.ensure_future(bus.emit("sniper.weakness", {
                    "description": fix.get("weakness", ""),
                    "original_claim": fix.get("stage", ""),
                    "suggestion": fix.get("suggestion", ""),
                }, source="sniper"))
        except Exception:
            pass

        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

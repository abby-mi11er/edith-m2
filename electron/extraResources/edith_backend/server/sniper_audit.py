"""
Methodological Sniper — Forensic Audit of Rival Research
===========================================================
Three-stage tactical audit that finds the single mathematical or
logical thread that unravels a rival's argument.

Stage 1: Causal Extraction — NLP → DAG → backdoor path detection
Stage 2: Statistical Re-Simulation — MLE recreation + sensitivity analysis
Stage 3: Data Integrity Audit — geospatial + cross-reference verification

Uses: AnthropicBridge | Gemini, StataBridge, DAGittyBridge, GoogleEarthBridge

Exposed as:
  POST /api/sniper/audit         — full 3-stage forensic audit
  POST /api/sniper/extract-dag   — Stage 1 only: causal extraction
  POST /api/sniper/sensitivity   — Stage 2 only: Oster/Monte Carlo
  POST /api/sniper/integrity     — Stage 3 only: data integrity
"""

import json
import logging
import os
import re
import time
from typing import Optional

log = logging.getLogger("edith.sniper_audit")


# ═══════════════════════════════════════════════════════════════════
# PDF Extraction — accept file path, auto-extract text
# ═══════════════════════════════════════════════════════════════════

def pdf_to_text(pdf_path: str) -> str:
    """Extract text from a PDF file using the existing forensic pipeline."""
    if not pdf_path or not os.path.isfile(pdf_path):
        return ""
    try:
        from server.forensic_audit import extract_text_from_pdf
        return extract_text_from_pdf(pdf_path)
    except ImportError:
        # Fallback: try PyMuPDF directly
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception:
            return ""
    except Exception as e:
        log.warning(f"PDF extraction failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Causal Extraction — extract the paper's DAG
# ═══════════════════════════════════════════════════════════════════

def extract_causal_structure(paper_text: str, max_chars: int = 15000) -> dict:
    """Extract the identification strategy and causal DAG from a paper.

    Uses LLM to parse methodology, then structures as a DAG
    for backdoor path analysis.
    """
    text = paper_text[:max_chars]

    prompt = (
        "You are a political science methodologist performing a forensic audit.\n\n"
        "Analyze this paper's methodology and extract:\n"
        "1. DEPENDENT VARIABLE (Y): The outcome being measured\n"
        "2. TREATMENT/INDEPENDENT VARIABLE (X): The main explanatory variable\n"
        "3. CONTROL VARIABLES (Z): Variables the author controls for\n"
        "4. IDENTIFICATION STRATEGY: How they claim causality (RCT, DiD, IV, RDD, MLE, OLS, etc.)\n"
        "5. POTENTIAL CONFOUNDERS: Variables NOT controlled for that could bias results\n"
        "6. DAG EDGES: List directed edges as JSON: [{\"from\": \"A\", \"to\": \"B\"}, ...]\n"
        "7. SUSPECTED BACKDOOR PATHS: Any unblocked paths from X to Y through confounders\n\n"
        "Return as JSON with keys: dependent_var, treatment_var, controls, strategy, "
        "confounders, dag_edges, backdoor_paths, fragility_notes\n\n"
        f"PAPER TEXT:\n{text}"
    )

    try:
        from server.backend_logic import generate_text_via_chain
        model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
        response, model = generate_text_via_chain(prompt, model_chain, temperature=0.1)

        # Parse JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            structure = json.loads(json_match.group())
        else:
            structure = {"raw_analysis": response}

        structure["model_used"] = model
        structure["stage"] = "causal_extraction"

        # Run DAGitty if edges were extracted
        if structure.get("dag_edges"):
            try:
                from server.dagitty_bridge import DAGittyBridge
                dag = DAGittyBridge()
                treatment = structure.get("treatment_var", "X")
                outcome = structure.get("dependent_var", "Y")
                edges = structure["dag_edges"]
                variables = list(set(
                    [e.get("from", "") for e in edges] + [e.get("to", "") for e in edges]
                ))

                dag_result = dag.build_dag(variables, edges)
                structure["dag_analysis"] = dag_result

                # Find adjustment sets
                dag_spec = dag_result.get("spec", "")
                if dag_spec and treatment and outcome:
                    adj = dag.find_adjustment_sets(dag_spec, treatment, outcome)
                    structure["adjustment_sets"] = adj
                    backdoors = dag.find_backdoor_paths(dag_spec, treatment, outcome)
                    structure["backdoor_analysis"] = backdoors
            except Exception as e:
                structure["dag_analysis_error"] = str(e)

        return structure

    except Exception as e:
        return {"error": str(e), "stage": "causal_extraction"}


# ═══════════════════════════════════════════════════════════════════
# Library Cross-Reference — find confounders from user's own papers
# ═══════════════════════════════════════════════════════════════════

def cross_reference_library(
    treatment_var: str,
    outcome_var: str,
    author_controls: list[str] = None,
) -> dict:
    """Query the user's indexed library (OpenAlex + local vault) to find
    known confounders that the rival paper missed.

    Checks:
    1. OpenAlex for papers studying the same X→Y relationship
    2. Local ChromaDB vault for variables mentioned in user's notes
    3. Compares author's control set against the literature's control set
    """
    author_controls = author_controls or []
    result = {"stage": "library_cross_reference", "missing_controls": [], "evidence": []}

    # Query OpenAlex for papers on the same topic
    query = f"{treatment_var} {outcome_var} causal effect control variables"
    try:
        from server.openalex import search_openalex
        papers = search_openalex(query, max_results=20)
        result["openalex_papers_found"] = len(papers) if papers else 0
    except Exception:
        papers = []
        result["openalex_note"] = "OpenAlex unavailable"

    # Use LLM to identify what the literature typically controls for
    if papers or treatment_var:
        try:
            from server.backend_logic import generate_text_via_chain
            model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

            papers_context = ""
            if papers:
                papers_context = "\n".join(
                    f"- {p.get('title', 'Untitled')} ({p.get('year', '?')})"
                    for p in papers[:10]
                )

            prompt = (
                f"In political science research studying the effect of '{treatment_var}' "
                f"on '{outcome_var}', what variables are typically controlled for?\n\n"
                f"The author of the paper being audited controls for: {', '.join(author_controls) if author_controls else 'NONE'}\n\n"
                f"{'Related papers found in the literature:' + chr(10) + papers_context if papers_context else ''}\n\n"
                f"Return a JSON object with:\n"
                f'- "standard_controls": list of variables typically controlled for\n'
                f'- "missing_from_author": list of variables the author should have controlled for but didn\'t\n'
                f'- "backdoor_risk": list of potential confounders that create backdoor paths\n'
                f'- "severity": "critical"/"moderate"/"minor"'
            )

            response, _ = generate_text_via_chain(prompt, model_chain, temperature=0.1)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                literature = json.loads(json_match.group())
                result["literature_controls"] = literature.get("standard_controls", [])
                result["missing_controls"] = literature.get("missing_from_author", [])
                result["backdoor_risk"] = literature.get("backdoor_risk", [])
                result["severity"] = literature.get("severity", "unknown")
        except Exception as e:
            result["cross_ref_error"] = str(e)

    # Also check local ChromaDB for variables in user's notes
    try:
        from server.retrieval_enhancements import retrieve_local_sources
        local_hits = retrieve_local_sources(
            queries=[f"{treatment_var} {outcome_var} confounders omitted variable bias"],
            chroma_dir=os.environ.get("CHROMA_DIR", ""),
            collection_name=os.environ.get("CHROMA_COLLECTION", "edith_vault"),
            embed_model=os.environ.get("EMBED_MODEL", "models/text-embedding-004"),
            top_k=5,
        )
        if local_hits:
            result["vault_evidence"] = [
                {"text": h.get("text", "")[:200], "source": h.get("source", "")} 
                for h in local_hits[:3]
            ]
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Statistical Re-Simulation — MLE + Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════

def run_sensitivity_analysis(
    dataset_path: str = "",
    treatment: str = "",
    outcome: str = "",
    controls: list[str] = None,
    strategy: str = "probit",
    n_simulations: int = 1000,
) -> dict:
    """Re-simulate a model and run sensitivity tests.

    Generates Stata code for:
    1. Original model estimation
    2. Oster sensitivity test (delta calculation)
    3. Monte Carlo perturbation (fragility index)
    4. Leave-one-out influence analysis
    """
    controls = controls or []
    # Coerce control items to strings — LLM may return dicts like {"name": "income"}
    controls = [
        (str(c.get("name") or c.get("variable") or c) if isinstance(c, dict) else str(c))
        for c in controls
    ]
    controls_str = " ".join(controls)

    # Build the comprehensive Stata audit script
    stata_code = f"""
* ══════════════════════════════════════════════════════════════
* E.D.I.T.H. Methodological Sniper — Sensitivity Audit
* ══════════════════════════════════════════════════════════════
clear all
set more off
set seed 42

* ── Load Data ────────────────────────────────────────────────
{"use " + '"' + dataset_path + '"' + ", clear" if dataset_path else "* No dataset specified — using current data"}

* ── Descriptive Statistics ───────────────────────────────────
summarize {outcome} {treatment} {controls_str}
correlate {outcome} {treatment} {controls_str}

* ── Stage 1: Original Model (Author's Specification) ────────
display "=== ORIGINAL MODEL ==="
{strategy} {outcome} {treatment} {controls_str}, robust
estimates store original
display "R-squared / Pseudo R-squared:"
ereturn list

* ── Stage 2: Oster Sensitivity Test ─────────────────────────
* Calculate how strong unobserved confounding would need to be
* to eliminate the treatment effect (delta)
display "=== OSTER SENSITIVITY TEST ==="

* Restricted model (without controls)
{strategy} {outcome} {treatment}, robust
estimates store restricted
scalar b_restricted = _b[{treatment}]

* Full model (with controls)
{strategy} {outcome} {treatment} {controls_str}, robust
estimates store full_model
scalar b_full = _b[{treatment}]

* Calculate proportional selection (delta approximation)
* delta = (b_full) / (b_restricted - b_full)
* If |delta| < 1, result is fragile to unobserved confounding
scalar delta = b_full / (b_restricted - b_full)
display "Delta (proportional selection): " delta
display "Interpretation: If |delta| < 1, result is FRAGILE"
display "Current |delta| = " abs(delta)

* ── Stage 3: Monte Carlo Fragility Test ─────────────────────
display "=== MONTE CARLO FRAGILITY TEST ==="
display "Running {min(n_simulations, 500)} perturbations..."

tempname sim_results
postfile `sim_results' b_treatment se_treatment p_value using sim_output, replace

forvalues i = 1/{min(n_simulations, 500)} {{
    preserve
    * Add random noise to treatment (±5% perturbation)
    gen noise = rnormal(0, 0.05 * abs({treatment}))
    replace {treatment} = {treatment} + noise
    quietly {strategy} {outcome} {treatment} {controls_str}, robust
    post `sim_results' (_b[{treatment}]) (_se[{treatment}]) (2*ttail(e(df_r), abs(_b[{treatment}]/_se[{treatment}])))
    restore
}}

postclose `sim_results'
use sim_output, clear
summarize b_treatment p_value

* Fragility Index: % of simulations where p > 0.05
count if p_value > 0.05
scalar pct_insignificant = r(N) / _N * 100
display "Fragility Index: " pct_insignificant "% of simulations lost significance"
display "Interpretation: >" 20 "% = FRAGILE, <" 5 "% = ROBUST"

* ── Stage 4: Leave-One-Out Influence ────────────────────────
display "=== INFLUENCE ANALYSIS ==="
{"use " + '"' + dataset_path + '"' + ", clear" if dataset_path else ""}
{strategy} {outcome} {treatment} {controls_str}, robust
predict leverage, hat
predict cooksd, cooksd
summarize leverage cooksd
display "Max Cook's D: " r(max)
display "Interpretation: Cook's D > " 4/_N " suggests influential outlier"
list if cooksd > 4/_N, noobs clean
"""

    # Execute via Stata bridge
    try:
        from server.stata_bridge import StataBridge
        bridge = StataBridge()
        status = bridge.status()
        if not status.get("available"):
            return {
                "stage": "sensitivity_analysis",
                "stata_available": False,
                "stata_code": stata_code,
                "note": "Stata not available — code generated but not executed. "
                        "Set STATA_PATH to enable live execution.",
            }

        result = bridge.execute(stata_code, timeout=300)
        output = result.get("output", "")

        # Parse key metrics from output
        audit = {
            "stage": "sensitivity_analysis",
            "stata_executed": True,
            "strategy": strategy,
            "treatment": treatment,
            "outcome": outcome,
            "controls": controls,
            "raw_output": output,
        }

        # Extract delta
        delta_match = re.search(r"Current \|delta\| = ([\d.]+)", output)
        if delta_match:
            delta_val = float(delta_match.group(1))
            audit["oster_delta"] = delta_val
            audit["oster_verdict"] = "FRAGILE" if delta_val < 1.0 else "ROBUST"

        # Extract fragility index
        frag_match = re.search(r"Fragility Index: ([\d.]+)%", output)
        if frag_match:
            frag_val = float(frag_match.group(1))
            audit["fragility_index"] = frag_val
            audit["fragility_verdict"] = (
                "CRITICAL — FRAGILE" if frag_val > 20 else
                "MARGINAL" if frag_val > 5 else
                "ROBUST"
            )

        # Extract Cook's D
        cooksd_match = re.search(r"Max Cook's D:\s*([\d.]+)", output)
        if cooksd_match:
            audit["max_cooks_d"] = float(cooksd_match.group(1))
            audit["influential_outliers"] = float(cooksd_match.group(1)) > 0.5

        return audit

    except Exception as e:
        return {
            "stage": "sensitivity_analysis",
            "error": str(e),
            "stata_code": stata_code,
        }


# ═══════════════════════════════════════════════════════════════════
# Stage 3: Data Integrity Audit — geospatial cross-reference
# ═══════════════════════════════════════════════════════════════════

def audit_data_integrity(
    paper_text: str,
    coordinates: list[dict] = None,
    max_chars: int = 10000,
) -> dict:
    """Verify data claims in a paper using satellite + bibliometric data.

    Checks:
    - Physical locations mentioned in the paper against satellite imagery
    - Cross-references citations against OpenAlex for retraction/correction
    - Flags data inconsistencies
    """
    audit = {"stage": "data_integrity", "checks": []}

    # Extract coordinates from paper if not provided
    if not coordinates:
        coord_prompt = (
            "Extract any geographic coordinates, city names, or specific locations "
            "mentioned in this research paper. Return as JSON array: "
            '[{"name": "location", "lat": 0.0, "lon": 0.0, "context": "why mentioned"}]\n\n'
            f"PAPER:\n{paper_text[:max_chars]}"
        )
        try:
            from server.backend_logic import generate_text_via_chain
            model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
            response, _ = generate_text_via_chain(coord_prompt, model_chain, temperature=0.1)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                coordinates = json.loads(json_match.group())
        except Exception as e:
            audit["coordinate_extraction_error"] = str(e)
            coordinates = []

    # Check each location via Google Earth Engine
    if coordinates:
        try:
            from server.google_earth_bridge import GoogleEarthBridge
            gee = GoogleEarthBridge()
            if gee.status().get("available"):
                for loc in coordinates[:5]:  # Cap at 5 locations
                    lat = loc.get("lat")
                    lon = loc.get("lon")
                    if lat and lon:
                        gee_result = gee.audit_location(
                            lat=float(lat), lon=float(lon),
                            claim=loc.get("context", ""),
                        )
                        audit["checks"].append({
                            "location": loc.get("name", f"{lat},{lon}"),
                            "context": loc.get("context", ""),
                            "satellite_audit": gee_result,
                        })
            else:
                audit["gee_note"] = "Google Earth Engine not available — skipping satellite checks"
        except Exception as e:
            audit["gee_error"] = str(e)

    return audit


# ═══════════════════════════════════════════════════════════════════
# Full Sniper Audit — all three stages
# ═══════════════════════════════════════════════════════════════════

def full_sniper_audit(
    paper_text: str = "",
    pdf_path: str = "",
    dataset_path: str = "",
    coordinates: list[dict] = None,
    n_simulations: int = 500,
) -> dict:
    """Execute the full 3-stage Methodological Sniper audit.

    Accepts either raw paper_text or a pdf_path (auto-extracts text).
    Returns a tactical briefing with verdicts for each audit dimension.
    """
    t0 = time.time()

    # #2: Auto-extract text from PDF if path provided
    if not paper_text and pdf_path:
        paper_text = pdf_to_text(pdf_path)
        if not paper_text:
            return {"error": f"Could not extract text from PDF: {pdf_path}"}

    if not paper_text:
        return {"error": "paper_text or pdf_path required"}

    briefing = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}

    # Stage 1: Causal Extraction
    log.info("Sniper Stage 1: Causal Extraction")
    stage1 = extract_causal_structure(paper_text)
    briefing["stages"]["causal_extraction"] = stage1

    # Stage 1.5: Library Cross-Reference (#3)
    treatment = stage1.get("treatment_var", "")
    outcome = stage1.get("dependent_var", "")
    controls = stage1.get("controls", [])
    if isinstance(controls, str):
        controls = [c.strip() for c in controls.split(",") if c.strip()]
    elif not isinstance(controls, list):
        controls = []
    strategy = stage1.get("strategy", "probit")
    if isinstance(strategy, dict):
        strategy = str(strategy.get("name") or strategy.get("type") or "probit")
    elif not isinstance(strategy, str):
        strategy = str(strategy)

    if treatment and outcome:
        log.info("Sniper Stage 1.5: Library Cross-Reference")
        xref = cross_reference_library(treatment, outcome, controls if isinstance(controls, list) else [])
        briefing["stages"]["library_cross_reference"] = xref

        # Merge library-found confounders into Stage 1 results
        if xref.get("missing_controls"):
            existing_confounders = stage1.get("confounders", [])
            combined = list(set(existing_confounders + xref["missing_controls"]))
            stage1["confounders"] = combined
            stage1["library_augmented"] = True

    # Normalize strategy name for Stata
    strategy_map = {
        "ols": "regress", "logistic": "logit", "logit": "logit",
        "probit": "probit", "mle": "probit", "did": "regress",
        "iv": "ivregress 2sls", "rdd": "regress", "fe": "xtreg",
    }
    stata_cmd = strategy_map.get(strategy.lower().split()[0] if strategy else "probit", "probit")

    # Stage 2: Statistical Re-Simulation
    log.info(f"Sniper Stage 2: Sensitivity Analysis ({stata_cmd})")
    stage2 = run_sensitivity_analysis(
        dataset_path=dataset_path,
        treatment=treatment,
        outcome=outcome,
        controls=[str(c.get("name") or c) if isinstance(c, dict) else str(c) for c in (controls if isinstance(controls, list) else [controls])],
        strategy=stata_cmd,
        n_simulations=n_simulations,
    )
    briefing["stages"]["sensitivity_analysis"] = stage2

    # Stage 3: Data Integrity
    log.info("Sniper Stage 3: Data Integrity Audit")
    stage3 = audit_data_integrity(paper_text, coordinates)
    briefing["stages"]["data_integrity"] = stage3

    # Compile Tactical Briefing
    briefing["elapsed_seconds"] = round(time.time() - t0, 1)
    briefing["tactical_summary"] = _compile_tactical_summary(stage1, stage2, stage3)

    return briefing


def _compile_tactical_summary(stage1: dict, stage2: dict, stage3: dict) -> dict:
    """Compile a one-page tactical summary from all three stages."""
    verdicts = []

    # Causal verdict
    backdoors = stage1.get("backdoor_analysis", {}).get("paths", [])
    confounders = stage1.get("confounders", [])
    if backdoors:
        verdicts.append({
            "audit": "Causal Path", "claim": "No confounders",
            "finding": f"{len(backdoors)} backdoor path(s) found", "result": "FAIL",
        })
    elif confounders:
        verdicts.append({
            "audit": "Causal Path", "claim": "Complete controls",
            "finding": f"{len(confounders)} potential confounder(s) missed",
            "result": "WARNING",
        })
    else:
        verdicts.append({
            "audit": "Causal Path", "claim": "No confounders",
            "finding": "No obvious backdoor paths detected", "result": "PASS",
        })

    # Sensitivity verdict
    delta = stage2.get("oster_delta")
    if delta is not None:
        verdicts.append({
            "audit": "Selection Bias (Oster)",
            "claim": "Robust results",
            "finding": f"δ = {delta:.2f} {'(< 1.0 = FRAGILE)' if delta < 1.0 else '(≥ 1.0 = stable)'}",
            "result": "FAIL" if delta < 1.0 else "PASS",
        })

    frag = stage2.get("fragility_index")
    if frag is not None:
        verdicts.append({
            "audit": "Monte Carlo Fragility",
            "claim": "Stable under perturbation",
            "finding": f"{frag:.1f}% of simulations lost significance",
            "result": "FAIL" if frag > 20 else "WARNING" if frag > 5 else "PASS",
        })

    if stage2.get("influential_outliers"):
        verdicts.append({
            "audit": "Influential Outliers",
            "claim": "No outlier dependency",
            "finding": f"Cook's D = {stage2.get('max_cooks_d', 0):.4f} (high influence)",
            "result": "FAIL",
        })

    # Data integrity verdict
    for check in stage3.get("checks", []):
        sat = check.get("satellite_audit", {})
        if sat.get("change_detected", {}).get("interpretation", "").startswith("Possible demolition"):
            verdicts.append({
                "audit": "Data Integrity",
                "claim": f"Location: {check.get('location', 'unknown')}",
                "finding": "Satellite shows possible demolition/clearing",
                "result": "FAIL",
            })

    fails = sum(1 for v in verdicts if v["result"] == "FAIL")
    warnings = sum(1 for v in verdicts if v["result"] == "WARNING")

    return {
        "verdicts": verdicts,
        "fails": fails,
        "warnings": warnings,
        "overall": (
            "CRITICAL — Multiple fatal flaws detected" if fails >= 2 else
            "VULNERABLE — Key weakness identified" if fails == 1 else
            "MARGINAL — Concerns but no fatal flaws" if warnings > 0 else
            "ROBUST — No significant issues found"
        ),
    }

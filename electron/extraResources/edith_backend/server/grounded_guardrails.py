"""
Grounded Guardrails — Keeping Winnie Sharp
=============================================
The "Stay Sharp" middleware that prevents semantic drift and ensures
Winnie remains an academic powerhouse rather than a generic AI.

Features:
  - Citation or Silence Rule — RAG-only enforcement
  - Persona Drift Audit — monthly persona benchmarking
  - Methodological Hawk Middleware — code rigor enforcement
  - Literature Stress Test — Winnie sharpness verification
  - Mandatory Citation Middleware — no uncited claims allowed
"""

import json
import logging
import os
import re
import time
from typing import Optional

log = logging.getLogger("edith.guardrails")


# ═══════════════════════════════════════════════════════════════════
# CITATION OR SILENCE — RAG-Only Enforcement
# "If she can't find a source on the Bolt, she must say so."
# ═══════════════════════════════════════════════════════════════════

def enforce_rag_only(
    response: str,
    sources: list[dict],
    min_sources: int = 1,
    strict: bool = True,
) -> dict:
    """Enforce the "Citation or Silence" rule.

    If the response lacks grounding in retrieved sources,
    either inject a disclaimer or block the response entirely.
    """
    has_sources = len(sources) >= min_sources

    # Check if response references any of the provided sources
    source_terms = set()
    for src in sources:
        # Handle both dict sources and plain strings
        if isinstance(src, str):
            src = {"text": src}
        meta = src.get("metadata", {})
        author = meta.get("author", "")
        title = meta.get("title", "")
        if author:
            source_terms.add(author.lower().split()[-1])  # Last name
        if title:
            source_terms.update(w.lower() for w in title.split() if len(w) > 4)

    response_lower = response.lower()
    cited_terms = [t for t in source_terms if t in response_lower]
    citation_ratio = len(cited_terms) / max(len(source_terms), 1)

    if not has_sources:
        if strict:
            return {
                "allowed": False,
                "response": (
                    "⚠️ **Citation Required**: I don't have the literature on your "
                    "Oyen Bolt to support a grounded answer to that question. "
                    "Please add the relevant papers to your library and re-ask."
                ),
                "reason": "no_sources_found",
                "original_response": response,
            }
        else:
            disclaimer = (
                "\n\n> ⚠️ *This response is not grounded in your local library. "
                "Consider adding relevant sources to your Bolt for verification.*"
            )
            return {
                "allowed": True,
                "response": response + disclaimer,
                "reason": "ungrounded_with_disclaimer",
                "citation_ratio": round(citation_ratio, 2),
            }

    return {
        "allowed": True,
        "response": response,
        "reason": "grounded",
        "sources_found": len(sources),
        "citation_ratio": round(citation_ratio, 2),
        "cited_terms": cited_terms[:10],
    }


# ═══════════════════════════════════════════════════════════════════
# PERSONA DRIFT AUDIT — Monthly Benchmarking
# Tests persona accuracy against "Gold Standard" prompts
# ═══════════════════════════════════════════════════════════════════

GOLD_STANDARD_BENCHMARKS = {
    "winnie": {
        "prompt": "What is the 'submerged state' according to Suzanne Mettler?",
        "must_contain": ["hidden", "invisible", "tax expenditure", "indirect",
                         "subsidized", "government role"],
        "must_not_contain": ["I don't know", "I'm not sure"],
    },
    "professor_stern": {
        "prompt": "Evaluate a student's claim: 'SNAP reduces poverty.'",
        "must_contain": ["identification", "endogen", "counterfactual",
                         "selection", "causal"],
        "must_not_contain": ["great point", "absolutely", "that's correct"],
    },
    "methods_tutor": {
        "prompt": "Explain instrumental variable estimation to a beginner.",
        "must_contain": ["instrument", "exogen", "exclusion", "two-stage",
                         "correlated"],
        "must_not_contain": ["obviously", "simply"],
    },
}


def run_persona_drift_audit(
    persona_key: str = "",
    model_chain: list[str] = None,
) -> dict:
    """Run a Persona Drift Audit — test if persona is still "sharp."

    Compare against Gold Standard benchmarks and flag drift.
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

    if persona_key:
        benchmarks = {persona_key: GOLD_STANDARD_BENCHMARKS.get(persona_key)}
        if not benchmarks[persona_key]:
            return {"error": f"No benchmark for persona: {persona_key}"}
    else:
        benchmarks = GOLD_STANDARD_BENCHMARKS

    results = {}
    for key, benchmark in benchmarks.items():
        if not benchmark:
            continue

        try:
            from server.cognitive_engine import PERSONA_VAULT, switch_persona
            if key in PERSONA_VAULT:
                switch_persona(key)

            from server.backend_logic import generate_text_via_chain
            persona_sys = PERSONA_VAULT.get(key, {}).get("system", "")
            response, model = generate_text_via_chain(
                benchmark["prompt"], model_chain,
                system_instruction=persona_sys,
                temperature=0.1,
            )

            # Score the response
            resp_lower = response.lower()
            hits = [w for w in benchmark["must_contain"] if w in resp_lower]
            violations = [w for w in benchmark["must_not_contain"] if w in resp_lower]

            score = (len(hits) / max(len(benchmark["must_contain"]), 1)) * 100
            drift_detected = score < 60 or len(violations) > 0

            results[key] = {
                "score": round(score, 1),
                "hits": hits,
                "misses": [w for w in benchmark["must_contain"]
                          if w not in resp_lower],
                "violations": violations,
                "drift_detected": drift_detected,
                "recommendation": (
                    "SHARP ✅" if not drift_detected
                    else f"DRIFT ⚠️ — re-weight against source material "
                         f"(missing: {', '.join(benchmark['must_contain'][:3])})"
                ),
                "response_preview": response[:200],
            }
        except Exception as e:
            results[key] = {"error": str(e)}

    # Reset to winnie
    try:
        from server.cognitive_engine import switch_persona
        switch_persona("winnie")
    except Exception:
        pass

    drifted = sum(1 for r in results.values()
                  if isinstance(r, dict) and r.get("drift_detected"))
    return {
        "results": results,
        "personas_tested": len(results),
        "drifted": drifted,
        "all_sharp": drifted == 0,
    }


# ═══════════════════════════════════════════════════════════════════
# METHODOLOGICAL HAWK — Code Rigor Middleware
# Every code snippet passes through the Hawk before reaching the user
# ═══════════════════════════════════════════════════════════════════

HAWK_CHECKS = [
    {
        "pattern": r"\bOLS\b.*(?!cluster|robust|HC\d)",
        "issue": "OLS without clustered/robust standard errors",
        "severity": "high",
        "suggestion": "Consider vce(cluster) or HC3 robust SEs if data is clustered",
    },
    {
        "pattern": r"(?:regress|lm\(|ols\().*(?:year|time).*(?:state|county|fips)",
        "issue": "Possible TWFE specification — check for staggered treatment",
        "severity": "medium",
        "suggestion": "If treatment adoption is staggered, consider Callaway-Sant'Anna or Sun-Abraham",
    },
    {
        "pattern": r"(?:log|ln)\s*\(",
        "issue": "Log transformation detected — check for zeros",
        "severity": "medium",
        "suggestion": "Use IHS (inverse hyperbolic sine) if data contains zeros",
    },
    {
        "pattern": r"drop\s+if|\.dropna|\.drop\(",
        "issue": "Dropping observations — potential selection bias",
        "severity": "high",
        "suggestion": "Document attrition pattern; consider bounds analysis or Lee bounds",
    },
    {
        "pattern": r"stepwise|forward|backward",
        "issue": "Stepwise selection detected — strong p-hacking risk",
        "severity": "critical",
        "suggestion": "Use LASSO/Ridge with cross-validation instead of stepwise",
    },
    {
        "pattern": r"(?:p\s*[<>]\s*0\.1|marginally significant|approaching significance)",
        "issue": "Marginal significance language — p-hacking red flag",
        "severity": "high",
        "suggestion": "Report exact p-values; don't cherry-pick thresholds",
    },
    {
        "pattern": r"(?:xtreg|xtlogit|mixed|lmer)\b",
        "issue": "Panel/multilevel model — verify nesting structure",
        "severity": "info",
        "suggestion": "Confirm sufficient clusters (>30) at each level; check ICC",
    },
    {
        "pattern": r"(?:ivregress|ivreg2|2sls|tsls)",
        "issue": "IV estimation — verify instrument validity",
        "severity": "high",
        "suggestion": "Run first-stage F-test (>10); test exclusion restriction; check overid",
    },
    {
        "pattern": r"(?:rdrobust|rdplot|rdd|discontinuity)",
        "issue": "RDD design — verify assumptions",
        "severity": "medium",
        "suggestion": "Check McCrary density test; verify no manipulation at cutoff",
    },
    {
        "pattern": r"(?:heteroskedastic|breusch|white\s*test)",
        "issue": "Heteroskedasticity concern flagged",
        "severity": "info",
        "suggestion": "Use HC-robust SEs; consider WLS if pattern is known",
    },
]


def methodological_hawk_review(code: str, language: str = "auto") -> dict:
    """Pass code through the Methodological Hawk.

    The Hawk's ONE job: find the most complex, annoying reason
    why the code might produce wrong results.
    """
    findings = []
    code_lower = code.lower()

    for check in HAWK_CHECKS:
        if re.search(check["pattern"], code, re.IGNORECASE):
            findings.append({
                "issue": check["issue"],
                "severity": check["severity"],
                "suggestion": check["suggestion"],
            })

    # Calculate risk level
    critical = sum(1 for f in findings if f["severity"] == "critical")
    high = sum(1 for f in findings if f["severity"] == "high")
    medium = sum(1 for f in findings if f["severity"] == "medium")

    if critical > 0:
        risk = "CRITICAL"
    elif high >= 2:
        risk = "HIGH"
    elif high >= 1 or medium >= 2:
        risk = "MEDIUM"
    elif findings:
        risk = "LOW"
    else:
        risk = "CLEAN"

    return {
        "risk_level": risk,
        "findings": findings,
        "total_flags": len(findings),
        "recommendation": (
            "✅ Code passes Hawk review" if risk == "CLEAN"
            else f"⚠️ {len(findings)} methodological concerns detected — review before running"
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# LITERATURE STRESS TEST — Verify Winnie is Still Sharp
# The "Am I Still Good?" health check for the AI
# ═══════════════════════════════════════════════════════════════════

STRESS_TEST_QUERIES = [
    {
        "query": "What is the relationship between SNAP and voter turnout?",
        "expected_fields": ["APE", "Voting"],
        "must_cite": ["Mettler", "Campbell"],
        "difficulty": "advanced",
    },
    {
        "query": "Compare DW-NOMINATE's spatial model with the cartel agenda model.",
        "expected_fields": ["Voting", "Methods"],
        "must_cite": ["Poole", "Rosenthal", "Cox"],
        "difficulty": "doctoral",
    },
    {
        "query": "How does criminal governance in Mexico relate to US charity operations?",
        "expected_fields": ["Criminal", "APE"],
        "must_cite": ["Arias", "Lessing"],
        "difficulty": "doctoral",
    },
]


def run_literature_stress_test(
    sources: list[dict] = None,
    model_chain: list[str] = None,
) -> dict:
    """Run a Literature Stress Test to verify Winnie's sharpness.

    Tests: cross-field retrieval, citation accuracy, depth of response.
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
    results = []

    for test in STRESS_TEST_QUERIES:
        t0 = time.time()
        test_result = {
            "query": test["query"],
            "difficulty": test["difficulty"],
        }

        # Test retrieval
        try:
            from server.chroma_backend import retrieve_local_sources
            retrieved = retrieve_local_sources(
                queries=[test["query"]],
                chroma_dir=os.environ.get("EDITH_CHROMA_DIR", ""),
                collection_name="edith",
                embed_model=os.environ.get("EDITH_EMBED_MODEL", ""),
                top_k=10,
            )
            test_result["sources_found"] = len(retrieved)
        except Exception:
            retrieved = []
            test_result["sources_found"] = 0

        # Check if expected authors appear in sources
        source_text = " ".join(
            (s.get("text", "") + str(s.get("metadata", {}))) for s in retrieved
        ).lower()
        cited = [a for a in test["must_cite"] if a.lower() in source_text]
        test_result["citations_found"] = cited
        test_result["citations_missing"] = [
            a for a in test["must_cite"] if a.lower() not in source_text
        ]

        # Score
        citation_score = len(cited) / max(len(test["must_cite"]), 1) * 100
        test_result["score"] = round(citation_score, 1)
        test_result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        test_result["pass"] = citation_score >= 50

        results.append(test_result)

    passed = sum(1 for r in results if r["pass"])
    return {
        "results": results,
        "tests_run": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "sharpness": "SHARP ✅" if passed == len(results) else f"NEEDS TUNING ⚠️ ({passed}/{len(results)})",
    }


# ═══════════════════════════════════════════════════════════════════
# MANDATORY CITATION MIDDLEWARE — wrap every response
# ═══════════════════════════════════════════════════════════════════

def citation_middleware(response: str, sources: list[dict]) -> str:
    """Add mandatory citation footnotes to a response.

    Scans for author mentions and appends inline citations.
    §FIX: Checks both top-level and metadata.author fields.
    §FIX: Fallback source list uses meaningful labels, not raw paths.
    """
    # Build author → source mapping (check both flat and nested structures)
    author_map = {}
    for i, src in enumerate(sources):
        meta = src.get("metadata", {}) if isinstance(src.get("metadata"), dict) else {}
        # Author can be at top level (enriched) or in metadata
        author = src.get("author", "") or meta.get("author", "")
        if author:
            last_name = author.split()[-1]
            if last_name not in author_map:
                author_map[last_name] = {
                    "full": author,
                    "source": src.get("title", "") or meta.get("source", meta.get("title", f"Source {i+1}")),
                    "page": src.get("page", "") or meta.get("page", ""),
                }

    # Add citation markers
    annotated = response
    citations_added = 0
    for last_name, info in author_map.items():
        if last_name in annotated and f"[{last_name}" not in annotated:
            page_ref = f", p. {info['page']}" if info["page"] else ""
            citation = f" [{info['full']}{page_ref}]"
            # Add citation after first mention
            annotated = annotated.replace(
                last_name, f"{last_name}{citation}", 1
            )
            citations_added += 1

    # §FIX: Build meaningful source labels for the "Sources consulted" fallback
    if citations_added == 0 and sources:
        source_labels = []
        for i, s in enumerate(sources[:5]):
            meta = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
            author = s.get("author", "") or meta.get("author", "")
            year = s.get("year", "") or meta.get("year", "")
            title = s.get("title", "") or meta.get("title", "")
            # If no title, try to extract from path
            if not title:
                path = (s.get("source", "") or s.get("path", "") or
                        meta.get("source", "") or meta.get("path", "") or
                        meta.get("rel_path", ""))
                if path:
                    fname = path.rsplit("/", 1)[-1] if "/" in path else path
                    import re as _re_cm
                    title = _re_cm.sub(r'\.(pdf|txt|docx|md|tex)$', '', fname, flags=_re_cm.IGNORECASE)
                    title = title.replace('_', ' ').replace('-', ' ').strip()
            # Build label
            if author and year:
                label = f"{author} ({year})"
            elif author:
                label = author
            elif title:
                label = f"{title} ({year})" if year else title
            else:
                label = f"Source {i+1}"
            source_labels.append(label[:60])
        annotated += "\n\n---\n**Sources consulted**: "
        annotated += "; ".join(source_labels)

    return annotated

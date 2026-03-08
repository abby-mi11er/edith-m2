"""
Proactive Suggestions — E.D.I.T.H.'s "What's Next?" Engine
============================================================
"After analyzing your data, you might also want to..."

Fires after tool completions and chat responses to suggest:
  - Related papers to read
  - Methods to try
  - Next steps in the research process
  - Connections to the user's dissertation
"""
import logging
import re

log = logging.getLogger("edith.suggestions")


# ═══════════════════════════════════════════════════════════════════
# §1: SUGGESTION TEMPLATES — rule-based fast suggestions
# No LLM needed — fires instantly after every interaction
# ═══════════════════════════════════════════════════════════════════

# Maps (detected pattern) → (suggestion type, template)
_SUGGESTION_RULES = [
    # After causal analysis → suggest robustness checks
    {
        "trigger": ["causal", "cause", "effect", "treatment", "outcome"],
        "suggestions": [
            {"type": "method", "text": "Consider a robustness check: try an alternative specification or placebo test."},
            {"type": "tool", "text": "Run a counterfactual analysis to stress-test this claim.", "action": "causal/counterfactual"},
        ],
    },
    # After DiD mention → suggest parallel trends
    {
        "trigger": ["difference-in-differences", "diff-in-diff", "did ", "parallel trends"],
        "suggestions": [
            {"type": "method", "text": "Check parallel trends assumption — plot pre-treatment trends for treatment and control groups."},
            {"type": "code", "text": "Generate a parallel trends plot in Stata.", "action": "antigrav/intent"},
        ],
    },
    # After regression → suggest diagnostics
    {
        "trigger": ["regression", "ols", "logit", "probit", "fixed effect"],
        "suggestions": [
            {"type": "method", "text": "Run standard diagnostics: heteroskedasticity, multicollinearity, and specification tests."},
            {"type": "code", "text": "Generate diagnostic tests for your model.", "action": "antigrav/intent"},
        ],
    },
    # After literature discussion → suggest related work
    {
        "trigger": ["literature", "paper", "study finds", "research shows", "scholar"],
        "suggestions": [
            {"type": "paper", "text": "Search for related papers on this topic to strengthen your argument."},
            {"type": "tool", "text": "Run a research gap analysis.", "action": "oracle/gaps"},
        ],
    },
    # After mentioning a dataset → suggest analysis
    {
        "trigger": ["dataset", "census", "survey", "panel data", "cross-section", "time series"],
        "suggestions": [
            {"type": "analysis", "text": "Explore descriptive statistics before running models — look at distributions, missingness, and outliers."},
            {"type": "code", "text": "Generate a data exploration script.", "action": "antigrav/intent"},
        ],
    },
    # After policy discussion → suggest counterfactual
    {
        "trigger": ["policy", "intervention", "program", "reform", "legislation"],
        "suggestions": [
            {"type": "method", "text": "Consider a counterfactual: what would have happened without this policy?"},
            {"type": "tool", "text": "Run causal forecasting for this scenario.", "action": "causal/forecast"},
        ],
    },
    # After writing/drafting → suggest peer review
    {
        "trigger": ["draft", "methods section", "results section", "introduction", "conclusion"],
        "suggestions": [
            {"type": "review", "text": "Submit this section for a faculty committee review."},
            {"type": "tool", "text": "Run peer review on your draft.", "action": "socratic/committee"},
        ],
    },
    # After code error → suggest self-heal
    {
        "trigger": ["error", "failed", "syntax error", "not working", "bug"],
        "suggestions": [
            {"type": "tool", "text": "Run self-heal to automatically diagnose and fix the error.", "action": "antigrav/heal"},
        ],
    },
]


def suggest_fast(text: str, context: dict = None) -> list[dict]:
    """Generate rule-based suggestions from text content.

    Fast — no LLM call. Runs in <1ms.
    Returns list of suggestion dicts.
    """
    text_lower = text.lower()
    suggestions = []
    seen_types = set()

    for rule in _SUGGESTION_RULES:
        # Check if any trigger matches
        if any(t in text_lower for t in rule["trigger"]):
            for s in rule["suggestions"]:
                # Avoid duplicate suggestion types
                if s["type"] not in seen_types:
                    suggestions.append(s)
                    seen_types.add(s["type"])

    # Personalize with profile context
    if context:
        dissertation = context.get("dissertation_topic", "")
        if dissertation and dissertation.lower() not in text_lower:
            suggestions.append({
                "type": "connection",
                "text": f"How does this relate to your dissertation on '{dissertation}'?",
            })

    return suggestions[:5]  # Max 5 suggestions


# ═══════════════════════════════════════════════════════════════════
# §2: LLM-POWERED SUGGESTIONS — deeper, personalized
# ═══════════════════════════════════════════════════════════════════

def suggest_deep(text: str, profile_context: str = "") -> list[dict]:
    """Generate LLM-powered suggestions based on content + profile.

    Slower (~5-10s) but much more personalized.
    """
    from server.agentic import safe_llm_call

    prompt = (
        f"Based on this research interaction, suggest 3 concrete next steps.\n\n"
        f"CONTENT:\n{text[:500]}\n\n"
        f"{profile_context + chr(10) + chr(10) if profile_context else ''}"
        f"Respond with exactly 3 suggestions, one per line, in this format:\n"
        f"TYPE: <paper|method|code|analysis|connection> | SUGGESTION: <concise text>\n"
    )

    result = safe_llm_call(
        prompt,
        system_instruction="You are a research advisor suggesting concrete next steps.",
        temperature=0.3,
        fallback_text="TYPE: method | SUGGESTION: Review your methodology section",
    )

    suggestions = []
    for line in result["text"].strip().split("\n"):
        line = line.strip()
        if "SUGGESTION:" in line:
            parts = line.split("|")
            stype = "method"
            stext = line
            for p in parts:
                p = p.strip()
                if p.upper().startswith("TYPE:"):
                    stype = p.split(":", 1)[1].strip().lower()
                elif p.upper().startswith("SUGGESTION:"):
                    stext = p.split(":", 1)[1].strip()
            suggestions.append({"type": stype, "text": stext})

    return suggestions[:3]


# ═══════════════════════════════════════════════════════════════════
# §3: POST-CHAT SUGGESTIONS — fire after chat response
# ═══════════════════════════════════════════════════════════════════

def get_chat_suggestions(query: str, answer: str, sources: list = None) -> list[dict]:
    """Generate suggestions after a chat response.

    Uses fast rule-based suggestions on combined query + answer text.
    """
    combined = f"{query}\n{answer}"

    # Get profile context for personalization
    profile_ctx = {}
    try:
        from server.research_profile import get_profile
        profile = get_profile()
        profile_ctx = profile.get_profile()
    except Exception:
        pass

    suggestions = suggest_fast(combined, context=profile_ctx)

    # Add source-based suggestions
    if sources:
        source_titles = [s.get("title", "") for s in sources[:5] if s.get("title")]
        if source_titles:
            suggestions.append({
                "type": "paper",
                "text": f"Explore related work: search for papers citing '{source_titles[0][:60]}'.",
            })

    return suggestions[:5]


# ═══════════════════════════════════════════════════════════════════
# §4: POST-PIPELINE SUGGESTIONS — fire after tool pipeline
# ═══════════════════════════════════════════════════════════════════

def get_pipeline_suggestions(intent: str, steps: list, context: dict) -> list[dict]:
    """Generate suggestions after a pipeline execution.

    More targeted — uses the specific intent and results.
    """
    suggestions = []

    # Intent-specific follow-ups
    follow_ups = {
        "causal_extract": [
            {"type": "tool", "text": "Build a causal graph from these claims.", "action": "causal/graph"},
            {"type": "tool", "text": "Stress-test the strongest claim.", "action": "causal/stress-test"},
        ],
        "causal_graph": [
            {"type": "tool", "text": "Run counterfactual analysis on key paths.", "action": "causal/counterfactual"},
            {"type": "method", "text": "Check if any paths suggest confounders — consider instrumental variables."},
        ],
        "code_generation": [
            {"type": "method", "text": "Review the generated code for robustness — add clustered standard errors."},
            {"type": "tool", "text": "Write up a research memo explaining the analysis.", "action": "antigrav/memo"},
        ],
        "synthesis": [
            {"type": "paper", "text": "Search for empirical studies that test this synthesis."},
            {"type": "tool", "text": "Identify research gaps in this area.", "action": "oracle/gaps"},
        ],
        "gap_analysis": [
            {"type": "method", "text": "Design a study to fill the most promising gap."},
            {"type": "tool", "text": "Generate a research design for the top gap.", "action": "antigrav/plan"},
        ],
    }

    if intent in follow_ups:
        suggestions.extend(follow_ups[intent])

    # Profile-based suggestions
    try:
        from server.research_profile import get_profile
        profile = get_profile().get_profile()
        dissertation = profile.get("dissertation_topic", "")
        if dissertation:
            suggestions.append({
                "type": "connection",
                "text": f"Consider how this connects to your dissertation on '{dissertation}'.",
            })
    except Exception:
        pass

    return suggestions[:4]

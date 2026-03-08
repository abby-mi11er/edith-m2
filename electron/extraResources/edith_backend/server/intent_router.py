"""
Intent Router — "Just tell Winnie what you want"
==================================================
Takes a natural language request, classifies the intent,
selects the right tools/pipeline, and executes it.

This is what makes E.D.I.T.H. feel like an agent instead
of a collection of buttons.
"""
import json
import logging
import os
import re
from typing import Optional

log = logging.getLogger("edith.intent_router")


# ═══════════════════════════════════════════════════════════════════
# §1: INTENT TAXONOMY
# Every tool E.D.I.T.H. has, classified by intent category
# ═══════════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    # ── Causal Analysis ──────────────────────────────────────────
    "causal_extract": {
        "signals": ["causal claim", "extract cause", "what causes", "identify causal",
                     "causal mechanism", "causal relationship"],
        "pipeline": ["causal/extract"],
        "description": "Extract causal claims from text",
    },
    "causal_graph": {
        "signals": ["causal graph", "causal map", "dag", "directed acyclic",
                     "causal diagram", "draw the causal"],
        "pipeline": ["causal/extract", "causal/graph"],
        "description": "Build a causal graph from claims",
    },
    "causal_stress": {
        "signals": ["stress test", "challenge", "falsify", "critique the causal",
                     "counterfactual", "what if"],
        "pipeline": ["causal/extract", "causal/stress-test"],
        "description": "Stress-test a causal claim",
    },
    "causal_full": {
        "signals": ["full causal analysis", "causal deep dive", "analyze the causal",
                     "causal evaluation"],
        "pipeline": ["causal/extract", "causal/graph", "causal/stress-test", "causal/counterfactual"],
        "description": "Full causal analysis pipeline",
    },
    "causal_forecast": {
        "signals": ["forecast", "predict", "what would happen if",
                     "project", "scenario analysis"],
        "pipeline": ["causal/forecast"],
        "description": "Causal forecasting / scenario analysis",
    },

    # ── Research & Writing ───────────────────────────────────────
    "code_generation": {
        "signals": ["write code", "generate script", "stata code", "r code",
                     "regression", "tab-to-intent", "run analysis",
                     "statistical script", "estimate model"],
        "pipeline": ["antigrav/intent"],
        "description": "Generate a statistical script from intent",
    },
    "artifact_plan": {
        "signals": ["write a plan", "outline", "structure", "organize",
                     "methods section", "results section", "draft"],
        "pipeline": ["antigrav/plan"],
        "description": "Generate an artifact plan / outline",
    },
    "research_memo": {
        "signals": ["research memo", "summarize findings", "write up results",
                     "interpret output", "explain results"],
        "pipeline": ["antigrav/memo"],
        "description": "Generate a research memo from analysis",
    },
    "self_heal": {
        "signals": ["fix error", "debug", "broken code", "error message",
                     "not working", "syntax error", "command failed"],
        "pipeline": ["antigrav/heal"],
        "description": "Diagnose and fix script errors",
    },

    # ── Oracle / Synthesis ───────────────────────────────────────
    "synthesis": {
        "signals": ["synthesize", "connect", "bridge between", "interdisciplinary",
                     "cross-domain", "link between", "relate to"],
        "pipeline": ["oracle/synthesis"],
        "description": "Cross-domain synthesis",
    },
    "gap_analysis": {
        "signals": ["research gap", "what's missing", "understudied",
                     "literature gap", "what hasn't been"],
        "pipeline": ["oracle/gaps"],
        "description": "Identify research gaps",
    },

    # ── Analysis ─────────────────────────────────────────────────
    "confidence_check": {
        "signals": ["how confident", "evidence strength", "confidence score",
                     "how reliable", "can we trust"],
        "pipeline": ["analysis/confidence"],
        "description": "Check confidence / evidence strength",
    },
    "contradiction_check": {
        "signals": ["contradiction", "conflicting", "inconsistent",
                     "disagree", "tension between"],
        "pipeline": ["analysis/contradictions"],
        "description": "Find contradictions in sources",
    },

    # ── Multi-step Compound Intents ──────────────────────────────
    "full_analysis": {
        "signals": ["analyze.*thoroughly", "full analysis", "deep dive",
                     "comprehensive analysis", "analyze everything"],
        "pipeline": ["causal/extract", "causal/graph", "analysis/confidence",
                      "oracle/gaps", "antigrav/memo"],
        "description": "Full comprehensive analysis pipeline",
    },
    "prepare_methods": {
        "signals": ["methods section", "methodology", "research design",
                     "prepare my methods"],
        "pipeline": ["causal/extract", "antigrav/intent", "antigrav/plan"],
        "description": "Prepare a methods section",
    },

    # ── Fallback: Chat ───────────────────────────────────────────
    "chat": {
        "signals": [],  # default fallback
        "pipeline": [],
        "description": "General chat — route to Winnie/Gemini",
    },
}


# ═══════════════════════════════════════════════════════════════════
# §2: PATTERN-BASED CLASSIFICATION (fast, no LLM needed)
# ═══════════════════════════════════════════════════════════════════

def classify_intent_fast(query: str) -> tuple[str, float]:
    """Classify intent using pattern matching. Returns (intent_key, confidence).

    Fast — no LLM call. Used as the primary classification layer.
    """
    q = query.lower().strip()
    best_intent = "chat"
    best_score = 0.0

    for intent_key, intent_def in INTENT_PATTERNS.items():
        if intent_key == "chat":
            continue
        signals = intent_def["signals"]
        matches = sum(1 for s in signals if s in q)
        if matches > 0:
            # Score: number of matches / total signals, scaled
            score = min(matches / max(len(signals) * 0.3, 1), 1.0)
            if score > best_score:
                best_score = score
                best_intent = intent_key

    return best_intent, round(best_score, 2)


# ═══════════════════════════════════════════════════════════════════
# §3: FUNCTION-CALLING CLASSIFICATION (Codex-style)
# The LLM picks which tool to call instead of outputting text.
# ═══════════════════════════════════════════════════════════════════

def _build_tool_schemas() -> list[dict]:
    """Generate OpenAI function-calling tool schemas from INTENT_PATTERNS."""
    tools = []
    for key, defn in INTENT_PATTERNS.items():
        if key == "chat" or not defn.get("pipeline"):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": key,
                "description": defn["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "One-line explanation of why this tool was chosen",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score from 0.0 to 1.0",
                        },
                    },
                    "required": ["reasoning", "confidence"],
                },
            },
        })
    return tools


TOOL_SCHEMAS = _build_tool_schemas()


def classify_intent_function_calling(query: str, context: str = "") -> tuple[str, float, str]:
    """Classify intent using OpenAI function calling.

    The LLM chooses which research tool to invoke — just like Codex.
    Returns (intent_key, confidence, reasoning).
    """
    try:
        import openai
        client = openai.OpenAI()

        messages = [
            {"role": "system", "content": (
                "You are the intent classifier for E.D.I.T.H., a political science research assistant. "
                "Given a user's research request, call the most appropriate tool. "
                "If none of the tools match, do NOT call any function."
            )},
        ]
        if context:
            messages.append({"role": "system", "content": f"Context: {context[:300]}"})
        messages.append({"role": "user", "content": query})

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_BASE_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=150,
        )

        choice = resp.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            intent = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            confidence = min(max(float(args.get("confidence", 0.8)), 0), 1)
            reasoning = args.get("reasoning", "Function calling selection")

            if intent in INTENT_PATTERNS:
                return intent, confidence, reasoning

        # No tool call = chat intent
        return "chat", 0.0, "No tool matched — general chat"

    except Exception as e:
        log.warning(f"Function calling classification failed: {e}")
        # Fall back to text-based classification
        return _classify_intent_llm_text(query, context)


def _classify_intent_llm_text(query: str, context: str = "") -> tuple[str, float, str]:
    """Fallback: text-based LLM classification if function calling fails."""
    from server.agentic import safe_llm_call

    intent_list = "\n".join(
        f"  {k}: {v['description']}"
        for k, v in INTENT_PATTERNS.items()
        if k != "chat"
    )
    prompt = (
        f"Classify this research request into exactly ONE intent category.\n\n"
        f"REQUEST: {query}\n\n"
        f"{'CONTEXT: ' + context[:300] + chr(10) + chr(10) if context else ''}"
        f"AVAILABLE INTENTS:\n{intent_list}\n\n"
        f"Respond in this exact format:\n"
        f"INTENT: <intent_key>\nCONFIDENCE: <0.0 to 1.0>\nREASONING: <one line>\n"
    )
    result = safe_llm_call(
        prompt,
        system_instruction="You are an intent classifier for E.D.I.T.H.",
        temperature=0.0,
        fallback_text="INTENT: chat\nCONFIDENCE: 0.0\nREASONING: classification failed",
    )
    text = result["text"]
    intent, confidence, reasoning = "chat", 0.0, ""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            parsed = line.split(":", 1)[1].strip().lower()
            if parsed in INTENT_PATTERNS:
                intent = parsed
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return intent, confidence, reasoning


# ═══════════════════════════════════════════════════════════════════
# §4: UNIFIED ROUTER — Fast first, LLM if uncertain
# ═══════════════════════════════════════════════════════════════════

def route_intent(
    query: str,
    context: str = "",
    fast_threshold: float = 0.5,
    use_llm_fallback: bool = True,
) -> dict:
    """Classify intent and return routing decision.

    Returns:
        {
            "intent": str,           # intent key
            "pipeline": list[str],   # pipeline steps to execute
            "confidence": float,
            "method": "pattern"|"llm",
            "description": str,
            "reasoning": str,
        }
    """
    # Layer 1: Fast pattern matching
    intent, confidence = classify_intent_fast(query)

    if confidence >= fast_threshold:
        return {
            "intent": intent,
            "pipeline": INTENT_PATTERNS[intent]["pipeline"],
            "confidence": confidence,
            "method": "pattern",
            "description": INTENT_PATTERNS[intent]["description"],
            "reasoning": f"Pattern match (confidence: {confidence})",
        }

    # Layer 2: Function-calling classification for ambiguous queries
    if use_llm_fallback and confidence < fast_threshold:
        fc_intent, fc_confidence, reasoning = classify_intent_function_calling(query, context)
        # Use function-calling result if it's more confident
        if fc_confidence > confidence:
            return {
                "intent": fc_intent,
                "pipeline": INTENT_PATTERNS[fc_intent]["pipeline"],
                "confidence": fc_confidence,
                "method": "function_calling",
                "description": INTENT_PATTERNS[fc_intent]["description"],
                "reasoning": reasoning,
            }

    # Fallback: return pattern match result (even if low confidence)
    return {
        "intent": intent,
        "pipeline": INTENT_PATTERNS[intent]["pipeline"],
        "confidence": confidence,
        "method": "pattern" if confidence > 0 else "default",
        "description": INTENT_PATTERNS[intent]["description"],
        "reasoning": f"Low confidence match ({confidence})",
    }


# ═══════════════════════════════════════════════════════════════════
# §5: AUTONOMOUS PLANNER — LLM plans its own steps
# "Winnie figures out what to do."
# ═══════════════════════════════════════════════════════════════════

def plan_execution(
    goal: str,
    constraints: dict = None,
    available_steps: list = None,
) -> dict:
    """Use LLM to plan a multi-step execution from a goal.

    Unlike route_intent (which maps to predefined pipelines),
    this generates a custom sequence of steps.

    Returns:
        {
            "goal": str,
            "steps": list[str],     # ordered pipeline step names
            "reasoning": str,
            "estimated_time_s": int,
        }
    """
    from server.agentic import safe_llm_call

    # Get available steps
    if not available_steps:
        try:
            from server.routes.pipeline import _get_registry
            available_steps = list(_get_registry().keys())
        except ImportError:
            available_steps = list(INTENT_PATTERNS.keys())

    constraints = constraints or {}
    constraint_str = "\n".join(f"  - {k}: {v}" for k, v in constraints.items())

    prompt = (
        f"You are the planning module for E.D.I.T.H., a research assistant.\n\n"
        f"GOAL: {goal}\n\n"
        f"{'CONSTRAINTS:' + chr(10) + constraint_str + chr(10) + chr(10) if constraint_str else ''}"
        f"AVAILABLE TOOLS:\n"
        + "\n".join(f"  - {s}" for s in available_steps) +
        f"\n\nSelect the tools needed to achieve this goal, in execution order.\n"
        f"Each tool's output feeds into the next.\n\n"
        f"Respond in this exact format:\n"
        f"STEPS: step1, step2, step3\n"
        f"REASONING: <one line explanation>\n"
        f"TIME: <estimated seconds>\n"
    )

    result = safe_llm_call(
        prompt,
        system_instruction="You are a research task planner. Select the minimum set of tools needed.",
        temperature=0.1,
        fallback_text=f"STEPS: {available_steps[0] if available_steps else 'chat'}\nREASONING: fallback\nTIME: 5",
    )

    text = result["text"]
    steps = []
    reasoning = ""
    estimated_time = 10

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("STEPS:"):
            raw_steps = line.split(":", 1)[1].strip()
            steps = [s.strip() for s in raw_steps.split(",") if s.strip()]
            # Validate against available steps
            steps = [s for s in steps if s in available_steps]
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TIME:"):
            try:
                estimated_time = int(re.search(r"\d+", line.split(":", 1)[1]).group())
            except (ValueError, AttributeError):
                pass

    return {
        "goal": goal,
        "steps": steps if steps else ([available_steps[0]] if available_steps else []),
        "reasoning": reasoning or "Planned via LLM",
        "estimated_time_s": estimated_time,
        "constraints": constraints,
    }

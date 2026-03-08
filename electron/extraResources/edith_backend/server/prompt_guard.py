"""
Prompt safety guardrails for Edith / Winnie LLM interactions.

Provides:
- Input sanitization (prompt injection detection)
- Output filtering (data leakage prevention)
- Source text guarding (§6.4: retrieval injection prevention)
- Tool-use allowlisting (§6.3: configurable)
- Prompt versioning (§6.2)
"""

from __future__ import annotations

import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt versioning (§6.2)
# ---------------------------------------------------------------------------
PROMPT_GUARD_VERSION = "2.1"


# ---------------------------------------------------------------------------
# Prompt injection detection
# ---------------------------------------------------------------------------

# Patterns that indicate prompt injection attempts.
# Each entry is (pattern, weight).  High-confidence attacks get weight=2.
_INJECTION_PATTERNS = [
    # Direct instruction override (high confidence)
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I), 2),
    (re.compile(r"disregard\s+(\w+\s+)?(previous|prior|above|everything)", re.I), 2),
    (re.compile(r"forget\s+(everything|all|your)\s+(previous|instructions?|rules?)", re.I), 2),
    # System prompt extraction (high confidence)
    (re.compile(r"(show|print|reveal|output|display|repeat)\s+(your\s+)?(system\s+prompt|instructions?|rules?)", re.I), 2),
    (re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)", re.I), 2),
    # Encoded/obfuscated injection (high confidence)
    (re.compile(r"<\s*/?\s*system\s*>", re.I), 2),
    (re.compile(r"\[INST\]|\[/INST\]", re.I), 2),
    (re.compile(r"###\s*(Human|Assistant|System)\s*:", re.I), 2),
    # Role manipulation (lower confidence — can appear in academic contexts)
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I), 1),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I), 1),
    (re.compile(r"act\s+as\s+(a|an|if)\s+", re.I), 1),
    # §6.1: Malicious intent – require proximity to jailbreak context
    # Only match if combined with role manipulation (weight=1 needs partner)
    (re.compile(r"(hack\s+into|break\s+into|bypass\s+security|phishing\s+attack|exploit\s+vulnerability)", re.I), 1),
    # Data exfiltration attempts (lower confidence)
    (re.compile(r"(send|transmit|post|curl|fetch|webhook)\s+.*(to\s+http|to\s+url)", re.I), 1),
]

# Block if score >= this threshold
_INJECTION_THRESHOLD = 2


def check_prompt_injection(text: str) -> dict:
    """Check input text for prompt injection patterns.

    Returns:
        dict with keys:
        - safe: bool (True if no injection detected)
        - score: int (0 = clean, higher = more suspicious)
        - matched: list of matched pattern descriptions
    """
    score = 0
    matched = []

    for pattern, weight in _INJECTION_PATTERNS:
        if pattern.search(text):
            score += weight
            matched.append(pattern.pattern[:60])

    return {
        "safe": score < _INJECTION_THRESHOLD,
        "score": score,
        "matched": matched,
    }


# ---------------------------------------------------------------------------
# Source text injection guard (§6.4)
# ---------------------------------------------------------------------------

def check_source_injection(sources: list[dict]) -> list[dict]:
    """Strip adversarial content from retrieved source snippets (§6.4).

    Scans each source snippet for injection patterns and redacts matched text.
    Returns the cleaned source list.
    """
    cleaned = []
    for s in sources:
        snippet = s.get("snippet", "") or s.get("text", "")
        result = check_prompt_injection(snippet)
        if not result["safe"]:
            # Redact the adversarial content
            for pat, _ in _INJECTION_PATTERNS:
                snippet = pat.sub("[REDACTED-INJECTION]", snippet)
            s = {**s}
            if "snippet" in s:
                s["snippet"] = snippet
            if "text" in s:
                s["text"] = snippet
        cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# Output filtering (prevent data leakage)
# ---------------------------------------------------------------------------

_LEAKAGE_PATTERNS = [
    # API keys
    re.compile(r"AIzaSy[A-Za-z0-9_-]{33}", re.I),      # Google API key
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.I),           # OpenAI API key
    # §6.7: Generic env var leak pattern
    re.compile(r"(GOOGLE_API_KEY|GEMINI_API_KEY|OPENAI_API_KEY)\s*=\s*\S+", re.I),
    # Secrets file paths
    re.compile(r"secrets\.json", re.I),
    # Internal system prompt markers
    re.compile(r"EDITH_ACCESS_PASSWORD", re.I),
    # Personal identifiers (SSN, credit card)
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),               # SSN
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
]


def filter_output(text: str) -> str:
    """Redact sensitive patterns from LLM output before returning to user."""
    for pattern in _LEAKAGE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Allowed tools / function calls (§6.3: configurable via env)
# ---------------------------------------------------------------------------

_DEFAULT_TOOLS = frozenset({
    "retrieve_sources",
    "search_chroma",
    "search_openalex",
    "generate_literature_review",
    "research_design",
})

# §6.3: Allow extending via EDITH_ALLOWED_TOOLS env var
_extra_tools = os.environ.get("EDITH_ALLOWED_TOOLS", "")
_ALLOWED_TOOLS = _DEFAULT_TOOLS | frozenset(
    t.strip() for t in _extra_tools.split(",") if t.strip()
)


def validate_tool_call(tool_name: str) -> bool:
    """Check if a tool/function call is in the allowlist."""
    return tool_name in _ALLOWED_TOOLS


# ---------------------------------------------------------------------------
# Combined input guard
# ---------------------------------------------------------------------------


def guard_input(query: str, max_length: int = 0) -> Optional[str]:
    """Validate and sanitize user input.

    §6.10: max_length configurable via EDITH_MAX_QUERY_LENGTH env var.

    Returns:
        None if input is safe, or an error message string if blocked.
    """
    if max_length <= 0:
        max_length = int(os.environ.get("EDITH_MAX_QUERY_LENGTH", "32000"))

    # Length check
    if len(query) > max_length:
        return f"Query too long ({len(query)} chars, max {max_length})"

    # Prompt injection check
    result = check_prompt_injection(query)
    if not result["safe"]:
        return "Query blocked: potential prompt injection detected"

    return None  # safe

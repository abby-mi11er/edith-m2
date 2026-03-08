"""
Input Sanitizer — Context-Aware Academic Whitelist
====================================================
Batch 4 — CE-42/43/44: Smart sanitization for academic research.

Standard sanitizers strip LaTeX, statistical notation, and code blocks.
This one understands academic context and preserves what matters.
"""

import logging
import re

log = logging.getLogger("edith.sanitizer")


# ═══════════════════════════════════════════════════════════════════
# Academic whitelist patterns — these should NOT be stripped
# ═══════════════════════════════════════════════════════════════════

ACADEMIC_PATTERNS = {
    "latex_inline": re.compile(r'\$[^$]+\$'),  # $x^2$
    "latex_block": re.compile(r'\$\$[^$]+\$\$'),  # $$\sum_{i}$$
    "citations": re.compile(r'\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?\)'),
    "stat_notation": re.compile(r'[pnNrRβα]\s*[=<>≤≥]\s*[\d.]+'),  # p < 0.05, N = 500
    "confidence_interval": re.compile(r'\[\s*[\d.]+\s*,\s*[\d.]+\s*\]'),  # [0.23, 0.45]
    "r_squared": re.compile(r'R[²2]\s*=\s*[\d.]+'),
    "regression": re.compile(r'β\d?\s*=\s*-?[\d.]+'),
    "footnote_refs": re.compile(r'\[\d+\]|\^?\d+$'),
}

# Dangerous patterns that SHOULD be stripped
DANGEROUS_PATTERNS = {
    "script_tags": re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE),
    "event_handlers": re.compile(r'\bon\w+\s*=', re.IGNORECASE),
    "javascript_uri": re.compile(r'javascript:', re.IGNORECASE),
    "data_uri": re.compile(r'data:text/html', re.IGNORECASE),
    # §FIX W2: Tightened SQL pattern — requires SQL syntax context (semicolons,
    # parentheses, quotes) to avoid false positives on academic prose like
    # "SELECT variables for regression" or "DELETE outliers from the dataset"
    "sql_injection": re.compile(
        r"(?:"
        r";\s*(?:SELECT|INSERT|DELETE|DROP|UPDATE|ALTER)\b"  # statement after semicolon
        r"|(?:UNION\s+(?:ALL\s+)?SELECT)\b"                 # UNION SELECT (classic SQLi)
        r"|(?:DROP\s+TABLE)\b"                               # DROP TABLE
        r"|(?:INSERT\s+INTO)\b"                              # INSERT INTO
        r"|(?:DELETE\s+FROM)\b"                              # DELETE FROM
        r"|(?:UPDATE\s+\S+\s+SET)\b"                         # UPDATE x SET
        r"|(?:SELECT\s+\S+\s+FROM)\b"                        # SELECT x FROM
        r")", re.IGNORECASE
    ),
    "command_injection": re.compile(r'[;&|`]\s*(rm|cat|wget|curl|bash|sh|python|eval)\b'),
}


def sanitize_academic_input(text: str, context: str = "chat") -> dict:
    """Sanitize input while preserving academic notation.

    CE-42: Context-aware — knows the difference between LaTeX and XSS.
    CE-43: Academic whitelist — preserves citations, stats, equations.
    CE-44: Audit logging — tracks what was sanitized and why.

    Returns dict with sanitized text and audit log.
    """
    audit = []
    sanitized = text

    # Step 1: Protect academic patterns (mark them so we don't strip them)
    preserved = {}
    marker_id = 0
    for pattern_name, pattern in ACADEMIC_PATTERNS.items():
        for match in pattern.finditer(sanitized):
            marker = f"__ACADEMIC_{marker_id}__"
            preserved[marker] = match.group()
            sanitized = sanitized.replace(match.group(), marker, 1)
            marker_id += 1

    # Step 2: Strip dangerous patterns
    for pattern_name, pattern in DANGEROUS_PATTERNS.items():
        matches = pattern.findall(sanitized)
        if matches:
            sanitized = pattern.sub("", sanitized)
            audit.append({
                "type": pattern_name,
                "matches": len(matches) if isinstance(matches[0], str) else len(matches),
                "action": "stripped",
            })

    # Step 3: Strip HTML tags (but keep content)
    html_tags = re.findall(r'<[^>]+>', sanitized)
    if html_tags:
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        audit.append({
            "type": "html_tags",
            "matches": len(html_tags),
            "action": "stripped",
        })

    # Step 4: Restore academic patterns
    for marker, original in preserved.items():
        sanitized = sanitized.replace(marker, original)

    # Step 5: Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Step 6: Length check
    max_length = {"chat": 10000, "deep_dive": 50000, "bulk": 100000}.get(context, 10000)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        audit.append({
            "type": "length_truncation",
            "original_length": len(text),
            "truncated_to": max_length,
        })

    is_safe = len(audit) == 0 or all(a["type"] in {"html_tags", "length_truncation"} for a in audit)

    return {
        "text": sanitized,
        "safe": is_safe,
        "audit": audit,
        "preserved_academic": len(preserved),
    }


def validate_query(query: str) -> dict:
    """Quick validation for chat queries."""
    if not query or not query.strip():
        return {"valid": False, "reason": "empty_query"}
    if len(query) > 15000:
        return {"valid": False, "reason": "too_long", "max": 15000}
    result = sanitize_academic_input(query, context="chat")
    if not result["safe"]:
        return {"valid": False, "reason": "unsafe_content", "audit": result["audit"]}
    return {"valid": True, "sanitized": result["text"]}

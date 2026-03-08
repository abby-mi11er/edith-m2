"""
Model Utilities — Text cleaning, JSON parsing, source block formatting.

§5.3: HTML entity decoding in clean_text
§5.4: Fallback-safe JSON parsing with logging
§5.8: Configurable max_snippet_chars via env
§CE-4: JSON repair — auto-fix malformed LLM output
§CE-5: Token counter — accurate counting for GPT/Gemini/MLX
§CE-6: Output schema validation — validate LLM output before returning
"""

import html
import json
import logging
import os
import re
from typing import Optional

log = logging.getLogger("edith.model_utils")

# §5.8: configurable default snippet length
_DEFAULT_MAX_SNIPPET = int(os.environ.get("EDITH_MAX_SNIPPET_CHARS", "900"))


def clean_text(text: str) -> str:
    """Clean and normalize text.  §5.3: also decodes HTML entities."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = html.unescape(text)  # §5.3
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_retryable_model_error(exc: Exception) -> bool:
    """Check if an exception is retryable (transient API errors)."""
    msg = str(exc).lower()
    retry_markers = [
        "429", "500", "503", "timeout", "deadline", "temporar",
        "unavailable", "rate", "quota", "resource exhausted",
        "model not found", "404",
    ]
    return any(marker in msg for marker in retry_markers)


# ═══════════════════════════════════════════════════════════════════
# §CE-4: JSON Repair — Auto-fix malformed LLM output
# ═══════════════════════════════════════════════════════════════════

def repair_json(text: str) -> Optional[str]:
    """Attempt to repair malformed JSON from LLM output.

    Common LLM JSON errors:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing closing braces/brackets
    - Comments in JSON
    - Truncated output
    """
    if not text:
        return None

    repaired = text.strip()

    # Remove markdown fencing
    repaired = re.sub(r'^```(?:json)?\s*', '', repaired)
    repaired = re.sub(r'\s*```$', '', repaired)

    # Remove single-line comments (// ...)
    repaired = re.sub(r'//[^\n]*', '', repaired)

    # Remove multi-line comments (/* ... */)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.S)

    # Replace single quotes with double quotes (but not inside strings)
    # Simple heuristic: replace ' with " when it's around keys/values
    repaired = re.sub(r"(?<![\\])'", '"', repaired)

    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # Fix unquoted keys: {key: "value"} → {"key": "value"}
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', repaired)

    # Try to balance braces/brackets
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_braces > 0:
        repaired += '}' * open_braces
    if open_brackets > 0:
        repaired += ']' * open_brackets

    # Verify it's valid now
    try:
        json.loads(repaired)
        log.debug("repair_json: successfully repaired malformed JSON")
        return repaired
    except json.JSONDecodeError:
        log.warning("repair_json: could not repair JSON")
        return None


# ═══════════════════════════════════════════════════════════════════
# §CE-5: Token Counter — Accurate counting for all models
# ═══════════════════════════════════════════════════════════════════

# Lazy-loaded tokenizer cache
_tokenizer_cache: dict[str, object] = {}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a given model.

    Supports:
    - OpenAI models (gpt-4o, gpt-4o-mini, etc.) via tiktoken
    - Gemini models via character-based estimation (4 chars ≈ 1 token)
    - MLX/local models via word-based estimation (0.75 words ≈ 1 token)

    Returns accurate count for OpenAI, reasonable estimate for others.
    """
    if not text:
        return 0

    model_lower = model.lower()

    # OpenAI models — use tiktoken if available
    if any(m in model_lower for m in ["gpt", "ft:", "o1", "o3"]):
        try:
            import tiktoken
            enc_name = model_lower
            if enc_name not in _tokenizer_cache:
                try:
                    _tokenizer_cache[enc_name] = tiktoken.encoding_for_model(model)
                except KeyError:
                    _tokenizer_cache[enc_name] = tiktoken.get_encoding("cl100k_base")
            return len(_tokenizer_cache[enc_name].encode(text))
        except ImportError:
            pass  # Fall through to estimation

    # Gemini models — character-based estimation
    if "gemini" in model_lower:
        return max(1, len(text) // 4)

    # MLX/local — word-based estimation
    words = len(text.split())
    return max(1, int(words * 1.33))


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
) -> dict:
    """Estimate API cost for a completion.

    Returns dict with input_cost, output_cost, total_cost in USD.
    """
    # Pricing per 1M tokens (as of early 2026)
    _PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }

    model_lower = model.lower()
    pricing = None
    for key, p in _PRICING.items():
        if key in model_lower:
            pricing = p
            break
    if not pricing:
        # Default to gpt-4o-mini pricing for unknown models
        pricing = _PRICING["gpt-4o-mini"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "model": model,
    }


# ═══════════════════════════════════════════════════════════════════
# §CE-6: Output Schema Validation
# ═══════════════════════════════════════════════════════════════════

def validate_output(data: dict, required_fields: list[str]) -> dict:
    """Validate LLM output has required fields.

    Returns the data with a _valid flag and list of any missing fields.
    """
    missing = [f for f in required_fields if f not in data]
    data["_valid"] = len(missing) == 0
    if missing:
        data["_missing_fields"] = missing
        log.warning(f"validate_output: missing fields {missing}")
    return data


# ═══════════════════════════════════════════════════════════════════
# Original JSON parsing — now with §CE-4 repair fallback
# ═══════════════════════════════════════════════════════════════════

def parse_json_object(text: str) -> Optional[dict]:
    """Extract a JSON object from raw text.

    §5.4: Tries full text, fenced code block, regex extraction,
    then §CE-4 repair as final fallback.
    """
    if not text:
        return None
    raw = text.strip()

    # Try 1: direct parse
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try 2: fenced code block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Try 3: regex extraction (fallback)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        blob = m.group(0)
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                log.debug("parse_json_object: used regex fallback")
                return obj
        except Exception:
            pass

    # Try 4: §CE-4 JSON repair
    repaired = repair_json(text)
    if repaired:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                log.info("parse_json_object: used JSON repair fallback")
                return obj
        except Exception:
            pass

    log.warning(f"parse_json_object: all strategies failed for text[:{80}]={text[:80]}")
    return None


def parse_json_array(text: str) -> Optional[list]:
    """Extract a JSON array from raw text.

    §5.4: Tries full text, fenced code block, regex extraction,
    then §CE-4 repair as final fallback.
    """
    if not text:
        return None
    raw = text.strip()

    # Try 1: direct parse
    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else None
    except Exception:
        pass

    # Try 2: fenced code block
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            arr = json.loads(fenced.group(1))
            if isinstance(arr, list):
                return arr
        except Exception:
            pass

    # Try 3: regex extraction
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return arr
        except Exception:
            pass

    # Try 4: §CE-4 JSON repair
    repaired = repair_json(text)
    if repaired:
        try:
            arr = json.loads(repaired)
            if isinstance(arr, list):
                log.info("parse_json_array: used JSON repair fallback")
                return arr
        except Exception:
            pass

    log.warning(f"parse_json_array: all strategies failed for text[:{80}]={text[:80]}")
    return None


def _extract_label_from_path(path: str) -> str:
    """Extract a human-readable label from a file path."""
    if not path:
        return ""
    # Get filename from path
    fname = path.rsplit("/", 1)[-1] if "/" in path else path
    # Strip extension
    stem = re.sub(r'\.(pdf|txt|docx|md|tex|rtf|csv|xlsx|json)$', '', fname, flags=re.IGNORECASE)
    # Clean separators
    stem = stem.replace('_', ' ').replace('-', ' ').strip()
    stem = re.sub(r'\s+', ' ', stem).strip()
    return stem if stem else ""


def build_support_audit_source_blocks(
    sources: Optional[list[dict]] = None,
    max_sources: int = 14,
    max_snippet_chars: int = 0,
) -> str:
    """Build formatted source blocks for prompts and audits.

    §5.8: max_snippet_chars defaults to EDITH_MAX_SNIPPET_CHARS env or 900.
    §FIX: Uses meaningful labels from author/year/filename instead of generic
    'source_N' fallbacks. Format avoids raw metadata tags that models echo.
    """
    if max_snippet_chars <= 0:
        max_snippet_chars = _DEFAULT_MAX_SNIPPET
    if not sources:
        return ""

    blocks = []
    for i, s in enumerate(sources[:max_sources], start=1):
        meta = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}

        # Build a meaningful label: prefer author+year, then title, then filename
        author = s.get("author", "") or meta.get("author", "")
        year = s.get("year", "") or meta.get("year", "")
        title = (s.get("title") or "").strip()

        # Support nested meta structures
        if isinstance(s.get("meta"), dict):
            title = s["meta"].get("title", title) or title

        # If no title, extract from file path
        if not title or title.startswith("source_"):
            path = (s.get("source") or s.get("path") or s.get("uri") or
                    meta.get("source") or meta.get("path") or
                    meta.get("rel_path") or "")
            path_label = _extract_label_from_path(path)
            if path_label:
                title = path_label

        # Build the display label
        if author and year:
            label = f"{author} ({year})"
            if title and title != author:
                label += f" — {title}"
        elif author:
            label = f"{author} — {title}" if title else author
        elif title:
            label = f"{title} ({year})" if year else title
        else:
            label = f"Source {i}"

        # Support 'snippet' or 'text' fields
        snippet = (s.get("snippet") or s.get("text") or "").strip()
        topic = s.get("academic_topic")
        if snippet:
            snippet = snippet[:max_snippet_chars]

        # Clean format: [S#] Label, then content on next line
        block = f"[S{i}] {label}\n"
        if topic:
            block += f"Topic: {topic}\n"
        if snippet:
            block += f"Content: {snippet}\n"
        blocks.append(block.strip())
    return "\n\n".join(blocks)

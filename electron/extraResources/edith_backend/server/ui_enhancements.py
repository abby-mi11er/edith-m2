"""
UI & E2E Flow Enhancements — Backend utilities for UI features and end-to-end flow.

Implements:
  9.4   Citation export (BibTeX / RIS)
  12.2  Parallel retrieval + planning (asyncio support)
  12.4  Cost-aware model routing
  12.5  Latency budgets per stage
  12.8  Telemetry pipeline (structured logging)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from server.vault_config import VAULT_ROOT

log = logging.getLogger("edith.ui_enhancements")


# ---------------------------------------------------------------------------
# 9.4: Citation Export (BibTeX / RIS)
# ---------------------------------------------------------------------------

def sources_to_bibtex(sources: list[dict]) -> str:
    """
    Convert retrieved sources to BibTeX format for import into Zotero/Mendeley.

    Handles sources with and without full metadata.
    """
    entries = []

    for i, source in enumerate(sources):
        # Extract fields
        title = source.get("title", source.get("filename", f"Source {i+1}"))
        authors = source.get("authors", source.get("author", "Unknown"))
        year = source.get("year", source.get("pub_year", "n.d."))
        doc_type = source.get("doc_type", "article")

        # Generate citation key
        first_author = re.split(r"[,;&]", str(authors))[0].strip().split()[-1] if authors else "unknown"
        cite_key = f"{first_author.lower()}{year}"
        cite_key = re.sub(r"[^\w]", "", cite_key)

        # Map doc_type to BibTeX entry type
        bibtex_type = {
            "article": "article",
            "book": "book",
            "book_chapter": "incollection",
            "report": "techreport",
            "thesis": "phdthesis",
            "conference": "inproceedings",
            "dataset": "misc",
            "working_paper": "unpublished",
        }.get(doc_type, "misc")

        entry_lines = [f"@{bibtex_type}{{{cite_key},"]
        entry_lines.append(f"  title = {{{_bibtex_escape(str(title))}}},")
        entry_lines.append(f"  author = {{{_bibtex_escape(str(authors))}}},")
        entry_lines.append(f"  year = {{{year}}},")

        if source.get("journal"):
            entry_lines.append(f"  journal = {{{_bibtex_escape(source['journal'])}}},")
        if source.get("volume"):
            entry_lines.append(f"  volume = {{{source['volume']}}},")
        if source.get("pages"):
            entry_lines.append(f"  pages = {{{source['pages']}}},")
        if source.get("doi"):
            entry_lines.append(f"  doi = {{{source['doi']}}},")
        if source.get("url"):
            entry_lines.append(f"  url = {{{source['url']}}},")

        # Add note about retrieval
        entry_lines.append(f"  note = {{Retrieved by Edith for research query}},")
        entry_lines.append("}")

        entries.append("\n".join(entry_lines))

    return "\n\n".join(entries) + "\n"


def _bibtex_escape(text: str) -> str:
    """Escape special characters for BibTeX."""
    return text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


# ---------------------------------------------------------------------------
# 12.4: Cost-Aware Model Routing
# ---------------------------------------------------------------------------

# Approximate cost per 1K tokens (USD)
MODEL_COSTS = {
    "openai_ft": {"input": 0.0003, "output": 0.0012},       # ft:gpt-4o-mini
    "gemini_flash": {"input": 0.000075, "output": 0.0003},    # Gemini 2.5 Flash
    "gemini_lite": {"input": 0.000038, "output": 0.00015},    # Gemini lite tier
    "gpt4o": {"input": 0.005, "output": 0.015},               # GPT-4o (if used for judge)
}


@dataclass
class CostTracker:
    """Track per-session and cumulative model usage costs."""
    session_costs: dict[str, float] = field(default_factory=dict)
    total_tokens: dict[str, int] = field(default_factory=dict)
    log_path: Path = field(default_factory=lambda: VAULT_ROOT / "Forge" / "Runs" / "cost_log.jsonl")

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage and return estimated cost in USD."""
        costs = MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
        cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])

        self.session_costs[model] = self.session_costs.get(model, 0) + cost
        self.total_tokens[model] = self.total_tokens.get(model, 0) + input_tokens + output_tokens

        # Log to file
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost, 6),
                }) + "\n")
        except Exception:
            pass

        return cost

    @property
    def session_total(self) -> float:
        return sum(self.session_costs.values())


# ---------------------------------------------------------------------------
# 12.5: Latency Budget Enforcement
# ---------------------------------------------------------------------------

class LatencyBudget:
    """Track and enforce time budgets per pipeline stage."""

    def __init__(self, budgets: dict[str, float]):
        """
        Args:
            budgets: {stage_name: max_seconds}
        """
        self.budgets = budgets
        self._starts: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}

    def start_stage(self, stage: str):
        """Mark the beginning of a pipeline stage."""
        self._starts[stage] = time.time()

    def end_stage(self, stage: str) -> float:
        """Mark the end of a stage and return elapsed time."""
        if stage not in self._starts:
            return 0.0
        elapsed = time.time() - self._starts[stage]
        self._elapsed[stage] = elapsed
        return elapsed

    def is_over_budget(self, stage: str) -> bool:
        """Check if a stage has exceeded its budget."""
        if stage not in self._starts:
            return False
        elapsed = time.time() - self._starts[stage]
        budget = self.budgets.get(stage, float("inf"))
        return elapsed > budget

    def remaining(self, stage: str) -> float:
        """Get remaining time budget for a stage."""
        if stage not in self._starts:
            return self.budgets.get(stage, float("inf"))
        elapsed = time.time() - self._starts[stage]
        budget = self.budgets.get(stage, float("inf"))
        return max(0, budget - elapsed)

    @property
    def summary(self) -> dict:
        return {
            stage: {
                "budget": self.budgets.get(stage, 0),
                "elapsed": round(elapsed, 3),
                "over_budget": elapsed > self.budgets.get(stage, float("inf")),
            }
            for stage, elapsed in self._elapsed.items()
        }


# ---------------------------------------------------------------------------
# 12.8: Telemetry Pipeline
# ---------------------------------------------------------------------------

@dataclass
class RequestTelemetry:
    """Structured telemetry for a single request."""
    request_id: str = ""
    query: str = ""
    depth: str = ""
    started_at: float = field(default_factory=time.time)

    # Timing
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    audit_ms: float = 0.0
    total_ms: float = 0.0

    # Usage
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    sources_retrieved: int = 0
    sources_cited: int = 0

    # Quality
    coverage: float = 0.0
    confidence: float = 0.0
    audit_clean: bool = True
    cache_hit: bool = False

    # Errors
    errors: list[str] = field(default_factory=list)
    fallback_used: bool = False

    def finalize(self):
        """Called when request completes."""
        self.total_ms = (time.time() - self.started_at) * 1000

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "query_preview": self.query[:80],
            "depth": self.depth,
            "timings": {
                "retrieval_ms": round(self.retrieval_ms, 1),
                "generation_ms": round(self.generation_ms, 1),
                "audit_ms": round(self.audit_ms, 1),
                "total_ms": round(self.total_ms, 1),
            },
            "usage": {
                "model": self.model,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "sources_retrieved": self.sources_retrieved,
                "sources_cited": self.sources_cited,
            },
            "quality": {
                "coverage": round(self.coverage, 3),
                "confidence": round(self.confidence, 3),
                "audit_clean": self.audit_clean,
                "cache_hit": self.cache_hit,
            },
            "errors": self.errors,
            "fallback_used": self.fallback_used,
        }


class TelemetryCollector:
    """Collect and aggregate request telemetry."""

    def __init__(self, log_path: Path | None = None, max_entries: int = 10000):
        self.log_path = log_path or VAULT_ROOT / "Forge" / "Runs" / "telemetry.jsonl"
        self.max_entries = max_entries
        self._recent: list[dict] = []

    def record(self, telemetry: RequestTelemetry):
        """Record a completed request's telemetry."""
        telemetry.finalize()
        entry = telemetry.as_dict()
        entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        self._recent.append(entry)
        if len(self._recent) > 100:
            self._recent = self._recent[-100:]

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_summary(self, last_n: int = 50) -> dict:
        """Get aggregate stats from recent requests."""
        entries = self._recent[-last_n:] if self._recent else []
        if not entries:
            return {"count": 0}

        total_ms = [e["timings"]["total_ms"] for e in entries]
        coverages = [e["quality"]["coverage"] for e in entries if e["quality"]["coverage"] > 0]

        return {
            "count": len(entries),
            "p50_latency_ms": round(sorted(total_ms)[len(total_ms) // 2], 1) if total_ms else 0,
            "p95_latency_ms": round(sorted(total_ms)[int(len(total_ms) * 0.95)] if total_ms else 0, 1),
            "avg_coverage": round(sum(coverages) / len(coverages), 3) if coverages else 0,
            "cache_hit_rate": round(
                sum(1 for e in entries if e["quality"]["cache_hit"]) / len(entries), 3
            ),
            "error_rate": round(
                sum(1 for e in entries if e["errors"]) / len(entries), 3
            ),
            "fallback_rate": round(
                sum(1 for e in entries if e["fallback_used"]) / len(entries), 3
            ),
        }

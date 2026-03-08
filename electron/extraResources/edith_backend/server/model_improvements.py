#!/usr/bin/env python3
"""
Models & AI Improvements Module
=================================
Enhancements for the model layer:
  7.1  Dynamic model routing
  7.2  Structured output mode
  7.3  Streaming chain-of-thought
  7.4  Model fallback chain
  7.5  Temperature/sampling controls
  7.6  Local model option (Ollama)
  7.7  Multi-turn context window management
  7.8  Prompt versioning and A/B testing
  7.9  Citation entailment at inference
  7.10 Cost tracking per query
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 7.1  Dynamic Model Routing
# ---------------------------------------------------------------------------

@dataclass
class ModelRoute:
    """Route queries to the best model based on task characteristics."""
    name: str
    model_id: str
    strengths: list[str] = field(default_factory=list)
    max_tokens: int = 128000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    latency_ms: float = 0.0

MODEL_ROUTES = {
    "fast": ModelRoute(
        name="fast", model_id="gemini-2.5-flash",
        strengths=["speed", "summarization", "classification"],
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
    ),
    "reasoning": ModelRoute(
        name="reasoning", model_id="gemini-2.5-pro",
        strengths=["reasoning", "math", "code", "analysis", "synthesis"],
        cost_per_1k_input=0.00125, cost_per_1k_output=0.005,
    ),
    "creative": ModelRoute(
        name="creative", model_id="gpt-4o",
        strengths=["creative", "writing", "nuance", "debate"],
        cost_per_1k_input=0.005, cost_per_1k_output=0.015,
    ),
    "finetuned": ModelRoute(
        name="finetuned",
        model_id="ft:gpt-4o-mini-2024-07-18:personal:winnie-v1:D9xqwC8p",
        strengths=["domain", "phd", "research", "factual", "policy"],
        cost_per_1k_input=0.0003, cost_per_1k_output=0.0012,
    ),
}


def route_query(query: str, task_type: str = "chat") -> ModelRoute:
    """Select the best model for a query based on content analysis."""
    q_lower = query.lower()

    # Code/math queries → reasoning model
    if any(kw in q_lower for kw in ["code", "implement", "algorithm", "proof", "equation",
                                     "calculate", "derive", "solve"]):
        return MODEL_ROUTES["reasoning"]

    # Domain-specific research queries → fine-tuned
    if any(kw in q_lower for kw in ["time series", "forecasting", "arima", "lstm",
                                     "transformer", "paper", "literature"]):
        return MODEL_ROUTES["finetuned"]

    # Creative tasks → GPT-4o
    if any(kw in q_lower for kw in ["write", "essay", "creative", "draft", "rewrite",
                                     "critique", "review"]):
        return MODEL_ROUTES["creative"]

    # Default: fast model for general queries
    return MODEL_ROUTES["fast"]


# ---------------------------------------------------------------------------
# 7.2  Structured Output Mode
# ---------------------------------------------------------------------------

STRUCTURED_SCHEMAS = {
    "claim_table": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "evidence": {"type": "string"},
                "confidence": {"type": "number"},
                "source": {"type": "string"},
            },
        },
    },
    "comparison": {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
            "criteria": {"type": "array", "items": {"type": "string"}},
            "matrix": {"type": "array"},
            "recommendation": {"type": "string"},
        },
    },
    "summary": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "methodology": {"type": "string"},
            "findings": {"type": "string"},
            "limitations": {"type": "string"},
        },
    },
}


# ---------------------------------------------------------------------------
# 7.3  Streaming Chain-of-Thought
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7.4  Model Fallback Chain
# ---------------------------------------------------------------------------

class ModelFallbackChain:
    """Try models in sequence, falling back on failure."""

    def __init__(self, models: list[str] = None):
        self.models = models or [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gpt-4o-mini",
        ]
        self.last_used = None
        self.failures: dict[str, int] = {}

    async def generate(self, prompt: str, generate_fn) -> dict:
        """Try each model in the chain until one succeeds."""
        for model in self.models:
            try:
                result = await generate_fn(prompt, model=model)
                self.last_used = model
                return {"response": result, "model": model, "fallback": model != self.models[0]}
            except Exception as e:
                self.failures[model] = self.failures.get(model, 0) + 1
                continue

        return {"response": "All models failed. Please try again.", "model": "none", "error": True}


# ---------------------------------------------------------------------------
# 7.5  Temperature / Sampling Controls
# ---------------------------------------------------------------------------

SAMPLING_PRESETS = {
    "precise": {"temperature": 0.1, "top_p": 0.9, "top_k": 10},
    "balanced": {"temperature": 0.5, "top_p": 0.95, "top_k": 40},
    "creative": {"temperature": 0.9, "top_p": 0.98, "top_k": 100},
    "research": {"temperature": 0.3, "top_p": 0.92, "top_k": 20},
    "coding": {"temperature": 0.2, "top_p": 0.9, "top_k": 10},
}


def get_sampling_params(preset: str = "balanced", overrides: dict = None) -> dict:
    """Get sampling parameters from preset with optional overrides."""
    params = dict(SAMPLING_PRESETS.get(preset, SAMPLING_PRESETS["balanced"]))
    if overrides:
        params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# 7.6  Local Model (Ollama)
# ---------------------------------------------------------------------------

class OllamaClient:
    """Client for local Ollama models."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        if self._available is not None:
            return self._available
        try:
            import requests  # type: ignore
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def list_models(self) -> list[str]:
        """List available local models."""
        try:
            import requests  # type: ignore
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m.get("name", "") for m in models]
        except Exception:
            pass
        return []

    def generate(self, prompt: str, model: str = "llama3.2", **kwargs) -> str:
        """Generate text using a local Ollama model."""
        import requests  # type: ignore
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, **kwargs},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return ""


# ---------------------------------------------------------------------------
# 7.7  Context Window Manager
# ---------------------------------------------------------------------------

class ContextWindowManager:
    """Manage multi-turn context within model token limits."""

    MODEL_LIMITS = {
        "gemini-2.5-flash": 1048576,
        "gemini-2.5-pro": 2097152,
        "gemini-2.0-flash": 1048576,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.max_tokens = self.MODEL_LIMITS.get(model, 128000)
        self.reserved_output = 8192  # Reserve for response

    def fit_messages(self, messages: list[dict], max_tokens: int = 0) -> list[dict]:
        """Trim messages to fit within context window, preserving system + recent."""
        limit = max_tokens or (self.max_tokens - self.reserved_output)

        # Rough token estimate: 4 chars per token
        def _estimate_tokens(msgs):
            return sum(len(m.get("content", "")) // 4 for m in msgs)

        total = _estimate_tokens(messages)
        if total <= limit:
            return messages

        # Strategy: keep system message + first user message + recent messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Keep most recent messages, sliding window
        fitted = list(system_msgs)
        budget = limit - _estimate_tokens(system_msgs)

        for msg in reversed(non_system):
            msg_tokens = len(msg.get("content", "")) // 4
            if budget - msg_tokens < 0:
                break
            fitted.insert(len(system_msgs), msg)
            budget -= msg_tokens

        return fitted


# ---------------------------------------------------------------------------
# 7.8  Prompt Versioning & A/B Testing
# ---------------------------------------------------------------------------

class PromptRegistry:
    """Version and A/B test system prompts."""

    def __init__(self, prompts_dir: Path = None):
        self.dir = prompts_dir or Path("prompts")
        self._prompts: dict[str, list[dict]] = {}

    def register(self, name: str, version: str, template: str, metadata: dict = None):
        """Register a prompt version."""
        if name not in self._prompts:
            self._prompts[name] = []
        self._prompts[name].append({
            "version": version,
            "template": template,
            "metadata": metadata or {},
            "created_at": time.strftime("%Y-%m-%d"),
        })

    def get(self, name: str, version: str = "latest") -> Optional[str]:
        """Get a specific prompt version."""
        versions = self._prompts.get(name, [])
        if not versions:
            return None
        if version == "latest":
            return versions[-1]["template"]
        for v in versions:
            if v["version"] == version:
                return v["template"]
        return None

    def ab_select(self, name: str, user_hash: str) -> dict:
        """Select a prompt variant for A/B testing based on user hash."""
        versions = self._prompts.get(name, [])
        if len(versions) < 2:
            return versions[-1] if versions else {}
        idx = hash(user_hash) % len(versions)
        return versions[idx]


# ---------------------------------------------------------------------------
# 7.9  Citation Entailment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7.10  Cost Tracking
# ---------------------------------------------------------------------------

class CostTracker:
    """Track API costs per query and in aggregate."""

    PRICING = {
        "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gemini-embedding-001": {"input": 0.00001, "output": 0},
    }

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.session_costs: list[dict] = []
        self.total_cost = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int, query: str = ""):
        """Record a cost entry."""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "query": query[:100],
        }
        self.session_costs.append(entry)
        self.total_cost += cost

        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

        return entry

    @property
    def summary(self) -> dict:
        return {
            "total_usd": round(self.total_cost, 4),
            "query_count": len(self.session_costs),
            "avg_cost_per_query": round(self.total_cost / max(len(self.session_costs), 1), 6),
            "by_model": self._by_model(),
        }

    def _by_model(self) -> dict:
        by_model = {}
        for entry in self.session_costs:
            model = entry["model"]
            if model not in by_model:
                by_model[model] = {"count": 0, "cost_usd": 0.0}
            by_model[model]["count"] += 1
            by_model[model]["cost_usd"] += entry["cost_usd"]
        return by_model

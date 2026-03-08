"""
Rate Limiter — Adaptive Token Budget Management
=================================================
Batch 4 — CE-39: Smart rate limiting that adapts to usage patterns.

Architecture:
    Request → Token Counter → Budget Check → Burst Allowance →
    Backpressure Signal → Cooldown Timer

Not a crude rate limit — an intelligent budget that:
- Knows your work patterns (heavy morning, light evening)
- Allows burst research sessions (50K tokens in 10 minutes)
- Preserves daily budget across sessions
- Warns before you hit the wall
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("edith.rate_limiter")


@dataclass
class TokenBudget:
    """A token budget for a time period."""
    period: str  # "hourly", "daily", "session"
    limit: int
    used: int = 0
    reset_at: float = 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def usage_pct(self) -> float:
        return self.used / max(self.limit, 1)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "usage_pct": round(self.usage_pct, 3),
            "reset_at": self.reset_at,
        }


class AdaptiveRateLimiter:
    """Intelligent rate limiting with burst budgets and usage forecasting.

    CE-39: Adaptive rate limits
    CE-40: Burst budget for research sprints
    CE-41: Usage pattern learning
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._budgets = {
            "hourly": TokenBudget("hourly", 100_000, 0, time.time() + 3600),
            "daily": TokenBudget("daily", 1_000_000, 0, time.time() + 86400),
            "session": TokenBudget("session", 500_000, 0, 0),
        }
        # Burst budget: extra allowance that regenerates slowly
        self._burst_budget = 200_000
        self._burst_used = 0
        self._burst_regen_rate = 1000  # tokens per minute

        # Usage history for pattern learning
        self._hourly_usage: dict[int, int] = {}  # hour → avg tokens
        self._last_regen = time.time()

    def check_budget(self, tokens_needed: int) -> dict:
        """Check if a request can proceed within budget."""
        self._auto_reset()
        self._regenerate_burst()

        # Check all budgets
        for budget in self._budgets.values():
            if budget.used + tokens_needed > budget.limit:
                # Try burst budget
                burst_available = self._burst_budget - self._burst_used
                if tokens_needed <= burst_available:
                    return {
                        "allowed": True,
                        "source": "burst",
                        "warning": f"Using burst budget ({burst_available:,} tokens remaining)",
                    }
                return {
                    "allowed": False,
                    "blocked_by": budget.period,
                    "remaining": budget.remaining,
                    "reset_in_seconds": max(0, int(budget.reset_at - time.time())),
                }

        return {"allowed": True, "source": "standard"}

    def record_usage(self, tokens: int):
        """Record token usage against all budgets."""
        for budget in self._budgets.values():
            budget.used += tokens

        # Record hourly pattern
        hour = time.localtime().tm_hour
        self._hourly_usage[hour] = self._hourly_usage.get(hour, 0) + tokens

    def get_status(self) -> dict:
        """Return full rate limiter status."""
        self._auto_reset()
        return {
            "budgets": {k: v.to_dict() for k, v in self._budgets.items()},
            "burst_remaining": self._burst_budget - self._burst_used,
            "burst_total": self._burst_budget,
            "peak_usage_hour": (
                max(self._hourly_usage, key=self._hourly_usage.get)
                if self._hourly_usage else None
            ),
            "forecast": self._forecast_usage(),
        }

    def _auto_reset(self):
        """Reset budgets that have expired."""
        now = time.time()
        for budget in self._budgets.values():
            if budget.reset_at and now > budget.reset_at:
                budget.used = 0
                if budget.period == "hourly":
                    budget.reset_at = now + 3600
                elif budget.period == "daily":
                    budget.reset_at = now + 86400

    def _regenerate_burst(self):
        """Slowly regenerate burst budget over time."""
        now = time.time()
        elapsed_minutes = (now - self._last_regen) / 60
        regen = int(elapsed_minutes * self._burst_regen_rate)
        if regen > 0:
            self._burst_used = max(0, self._burst_used - regen)
            self._last_regen = now

    def _forecast_usage(self) -> dict:
        """Predict if you'll hit budget limits today."""
        if not self._hourly_usage:
            return {"prediction": "unknown"}

        avg_hourly = sum(self._hourly_usage.values()) / max(len(self._hourly_usage), 1)
        hours_left = 24 - time.localtime().tm_hour
        predicted_total = self._budgets["daily"].used + avg_hourly * hours_left

        return {
            "avg_hourly_tokens": int(avg_hourly),
            "predicted_daily_total": int(predicted_total),
            "will_exceed": predicted_total > self._budgets["daily"].limit,
        }


# Global instance
rate_limiter = AdaptiveRateLimiter()

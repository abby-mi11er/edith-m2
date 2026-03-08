"""
Anomaly Detection — Cost and Usage Pattern Alerting
=====================================================
Batch 4 — CE-45/46/47: Detect unusual spending and usage patterns.

If your API costs suddenly spike 300% or you're making 10x your normal
requests, something is wrong. This module watches for that.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("edith.anomaly")


@dataclass
class UsageDataPoint:
    """A single usage observation."""
    timestamp: float
    metric: str  # "api_cost", "tokens", "requests", "latency"
    value: float
    context: str = ""  # "chat", "deep_dive", "committee", "discovery"


class AnomalyDetector:
    """Detect anomalous usage patterns using simple Z-score method.

    CE-45: Cost anomaly detection (API spending spikes)
    CE-46: Usage pattern alerting (unusual request volumes)
    CE-47: Latency degradation detection
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 2.5):
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._history: dict[str, list[float]] = {}  # metric → [values]
        self._alerts: list[dict] = []
        self._alert_cooldown: dict[str, float] = {}  # metric → last alert time

    def record(self, metric: str, value: float, context: str = "") -> dict | None:
        """Record a metric value and check for anomalies.

        Returns an alert dict if anomaly detected, None otherwise.
        """
        if metric not in self._history:
            self._history[metric] = []

        self._history[metric].append(value)
        # Keep within window
        if len(self._history[metric]) > self._window_size:
            self._history[metric] = self._history[metric][-self._window_size:]

        # Need at least 10 data points for anomaly detection
        if len(self._history[metric]) < 10:
            return None

        # Z-score check
        values = self._history[metric]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.01

        z_score = (value - mean) / std

        if abs(z_score) > self._z_threshold:
            # Check cooldown (don't spam alerts)
            now = time.time()
            if self._alert_cooldown.get(metric, 0) > now - 300:  # 5 minute cooldown
                return None

            self._alert_cooldown[metric] = now
            alert = {
                "type": "anomaly",
                "metric": metric,
                "value": round(value, 4),
                "mean": round(mean, 4),
                "z_score": round(z_score, 2),
                "direction": "spike" if z_score > 0 else "drop",
                "context": context,
                "timestamp": now,
                "severity": (
                    "critical" if abs(z_score) > 4 else
                    "warning" if abs(z_score) > 3 else "info"
                ),
                "message": (
                    f"{'⚠️ ' if abs(z_score) > 3 else '📊 '}"
                    f"{metric} {'spike' if z_score > 0 else 'drop'}: "
                    f"{value:.2f} (mean: {mean:.2f}, {abs(z_score):.1f}σ)"
                ),
            }
            self._alerts.append(alert)
            self._alerts = self._alerts[-50:]  # Keep last 50 alerts
            return alert

        return None

    def get_alerts(self, severity: str = "", limit: int = 10) -> list[dict]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
        return alerts[-limit:]

    def get_stats(self, metric: str) -> dict:
        """Get descriptive stats for a metric."""
        values = self._history.get(metric, [])
        if not values:
            return {"metric": metric, "data_points": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "metric": metric,
            "data_points": n,
            "mean": round(sum(values) / n, 4),
            "median": round(sorted_vals[n // 2], 4),
            "min": round(sorted_vals[0], 4),
            "max": round(sorted_vals[-1], 4),
            "p95": round(sorted_vals[int(n * 0.95)], 4) if n > 20 else None,
        }

    @property
    def status(self) -> dict:
        return {
            "metrics_tracked": list(self._history.keys()),
            "total_alerts": len(self._alerts),
            "active_alerts": [a for a in self._alerts if time.time() - a.get("timestamp", 0) < 3600],
            "stats": {m: self.get_stats(m) for m in self._history},
        }


# Global instance
anomaly_detector = AnomalyDetector()

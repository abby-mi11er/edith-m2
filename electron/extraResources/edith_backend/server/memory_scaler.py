"""
Dynamic Context Window Scaling — Feature #1
=============================================
Monitors macOS unified memory pressure and dynamically scales
the compute profile's context window based on available headroom.

On M4 with nominal pressure: expands to 24 sources × 2000 chars (3× more)
On warn/critical: contracts back to base or below to prevent swapping.
"""

import logging
import os
import re
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger("edith.memory_scaler")

# Cache the pressure reading for 10 seconds
_pressure_cache: Optional[dict] = None
_pressure_cache_time: float = 0
_CACHE_TTL = 10


def get_memory_pressure() -> dict:
    """Read macOS memory pressure and unified memory stats.

    Returns dict with:
        pressure_level: "nominal" | "warn" | "critical"
        free_mb: approximate free memory in MB
        used_pct: memory used percentage
        pages_free: raw pages free from vm_stat
    """
    global _pressure_cache, _pressure_cache_time

    now = time.time()
    if _pressure_cache and now - _pressure_cache_time < _CACHE_TTL:
        return _pressure_cache

    result = {
        "pressure_level": "nominal",
        "free_mb": 0,
        "used_pct": 0,
        "pages_free": 0,
    }

    # Method 1: memory_pressure command (most reliable on macOS)
    try:
        mp = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True, text=True, timeout=5,
        )
        output = mp.stdout.lower()
        if "critical" in output:
            result["pressure_level"] = "critical"
        elif "warn" in output:
            result["pressure_level"] = "warn"
        else:
            result["pressure_level"] = "nominal"
    except Exception:
        pass

    # Method 2: vm_stat for detailed numbers
    try:
        vs = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, timeout=5,
        )
        lines = vs.stdout.strip().split("\n")

        # Parse page size
        page_size = 16384  # Default for Apple Silicon
        if lines and "page size" in lines[0]:
            ps_match = re.search(r"(\d+)\s+bytes", lines[0])
            if ps_match:
                page_size = int(ps_match.group(1))

        # Parse page counts
        pages = {}
        for line in lines[1:]:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip().lower()
                val = parts[1].strip().rstrip(".")
                try:
                    pages[key] = int(val)
                except ValueError:
                    pass

        free = pages.get("pages free", 0)
        inactive = pages.get("pages inactive", 0)
        speculative = pages.get("pages speculative", 0)
        active = pages.get("pages active", 0)
        wired = pages.get("pages wired down", 0)
        compressed = pages.get("pages stored in compressor", 0)

        available = free + inactive + speculative
        total_used = active + wired + compressed
        total = available + total_used

        result["pages_free"] = free
        result["free_mb"] = round(available * page_size / (1024 * 1024))
        result["used_pct"] = round(total_used / max(total, 1) * 100)

    except Exception as e:
        log.debug(f"§MEM: vm_stat failed: {e}")

    _pressure_cache = result
    _pressure_cache_time = now
    return result


def scale_context_window(base_profile: dict) -> dict:
    """Scale the compute profile's context window based on memory pressure.

    On M4 with nominal pressure: EXPAND working memory (3× more context)
    On warn: use BASE profile (no change)
    On critical: CONTRACT to prevent swapping

    Args:
        base_profile: the hardware-adaptive profile from get_compute_profile()

    Returns:
        profile with adjusted context window parameters
    """
    pressure = get_memory_pressure()
    is_high_compute = any(
        gen in base_profile.get("chip", "")
        for gen in ["m4", "m3 max", "m3 pro", "m2 ultra"]
    )

    adjusted = dict(base_profile)
    adjusted["memory_pressure"] = pressure["pressure_level"]
    adjusted["memory_free_mb"] = pressure["free_mb"]

    if pressure["pressure_level"] == "nominal" and is_high_compute:
        # ═══ EXPAND: M4 with headroom — 3× more context ═══
        adjusted["max_sources_compressed"] = min(
            24, int(base_profile.get("max_sources_compressed", 16) * 1.5)
        )
        adjusted["max_chars_per_source"] = min(
            2000, int(base_profile.get("max_chars_per_source", 1200) * 1.6)
        )
        adjusted["max_conversation_turns"] = min(
            30, int(base_profile.get("max_conversation_turns", 20) * 1.5)
        )
        adjusted["max_tokens_debate"] = min(
            8000, int(base_profile.get("max_tokens_debate", 5000) * 1.4)
        )
        # Also expand retrieval pool
        adjusted["top_k"] = min(
            80, int(base_profile.get("top_k", 50) * 1.5)
        )
        adjusted["context_scaling"] = "expanded"
        log.info(
            f"§MEM: EXPANDED context — {adjusted['max_sources_compressed']} sources × "
            f"{adjusted['max_chars_per_source']} chars, top_k={adjusted['top_k']}, "
            f"free={pressure['free_mb']}MB"
        )

    elif pressure["pressure_level"] == "nominal":
        # M2 with headroom — modest expansion
        adjusted["max_sources_compressed"] = min(
            12, int(base_profile.get("max_sources_compressed", 8) * 1.25)
        )
        adjusted["max_chars_per_source"] = min(
            1000, int(base_profile.get("max_chars_per_source", 800) * 1.2)
        )
        adjusted["context_scaling"] = "modest"

    elif pressure["pressure_level"] == "warn":
        # Use base profile as-is
        adjusted["context_scaling"] = "base"
        log.info(f"§MEM: WARN pressure — using base profile")

    elif pressure["pressure_level"] == "critical":
        # ═══ CONTRACT: prevent swapping ═══
        adjusted["max_sources_compressed"] = max(
            4, int(base_profile.get("max_sources_compressed", 8) * 0.5)
        )
        adjusted["max_chars_per_source"] = max(
            400, int(base_profile.get("max_chars_per_source", 800) * 0.5)
        )
        adjusted["max_conversation_turns"] = max(
            5, int(base_profile.get("max_conversation_turns", 10) * 0.5)
        )
        adjusted["top_k"] = max(
            6, int(base_profile.get("top_k", 12) * 0.5)
        )
        adjusted["agents"] = 1  # Single agent to reduce memory
        adjusted["context_scaling"] = "contracted"
        log.warning(
            f"§MEM: CRITICAL pressure — contracted to {adjusted['max_sources_compressed']} sources, "
            f"agents=1, free={pressure['free_mb']}MB"
        )

    return adjusted


# ═══════════════════════════════════════════════════════════════════
# Background Memory Monitor
# ═══════════════════════════════════════════════════════════════════

class MemoryMonitor:
    """Background thread that monitors memory pressure and alerts on changes."""

    def __init__(self, poll_interval: int = 30):
        self._interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_level = "nominal"
        self._transitions: list[dict] = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        log.info("§MEM: Background memory monitor started")

    def stop(self):
        self._running = False

    def _poll(self):
        while self._running:
            try:
                pressure = get_memory_pressure()
                level = pressure["pressure_level"]
                if level != self._last_level:
                    self._transitions.append({
                        "from": self._last_level,
                        "to": level,
                        "time": time.time(),
                        "free_mb": pressure["free_mb"],
                    })
                    log.info(
                        f"§MEM: Pressure changed: {self._last_level} → {level} "
                        f"(free={pressure['free_mb']}MB)"
                    )
                    self._last_level = level
            except Exception:
                pass
            time.sleep(self._interval)

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "current_level": self._last_level,
            "transitions": self._transitions[-5:],
        }


# Singleton
memory_monitor = MemoryMonitor()

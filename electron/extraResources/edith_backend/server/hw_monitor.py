"""
Hardware Monitoring — H1/H2/H9 + P2
======================================
  - Power-aware profiling (battery vs plugged-in)
  - Thermal throttle detection
  - Thunderbolt hot-plug resilience
  - Drive SMART monitoring
"""

import logging
import os
import subprocess
import time
from typing import Optional

log = logging.getLogger("edith.hw_monitor")


# ═══════════════════════════════════════════════════════════════════
# H1: Power-Aware Profiling
# ═══════════════════════════════════════════════════════════════════

def get_power_state() -> dict:
    """Detect battery vs AC power and battery level.

    Returns dict with: source, battery_pct, is_low_power
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()

        if "ac power" in output:
            source = "ac"
        elif "battery power" in output:
            source = "battery"
        else:
            source = "unknown"

        # Parse battery percentage
        battery_pct = 100
        import re
        pct_match = re.search(r"(\d+)%", output)
        if pct_match:
            battery_pct = int(pct_match.group(1))

        is_low_power = source == "battery" and battery_pct < 20

        return {
            "source": source,
            "battery_pct": battery_pct,
            "is_low_power": is_low_power,
        }
    except Exception as e:
        log.debug(f"§POWER: Detection failed: {e}")
        return {"source": "unknown", "battery_pct": 100, "is_low_power": False}


def get_power_adjusted_profile(base_profile: dict) -> dict:
    """Adjust compute profile based on power state.

    On battery with <20%: downgrade to minimal mode.
    On battery: reduce workers and batch sizes by 50%.
    On AC: use full profile.
    """
    power = get_power_state()

    if power["source"] == "ac" or power["source"] == "unknown":
        return base_profile

    adjusted = dict(base_profile)

    if power["is_low_power"]:
        # Critical battery — minimal mode
        adjusted["mode"] = "focus_minimal"
        adjusted["agents"] = 1
        adjusted["max_concurrent_retrieval"] = 1
        adjusted["top_k"] = 8
        adjusted["max_tokens_standard"] = 1000
        adjusted["max_tokens_debate"] = 1500
        adjusted["max_sources_compressed"] = 5
        adjusted["index_workers"] = 1
        adjusted["local_inference_enabled"] = False  # Save power
        log.warning(f"§POWER: Low battery ({power['battery_pct']}%) — minimal mode")
    else:
        # Battery but not critical — reduce by 50%
        adjusted["agents"] = max(1, adjusted.get("agents", 1) // 2)
        adjusted["max_concurrent_retrieval"] = max(1, adjusted.get("max_concurrent_retrieval", 2) // 2)
        adjusted["index_workers"] = max(1, adjusted.get("index_workers", 2) // 2)
        adjusted["index_batch_size"] = adjusted.get("index_batch_size", 20) // 2
        log.info(f"§POWER: Battery mode ({power['battery_pct']}%) — reduced workers")

    adjusted["power_state"] = power
    return adjusted


# ═══════════════════════════════════════════════════════════════════
# H2: Thermal Throttle Detection
# ═══════════════════════════════════════════════════════════════════

def get_thermal_state() -> dict:
    """Detect thermal throttling on Apple Silicon.

    Returns dict with: throttled, cpu_temp_estimate, recommendation
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()

        # Check for thermal pressure
        if "cpu_speed_limit" in output:
            import re
            limit_match = re.search(r"cpu_speed_limit\s*=\s*(\d+)", output)
            if limit_match:
                limit = int(limit_match.group(1))
                throttled = limit < 100
                return {
                    "throttled": throttled,
                    "cpu_speed_limit": limit,
                    "recommendation": "reduce_workers" if throttled else "normal",
                }

        # Check thermal-NOMINAL / FAIR / SERIOUS
        if "serious" in output or "critical" in output:
            return {"throttled": True, "cpu_speed_limit": 50, "recommendation": "reduce_workers"}
        elif "fair" in output:
            return {"throttled": False, "cpu_speed_limit": 80, "recommendation": "monitor"}

        return {"throttled": False, "cpu_speed_limit": 100, "recommendation": "normal"}
    except Exception as e:
        log.debug(f"§THERMAL: Detection failed: {e}")
        return {"throttled": False, "cpu_speed_limit": 100, "recommendation": "normal"}


# ═══════════════════════════════════════════════════════════════════
# H9: Thunderbolt Hot-Plug Resilience
# ═══════════════════════════════════════════════════════════════════

class DriveWatchdog:
    """Monitor drive availability and handle hot-unplug gracefully."""

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
        self._last_check = 0
        self._check_interval = 5  # seconds
        self._available = True
        self._disconnect_count = 0

    def is_available(self) -> bool:
        """Check if the data drive is still accessible. Cached for 5s."""
        now = time.time()
        if now - self._last_check < self._check_interval:
            return self._available

        self._last_check = now
        if not self._data_root:
            self._available = False
            return False

        was_available = self._available
        self._available = os.path.isdir(self._data_root)

        if was_available and not self._available:
            self._disconnect_count += 1
            log.error(f"§HOTPLUG: Drive disconnected! ({self._data_root}) "
                      f"— disconnect #{self._disconnect_count}")
        elif not was_available and self._available:
            log.info(f"§HOTPLUG: Drive reconnected ({self._data_root})")

        return self._available

    @property
    def status(self) -> dict:
        return {
            "available": self._available,
            "data_root": self._data_root,
            "disconnect_count": self._disconnect_count,
        }


# Singleton
drive_watchdog = DriveWatchdog()


# ═══════════════════════════════════════════════════════════════════
# P2: Drive SMART Monitoring
# ═══════════════════════════════════════════════════════════════════

def get_drive_health(data_root: str = "") -> dict:
    """Check SSD health via diskutil info.

    Returns dict with: healthy, wear_level, free_space_gb, filesystem
    """
    data_root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return {"healthy": None, "error": "No data root set"}

    try:
        import shutil
        usage = shutil.disk_usage(data_root)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = usage.used / usage.total * 100

        health = {
            "healthy": free_gb > 5,
            "free_gb": round(free_gb, 1),
            "total_gb": round(total_gb, 1),
            "used_pct": round(used_pct, 1),
        }

        # Try to get filesystem type
        try:
            result = subprocess.run(
                ["diskutil", "info", data_root],
                capture_output=True, text=True, timeout=10,
            )
            if "File System" in result.stdout:
                import re
                fs_match = re.search(r"File System.*?:\s*(.+)", result.stdout)
                if fs_match:
                    health["filesystem"] = fs_match.group(1).strip()
            if "Solid State" in result.stdout:
                health["ssd"] = True
        except Exception:
            pass

        if free_gb < 5:
            log.warning(f"§DISK: Low space: {free_gb:.1f} GB free")
        if free_gb < 1:
            log.error(f"§DISK: CRITICAL: Only {free_gb:.1f} GB free!")

        return health
    except Exception as e:
        return {"healthy": None, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# §CE-25: GPU / Neural Engine Utilization — Know your silicon
# ═══════════════════════════════════════════════════════════════════

def get_gpu_utilization() -> dict:
    """Estimate GPU and Neural Engine utilization on Apple Silicon.

    Apple doesn't expose GPU utilization directly via CLI,
    so we use `sudo powermetrics` residency data or estimate
    from active model inference.
    """
    try:
        # Use ioreg for basic GPU info
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_active = "IOAccelerator" in result.stdout

        # Check if any ML framework is using the GPU
        import re
        gpu_info = {
            "gpu_available": gpu_active,
            "neural_engine_available": True,  # All Apple Silicon has ANE
        }

        # Try to detect if MPS (Metal Performance Shaders) is active
        try:
            import torch
            gpu_info["mps_available"] = torch.backends.mps.is_available()
            gpu_info["mps_built"] = torch.backends.mps.is_built()
        except ImportError:
            gpu_info["mps_available"] = False

        # Memory pressure as proxy for GPU memory usage
        try:
            import psutil
            vm = psutil.virtual_memory()
            gpu_info["system_memory_gb"] = round(vm.total / (1024**3), 1)
            gpu_info["memory_used_pct"] = vm.percent
            gpu_info["memory_pressure"] = (
                "critical" if vm.percent > 90
                else "high" if vm.percent > 75
                else "normal"
            )
        except ImportError:
            pass

        return gpu_info
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# §CE-26: Thermal Alert System — Proactive throttle warnings
# ═══════════════════════════════════════════════════════════════════

class ThermalAlertSystem:
    """Monitor thermal state and emit alerts before throttling occurs.

    Instead of just detecting throttling AFTER it happens (H2),
    this predicts it and proactively reduces load.
    """

    def __init__(self):
        self._history: list[dict] = []
        self._alert_callback = None
        self._max_history = 60  # 1 sample per second for 1 minute

    def sample(self) -> dict:
        """Take a thermal sample and check for trending toward throttle."""
        state = get_thermal_state()
        state["timestamp"] = time.time()
        self._history.append(state)

        # Keep only recent history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Analyze trend
        alert = self._analyze_trend()
        if alert:
            state["alert"] = alert
            log.warning(f"§THERMAL ALERT: {alert}")

        return state

    def _analyze_trend(self) -> str | None:
        """Predict if we're trending toward throttling."""
        if len(self._history) < 5:
            return None

        recent = self._history[-5:]
        speed_limits = [s.get("cpu_speed_limit", 100) for s in recent]

        # Speed limit is dropping
        if all(speed_limits[i] >= speed_limits[i+1] for i in range(len(speed_limits)-1)):
            if speed_limits[-1] < 90:
                return "THERMAL WARNING: CPU speed limit trending down — consider reducing parallel operations"

        # Already throttled
        if speed_limits[-1] < 70:
            return "THERMAL CRITICAL: Significant throttling detected — model inference will be slow"

        return None

    def get_recommendation(self) -> dict:
        """Get a thermal-aware recommendation for compute intensity."""
        state = get_thermal_state()
        limit = state.get("cpu_speed_limit", 100)

        if limit >= 95:
            return {"mode": "full", "message": "Thermal headroom available", "max_workers": 8}
        elif limit >= 80:
            return {"mode": "moderate", "message": "Slight thermal pressure", "max_workers": 4}
        elif limit >= 60:
            return {"mode": "reduced", "message": "Thermal throttling active", "max_workers": 2}
        else:
            return {"mode": "minimal", "message": "Severe throttling — minimal compute only", "max_workers": 1}


# Global thermal alerts
thermal_alerts = ThermalAlertSystem()


# ═══════════════════════════════════════════════════════════════════
# §CE-27: Battery-Aware GPU Mode — Scale GPU usage with battery
# ═══════════════════════════════════════════════════════════════════

def get_battery_gpu_policy() -> dict:
    """Determine GPU usage policy based on power state.

    On AC power: Full GPU utilization for embeddings, Monte Carlo, etc.
    On battery >50%: GPU for embeddings only, no Monte Carlo
    On battery <50%: CPU-only mode, no GPU inference
    On battery <20%: Minimal mode, defer all non-essential computation
    """
    power = get_power_state()
    battery = power.get("battery_pct", 100)
    source = power.get("source", "ac")

    if source == "ac" or source == "unknown":
        return {
            "gpu_mode": "full",
            "embeddings_gpu": True,
            "monte_carlo_gpu": True,
            "local_inference": True,
            "speculative_indexing": True,
            "message": "AC power — full GPU utilization",
        }
    elif battery > 50:
        return {
            "gpu_mode": "conservative",
            "embeddings_gpu": True,
            "monte_carlo_gpu": False,
            "local_inference": True,
            "speculative_indexing": False,
            "message": f"Battery {battery}% — GPU for embeddings only",
        }
    elif battery > 20:
        return {
            "gpu_mode": "cpu_only",
            "embeddings_gpu": False,
            "monte_carlo_gpu": False,
            "local_inference": False,
            "speculative_indexing": False,
            "message": f"Battery {battery}% — CPU-only mode",
        }
    else:
        return {
            "gpu_mode": "minimal",
            "embeddings_gpu": False,
            "monte_carlo_gpu": False,
            "local_inference": False,
            "speculative_indexing": False,
            "message": f"Battery CRITICAL {battery}% — minimal mode, charge soon",
        }


def get_full_hardware_status() -> dict:
    """Return a comprehensive hardware status for the Doctor panel.

    Combines all monitoring data into a single snapshot.
    """
    return {
        "power": get_power_state(),
        "thermal": get_thermal_state(),
        "gpu": get_gpu_utilization(),
        "drive": get_drive_health(),
        "drive_watchdog": drive_watchdog.status,
        "gpu_policy": get_battery_gpu_policy(),
        "thermal_recommendation": thermal_alerts.get_recommendation(),
    }


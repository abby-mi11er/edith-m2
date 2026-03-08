"""
Citadel Theme Engine — Solar-Paper Non-Fatigue Design
======================================================
When you stare at 1TB of data for 10 hours, blue light is the enemy.

Features:
  1. Solar-Paper Token System — Deep Slate + Solar-White + muted accents
  2. Time-of-Day Auto-Shift — warmer tones at night
  3. Arc Reactor Pulse — Bolt I/O status as subtle color state
  4. E-Ink Context Strip — data format for external e-ink display
"""

import logging
import os
import time
from datetime import datetime

log = logging.getLogger("edith.theme")


# ═══════════════════════════════════════════════════════════════════
# §1: SOLAR-PAPER TOKEN SYSTEM — Non-Fatigue Color Palette
# ═══════════════════════════════════════════════════════════════════

class CitadelTheme:
    """Solar-Paper theme token system for all-day research.

    Deep Slate backgrounds, Solar-White text, muted accents.
    Auto-shifts to warmer tones after sunset to reduce eye strain.

    Usage:
        theme = CitadelTheme()
        tokens = theme.get_tokens()           # Current palette
        night  = theme.get_tokens(hour=22)    # Force night mode
    """

    # ── Core Palette ──
    SOLAR_PAPER = {
        "day": {
            # Backgrounds (deepest → lightest)
            "bg_primary": "#0F1419",        # Deep Slate — main canvas
            "bg_secondary": "#151B23",      # Slate — panels, sidebars
            "bg_tertiary": "#1C2430",       # Soft Slate — cards, inputs
            "bg_elevated": "#232D3B",       # Elevated — modals, tooltips

            # Text (brightest → dimmest)
            "text_primary": "#E8E6DF",      # Solar-White — body text
            "text_secondary": "#A0A8B4",    # Warm Ash — labels, metadata
            "text_muted": "#6B7280",        # Muted — timestamps, hints
            "text_accent": "#C8B88A",       # Parchment Gold — highlights

            # Accents
            "accent_cerulean": "#2D7D9A",   # Cerulean — links, Arc Reactor idle
            "accent_cerulean_deep": "#1A5F7A",  # Deep Cerulean — heavy load
            "accent_amber": "#D4A853",      # Amber — warnings, errors
            "accent_sage": "#6B8F71",       # Sage — success, complete
            "accent_rose": "#9B6B6B",       # Dusty Rose — critical alerts

            # Borders
            "border_subtle": "#1E2A36",     # Barely visible separator
            "border_active": "#2D7D9A",     # Active state border

            # Shadows
            "shadow": "rgba(0, 0, 0, 0.4)",
        },
        "night": {
            # Warmer, dimmer variant for post-sunset
            "bg_primary": "#0D1117",
            "bg_secondary": "#131820",
            "bg_tertiary": "#1A1F2B",
            "bg_elevated": "#202735",

            "text_primary": "#DDD8CE",      # Warmer white
            "text_secondary": "#8F9198",
            "text_muted": "#5D6068",
            "text_accent": "#BFA76F",       # Dimmer gold

            "accent_cerulean": "#245F75",   # Muted cerulean
            "accent_cerulean_deep": "#153D52",
            "accent_amber": "#B8904A",
            "accent_sage": "#567358",
            "accent_rose": "#7D5555",

            "border_subtle": "#181E28",
            "border_active": "#245F75",

            "shadow": "rgba(0, 0, 0, 0.6)",
        },
    }

    # Typography
    TYPOGRAPHY = {
        "font_primary": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        "font_mono": "'SF Mono', 'JetBrains Mono', 'Fira Code', monospace",
        "font_size_base": "14px",
        "font_size_sm": "12px",
        "font_size_lg": "16px",
        "font_size_xl": "20px",
        "line_height": "1.6",
        "letter_spacing": "0.01em",
    }

    # Transition timing for smooth shifts
    TRANSITIONS = {
        "color_shift": "0.8s ease-in-out",
        "hover": "0.2s ease",
        "panel": "0.3s ease-in-out",
    }

    def __init__(self, sunset_hour: int = 18, sunrise_hour: int = 6):
        self._sunset = sunset_hour
        self._sunrise = sunrise_hour

    def _is_night(self, hour: int = None) -> bool:
        h = hour if hour is not None else datetime.now().hour
        return h >= self._sunset or h < self._sunrise

    def get_tokens(self, hour: int = None) -> dict:
        """Get the full theme token set for the current time of day.

        Returns dict with: colors, typography, transitions, mode.
        """
        mode = "night" if self._is_night(hour) else "day"
        return {
            "mode": mode,
            "colors": self.SOLAR_PAPER[mode],
            "typography": self.TYPOGRAPHY,
            "transitions": self.TRANSITIONS,
        }

    def get_css_variables(self, hour: int = None) -> str:
        """Export tokens as CSS custom properties for the frontend."""
        tokens = self.get_tokens(hour)
        lines = [":root {"]
        for key, value in tokens["colors"].items():
            lines.append(f"  --citadel-{key.replace('_', '-')}: {value};")
        for key, value in tokens["typography"].items():
            lines.append(f"  --citadel-{key.replace('_', '-')}: {value};")
        lines.append("}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# §2: ARC REACTOR PULSE — Bolt I/O Status Signal
# ═══════════════════════════════════════════════════════════════════

def arc_reactor_pulse() -> dict:
    """Read Bolt I/O status and return a color state for the HUD ring.

    States:
        idle:   Soft Slate    — no significant I/O
        active: Cerulean      — normal Bolt reads/writes
        heavy:  Deep Cerulean — high throughput (indexing, backup)
        error:  Amber         — drive not detected or I/O error

    Returns:
        {color, intensity, bpm, state, io_rate_mbps}
    """
    data_root = os.environ.get("EDITH_DATA_ROOT", "")

    # Check drive availability
    if not data_root or not os.path.isdir(data_root):
        return {
            "state": "error",
            "color": "#D4A853",
            "intensity": 0.8,
            "bpm": 40,
            "io_rate_mbps": 0,
        }

    # Estimate I/O activity by monitoring disk stats
    try:
        import shutil
        usage = shutil.disk_usage(data_root)
        # Use a proxy: check if any .lock or .wal files indicate active writes
        active_writes = False
        for f in os.listdir(data_root):
            if f.endswith((".lock", ".wal", ".tmp")):
                active_writes = True
                break

        # Check for heavy activity (ChromaDB indexing)
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        heavy = False
        if chroma_dir and os.path.isdir(chroma_dir):
            for f in os.listdir(chroma_dir):
                if f.endswith((".wal", ".tmp")):
                    heavy = True
                    break

        if heavy:
            return {
                "state": "heavy",
                "color": "#1A5F7A",
                "intensity": 0.9,
                "bpm": 8,            # Slow, deep breathe
                "io_rate_mbps": 1500,
                "disk_used_pct": round(usage.used / usage.total * 100, 1),
            }
        elif active_writes:
            return {
                "state": "active",
                "color": "#2D7D9A",
                "intensity": 0.6,
                "bpm": 12,
                "io_rate_mbps": 500,
                "disk_used_pct": round(usage.used / usage.total * 100, 1),
            }
        else:
            return {
                "state": "idle",
                "color": "#1C2430",
                "intensity": 0.3,
                "bpm": 15,           # Calm breathe
                "io_rate_mbps": 0,
                "disk_used_pct": round(usage.used / usage.total * 100, 1),
            }

    except Exception as e:
        return {
            "state": "error",
            "color": "#D4A853",
            "intensity": 0.8,
            "bpm": 40,
            "io_rate_mbps": 0,
            "error": str(e)[:100],
        }


# ═══════════════════════════════════════════════════════════════════
# §3: E-INK CONTEXT STRIP — Data for Secondary Display
# ═══════════════════════════════════════════════════════════════════

def get_context_strip_data() -> dict:
    """Generate minimal data for a secondary e-ink display.

    Designed for a 13-inch e-ink monitor (Boox Mira) positioned
    below the main screen. Shows: daily rhythm status, active
    bibliography, and system health — all refreshed slowly.
    """
    from datetime import datetime

    data = {
        "refreshed": datetime.now().strftime("%H:%M"),
        "sections": {},
    }

    # Current rhythm block
    try:
        hour = datetime.now().hour
        if hour < 9:
            block = "☕ Morning Boot"
        elif hour < 12:
            block = "🔬 Deep Work A"
        elif hour < 13:
            block = "🍽️ Break"
        elif hour < 15:
            block = "👥 Committee Hour"
        elif hour < 18:
            block = "🔬 Deep Work B"
        elif hour < 19:
            block = "📝 Evening Review"
        else:
            block = "🌙 Shutdown"
        data["sections"]["rhythm"] = {"current_block": block}
    except Exception:
        pass

    # Bolt status
    data["sections"]["bolt"] = arc_reactor_pulse()

    return data


# Global theme instance
citadel_theme = CitadelTheme()

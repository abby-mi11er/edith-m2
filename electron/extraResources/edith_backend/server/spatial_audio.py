"""
Spatial Audio Engine — Directional Persona Mapping
====================================================
Frontier Feature 2: The Committee lives in your room.

Instead of reading Mettler's critique in a chat box, you hear her voice
coming from the 2 o'clock position near your bookshelf. When Aldrich
disagrees, his voice comes from 10 o'clock. You can close your eyes
and have a verbal debate with your sources.

Architecture:
    Persona → Virtual 3D Coordinate → HRTF Processing → Spatial Audio Output

Uses the M4's Advanced Audio Engine for hardware-accelerated
Head-Related Transfer Functions (HRTF). Falls back to software
panning on M2.

The key insight: Screen Fatigue is the enemy of deep thinking.
By mapping personas to physical space, we reduce visual overload
and engage the auditory cortex for richer, more embodied reasoning.
"""

import json
import logging
import math
import os
import subprocess
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.spatial_audio")


# ═══════════════════════════════════════════════════════════════════
# Spatial Coordinates — Where each persona "sits"
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SpatialPosition:
    """3D position in the user's room.

    Uses clock-position notation for intuitive placement:
    - 12 o'clock = directly in front (0°)
    - 3 o'clock = right (90°)
    - 6 o'clock = behind (180°)
    - 9 o'clock = left (270°)

    Elevation: -1 (below) to +1 (above), 0 = ear level
    Distance: 0.5 (intimate) to 5.0 (far), 2.0 = conversational
    """
    azimuth_degrees: float  # 0-360, clock position
    elevation: float = 0.0  # -1 to +1
    distance: float = 2.0   # meters

    @property
    def clock_position(self) -> str:
        """Human-readable clock position."""
        hour = int((self.azimuth_degrees / 30) % 12) or 12
        return f"{hour} o'clock"

    @property
    def azimuth_radians(self) -> float:
        return math.radians(self.azimuth_degrees)

    def to_cartesian(self) -> tuple[float, float, float]:
        """Convert to XYZ for 3D rendering."""
        x = self.distance * math.sin(self.azimuth_radians)
        y = self.elevation * self.distance
        z = self.distance * math.cos(self.azimuth_radians)
        return (round(x, 3), round(y, 3), round(z, 3))

    def to_dict(self) -> dict:
        return {
            "azimuth_degrees": self.azimuth_degrees,
            "elevation": self.elevation,
            "distance": self.distance,
            "clock_position": self.clock_position,
            "cartesian": self.to_cartesian(),
        }


# ═══════════════════════════════════════════════════════════════════
# Persona Audio Profiles — Voice characteristics per persona
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VoiceProfile:
    """Audio characteristics for a Committee persona."""
    persona_name: str
    position: SpatialPosition
    pitch_shift: float = 1.0   # 0.8 = deeper, 1.2 = higher
    speaking_rate: float = 1.0  # 0.8 = slower, 1.2 = faster
    voice_id: str = ""          # TTS voice ID (system-specific)
    reverb: float = 0.1        # Room reflections, 0-1
    warmth: float = 0.5        # EQ warmth, 0-1

    def to_dict(self) -> dict:
        return {
            "persona": self.persona_name,
            "position": self.position.to_dict(),
            "pitch_shift": self.pitch_shift,
            "speaking_rate": self.speaking_rate,
            "voice_id": self.voice_id,
            "reverb": self.reverb,
            "warmth": self.warmth,
        }


# ═══════════════════════════════════════════════════════════════════
# Default Persona Positions — The Lubbock Office Layout
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PERSONAS: dict[str, VoiceProfile] = {
    "mettler": VoiceProfile(
        persona_name="Mettler",
        position=SpatialPosition(azimuth_degrees=60, elevation=0.1, distance=2.0),
        pitch_shift=1.05,
        speaking_rate=0.95,
        voice_id="com.apple.voice.premium.en-US.Samantha",
        reverb=0.15,
        warmth=0.6,
    ),
    "aldrich": VoiceProfile(
        persona_name="Aldrich",
        position=SpatialPosition(azimuth_degrees=300, elevation=0.0, distance=2.5),
        pitch_shift=0.85,
        speaking_rate=1.0,
        voice_id="com.apple.voice.premium.en-US.Daniel",
        reverb=0.2,
        warmth=0.4,
    ),
    "scholar": VoiceProfile(
        persona_name="Scholar",
        position=SpatialPosition(azimuth_degrees=0, elevation=0.2, distance=1.8),
        pitch_shift=1.0,
        speaking_rate=0.9,
        voice_id="com.apple.voice.premium.en-US.Ava",
        reverb=0.1,
        warmth=0.7,
    ),
    "critic": VoiceProfile(
        persona_name="Critic",
        position=SpatialPosition(azimuth_degrees=180, elevation=-0.1, distance=3.0),
        pitch_shift=0.9,
        speaking_rate=1.1,
        voice_id="com.apple.voice.premium.en-US.Tom",
        reverb=0.25,
        warmth=0.3,
    ),
    "mentor": VoiceProfile(
        persona_name="Mentor",
        position=SpatialPosition(azimuth_degrees=330, elevation=0.3, distance=1.5),
        pitch_shift=0.95,
        speaking_rate=0.85,
        voice_id="com.apple.voice.premium.en-US.Karen",
        reverb=0.1,
        warmth=0.8,
    ),
    "methodologist": VoiceProfile(
        persona_name="Methodologist",
        position=SpatialPosition(azimuth_degrees=120, elevation=0.0, distance=2.2),
        pitch_shift=1.0,
        speaking_rate=1.05,
        voice_id="com.apple.voice.premium.en-US.Alex",
        reverb=0.15,
        warmth=0.5,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Spatial Audio Engine — The Core
# ═══════════════════════════════════════════════════════════════════

class SpatialAudioEngine:
    """Manages spatial audio rendering for the Committee.

    On M4: Uses hardware-accelerated HRTF via AVAudioEngine
    On M2: Uses software-based stereo panning as fallback
    On web: Sends spatial coordinates to WebAudio API in frontend
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._personas = dict(DEFAULT_PERSONAS)
        self._active_voices: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._audio_backend = self._detect_backend()
        self._enabled = True
        self._load_custom_positions()

    def _detect_backend(self) -> str:
        """Detect available audio backend."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            chip = result.stdout.strip()
            if "M4" in chip:
                return "m4_spatial"  # Full hardware HRTF
            elif "M2" in chip or "M3" in chip or "M1" in chip:
                return "apple_silicon"  # Software spatial
            elif "Apple" in chip:
                return "apple_silicon"
        except Exception:
            pass
        return "web_audio"  # Fallback to frontend WebAudio API

    def _load_custom_positions(self):
        """Load user-customized persona positions from disk."""
        if not self._data_root:
            return
        config_path = os.path.join(self._data_root, "VAULT", "CONFIG", "spatial_audio.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r") as f:
                custom = json.load(f)
            for name, pos_data in custom.get("personas", {}).items():
                if name in self._personas:
                    p = self._personas[name]
                    p.position = SpatialPosition(
                        azimuth_degrees=pos_data.get("azimuth", p.position.azimuth_degrees),
                        elevation=pos_data.get("elevation", p.position.elevation),
                        distance=pos_data.get("distance", p.position.distance),
                    )
            log.info("Loaded custom spatial audio positions")
        except Exception as e:
            log.warning(f"Failed to load spatial audio config: {e}")

    def set_persona_position(self, persona_name: str, azimuth: float,
                              elevation: float = 0.0, distance: float = 2.0) -> dict:
        """Reposition a persona in 3D space.

        Called from the UI when the user drags a persona to a new position.
        """
        name_lower = persona_name.lower()
        if name_lower not in self._personas:
            return {"error": f"Unknown persona: {persona_name}"}

        self._personas[name_lower].position = SpatialPosition(
            azimuth_degrees=azimuth % 360,
            elevation=max(-1, min(1, elevation)),
            distance=max(0.5, min(5.0, distance)),
        )

        # Save to disk
        self._save_positions()
        return self._personas[name_lower].to_dict()

    def _save_positions(self):
        """Persist persona positions to disk."""
        if not self._data_root:
            return
        config_path = os.path.join(self._data_root, "VAULT", "CONFIG", "spatial_audio.json")
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            data = {
                "personas": {
                    name: {
                        "azimuth": p.position.azimuth_degrees,
                        "elevation": p.position.elevation,
                        "distance": p.position.distance,
                    }
                    for name, p in self._personas.items()
                }
            }
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save spatial positions: {e}")

    def get_spatial_scene(self) -> dict:
        """Return the full spatial scene for the frontend to render.

        The frontend uses this to:
        1. Position persona avatars in 3D space
        2. Configure WebAudio panners for each voice
        3. Show the "room layout" visualization
        """
        return {
            "backend": self._audio_backend,
            "enabled": self._enabled,
            "personas": {
                name: profile.to_dict()
                for name, profile in self._personas.items()
            },
            "room": {
                "width": 4.0,
                "depth": 5.0,
                "height": 2.8,
                "user_position": [0, 0, 0],
            },
        }

    def prepare_speech(self, persona_name: str, text: str) -> dict:
        """Prepare spatial audio parameters for a persona's speech.

        Returns configuration for the frontend's WebAudio API or
        the native macOS speech synthesizer.
        """
        name_lower = persona_name.lower()
        profile = self._personas.get(name_lower)
        if not profile:
            # Fall back to center position for unknown personas
            profile = VoiceProfile(
                persona_name=persona_name,
                position=SpatialPosition(azimuth_degrees=0),
            )

        pos = profile.position
        cart = pos.to_cartesian()

        # Build Web Audio API panner config
        panner_config = {
            "panningModel": "HRTF",
            "distanceModel": "inverse",
            "positionX": cart[0],
            "positionY": cart[1],
            "positionZ": cart[2],
            "refDistance": 1.0,
            "maxDistance": 10.0,
            "rolloffFactor": 1.0,
            "coneInnerAngle": 360,
            "coneOuterAngle": 0,
            "coneOuterGain": 0,
        }

        # macOS native speech params
        native_speech = {
            "voice": profile.voice_id,
            "rate": profile.speaking_rate,
            "pitch": profile.pitch_shift,
        }

        return {
            "persona": profile.persona_name,
            "text": text,
            "position": pos.to_dict(),
            "panner": panner_config,
            "native_speech": native_speech,
            "reverb": profile.reverb,
            "warmth": profile.warmth,
            "backend": self._audio_backend,
        }

    def prepare_debate(self, exchanges: list[dict]) -> list[dict]:
        """Prepare a multi-persona debate for spatial audio playback.

        Input: [{"persona": "mettler", "text": "..."}, {"persona": "aldrich", "text": "..."}]
        Output: List of spatial audio commands with timing and positioning
        """
        commands = []
        estimated_time_ms = 0

        for i, exchange in enumerate(exchanges):
            persona = exchange.get("persona", "scholar")
            text = exchange.get("text", "")

            speech = self.prepare_speech(persona, text)
            # Estimate duration: ~150 words per minute
            word_count = len(text.split())
            duration_ms = int((word_count / 150) * 60 * 1000)

            commands.append({
                **speech,
                "sequence": i,
                "start_time_ms": estimated_time_ms,
                "estimated_duration_ms": duration_ms,
            })
            estimated_time_ms += duration_ms + 500  # 500ms pause between speakers

        return commands

    def toggle(self, enabled: bool) -> dict:
        """Enable or disable spatial audio."""
        self._enabled = enabled
        return {"enabled": self._enabled, "backend": self._audio_backend}

    def process(self, input_text: str, mode: str = "ambient") -> dict:
        """Process spatial audio request — called by /api/audio/spatialize.

        Modes:
          - 'ambient': return the full spatial scene for WebAudio setup
          - 'debate': parse JSON exchanges and prepare a multi-persona debate
          - any persona name: prepare speech for that persona
        """
        if mode == "ambient" or not input_text:
            return self.get_spatial_scene()
        if mode == "debate":
            try:
                exchanges = json.loads(input_text)
                if isinstance(exchanges, list):
                    return {"commands": self.prepare_debate(exchanges)}
            except (json.JSONDecodeError, TypeError):
                pass
            return self.get_spatial_scene()
        # Treat mode as persona name, input as text
        return self.prepare_speech(mode, input_text)

    def get_status(self) -> dict:
        """Return spatial audio system status."""
        return {
            "enabled": self._enabled,
            "backend": self._audio_backend,
            "persona_count": len(self._personas),
            "positions": {
                name: p.position.clock_position
                for name, p in self._personas.items()
            },
        }


# Global instance
spatial_audio = SpatialAudioEngine()

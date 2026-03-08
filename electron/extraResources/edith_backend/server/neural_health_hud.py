"""
Neural Health HUD — Real-Time Brain Visualization
====================================================
A minimalist, glowing overlay for your second monitor.

Shows in real-time:
  - Which lobes are active (Thalamus, Hippocampus, Cortex, REM)
  - Synaptic connection count and average weight
  - Hidden Bridges found by the Dream Engine
  - Confidence Score of your current chapter
  - Theoretical Voice drift alerts
  - Logic Conflict warnings
  - Bolt I/O throughput

Designed to be served as a lightweight web page that auto-refreshes.
"""

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("edith.neural_hud")


# ═══════════════════════════════════════════════════════════════════
# HUD Data Collector — Gathers state from all modules
# ═══════════════════════════════════════════════════════════════════

class NeuralHealthHUD:
    """Collects health data from all Citadel modules.

    Usage:
        hud = NeuralHealthHUD()
        state = hud.snapshot()  # Full brain state
        html = hud.render_html()  # Self-contained HTML dashboard
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default

    def snapshot(self) -> dict:
        """Capture a full brain health snapshot."""
        snap = {
            "timestamp": time.time(),
            "formatted_time": time.strftime("%H:%M:%S"),
        }

        # Brain status
        try:
            from server.citadel_neural_net import edith_brain
            snap["brain"] = edith_brain.health_check()
        except Exception:
            snap["brain"] = {"state": "offline"}

        # Synapse Bridge status
        try:
            from server.synapse_bridge import synapse_bridge
            snap["synapse"] = synapse_bridge.status
        except Exception:
            snap["synapse"] = {"status": "offline"}

        # Connectome status
        try:
            from server.connectome import connectome
            snap["connectome"] = connectome.status
        except Exception:
            snap["connectome"] = {"status": "offline"}

        # Dream Engine status
        try:
            from server.dream_engine import dream_engine
            snap["dreams"] = dream_engine.status
        except Exception:
            snap["dreams"] = {"status": "offline"}

        # Graph Engine status
        try:
            from server.graph_vector_engine import graph_engine
            snap["graph"] = graph_engine.get_graph_stats()
        except Exception:
            snap["graph"] = {"status": "offline"}

        # Bolt SSD status
        snap["bolt"] = self._bolt_status()

        # Truth Maintenance — weak claims
        try:
            from server.synapse_bridge import synapse_bridge
            snap["weak_claims"] = synapse_bridge.truth_maintenance_scan()[:5]
        except Exception:
            snap["weak_claims"] = []

        return snap

    def _bolt_status(self) -> dict:
        """Check Bolt SSD connection, space, and I/O latency."""
        bolt = Path(self._bolt_path)
        if not bolt.exists():
            return {"connected": False}

        try:
            stat = os.statvfs(str(bolt))
            total = stat.f_frsize * stat.f_blocks
            free = stat.f_frsize * stat.f_bavail
            used = total - free
            result = {
                "connected": True,
                "total_gb": round(total / (1024 ** 3), 1),
                "used_gb": round(used / (1024 ** 3), 1),
                "free_gb": round(free / (1024 ** 3), 1),
                "usage_pct": round((used / max(total, 1)) * 100, 1),
            }

            # §DRIVE-IO: Measure actual I/O latency with a 4KB probe
            try:
                import tempfile
                probe_path = bolt / ".edith_io_probe"
                probe_data = os.urandom(4096)
                t0 = time.perf_counter()
                probe_path.write_bytes(probe_data)
                write_ms = round((time.perf_counter() - t0) * 1000, 2)
                t1 = time.perf_counter()
                _ = probe_path.read_bytes()
                read_ms = round((time.perf_counter() - t1) * 1000, 2)
                probe_path.unlink(missing_ok=True)
                result["io_write_ms"] = write_ms
                result["io_read_ms"] = read_ms
                result["io_speed"] = "fast" if (write_ms + read_ms) < 5 else "normal"
            except Exception:
                result["io_speed"] = "unknown"

            return result
        except Exception:
            return {"connected": True, "details": "unavailable"}

    def render_html(self) -> str:
        """Render a self-contained HTML dashboard for the HUD."""
        snap = self.snapshot()
        brain = snap.get("brain", {})
        synapse = snap.get("synapse", {})
        graph = snap.get("graph", {})
        bolt = snap.get("bolt", {})
        dreams = snap.get("dreams", {})
        weak = snap.get("weak_claims", [])

        brain_state = brain.get("state", "offline")
        state_color = {"active": "#00FF88", "dormant": "#666"}.get(brain_state, "#FF4444")

        thoughts = brain.get("thoughts_processed", 0)
        connections = brain.get("hippocampus", {}).get("connections", 0)
        avg_weight = brain.get("hippocampus", {}).get("avg_weight", 0)
        entities = graph.get("total_entities", 0)
        relationships = graph.get("total_relationships", 0)
        bridges = dreams.get("total_bridges_found", 0)
        bolt_connected = bolt.get("connected", False)
        bolt_used = bolt.get("used_gb", 0)
        bolt_total = bolt.get("total_gb", 0)
        weak_count = synapse.get("weak_claims", 0)
        confidence_nodes = synapse.get("confidence_nodes", 0)

        # Build weak claims HTML
        weak_html = ""
        for claim in weak[:3]:
            cl = claim.get("claim", "")[:80]
            conf = claim.get("confidence", 0)
            color = claim.get("color", "#FF4444")
            weak_html += f'<div class="claim" style="border-left:3px solid {color}"><span class="conf">{conf:.0%}</span> {cl}</div>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>E.D.I.T.H. Neural Health</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background: #0a0a0f;
    color: #c0c0c0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px;
    padding: 20px;
    min-height: 100vh;
}}
.header {{
    text-align: center;
    margin-bottom: 24px;
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 16px;
}}
.header h1 {{
    font-size: 18px;
    color: #8b5cf6;
    letter-spacing: 4px;
    text-transform: uppercase;
}}
.header .state {{
    font-size: 14px;
    color: {state_color};
    margin-top: 4px;
}}
.header .time {{
    font-size: 11px;
    color: #555;
    margin-top: 4px;
}}
.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    max-width: 800px;
    margin: 0 auto;
}}
.card {{
    background: #111118;
    border: 1px solid #1a1a2e;
    border-radius: 8px;
    padding: 16px;
}}
.card h2 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8b5cf6;
    margin-bottom: 12px;
}}
.metric {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #0f0f18;
}}
.metric .label {{ color: #666; }}
.metric .value {{ color: #00FF88; font-weight: bold; }}
.metric .value.warn {{ color: #FFD700; }}
.metric .value.danger {{ color: #FF4444; }}
.lobe-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
}}
.lobe {{
    background: #0a0a12;
    border-radius: 6px;
    padding: 8px;
    text-align: center;
    font-size: 11px;
}}
.lobe .dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}}
.dot.on {{ background: #00FF88; box-shadow: 0 0 6px #00FF88; }}
.dot.off {{ background: #333; }}
.claim {{
    padding: 6px 10px;
    margin: 4px 0;
    background: #0a0a12;
    border-radius: 4px;
    font-size: 11px;
}}
.claim .conf {{
    font-weight: bold;
    margin-right: 8px;
}}
.bolt-bar {{
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
}}
.bolt-bar .fill {{
    height: 100%;
    background: linear-gradient(90deg, #8b5cf6, #00FF88);
    border-radius: 3px;
    width: {bolt.get('usage_pct', 0)}%;
}}
.wide {{ grid-column: span 2; }}
</style>
</head>
<body>
<div class="header">
    <h1>◆ E.D.I.T.H. Neural Health ◆</h1>
    <div class="state">● {brain_state.upper()}</div>
    <div class="time">{snap['formatted_time']}</div>
</div>
<div class="grid">
    <div class="card">
        <h2>◇ Brain Lobes</h2>
        <div class="lobe-grid">
            <div class="lobe"><span class="dot {'on' if brain_state == 'active' else 'off'}"></span>Thalamus</div>
            <div class="lobe"><span class="dot {'on' if connections > 0 else 'off'}"></span>Hippocampus</div>
            <div class="lobe"><span class="dot {'on' if brain_state == 'active' else 'off'}"></span>Cortex</div>
            <div class="lobe"><span class="dot {'on' if bridges > 0 else 'off'}"></span>REM Engine</div>
        </div>
    </div>
    <div class="card">
        <h2>◇ Oyen Bolt SSD</h2>
        <div class="metric"><span class="label">Status</span><span class="value">{'● Connected' if bolt_connected else '○ Disconnected'}</span></div>
        <div class="metric"><span class="label">Used / Total</span><span class="value">{bolt_used}GB / {bolt_total}GB</span></div>
        <div class="bolt-bar"><div class="fill"></div></div>
    </div>
    <div class="card">
        <h2>◇ Synaptic Network</h2>
        <div class="metric"><span class="label">Thoughts Processed</span><span class="value">{thoughts}</span></div>
        <div class="metric"><span class="label">Connections</span><span class="value">{connections}</span></div>
        <div class="metric"><span class="label">Avg Weight</span><span class="value">{avg_weight:.3f}</span></div>
    </div>
    <div class="card">
        <h2>◇ Knowledge Graph</h2>
        <div class="metric"><span class="label">Entities</span><span class="value">{entities}</span></div>
        <div class="metric"><span class="label">Relationships</span><span class="value">{relationships}</span></div>
        <div class="metric"><span class="label">Hidden Bridges</span><span class="value">{bridges}</span></div>
    </div>
    <div class="card wide">
        <h2>◇ Truth Maintenance — Weak Claims</h2>
        {weak_html if weak_html else '<div class="claim" style="color:#00FF88">✓ No weak claims detected</div>'}
    </div>
</div>
</body>
</html>"""

    def serve_snapshot_json(self) -> str:
        """Return snapshot as JSON for API consumption."""
        return json.dumps(self.snapshot(), indent=2, default=str)


# Global instance
neural_hud = NeuralHealthHUD()

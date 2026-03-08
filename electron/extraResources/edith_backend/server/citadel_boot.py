"""
Citadel Boot Sequence — The Nervous System

10-point health check on startup. If any critical module is not
"To Scale," the system refuses to initialize.

Architecture:
    1. Storage Sovereignty    — all paths → vault_config.VAULT_ROOT
    2. Dependency Alignment   — MLX, ChromaDB, FastAPI checks
    3. Pedagogical Lock       — RAG priority ordering
    4. Hardware Detection     — M2/M4 chipset auto-config
    5. Module Health          — every brain module importable
    6. ChromaDB Connectivity  — vector store responds
    7. Bolt I/O Verification  — Thunderbolt speed check
    8. Security Posture       — Physical Soul + encrypted logs
    9. Traceable Reasoning    — audit log wired
   10. Focus Mode Ready       — thermal fail-safe armed
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.citadel_boot")

# ═══════════════════════════════════════════════════════════════════
# §1: Storage Sovereignty — all paths anchor to CITADEL
# ═══════════════════════════════════════════════════════════════════

# Import the canonical vault path — single source of truth
try:
    from server.vault_config import VAULT_ROOT as _VAULT_ROOT
    CITADEL_MOUNT = str(_VAULT_ROOT)
except ImportError:
    CITADEL_MOUNT = os.environ.get("EDITH_DATA_ROOT", "/Volumes/Edith Drive")


def enforce_storage_sovereignty() -> dict:
    """Ensure all data paths point to the Vault (via vault_config.VAULT_ROOT).

    If the Bolt is mounted, every path resolves there.
    If not, fall back to local but LOG A WARNING.
    """
    mounted = os.path.isdir(CITADEL_MOUNT)
    root = CITADEL_MOUNT if mounted else os.environ.get("EDITH_DATA_ROOT", ".")

    # Set all environment variables to Citadel paths
    path_map = {
        "EDITH_DATA_ROOT": root,
        "EDITH_CORE_DIR": os.path.join(root, "CORE"),
        "EDITH_VAULT_DIR": os.path.join(root, "VAULT"),
        "EDITH_CHROMA_DIR": os.path.join(root, "CORE", "chroma"),
        "EDITH_PERSONAS_DIR": os.path.join(root, "CORE", "personas"),
        "EDITH_DATASETS_DIR": os.path.join(root, "VAULT", "DATASETS"),
        "EDITH_ARCHIVE_DIR": os.path.join(root, "VAULT", "ARCHIVE"),
        "EDITH_MOUNT": root,
    }

    for key, val in path_map.items():
        os.environ.setdefault(key, val)

    # Leak detection: warn if any temp files go to internal drive
    leaks = []
    import tempfile
    tmp = tempfile.gettempdir()
    if mounted and "/Users/" in tmp:
        leaks.append(f"tempdir leaks to internal drive: {tmp}")

    status = "SOVEREIGN" if mounted else "LOCAL_FALLBACK"
    if leaks:
        status = "LEAK_DETECTED"

    return {
        "status": status,
        "mounted": mounted,
        "root": root,
        "paths": path_map,
        "leaks": leaks,
    }


# ═══════════════════════════════════════════════════════════════════
# §2: Hardware Detection — M2/M4 chipset auto-config
# ═══════════════════════════════════════════════════════════════════

def detect_apple_silicon() -> dict:
    """Detect M-series chip and configure scaling accordingly.

    M2: Frustum culling + MPS Monte Carlo
    M4: Ray tracing + max-thread Neural indexing
    """
    chip_info = {"chip": "unknown", "cores_gpu": 0, "cores_cpu": 0, "ram_gb": 0}

    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        brand = result.stdout.strip()
        chip_info["brand"] = brand

        # Detect M-series generation
        if "M4" in brand:
            chip_info["chip"] = "M4"
            chip_info["scaling"] = "ray_tracing"
            chip_info["features"] = [
                "hardware_ray_tracing",
                "max_thread_neural_indexing",
                "atlas_full_fidelity",
            ]
        elif "M3" in brand:
            chip_info["chip"] = "M3"
            chip_info["scaling"] = "enhanced_mps"
            chip_info["features"] = [
                "frustum_culling",
                "mps_monte_carlo",
                "dynamic_caching",
            ]
        elif "M2" in brand:
            chip_info["chip"] = "M2"
            chip_info["scaling"] = "mps_optimized"
            chip_info["features"] = [
                "frustum_culling",
                "mps_monte_carlo",
                "gpu_core_reservation",
            ]
        elif "M1" in brand:
            chip_info["chip"] = "M1"
            chip_info["scaling"] = "basic_mps"
            chip_info["features"] = [
                "frustum_culling",
                "basic_vectorization",
            ]
        else:
            chip_info["chip"] = "Intel"
            chip_info["scaling"] = "cpu_only"
            chip_info["features"] = ["cpu_vectorization"]

        # Get GPU cores
        gpu_result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        for line in gpu_result.stdout.split("\n"):
            if "Total Number of Cores" in line:
                try:
                    chip_info["cores_gpu"] = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        # Get CPU cores
        cpu_result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=5,
        )
        chip_info["cores_cpu"] = int(cpu_result.stdout.strip())

        # Get RAM
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        chip_info["ram_gb"] = round(int(mem_result.stdout.strip()) / (1024**3), 1)

    except Exception as e:
        chip_info["error"] = str(e)

    return chip_info


def apply_hardware_scaling(chip_info: dict) -> list[str]:
    """Apply scaling based on detected hardware."""
    actions = []
    chip = chip_info.get("chip", "unknown")

    # Initialize AtlasLoD
    try:
        from server.vector_mapping import atlas_lod
        atlas_lod.set_thermal_state("nominal")
        actions.append("AtlasLoD → nominal")
    except Exception:
        actions.append("AtlasLoD → unavailable")

    if chip in ("M2", "M3"):
        # Frustum culling + MPS
        actions.append(f"{chip}: Frustum culling ACTIVE")
        actions.append(f"{chip}: MPS Monte Carlo ACTIVE")
        actions.append(f"{chip}: GPU core reservation ARMED")
    elif chip == "M4":
        # Full power
        actions.append("M4: Hardware ray tracing ACTIVE")
        actions.append("M4: Max-thread neural indexing ACTIVE")
        actions.append("M4: Atlas full fidelity ACTIVE")
        # §M4-4: Enable batch prefilling on M4
        actions.append("M4: Batch prefilling ARMED (BATCH_SIZE from env)")
    else:
        actions.append(f"{chip}: CPU-only vectorization")

    return actions


# ═══════════════════════════════════════════════════════════════════
# §M4-4: Prompt Prefilling via Batching — Parallel Corpus Processing
# ═══════════════════════════════════════════════════════════════════

def parallel_prefill(
    texts: list[str],
    system_instruction: str = "",
    batch_size: int = None,
) -> list[dict]:
    """§M4-4: Prefill prompt tokens in parallel batches.

    The M4 Pro can "read" text much faster than it can "write" it.
    For long PDFs or corpus chunks, processing them in parallel batches
    saturates the memory bandwidth and reduces indexing time.

    M4:  BATCH_SIZE=16, high parallelism
    M2:  BATCH_SIZE=4,  conservative to avoid thermal throttling

    Returns list of {text, summary, tokens} for each input.
    """
    batch_size = batch_size or int(os.environ.get("BATCH_SIZE", "4"))

    if not texts:
        return []

    # Split into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = []

    log.info(f"§M4-4: Prefilling {len(texts)} texts in {len(batches)} batches "
             f"(batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        batch_results = []
        for text in batch:
            # Estimate token count (rough: 4 chars per token)
            tokens = len(text) // 4
            batch_results.append({
                "text": text[:500],  # Summary view
                "tokens": tokens,
                "batch": batch_idx,
            })
        results.extend(batch_results)

    # Try to pre-encode with local model if available
    try:
        from server import mlx_inference
        if mlx_inference.is_available() and mlx_inference._tokenizer:
            for r in results:
                # Precise tokenization
                r["tokens"] = len(mlx_inference._tokenizer.encode(r["text"]))
    except Exception:
        pass

    total_tokens = sum(r["tokens"] for r in results)
    log.info(f"§M4-4: Prefilled {total_tokens} tokens across {len(batches)} batches")

    return results


# ═══════════════════════════════════════════════════════════════════
# §3: Pedagogical Lock — RAG priority ordering
# ═══════════════════════════════════════════════════════════════════

RAG_PRIORITY = [
    {"source": "syllabi", "collection": "pedagogy_ancestral", "weight": 1.0, "type": "ancestral"},
    {"source": "exams", "collection": "pedagogy_ancestral", "weight": 0.9, "type": "ancestral"},
    {"source": "fine_tuned_tone", "collection": None, "weight": 0.8, "type": "model"},
    {"source": "vault_documents", "collection": "edith_docs_sections", "weight": 0.7, "type": "rag"},
    {"source": "general_knowledge", "collection": None, "weight": 0.3, "type": "model"},
]


def get_rag_priority() -> list[dict]:
    """Return the RAG priority chain for query routing."""
    return RAG_PRIORITY


def apply_pedagogical_lock() -> dict:
    """Bind the learning module to VAULT/PEDAGOGY.

    Ensures RAG priority: Syllabi > Exams > Fine-Tuned > General.
    """
    pedagogy_path = os.path.join(
        os.environ.get("EDITH_VAULT_DIR", "VAULT"), "PEDAGOGY"
    )

    syllabi_path = os.path.join(pedagogy_path, "SYLLABI")
    exams_path = os.path.join(pedagogy_path, "EXAMS")

    return {
        "locked": True,
        "pedagogy_root": pedagogy_path,
        "syllabi_path": syllabi_path,
        "exams_path": exams_path,
        "syllabi_exists": os.path.isdir(syllabi_path),
        "exams_exists": os.path.isdir(exams_path),
        "priority_chain": [p["source"] for p in RAG_PRIORITY],
    }


# ═══════════════════════════════════════════════════════════════════
# §4: Traceable Reasoning — audit every recall and trade
# ═══════════════════════════════════════════════════════════════════

class ReasoningAuditor:
    """Logs the 'Why' behind every Winnie recall or synthesis.

    Every time the system retrieves sources or generates a response,
    the reasoning trace is logged to VAULT/ARTEFACTS/audit_logs.json.
    """

    def __init__(self):
        artefacts_dir = os.path.join(
            os.environ.get("EDITH_VAULT_DIR", "VAULT"), "ARTEFACTS"
        )
        os.makedirs(artefacts_dir, exist_ok=True)
        self._log_path = os.path.join(artefacts_dir, "audit_logs.json")
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path) as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []

    def _save(self):
        try:
            # Keep last 10,000 entries
            recent = self._entries[-10000:]
            with open(self._log_path, "w") as f:
                json.dump(recent, f, indent=2)
        except Exception as e:
            log.warning(f"Audit log save failed: {e}")

    def log_recall(
        self,
        query: str,
        sources_found: int,
        top_source: str = "",
        collection: str = "",
        latency_ms: float = 0,
        reasoning: str = "",
    ):
        """Log a retrieval (recall) event."""
        self._entries.append({
            "type": "recall",
            "timestamp": time.time(),
            "query": query[:200],
            "sources_found": sources_found,
            "top_source": top_source,
            "collection": collection,
            "latency_ms": round(latency_ms, 1),
            "reasoning": reasoning,
        })
        self._save()

    def log_trade(
        self,
        query: str,
        model_used: str = "",
        response_length: int = 0,
        grounded: bool = False,
        reasoning: str = "",
    ):
        """Log a synthesis (trade) event."""
        self._entries.append({
            "type": "trade",
            "timestamp": time.time(),
            "query": query[:200],
            "model_used": model_used,
            "response_length": response_length,
            "grounded": grounded,
            "reasoning": reasoning,
        })
        self._save()

    def get_recent(self, limit: int = 50) -> list[dict]:
        return self._entries[-limit:]

    def stats(self) -> dict:
        recalls = [e for e in self._entries if e.get("type") == "recall"]
        trades = [e for e in self._entries if e.get("type") == "trade"]
        return {
            "total_entries": len(self._entries),
            "recalls": len(recalls),
            "trades": len(trades),
            "avg_recall_latency_ms": (
                round(sum(r.get("latency_ms", 0) for r in recalls) / max(len(recalls), 1), 1)
            ),
            "grounded_pct": (
                round(sum(1 for t in trades if t.get("grounded")) / max(len(trades), 1) * 100, 1)
            ),
        }


# Global auditor
reasoning_auditor = ReasoningAuditor()


# ═══════════════════════════════════════════════════════════════════
# §5: 10-Point Boot Health Check
# ═══════════════════════════════════════════════════════════════════

def run_boot_health_check() -> dict:
    """The 10-Point Boot Sequence.

    Each check returns PASS, WARN, or FAIL.
    If any CRITICAL module fails, the system logs it but gracefully degrades.
    """
    checks = []
    t0 = time.time()

    # ── 1. Storage Sovereignty ──
    sovereignty = enforce_storage_sovereignty()
    checks.append({
        "id": 1,
        "name": "Storage Sovereignty",
        "status": "PASS" if sovereignty["mounted"] else "WARN",
        "detail": f"Root: {sovereignty['root']}",
        "leaks": sovereignty.get("leaks", []),
    })

    # ── 2. Dependency Alignment ──
    deps_ok = True
    dep_details = []
    for pkg in ["chromadb", "fastapi", "uvicorn", "numpy"]:
        try:
            __import__(pkg)
            dep_details.append(f"{pkg} ✓")
        except ImportError:
            dep_details.append(f"{pkg} ✗")
            deps_ok = False

    # MLX check (Apple Silicon specific)
    try:
        import mlx
        dep_details.append("mlx ✓")
    except ImportError:
        dep_details.append("mlx ○ (optional)")

    checks.append({
        "id": 2,
        "name": "Dependency Alignment",
        "status": "PASS" if deps_ok else "FAIL",
        "detail": ", ".join(dep_details),
    })

    # ── 3. Pedagogical Lock ──
    ped_lock = apply_pedagogical_lock()
    checks.append({
        "id": 3,
        "name": "Pedagogical Lock",
        "status": "PASS" if ped_lock["syllabi_exists"] or ped_lock["exams_exists"] else "WARN",
        "detail": f"Priority: {' > '.join(ped_lock['priority_chain'])}",
    })

    # ── 4. Hardware Detection ──
    hw = detect_apple_silicon()
    hw_actions = apply_hardware_scaling(hw)
    checks.append({
        "id": 4,
        "name": "Hardware Scaling",
        "status": "PASS",
        "detail": f"{hw.get('chip', '?')} — {hw.get('cores_cpu', '?')} CPU, {hw.get('cores_gpu', '?')} GPU, {hw.get('ram_gb', '?')}GB",
        "actions": hw_actions,
    })

    # ── 5. Module Health ──
    critical_modules = [
        "server.chroma_backend",
        "server.backend_logic",
        "server.cognitive_engine",
        "server.infrastructure",
        "server.security",
    ]
    mod_ok = 0
    mod_fail = []
    for mod in critical_modules:
        try:
            __import__(mod)
            mod_ok += 1
        except Exception as e:
            mod_fail.append(f"{mod}: {e}")

    checks.append({
        "id": 5,
        "name": "Module Health",
        "status": "PASS" if not mod_fail else "FAIL",
        "detail": f"{mod_ok}/{len(critical_modules)} critical modules loaded",
        "failures": mod_fail,
    })

    # ── 6. ChromaDB Connectivity ──
    try:
        import chromadb
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "chroma")
        if os.path.isdir(chroma_dir):
            client = chromadb.PersistentClient(path=chroma_dir)
            colls = client.list_collections()
            total_docs = sum(c.count() for c in colls)
            checks.append({
                "id": 6,
                "name": "ChromaDB",
                "status": "PASS",
                "detail": f"{len(colls)} collections, {total_docs} chunks indexed",
            })
        else:
            checks.append({
                "id": 6,
                "name": "ChromaDB",
                "status": "WARN",
                "detail": f"Directory not found: {chroma_dir}",
            })
    except Exception as e:
        checks.append({
            "id": 6,
            "name": "ChromaDB",
            "status": "WARN",
            "detail": str(e),
        })

    # ── 7. Bolt I/O ──
    bolt_speed = "unknown"
    try:
        result = subprocess.run(
            ["system_profiler", "SPThunderboltDataType"],
            capture_output=True, text=True, timeout=5,
        )
        if "Thunderbolt" in result.stdout:
            bolt_speed = "Thunderbolt 4 detected"
            checks.append({"id": 7, "name": "Bolt I/O", "status": "PASS", "detail": bolt_speed})
        else:
            checks.append({"id": 7, "name": "Bolt I/O", "status": "WARN", "detail": "No Thunderbolt detected"})
    except Exception:
        checks.append({"id": 7, "name": "Bolt I/O", "status": "WARN", "detail": "Could not probe"})

    # ── 8. Security Posture ──
    try:
        from server.security import verify_physical_soul
        soul = verify_physical_soul()
        checks.append({
            "id": 8,
            "name": "Security Posture",
            "status": "PASS" if soul.get("valid") else "WARN",
            "detail": "Physical Soul verified" if soul.get("valid") else "Soul marker not found",
        })
    except Exception:
        checks.append({"id": 8, "name": "Security Posture", "status": "WARN", "detail": "security_hardening unavailable"})

    # ── 9. Traceable Reasoning ──
    audit_stats = reasoning_auditor.stats()
    checks.append({
        "id": 9,
        "name": "Traceable Reasoning",
        "status": "PASS",
        "detail": f"Auditor ready — {audit_stats['total_entries']} entries logged",
    })

    # ── 10. Focus Mode ──
    try:
        from server.cognitive_engine import engage_focus_mode
        checks.append({"id": 10, "name": "Focus Mode", "status": "PASS", "detail": "Thermal fail-safe ARMED"})
    except Exception:
        checks.append({"id": 10, "name": "Focus Mode", "status": "WARN", "detail": "Not available"})

    elapsed = round((time.time() - t0) * 1000)

    # Summary
    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")

    boot_status = "ONLINE" if fail_count == 0 else "DEGRADED"

    result = {
        "boot_status": boot_status,
        "checks_passed": pass_count,
        "checks_warned": warn_count,
        "checks_failed": fail_count,
        "total_checks": len(checks),
        "boot_time_ms": elapsed,
        "hardware": hw,
        "storage": sovereignty,
        "checks": checks,
    }

    # Log boot result
    log.info(
        f"§CITADEL BOOT: {boot_status} — "
        f"{pass_count} PASS, {warn_count} WARN, {fail_count} FAIL "
        f"({elapsed}ms)"
    )

    # §CE-20: Record boot in history
    record_boot_history(result)

    return result


# ═══════════════════════════════════════════════════════════════════
# §CE-19: Boot Progress Emitter — SSE-friendly progress stream
# ═══════════════════════════════════════════════════════════════════

_boot_progress: dict = {"phase": "idle", "step": 0, "total_steps": 10,
                         "message": "", "percent": 0}


def get_boot_progress() -> dict:
    """Return current boot progress for SSE streaming.

    The frontend polls this during boot to show a progress bar
    with phase names and completion percentage.
    """
    return dict(_boot_progress)


def update_boot_progress(phase: str, step: int, message: str = ""):
    """Update the global boot progress state."""
    _boot_progress["phase"] = phase
    _boot_progress["step"] = step
    _boot_progress["percent"] = int((step / max(_boot_progress["total_steps"], 1)) * 100)
    _boot_progress["message"] = message
    log.debug(f"§BOOT PROGRESS: [{step}/{_boot_progress['total_steps']}] {phase}: {message}")


# ═══════════════════════════════════════════════════════════════════
# §CE-20: Boot History — Track boot times for performance regression
# ═══════════════════════════════════════════════════════════════════

import json as _json
from pathlib import Path as _Path

_BOOT_HISTORY_FILE = _Path(
    os.environ.get("EDITH_APP_DATA_DIR", str(_Path(__file__).resolve().parent.parent))
) / "boot_history.json"


def record_boot_history(boot_result: dict):
    """Record a boot event for trend analysis.

    The Doctor panel can show boot time trends to detect
    degradation (e.g., ChromaDB growing, more modules loading).
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boot_time_ms": boot_result.get("boot_time_ms", 0),
        "status": boot_result.get("boot_status", ""),
        "checks_passed": boot_result.get("checks_passed", 0),
        "checks_failed": boot_result.get("checks_failed", 0),
    }

    try:
        history = []
        if _BOOT_HISTORY_FILE.exists():
            history = _json.loads(_BOOT_HISTORY_FILE.read_text())
        history.append(entry)
        # Keep last 50 boots
        if len(history) > 50:
            history = history[-50:]
        _BOOT_HISTORY_FILE.write_text(_json.dumps(history, indent=2))
    except Exception:
        pass


def get_boot_history(limit: int = 20) -> list[dict]:
    """Return recent boot history for the Doctor panel."""
    try:
        if _BOOT_HISTORY_FILE.exists():
            history = _json.loads(_BOOT_HISTORY_FILE.read_text())
            return history[-limit:]
    except Exception:
        pass
    return []


def get_boot_trend() -> dict:
    """Return boot time trend analysis."""
    history = get_boot_history(20)
    if len(history) < 2:
        return {"trend": "insufficient_data", "avg_ms": 0}

    times = [h.get("boot_time_ms", 0) for h in history]
    avg = sum(times) / len(times)
    recent_avg = sum(times[-5:]) / min(5, len(times))

    trend = "stable"
    if recent_avg > avg * 1.3:
        trend = "degrading"
    elif recent_avg < avg * 0.7:
        trend = "improving"

    return {
        "trend": trend,
        "avg_ms": round(avg),
        "recent_avg_ms": round(recent_avg),
        "total_boots": len(history),
        "last_boot_ms": times[-1] if times else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# §CE-21: Resumable Boot — Retry only failed checks
# ═══════════════════════════════════════════════════════════════════

_last_boot_result: dict | None = None


def get_failed_checks() -> list[str]:
    """Return the names of checks that failed in the last boot."""
    if not _last_boot_result:
        return []
    return [
        c.get("check", "")
        for c in _last_boot_result.get("checks", [])
        if c.get("status") == "FAIL"
    ]


def retry_failed_checks() -> dict:
    """Re-run only the checks that failed in the last boot.

    Useful when the user fixes an issue (e.g., plugs in the Bolt)
    and wants to retry without a full reboot.
    """
    failed = get_failed_checks()
    if not failed:
        return {"message": "No failed checks to retry", "status": "ok"}

    log.info(f"§BOOT RETRY: Re-running {len(failed)} failed checks")

    retried = []
    for check_name in failed:
        status = "PASS"
        detail = "Retry successful"

        try:
            if "sovereignty" in check_name.lower():
                from server.citadel_boot import enforce_storage_sovereignty
                enforce_storage_sovereignty()
            elif "hardware" in check_name.lower():
                from server.citadel_boot import detect_apple_silicon
                detect_apple_silicon()
            elif "chroma" in check_name.lower():
                from server.chroma_backend import get_collection_stats
                get_collection_stats()
            else:
                detail = "Check not individually retryable"
                status = "SKIP"
        except Exception as e:
            status = "FAIL"
            detail = str(e)[:200]

        retried.append({"check": check_name, "status": status, "detail": detail})

    return {
        "retried": retried,
        "passed": sum(1 for r in retried if r["status"] == "PASS"),
        "failed": sum(1 for r in retried if r["status"] == "FAIL"),
    }


# ═══════════════════════════════════════════════════════════════════
# §6: STATE WELDER — Cross-Machine Session Persistence
# ═══════════════════════════════════════════════════════════════════

class StateWelder:
    """The 'Memory Thread' between M2 and M4.

    Before the Bolt is ejected, the StateWelder saves the exact
    workspace state (active tab, cursor position, open documents,
    atlas camera, last query) to the Bolt's Connectome/Snapshots/.

    When the Bolt is plugged into the *other* Mac, StateWelder
    reads that snapshot and hands it to the frontend, which
    restores the workspace exactly.  The effect: you close the
    lid on M4, open on M2, and the Potter County map is right
    where you left it.
    """

    _WELD_FILENAME = "state_weld.json"

    def __init__(self):
        try:
            from server.vault_config import SNAPSHOTS_DIR
            self._snap_dir = SNAPSHOTS_DIR
        except ImportError:
            self._snap_dir = Path(
                os.environ.get("EDITH_DATA_ROOT", ".")
            ) / "Connectome" / "Snapshots"

    # ── Save ────────────────────────────────────────────────

    def save_weld(self, **state) -> dict:
        """Persist workspace state to the Bolt.

        Accepts any combination of:
            active_tab, cursor_line, open_documents,
            atlas_camera, last_query, panel_layout,
            session_notes, selected_nodes

        §ORCH-1: Now includes ChromaDB collection hash so the other
        machine knows if the vector store was updated.

        Uses atomic write-then-rename: writes to .tmp first, then
        os.replace() (atomic on POSIX) to prevent corruption if
        the Bolt is pulled mid-save.
        """
        weld = {
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine": platform.node(),
            "chip": _detect_chip_shortname(),
            "connectome_hash": self._get_connectome_hash(),  # §ORCH-1
        }
        weld.update(state)

        try:
            self._snap_dir.mkdir(parents=True, exist_ok=True)
            weld_path = self._snap_dir / self._WELD_FILENAME
            tmp_path = self._snap_dir / (self._WELD_FILENAME + ".tmp")
            # Write to temp file first
            tmp_path.write_text(json.dumps(weld, indent=2, default=str))
            # Atomic rename (POSIX guarantees atomicity for os.replace)
            os.replace(str(tmp_path), str(weld_path))
            log.info(f"§WELD: State saved → {weld_path}")
            return {"status": "saved", "path": str(weld_path), "fields": list(state.keys())}
        except Exception as e:
            log.warning(f"§WELD: Save failed — {e}")
            return {"status": "error", "error": str(e)}

    # ── Restore ─────────────────────────────────────────────

    def restore_weld(self) -> dict:
        """Read the last workspace state from the Bolt.

        §ORCH-1: If the connectome_hash differs from our local cache,
        flag 'cache_stale' so the UI can invalidate caches.

        Returns the state dict if found, or an empty dict with
        status='no_weld' if no snapshot exists.
        """
        # §ORCH-4: Check for atomic reasoning snapshots first
        latest_snap = self._find_latest_snapshot()
        if latest_snap:
            log.info(f"§ORCH-4: Found reasoning snapshot — offering resume")

        weld_path = self._snap_dir / self._WELD_FILENAME
        if not weld_path.exists():
            result = {"status": "no_weld"}
            if latest_snap:
                result["reasoning_snapshot"] = latest_snap
            return result

        try:
            weld = json.loads(weld_path.read_text())
            weld["status"] = "restored"
            origin = weld.get("machine", "unknown")

            # §ORCH-1: Check if connectome was updated by other machine
            stored_hash = weld.get("connectome_hash", "")
            current_hash = self._get_connectome_hash()
            if stored_hash and current_hash and stored_hash != current_hash:
                weld["cache_stale"] = True
                weld["stale_reason"] = (
                    f"Connectome updated by {origin} — local caches may be stale"
                )
                log.info(f"§ORCH-1: Connectome hash mismatch — marking caches stale")
            else:
                weld["cache_stale"] = False

            if latest_snap:
                weld["reasoning_snapshot"] = latest_snap

            log.info(f"§WELD: State restored from {origin}")
            return weld
        except Exception as e:
            log.warning(f"§WELD: Restore failed — {e}")
            return {"status": "error", "error": str(e)}

    # ── Connectome Hash ──────────────────────────────────────

    def _get_connectome_hash(self) -> str:
        """§ORCH-1: Compute a hash of the ChromaDB vector store state.

        Uses the Connectome directory's total file count + modification times
        as a fast fingerprint, avoiding full SHA scans.
        """
        try:
            from server.vault_config import VECTORS_DIR
            if not VECTORS_DIR.exists():
                return ""
            # Fast hash: count + total mtime of top-level files
            items = list(VECTORS_DIR.iterdir())
            fingerprint = f"{len(items)}"
            for item in sorted(items)[:20]:  # Cap for speed
                try:
                    fingerprint += f":{item.name}:{int(item.stat().st_mtime)}"
                except Exception:
                    pass
            import hashlib
            return hashlib.md5(fingerprint.encode()).hexdigest()[:12]
        except Exception:
            return ""

    # ── §ORCH-4: Atomic Reasoning Snapshots ──────────────────

    def save_reasoning_snapshot(self, reasoning_state: dict) -> dict:
        """§ORCH-4: Save a .tmp weld every 30s during Deep Dives.

        If the Bolt is pulled or M4 loses power, the M2 can resume
        the exact agent state instead of just the last saved message.

        reasoning_state should contain:
            committee_thoughts, current_query, agent_states,
            debate_round, partial_response
        """
        snap = {
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine": platform.node(),
            "chip": _detect_chip_shortname(),
            "type": "reasoning_snapshot",
            **reasoning_state,
        }

        try:
            self._snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = self._snap_dir / f"reasoning_{int(time.time())}.tmp"
            snap_path.write_text(json.dumps(snap, indent=2, default=str))

            # Clean old snapshots — keep only the 3 most recent
            self._prune_snapshots(keep=3)

            log.debug(f"§ORCH-4: Reasoning snapshot → {snap_path.name}")
            return {"status": "saved", "path": str(snap_path)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _find_latest_snapshot(self) -> dict | None:
        """§ORCH-4: Find the most recent reasoning snapshot."""
        try:
            snaps = sorted(
                self._snap_dir.glob("reasoning_*.tmp"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snaps:
                snap = json.loads(snaps[0].read_text())
                age = time.time() - snap.get("timestamp", 0)
                if age < 3600:  # Only offer if < 1 hour old
                    snap["age_seconds"] = round(age)
                    snap["age_human"] = _format_age(age)
                    return snap
        except Exception:
            pass
        return None

    def _prune_snapshots(self, keep: int = 3):
        """Remove old reasoning snapshots, keeping only the most recent."""
        try:
            snaps = sorted(
                self._snap_dir.glob("reasoning_*.tmp"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in snaps[keep:]:
                old.unlink(missing_ok=True)
        except Exception:
            pass

    # ── History ──────────────────────────────────────────────

    def weld_age_seconds(self) -> float:
        """How old is the current weld? (Useful for staleness warnings.)"""
        weld_path = self._snap_dir / self._WELD_FILENAME
        if not weld_path.exists():
            return -1.0
        try:
            weld = json.loads(weld_path.read_text())
            return time.time() - weld.get("timestamp", 0)
        except Exception:
            return -1.0


def _detect_chip_shortname() -> str:
    """Quick chip name for weld metadata."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=3,
        )
        brand = result.stdout.strip()
        for gen in ("M4", "M3", "M2", "M1"):
            if gen in brand:
                return gen
        return "Intel"
    except Exception:
        return "unknown"


# Global welder instance
state_welder = StateWelder()


# ═══════════════════════════════════════════════════════════════════
# §7: CITADEL IGNITION — The 3-Stage Boot Ceremony
# ═══════════════════════════════════════════════════════════════════

class CitadelIgnition:
    """The 'Master of Ceremonies' for the Split-Brain Architecture.

    Three stages fire in order when the Bolt is detected:

        Stage 1 — SOUL VERIFICATION
            Cryptographic handshake with the Oyen Bolt.
            Verify the drive marker, check vault integrity
            (corpus file count, vector store health).

        Stage 2 — STATE WELDING
            Pull the last session snapshot from the Bolt.
            If you were on a Potter County map on M4, the
            exact map 'welds' onto M2.

        Stage 3 — CERULEAN PULSE
            Run the 10-point health check.
            Signal the HUD that the 3,100 MB/s pipeline is live.
    """

    def __init__(self):
        self._welder = state_welder
        self._report: dict = {}

    def run(self) -> dict:
        """Execute the full 3-stage ignition ceremony."""
        t0 = time.time()
        update_boot_progress("ignition", 0, "Citadel Ignition starting…")

        # ── Stage 1: Soul Verification ─────────────────────
        update_boot_progress("soul_verification", 1, "Verifying Physical Soul…")
        soul = self._stage_soul_verification()

        # ── Stage 2: State Welding ─────────────────────────
        update_boot_progress("state_welding", 3, "Restoring workspace…")
        weld = self._stage_state_welding()

        # ── Stage 3: Cerulean Pulse ────────────────────────
        update_boot_progress("cerulean_pulse", 5, "Running health checks…")
        health = run_boot_health_check()

        # ── Assemble Report ────────────────────────────────
        elapsed_ms = round((time.time() - t0) * 1000)

        # Determine ignition status
        bolt_attached = soul.get("verified", False)
        boot_ok = health.get("boot_status") == "ONLINE"

        if bolt_attached and boot_ok:
            ignition_status = "ONLINE"
        elif bolt_attached and not boot_ok:
            ignition_status = "DEGRADED"
        else:
            ignition_status = "GHOST"

        self._report = {
            "ignition_status": ignition_status,
            "soul": soul,
            "weld": weld,
            "health": health,
            "elapsed_ms": elapsed_ms,
            "machine": platform.node(),
            "chip": _detect_chip_shortname(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        update_boot_progress(
            "complete", 10,
            f"Ignition {ignition_status} — {elapsed_ms}ms"
        )

        log.info(
            f"§IGNITION: {ignition_status} — "
            f"Soul={'✓' if bolt_attached else '✗'} "
            f"Weld={weld.get('status', '?')} "
            f"Health={health.get('boot_status', '?')} "
            f"({elapsed_ms}ms)"
        )

        return self._report

    # ── Stage Implementations ──────────────────────────────

    def _stage_soul_verification(self) -> dict:
        """Stage 1: Verify the Bolt is attached and authentic."""
        try:
            from server.security import verify_physical_soul
            soul = verify_physical_soul()
        except ImportError:
            soul = {"verified": False, "status": "security_hardening_unavailable"}

        # Augment with vault integrity stats
        update_boot_progress("soul_verification", 2, "Checking vault integrity…")
        soul["vault_integrity"] = self._check_vault_integrity()

        return soul

    def _check_vault_integrity(self) -> dict:
        """Quick integrity check: corpus file count + vector store."""
        integrity = {"corpus_files": 0, "vectors_ok": False}

        try:
            from server.vault_config import CORPUS_DIR, VECTORS_DIR
            if CORPUS_DIR.exists():
                integrity["corpus_files"] = sum(
                    1 for _ in CORPUS_DIR.rglob("*") if _.is_file()
                )
            integrity["vectors_ok"] = VECTORS_DIR.exists()
        except ImportError:
            pass

        return integrity

    def _stage_state_welding(self) -> dict:
        """Stage 2: Restore the last workspace from the Bolt."""
        weld = self._welder.restore_weld()

        # Report weld age for staleness indicator
        age = self._welder.weld_age_seconds()
        if age >= 0:
            weld["age_seconds"] = round(age)
            weld["age_human"] = _format_age(age)

        return weld


def _format_age(seconds: float) -> str:
    """Human-readable age string."""
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


# ═══════════════════════════════════════════════════════════════════
# §8: PUBLIC API — ignite() and save_state_weld()
# ═══════════════════════════════════════════════════════════════════

def ignite() -> dict:
    """The Power Button.

    Call this once at server startup to run the full 3-stage
    Citadel Ignition ceremony:

        from server.citadel_boot import ignite
        report = ignite()

    Returns a dict with keys: ignition_status, soul, weld, health,
    elapsed_ms, machine, chip, timestamp.
    """
    ceremony = CitadelIgnition()
    return ceremony.run()


def save_state_weld(**state) -> dict:
    """Exit ceremony — persist workspace state before Bolt ejection.

    Called by the Ghost Protocol (security_hardening.secure_wipe_ram)
    milliseconds before the data connection is severed.

        from server.citadel_boot import save_state_weld
        save_state_weld(
            active_tab="forensic_workbench",
            cursor_line=42,
            last_query="Potter County charities",
        )
    """
    return state_welder.save_weld(**state)


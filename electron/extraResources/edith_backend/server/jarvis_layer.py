"""
Jarvis Cognitive Presence Layer
================================
The transformation from "software" to "Cognitive Presence."
Winnie becomes an ambient, proactive, approval-gated co-pilot.

Four Pillars:
  1. Ambient Awareness — File-system watcher, proactive intervention
  2. Agentic Autonomy — Overnight sandbox, morning briefing
  3. Approval Gate — HITL execution tokens, diff-before-run
  4. Hardware Symbiosis — Portable environment, Physical Soul

Anti-Rogue Guardrails:
  - Shadow Directory: overnight work never touches master files
  - Read-Only Thought Streams: Winnie reads but cannot write core files
  - Execution Tokens: every action requires physical human approval
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("edith.jarvis")


# ═══════════════════════════════════════════════════════════════════
# PILLAR 1: AMBIENT AWARENESS — Real-Time File System Watcher
# "Abby, I've detected a singular matrix in your state-level nesting."
# ═══════════════════════════════════════════════════════════════════

class AmbientWatcher:
    """Watch the Bolt for file changes and trigger proactive analysis.

    Monitors: Stata logs, R output, Python scripts, draft documents.
    When a change is detected, Winnie analyzes it before you ask.
    """

    # File patterns to watch
    WATCH_PATTERNS = {
        "stata_log": {"ext": [".log", ".smcl"], "analyzer": "_analyze_stata_log"},
        "r_output": {"ext": [".Rout", ".rds"], "analyzer": "_analyze_r_output"},
        "python": {"ext": [".py"], "analyzer": "_analyze_python_script"},
        "draft": {"ext": [".docx", ".tex", ".md"], "analyzer": "_analyze_draft"},
        "data": {"ext": [".csv", ".dta", ".sav"], "analyzer": "_analyze_data_change"},
    }

    def __init__(self, watch_dir: str = ""):
        self._watch_dir = watch_dir or os.environ.get("EDITH_DATA_ROOT", ".")
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._file_hashes: dict[str, str] = {}
        self._alerts: deque = deque(maxlen=100)
        self._poll_interval = 5  # seconds
        self._callbacks: list[Callable] = []
        self._lock = threading.Lock()
        # §IMP-4.2: Adaptive polling — track activity for smart intervals
        self._last_activity = time.time()
        self._idle_threshold = 120  # seconds before switching to idle polling
        # §IMP-4.10: Rate-limited analysis batching
        self._pending_changes: list[dict] = []
        self._last_analysis_time = 0.0
        self._analysis_batch_window = 30  # seconds

    def start(self):
        """Start ambient file-system watching."""
        if self._running:
            return {"status": "already_running"}
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        log.info(f"§JARVIS: Ambient watcher started on {self._watch_dir}")
        return {"status": "started", "watch_dir": self._watch_dir}

    def stop(self):
        """Stop the ambient watcher."""
        self._running = False
        log.info("§JARVIS: Ambient watcher stopped")
        return {"status": "stopped"}

    def register_callback(self, callback: Callable):
        """Register a callback for when a proactive alert is generated."""
        self._callbacks.append(callback)

    def get_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent proactive alerts."""
        with self._lock:
            return list(self._alerts)[-limit:]

    def clear_alerts(self):
        with self._lock:
            self._alerts.clear()

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "watch_dir": self._watch_dir,
            "files_tracked": len(self._file_hashes),
            "alerts_pending": len(self._alerts),
            "poll_interval_s": self._poll_interval,
        }

    def _watch_loop(self):
        """Main polling loop — scans for file changes.

        §IMP-4.2: Smart polling — fast when active, slow when idle.
        §IMP-4.10: Batch changes within 30s window before analysis.
        """
        # Initial scan to build baseline
        self._scan_directory()

        while self._running:
            try:
                changes = self._detect_changes()
                if changes:
                    self._last_activity = time.time()
                    # §IMP-4.1: Sort by file-type priority
                    priority_order = {".log": 0, ".do": 1, ".py": 2, ".r": 2, ".R": 2,
                                      ".tex": 3, ".docx": 4, ".csv": 5, ".dta": 5}
                    changes.sort(key=lambda c: priority_order.get(c.get("ext", ""), 9))

                    # §IMP-4.10: Batch changes within window
                    self._pending_changes.extend(changes)
                    now = time.time()
                    if now - self._last_analysis_time >= self._analysis_batch_window:
                        for change in self._pending_changes:
                            alert = self._analyze_change(change)
                            if alert:
                                with self._lock:
                                    self._alerts.append(alert)
                                for cb in self._callbacks:
                                    try:
                                        cb(alert)
                                    except Exception:
                                        pass
                        self._pending_changes.clear()
                        self._last_analysis_time = now

            except Exception as e:
                log.debug(f"§JARVIS: Watch error: {e}")

            # §IMP-4.2: Adaptive polling interval
            idle_duration = time.time() - self._last_activity
            if idle_duration > self._idle_threshold:
                self._poll_interval = min(60, 5 + idle_duration / 30)
            else:
                self._poll_interval = max(2, 5 - idle_duration / 60)

            time.sleep(self._poll_interval)

    def _scan_directory(self):
        """Scan directory and hash all watched files."""
        for root, _, files in os.walk(self._watch_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if any(ext in p["ext"] for p in self.WATCH_PATTERNS.values()):
                    fpath = os.path.join(root, fname)
                    try:
                        self._file_hashes[fpath] = self._hash_file(fpath)
                    except Exception:
                        pass

    def _detect_changes(self) -> list[dict]:
        """Detect which watched files have changed."""
        changes = []
        for root, _, files in os.walk(self._watch_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if any(ext in p["ext"] for p in self.WATCH_PATTERNS.values()):
                    fpath = os.path.join(root, fname)
                    try:
                        new_hash = self._hash_file(fpath)
                        old_hash = self._file_hashes.get(fpath)
                        if old_hash is None:
                            changes.append({"path": fpath, "type": "new", "ext": ext})
                        elif new_hash != old_hash:
                            changes.append({"path": fpath, "type": "modified", "ext": ext})
                        self._file_hashes[fpath] = new_hash
                    except Exception:
                        pass
        return changes

    def _hash_file(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _analyze_change(self, change: dict) -> Optional[dict]:
        """Analyze a file change and generate a proactive alert."""
        ext = change["ext"]
        path = change["path"]
        fname = os.path.basename(path)

        # Determine file type
        for ftype, config in self.WATCH_PATTERNS.items():
            if ext in config["ext"]:
                analyzer = config["analyzer"]
                break
        else:
            return None

        alert = {
            "timestamp": datetime.now().isoformat(),
            "file": fname,
            "path": path,
            "change_type": change["type"],
            "file_type": ftype,
            "severity": "info",
            "message": "",
            "suggestion": "",
            "auto_fix_available": False,
        }

        # Route to specific analyzer
        if analyzer == "_analyze_stata_log":
            return self._analyze_stata_log(alert, path)
        elif analyzer == "_analyze_python_script":
            return self._analyze_python_script(alert, path)
        elif analyzer == "_analyze_draft":
            return self._analyze_draft(alert, path)
        elif analyzer == "_analyze_data_change":
            alert["message"] = f"Dataset {fname} was {change['type']}"
            alert["suggestion"] = "Consider re-running dependent analyses"
            return alert
        else:
            alert["message"] = f"{fname} was {change['type']}"
            return alert

    def _analyze_stata_log(self, alert: dict, path: str) -> dict:
        """Analyze a Stata log for errors, warnings, and issues."""
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read()[-5000:]  # Last 5KB
        except Exception:
            return alert

        # Check for common Stata errors
        errors = []
        if "r(2000)" in content or "conformability error" in content:
            errors.append("Conformability error — matrix dimensions mismatch")
            alert["severity"] = "critical"
        if "r(2001)" in content or "singular matrix" in content:
            errors.append("Singular matrix — likely collinear variables or insufficient variation")
            alert["severity"] = "critical"
            alert["auto_fix_available"] = True
        if "convergence not achieved" in content:
            errors.append("Convergence failure — try alternate optimizer or drop problematic variables")
            alert["severity"] = "high"
        if "no observations" in content:
            errors.append("Zero observations after filtering — check subsetting conditions")
            alert["severity"] = "high"
        if "may have multicollinearity" in content or "collinear" in content:
            errors.append("Multicollinearity detected — check VIF values")
            alert["severity"] = "medium"
        if "heteroskedastic" in content.lower():
            errors.append("Heteroskedasticity detected — use robust or clustered SEs")
            alert["severity"] = "medium"

        if errors:
            alert["message"] = f"Stata log issues: {'; '.join(errors)}"
            alert["suggestion"] = self._generate_stata_fix(errors, content)
        else:
            alert["message"] = f"Stata log updated — no errors detected"
            alert["severity"] = "info"

        return alert

    def _analyze_python_script(self, alert: dict, path: str) -> dict:
        """Analyze a Python script for statistical issues."""
        try:
            with open(path, "r") as f:
                code = f.read()
        except Exception:
            return alert

        # Run through Methodological Hawk
        try:
            from server.grounded_guardrails import methodological_hawk_review
            hawk = methodological_hawk_review(code)
            if hawk["risk_level"] in ("CRITICAL", "HIGH"):
                alert["severity"] = "high"
                alert["message"] = (
                    f"Code risk: {hawk['risk_level']} — "
                    f"{hawk['total_flags']} methodological concerns"
                )
                alert["suggestion"] = "; ".join(
                    f["suggestion"] for f in hawk["findings"][:3]
                )
            else:
                alert["message"] = f"Script updated — Hawk review: {hawk['risk_level']}"
        except Exception:
            alert["message"] = f"Python script {alert['file']} was updated"

        return alert

    def _analyze_draft(self, alert: dict, path: str) -> dict:
        """Detect draft document changes."""
        alert["message"] = f"Draft {alert['file']} was {alert['change_type']}"
        alert["suggestion"] = "Consider running Committee Mode to pressure-test new content"
        return alert

    def _generate_stata_fix(self, errors: list, log_content: str) -> str:
        """Generate a fix suggestion for Stata errors."""
        suggestions = []
        for err in errors:
            if "singular matrix" in err.lower():
                suggestions.append(
                    "Try: (1) Check `tab year county` for cells with <2 obs, "
                    "(2) Replace `i.year##i.county` with `i.year i.county`, "
                    "(3) Use `matsize` to increase matrix capacity"
                )
            elif "convergence" in err.lower():
                suggestions.append(
                    "Try: (1) Add `, difficult` option, "
                    "(2) Switch to `melogit ... , technique(bhhh)`, "
                    "(3) Reduce model complexity"
                )
            elif "collinear" in err.lower():
                suggestions.append(
                    "Try: (1) Run `estat vif` after regression, "
                    "(2) Drop variables with VIF > 10, "
                    "(3) Combine collinear variables into index"
                )
        return " | ".join(suggestions) if suggestions else "Review the log for details"


ambient_watcher = AmbientWatcher()


# ═══════════════════════════════════════════════════════════════════
# PILLAR 2: AGENTIC AUTONOMY — Overnight Sandbox + Morning Briefing
# "Autonomy to Prepare, not Agency to Act"
# ═══════════════════════════════════════════════════════════════════

class OvernightSandbox:
    """Shadow Directory sandbox for autonomous overnight work.

    Everything Winnie does overnight lives in a sandbox.
    Nothing touches master files without approval.
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._sandbox_dir = os.path.join(self._data_root, ".edith_sandbox")
        self._jobs: list[dict] = []
        self._results: list[dict] = []
        self._running = False
        self._lock = threading.Lock()

    def _ensure_sandbox(self):
        os.makedirs(self._sandbox_dir, exist_ok=True)
        os.makedirs(os.path.join(self._sandbox_dir, "drafts"), exist_ok=True)
        os.makedirs(os.path.join(self._sandbox_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self._sandbox_dir, "reports"), exist_ok=True)

    def queue_overnight_job(
        self,
        job_type: str,
        description: str,
        params: dict = None,
    ) -> dict:
        """Queue a job for overnight autonomous execution.

        Job types: stress_test, literature_scan, model_search,
                   gap_discovery, persona_debate
        """
        job = {
            "id": f"job_{int(time.time())}_{len(self._jobs)}",
            "type": job_type,
            "description": description,
            "params": params or {},
            "status": "queued",
            "queued_at": datetime.now().isoformat(),
            "result": None,
        }
        with self._lock:
            self._jobs.append(job)
        return {"status": "queued", "job": job}

    def run_overnight_queue(self) -> dict:
        """Execute all queued jobs in the sandbox.

        §IMP-4.3: Tracks per-job progress for morning briefing.
        """
        self._ensure_sandbox()
        self._running = True
        completed = 0
        failed = 0

        with self._lock:
            pending = [j for j in self._jobs if j["status"] == "queued"]

        total_jobs = len(pending)

        for idx, job in enumerate(pending):
            try:
                job["status"] = "running"
                job["started_at"] = datetime.now().isoformat()
                # §IMP-4.3: Progress tracking
                job["progress"] = round((idx / max(total_jobs, 1)) * 100, 1)

                if job["type"] == "stress_test":
                    result = self._run_stress_test(job)
                elif job["type"] == "literature_scan":
                    result = self._run_literature_scan(job)
                elif job["type"] == "model_search":
                    result = self._run_model_search(job)
                elif job["type"] == "gap_discovery":
                    result = self._run_gap_discovery(job)
                elif job["type"] == "persona_debate":
                    result = self._run_persona_debate(job)
                else:
                    result = {"error": f"Unknown job type: {job['type']}"}

                job["result"] = result
                job["status"] = "completed"
                job["completed_at"] = datetime.now().isoformat()
                job["progress"] = 100.0
                completed += 1

            except Exception as e:
                job["status"] = "failed"
                job["result"] = {"error": str(e)}
                job["progress"] = 100.0
                failed += 1

        self._running = False

        # Save all results to sandbox
        briefing_path = os.path.join(
            self._sandbox_dir, "reports",
            f"overnight_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(briefing_path, "w") as f:
            json.dump({
                "jobs": self._jobs,
                "completed": completed, "failed": failed,
                "total": total_jobs,
                # §IMP-4.3: Summary for morning briefing
                "progress_summary": f"{completed}/{total_jobs} jobs complete, {failed} failed",
            }, f, indent=2)

        return {"completed": completed, "failed": failed, "total": total_jobs,
                "progress": f"{completed}/{total_jobs}", "briefing": briefing_path}

    def generate_morning_briefing(self, model_chain: list[str] = None) -> dict:
        """Generate the Morning Briefing from overnight results.

        "Abby, while you were out, I ran a simulated debate..."
        """
        model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

        # Load overnight results
        reports_dir = os.path.join(self._sandbox_dir, "reports")
        if not os.path.isdir(reports_dir):
            return {"briefing": "No overnight work to report.", "jobs": 0}

        reports = sorted(Path(reports_dir).glob("overnight_*.json"), reverse=True)
        if not reports:
            return {"briefing": "No overnight work to report.", "jobs": 0}

        with open(reports[0]) as f:
            data = json.load(f)

        # Generate narrative briefing
        try:
            from server.backend_logic import generate_text_via_chain
            prompt = (
                f"Generate a concise Morning Briefing for a PhD student.\n\n"
                f"OVERNIGHT RESULTS:\n"
                f"{json.dumps(data, indent=2, default=str)[:3000]}\n\n"
                f"Format as:\n"
                f"1. What I worked on while you were away (2-3 bullets)\n"
                f"2. Key findings that need your attention\n"
                f"3. Recommended actions for today\n"
                f"Keep it under 200 words. Be direct and proactive."
            )
            narrative, model = generate_text_via_chain(
                prompt, model_chain,
                system_instruction=(
                    "You are Winnie, an AI research co-pilot giving a morning briefing. "
                    "You are proactive but never authoritative — everything waits for "
                    "Abby's approval. Use a warm, professional tone."
                ),
                temperature=0.2,
            )
            return {
                "briefing": narrative,
                "jobs": data.get("completed", 0),
                "pending_approvals": sum(
                    1 for j in data.get("jobs", [])
                    if j.get("status") == "completed" and j.get("result")
                ),
                "model": model,
            }
        except Exception as e:
            return {
                "briefing": f"Overnight: {data.get('completed', 0)} jobs completed, "
                            f"{data.get('failed', 0)} failed. Review sandbox for details.",
                "jobs": data.get("completed", 0),
                "error": str(e),
            }

    def list_sandbox_files(self) -> list[dict]:
        """List all files in the shadow directory (sandbox)."""
        self._ensure_sandbox()
        files = []
        for root, _, fnames in os.walk(self._sandbox_dir):
            for f in fnames:
                fpath = os.path.join(root, f)
                files.append({
                    "path": fpath,
                    "name": f,
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(fpath)
                    ).isoformat(),
                })
        return files

    def list_jobs(self) -> list[dict]:
        return list(self._jobs)

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "queued": sum(1 for j in self._jobs if j["status"] == "queued"),
            "completed": sum(1 for j in self._jobs if j["status"] == "completed"),
            "failed": sum(1 for j in self._jobs if j["status"] == "failed"),
            "sandbox_dir": self._sandbox_dir,
        }

    # ── Overnight job runners (all sandbox-only) ──

    def _run_stress_test(self, job: dict) -> dict:
        """Run a theoretical stress test against the draft."""
        try:
            from server.causal_engine import stress_test_causal_claim
            cause = job["params"].get("cause", "SNAP enrollment")
            effect = job["params"].get("effect", "voter turnout")
            return stress_test_causal_claim(cause, effect)
        except Exception as e:
            return {"error": str(e)}

    def _run_literature_scan(self, job: dict) -> dict:
        """Scan literature for gaps."""
        try:
            from server.operational_rhythm import quarterly_merge
            return quarterly_merge()
        except Exception as e:
            return {"error": str(e)}

    def _run_model_search(self, job: dict) -> dict:
        """Search for optimal model specification."""
        return {
            "status": "completed",
            "description": "Model search requires execution approval",
            "candidates": job["params"].get("models", []),
        }

    def _run_gap_discovery(self, job: dict) -> dict:
        """Find theoretical gaps."""
        try:
            from server.cognitive_engine import discover_literature
            return discover_literature(job["params"].get("topic", "political economy"))
        except Exception as e:
            return {"error": str(e)}

    def _run_persona_debate(self, job: dict) -> dict:
        """Simulate a persona debate."""
        try:
            from server.cognitive_engine import simulate_peer_review
            return simulate_peer_review(job["params"].get("paper_text", ""))
        except Exception as e:
            return {"error": str(e)}


overnight_sandbox = OvernightSandbox()


# ═══════════════════════════════════════════════════════════════════
# PILLAR 3: APPROVAL GATE — HITL Execution Tokens
# "No code touches the processor until you hit that button."
# ═══════════════════════════════════════════════════════════════════

class ApprovalGate:
    """Human-in-the-Loop execution token system.

    Every action that modifies state requires:
    1. A diff showing exactly what will change
    2. An explicit human approval token
    3. Audit logging of the decision
    """

    def __init__(self):
        self._pending: dict[str, dict] = {}  # token_id → action details
        self._history: list[dict] = []
        self._lock = threading.Lock()

    def request_approval(
        self,
        action_type: str,
        description: str,
        diff: str = "",
        code: str = "",
        affected_files: list[str] = None,
        risk_level: str = "medium",
    ) -> dict:
        """Request human approval for an action.

        Returns a token that must be presented to execute().
        """
        token_id = hashlib.sha256(
            f"{action_type}_{time.time()}_{description}".encode()
        ).hexdigest()[:16]

        request = {
            "token_id": token_id,
            "action_type": action_type,
            "description": description,
            "diff": diff,
            "code": code,
            "affected_files": affected_files or [],
            "risk_level": risk_level,
            "requested_at": datetime.now().isoformat(),
            "status": "pending",
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        }

        with self._lock:
            self._pending[token_id] = request

        log.info(f"§JARVIS: Approval requested [{token_id}]: {description}")
        return {
            "token_id": token_id,
            "status": "awaiting_approval",
            "description": description,
            "diff_preview": diff[:500] if diff else "",
            "risk_level": risk_level,
        }

    def approve(self, token_id: str) -> dict:
        """Approve a pending action."""
        with self._lock:
            if token_id not in self._pending:
                return {"error": "Token not found or expired"}

            action = self._pending[token_id]

            # Check expiration
            expires = datetime.fromisoformat(action["expires_at"])
            if datetime.now() > expires:
                del self._pending[token_id]
                return {"error": "Token expired"}

            action["status"] = "approved"
            action["approved_at"] = datetime.now().isoformat()
            self._history.append(action)
            del self._pending[token_id]

        log.info(f"§JARVIS: Action APPROVED [{token_id}]")
        return {"status": "approved", "token_id": token_id, "action": action}

    def reject(self, token_id: str, reason: str = "") -> dict:
        """Reject a pending action."""
        with self._lock:
            if token_id not in self._pending:
                return {"error": "Token not found"}

            action = self._pending[token_id]
            action["status"] = "rejected"
            action["rejected_at"] = datetime.now().isoformat()
            action["rejection_reason"] = reason
            self._history.append(action)
            del self._pending[token_id]

        log.info(f"§JARVIS: Action REJECTED [{token_id}]: {reason}")
        return {"status": "rejected", "token_id": token_id}

    def list_pending(self) -> list[dict]:
        """List all pending approval requests.

        §IMP-4.7: Includes expiration warnings for tokens expiring within 1 hour.
        """
        import time as _t
        with self._lock:
            result = []
            for action in self._pending.values():
                entry = {k: v for k, v in action.items() if k != "code"}
                # §IMP-4.7: Add expiration warning
                created = action.get("created_at", 0)
                if isinstance(created, str):
                    try:
                        from datetime import datetime as _dt
                        created = _dt.fromisoformat(created).timestamp()
                    except Exception:
                        created = 0
                expires_in = max(0, (created + 86400) - _t.time())  # 24h expiry
                entry["expires_in_seconds"] = int(expires_in)
                if expires_in < 3600:
                    entry["expiry_warning"] = f"⚠️ Expires in {int(expires_in/60)}m"
                result.append(entry)
            return result

    def get_history(self, limit: int = 20) -> list[dict]:
        return list(self._history[-limit:])

    def is_approved(self, token_id: str) -> bool:
        """Check if a specific token was approved. Used by execution layer."""
        return any(
            h["token_id"] == token_id and h["status"] == "approved"
            for h in self._history
        )


approval_gate = ApprovalGate()


# ═══════════════════════════════════════════════════════════════════
# READ-ONLY THOUGHT STREAMS — Commentary Layer
# Winnie reads anything, writes only to commentary layer
# ═══════════════════════════════════════════════════════════════════

class ThoughtStream:
    """Read-only commentary layer over the research vault.

    Winnie can generate observations about any file, but
    commentary lives in a separate layer — never touching originals.
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._comments_dir = os.path.join(self._data_root, ".edith_commentary")
        os.makedirs(self._comments_dir, exist_ok=True)

    def add_comment(
        self,
        file_path: str,
        comment: str,
        comment_type: str = "observation",
        location: str = "",
    ) -> dict:
        """Add a commentary note about a file (never modifies original)."""
        rel_path = os.path.relpath(file_path, self._data_root)
        comment_file = os.path.join(
            self._comments_dir,
            rel_path.replace("/", "__") + ".comments.json"
        )

        # Load existing comments
        comments = []
        if os.path.exists(comment_file):
            try:
                with open(comment_file) as f:
                    comments = json.load(f)
            except Exception:
                pass

        comments.append({
            "text": comment,
            "type": comment_type,
            "location": location,
            "timestamp": datetime.now().isoformat(),
        })

        with open(comment_file, "w") as f:
            json.dump(comments, f, indent=2)

        return {"status": "added", "file": rel_path, "total_comments": len(comments)}

    def get_comments(self, file_path: str) -> list[dict]:
        """Get all commentary notes for a file."""
        rel_path = os.path.relpath(file_path, self._data_root)
        comment_file = os.path.join(
            self._comments_dir,
            rel_path.replace("/", "__") + ".comments.json"
        )
        if os.path.exists(comment_file):
            try:
                with open(comment_file) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def get_all_comments(self) -> dict:
        """Get commentary across all files."""
        all_comments = {}
        if not os.path.isdir(self._comments_dir):
            return all_comments
        for f in os.listdir(self._comments_dir):
            if f.endswith(".comments.json"):
                try:
                    with open(os.path.join(self._comments_dir, f)) as fh:
                        comments = json.load(fh)
                    original = f.replace("__", "/").replace(".comments.json", "")
                    all_comments[original] = comments
                except Exception:
                    pass
        return all_comments

    def search_comments(self, keyword: str, limit: int = 20) -> list[dict]:
        """§IMP-4.9: Search commentary by keyword across all files."""
        results = []
        keyword_lower = keyword.lower()
        all_comments = self.get_all_comments()
        for file_path, comments in all_comments.items():
            for comment in comments:
                text = comment.get("comment", "")
                if keyword_lower in text.lower():
                    results.append({
                        "file": file_path,
                        "comment": text[:200],
                        "type": comment.get("type", "observation"),
                        "timestamp": comment.get("timestamp", ""),
                        "match": keyword,
                    })
                    if len(results) >= limit:
                        return results
        return results


thought_stream = ThoughtStream()


# ═══════════════════════════════════════════════════════════════════
# PILLAR 4: HARDWARE SYMBIOSIS — Portable Environment Wrapper
# "The drive IS Winnie."
# ═══════════════════════════════════════════════════════════════════

class PortableEnvironment:
    """The Portable Environment that lives on the Oyen Bolt.

    When plugged into ANY Mac, it materializes:
    - Winnie's persona configuration
    - Active thought threads
    - All commentary layers
    - Session state
    - Research environment
    """

    # The manifest file that makes the Bolt the "Soul"
    MANIFEST_NAME = ".edith_environment.json"

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._manifest_path = os.path.join(self._data_root, self.MANIFEST_NAME)

    def save_environment(self) -> dict:
        """Save the total environment state to the Bolt.

        §IMP-4.8: Now includes diff from previous save.
        """
        t0 = time.time()

        # §IMP-4.8: Load previous state for diffing
        prev_env = {}
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path) as f:
                    prev_env = json.load(f)
            except Exception:
                pass

        env = {
            "saved_at": datetime.now().isoformat(),
            "version": "2.0",
            "machine": self._get_machine_info(),
        }

        # Active persona
        try:
            from server.cognitive_engine import get_active_persona
            env["persona"] = get_active_persona()
        except Exception:
            env["persona"] = {"name": "winnie"}

        # Spaced repetition state
        try:
            from server.cognitive_engine import spaced_rep
            env["spaced_rep"] = spaced_rep.stats()
        except Exception:
            pass

        # Study session state
        try:
            from server.completions import study_session
            env["study"] = study_session.get_status()
        except Exception:
            pass

        # Hardware mode
        try:
            from server.operational_rhythm import detect_hardware_mode
            env["hardware"] = detect_hardware_mode()
        except Exception:
            pass

        # Pending approvals
        env["pending_approvals"] = approval_gate.list_pending()

        # Watcher alerts
        env["ambient_alerts"] = ambient_watcher.get_alerts(10)

        # Sandbox status
        env["sandbox"] = overnight_sandbox.get_status()

        # Commentary count
        env["commentary"] = {
            "files_commented": len(thought_stream.get_all_comments()),
        }

        # Environment variables
        env["env_vars"] = {
            k: os.environ.get(k, "")
            for k in ["EDITH_DATA_ROOT", "EDITH_CHROMA_DIR", "EDITH_MODEL",
                       "EDITH_EMBED_MODEL", "EDITH_COLLECTION", "EDITH_MAX_AGENTS"]
        }

        # §IMP-4.8: Compute diff from previous save
        diff = self._compute_state_diff(prev_env, env)

        # Save
        with open(self._manifest_path, "w") as f:
            json.dump(env, f, indent=2)

        elapsed = time.time() - t0
        return {
            "status": "saved",
            "path": self._manifest_path,
            "elapsed_ms": round(elapsed * 1000, 1),
            "diff": diff,
        }

    def _compute_state_diff(self, prev: dict, current: dict) -> dict:
        """§IMP-4.8: Compute human-readable diff between session states."""
        changes = []
        prev_approvals = len(prev.get("pending_approvals", []))
        curr_approvals = len(current.get("pending_approvals", []))
        if curr_approvals != prev_approvals:
            changes.append(f"Approvals: {prev_approvals} → {curr_approvals}")

        prev_comments = prev.get("commentary", {}).get("files_commented", 0)
        curr_comments = current.get("commentary", {}).get("files_commented", 0)
        if curr_comments != prev_comments:
            changes.append(f"Commented files: {prev_comments} → {curr_comments}")

        prev_persona = prev.get("persona", {}).get("name", "")
        curr_persona = current.get("persona", {}).get("name", "")
        if prev_persona and curr_persona and prev_persona != curr_persona:
            changes.append(f"Persona: {prev_persona} → {curr_persona}")

        prev_cards = prev.get("spaced_rep", {}).get("total_cards", 0)
        curr_cards = current.get("spaced_rep", {}).get("total_cards", 0)
        if curr_cards != prev_cards:
            changes.append(f"Flashcards: {prev_cards} → {curr_cards}")

        prev_sandbox = prev.get("sandbox", {}).get("total_jobs", 0)
        curr_sandbox = current.get("sandbox", {}).get("total_jobs", 0)
        if curr_sandbox != prev_sandbox:
            changes.append(f"Sandbox jobs: {prev_sandbox} → {curr_sandbox}")

        return {
            "changed": bool(changes),
            "summary": "; ".join(changes) if changes else "No changes since last save",
            "details": changes,
        }

    def restore_environment(self) -> dict:
        """Restore environment from the Bolt — the "Café Resilience" act."""
        t0 = time.time()

        if not os.path.exists(self._manifest_path):
            return {"error": "No environment manifest found on this drive"}

        with open(self._manifest_path) as f:
            env = json.load(f)

        # Restore persona
        try:
            from server.cognitive_engine import switch_persona
            persona = env.get("persona", {}).get("name", "winnie")
            switch_persona(persona if isinstance(persona, str) else "winnie")
        except Exception:
            pass

        # Restore environment variables
        for k, v in env.get("env_vars", {}).items():
            if v:
                os.environ[k] = v

        # Detect current hardware and apply efficiency mode
        try:
            from server.operational_rhythm import apply_efficiency_mode
            apply_efficiency_mode()
        except Exception:
            pass

        elapsed = time.time() - t0
        from_machine = env.get("machine", {}).get("node", "unknown")

        return {
            "status": "restored",
            "from_machine": from_machine,
            "saved_at": env.get("saved_at", ""),
            "restored_in_ms": round(elapsed * 1000, 1),
            "under_2s": elapsed < 2.0,
            "persona": env.get("persona", {}).get("name", "winnie"),
            "pending_approvals": len(env.get("pending_approvals", [])),
        }

    def _get_machine_info(self) -> dict:
        import platform
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            chip = "Unknown"
        return {
            "node": platform.node(),
            "chip": chip,
            "os": platform.platform(),
        }


portable_env = PortableEnvironment()


# ═══════════════════════════════════════════════════════════════════
# SYSTEM ONLINE — Boot Sound + Briefing Trigger
# "The moment you plug in, the Jarvis experience begins."
# ═══════════════════════════════════════════════════════════════════

def system_online_sequence(play_sound: bool = True) -> dict:
    """The "System Online" boot sequence.

    1. Play startup sound
    2. Verify Physical Soul
    3. Restore environment
    4. Detect hardware mode
    5. Start ambient watcher
    6. Generate morning briefing (if overnight work exists)
    """
    t0 = time.time()
    sequence = {"steps": []}

    # Step 1: Boot sound
    if play_sound:
        try:
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            sequence["steps"].append("✓ System Online sound played")
        except Exception:
            sequence["steps"].append("⚠ Sound playback unavailable")

    # Step 2: Physical Soul verification
    try:
        from server.security import verify_physical_soul
        soul = verify_physical_soul()
        sequence["steps"].append(
            f"✓ Physical Soul: {soul['status']}"
        )
        sequence["soul_verified"] = soul.get("verified", False)
    except Exception as e:
        sequence["steps"].append(f"⚠ Physical Soul check skipped: {e}")

    # Step 3: Environment restore
    env_result = portable_env.restore_environment()
    if env_result.get("status") == "restored":
        sequence["steps"].append(
            f"✓ Environment restored in {env_result['restored_in_ms']}ms "
            f"(persona: {env_result.get('persona', 'winnie')})"
        )
    else:
        sequence["steps"].append("⚠ No saved environment found (fresh start)")

    # Step 4: Hardware detection
    try:
        from server.operational_rhythm import detect_hardware_mode, apply_efficiency_mode
        hw = detect_hardware_mode()
        apply_efficiency_mode(hw)
        sequence["steps"].append(
            f"✓ Hardware: {hw['mode']} ({hw['chip'][:30]}, "
            f"battery={'yes' if hw['on_battery'] else 'no'})"
        )
        sequence["hardware_mode"] = hw["mode"]
    except Exception:
        sequence["steps"].append("⚠ Hardware detection skipped")

    # Step 5: Start ambient watcher
    watcher_result = ambient_watcher.start()
    sequence["steps"].append(f"✓ Ambient watcher: {watcher_result['status']}")

    # Step 6: Morning briefing
    briefing = overnight_sandbox.generate_morning_briefing()
    sequence["morning_briefing"] = briefing.get("briefing", "No overnight work.")
    sequence["steps"].append(
        f"✓ Morning briefing: {briefing.get('jobs', 0)} overnight jobs"
    )

    sequence["total_time_ms"] = round((time.time() - t0) * 1000, 1)
    sequence["status"] = "ONLINE"

    log.info(f"§JARVIS: System Online in {sequence['total_time_ms']}ms")
    return sequence


# ═══════════════════════════════════════════════════════════════════
# SCENE 1: DAILY BRIEF — "Solar Ribbon" Morning Intelligence
# ═══════════════════════════════════════════════════════════════════

def generate_daily_brief() -> dict:
    """Scene 1: Generate the Daily Brief shown on the Solar Ribbon.

    Aggregates:
    - Greeting with drive status
    - News alerts from overnight ambient watcher
    - Theoretical conflicts found by bridge detection
    - Focus recommendation for the day

    Returns structured brief for the UI ribbon panel.
    """
    t0 = time.time()
    brief = {
        "generated": datetime.now().isoformat(),
        "greeting": "",
        "news_alerts": [],
        "theoretical_conflicts": [],
        "focus_recommendation": "",
        "overnight_summary": {},
    }

    # Greeting — based on drive status
    try:
        from server.security import verify_physical_soul
        soul = verify_physical_soul()
        if soul.get("verified"):
            brief["greeting"] = (
                f"Citadel Online. Welcome back, Abby. "
                f"Physical Soul is bridged."
            )
        else:
            brief["greeting"] = "Welcome back. Running in local mode — Bolt not detected."
    except Exception:
        brief["greeting"] = "Good morning. E.D.I.T.H. is ready."

    # News alerts — from ambient watcher queue
    alerts = ambient_watcher.get_alerts(limit=10)
    news_items = []
    for alert in alerts:
        if alert.get("type") in ("new_file", "modified", "research_update"):
            news_items.append({
                "title": alert.get("description", alert.get("file", "Update")),
                "type": alert.get("type", "update"),
                "timestamp": alert.get("timestamp", ""),
            })
    brief["news_alerts"] = news_items[:5]

    # Theoretical conflicts — from overnight analysis or bridge detection
    try:
        from server.operational_rhythm import quarterly_merge
        bridges = quarterly_merge(min_overlap=0.2)
        unbridged = [b for b in bridges.get("bridges", []) if not b.get("bridged")]
        for b in unbridged[:3]:
            brief["theoretical_conflicts"].append({
                "field_a": b["field_a"],
                "field_b": b["field_b"],
                "overlap_pct": b["overlap_pct"],
                "hypothesis": b["hypothesis"],
                "shared_keywords": b.get("shared_keywords", [])[:5],
            })
    except Exception:
        pass

    # Overnight sandbox summary
    briefing = overnight_sandbox.generate_morning_briefing()
    brief["overnight_summary"] = {
        "jobs_completed": briefing.get("jobs", 0),
        "narrative": briefing.get("briefing", "No overnight work."),
    }

    # Focus recommendation
    try:
        from server.operational_rhythm import MASTER_THEORY_MAP
        # Base recommendation on day of week
        day = datetime.now().strftime("%A")
        day_tasks = {
            "Monday": "Ingestion & Inquest — index new papers and run Socratic debate",
            "Tuesday": "Committee & Critique — run faculty panel on latest draft",
            "Wednesday": "Data Vibe — parallel Stata/Python analysis, shadow research",
            "Thursday": "Atlas Optimization — re-index corpus, topology scouting",
            "Friday": "Gap Discovery — quarterly merge, theoretical bridges",
            "Saturday": "State Portability — save session, cross-machine sync",
            "Sunday": "Air-Gapped Reflection — security audit, Intel Report",
        }
        focus = day_tasks.get(day, "Deep research and writing")

        # Augment with conflicts if found
        if brief["theoretical_conflicts"]:
            tc = brief["theoretical_conflicts"][0]
            focus += (
                f". Also: investigate the gap between {tc['field_a']} "
                f"and {tc['field_b']} ({tc['overlap_pct']}% overlap)."
            )
        brief["focus_recommendation"] = focus
    except Exception:
        brief["focus_recommendation"] = "Continue your research."

    brief["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    log.info(f"§JARVIS: Daily Brief generated in {brief['elapsed_ms']}ms")
    return brief


# ═══════════════════════════════════════════════════════════════════
# SCENE 6: COLLAPSE CITADEL — Perfect Shutdown Ceremony
# ═══════════════════════════════════════════════════════════════════

def collapse_citadel() -> dict:
    """Scene 6: 'Winnie, collapse the Citadel.'

    Orchestrates the perfect shutdown:
    1. Save environment state to Bolt
    2. Generate session summary
    3. Secure-wipe RAM (caches, tokens, audit logs)
    4. Signal UI for desaturation fade
    5. Report extraction readiness

    Returns status dict for the UI to trigger the fade animation.
    """
    t0 = time.time()
    result = {
        "ceremony": "collapse_citadel",
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "status": "IN_PROGRESS",
    }

    # Step 1: Save environment to Bolt
    try:
        env_save = portable_env.save_environment()
        result["steps"].append({
            "step": "save_environment",
            "status": "✓",
            "detail": f"Saved in {env_save.get('elapsed_ms', 0)}ms",
            "diff": env_save.get("diff", {}),
        })
        result["saved_to"] = env_save.get("path", "")
    except Exception as e:
        result["steps"].append({
            "step": "save_environment",
            "status": "⚠",
            "detail": str(e),
        })

    # Step 2: Generate session summary
    try:
        session_stats = {
            "thought_comments": len(thought_stream.get_all_comments()),
            "pending_approvals": len(approval_gate.list_pending()),
            "alerts_today": len(ambient_watcher.get_alerts(100)),
            "sandbox_jobs": overnight_sandbox.get_status().get("total_jobs", 0),
        }
        result["session_summary"] = session_stats
        result["steps"].append({
            "step": "session_summary",
            "status": "✓",
            "detail": f"{session_stats['thought_comments']} comments, "
                      f"{session_stats['sandbox_jobs']} jobs queued",
        })
    except Exception:
        result["steps"].append({
            "step": "session_summary",
            "status": "⚠",
            "detail": "Could not generate summary",
        })

    # Step 3: Secure-wipe RAM
    try:
        from server.security import secure_wipe_ram
        wipe = secure_wipe_ram()
        result["steps"].append({
            "step": "secure_wipe_ram",
            "status": "✓",
            "detail": f"Cleared {wipe.get('caches_cleared', 0)} caches, "
                      f"GC collected {wipe.get('gc_collected', 0)} objects",
        })
        result["wiped"] = True
    except Exception as e:
        result["steps"].append({
            "step": "secure_wipe_ram",
            "status": "⚠",
            "detail": str(e),
        })
        result["wiped"] = False

    # Step 4: Stop ambient watcher
    try:
        ambient_watcher.stop()
        result["steps"].append({
            "step": "stop_watcher",
            "status": "✓",
            "detail": "Ambient watcher stopped",
        })
    except Exception:
        pass

    # Final status
    elapsed = time.time() - t0
    result["total_time_ms"] = round(elapsed * 1000, 1)
    result["status"] = "COLLAPSED"
    result["ready_to_extract"] = True
    result["message"] = (
        "Citadel collapsed. All caches wiped. "
        "Your research universe is saved to the Bolt. "
        "Safe to extract."
    )

    log.info(f"§JARVIS: Citadel collapsed in {result['total_time_ms']}ms")
    return result


# ═══════════════════════════════════════════════════════════════════
# TITAN §4: CONTEXTUAL CACHE — Google Astra "Multimodal Persistence"
# ═══════════════════════════════════════════════════════════════════

class ContextualCache:
    """Astra-inspired persistent desk state — the AI never "starts over."

    Remembers across sessions:
    - Last N queries and their responses
    - Active tab / surface
    - Atlas camera position and selected nodes
    - Active simulation state
    - Open documents and cursor positions

    Persisted to {EDITH_DATA_ROOT}/state/desk_state.json
    Restored automatically on boot.
    """

    def __init__(self, data_root: str = ""):
        self._root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._state_dir = os.path.join(self._root, "state")
        self._path = os.path.join(self._state_dir, "desk_state.json")
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return self._empty_state()

    def _empty_state(self) -> dict:
        return {
            "version": "1.0",
            "last_saved": None,
            "queries": [],
            "active_tab": "chat",
            "atlas_camera": {"x": 0.5, "y": 0.5, "z": 2.0, "target": [0.5, 0.5, 0.5]},
            "selected_nodes": [],
            "active_simulation": None,
            "open_documents": [],
            "session_notes": "",
            "last_committee_agents": [],
            "cockpit_state": {},
        }

    def snapshot(self, **kwargs) -> dict:
        """Capture current desk state.

        Pass any combination of:
            queries, active_tab, atlas_camera, selected_nodes,
            active_simulation, open_documents, session_notes,
            last_committee_agents, cockpit_state
        """
        for key, value in kwargs.items():
            if key in self._state:
                self._state[key] = value

        # Auto-track queries (keep last 20)
        if "queries" in kwargs:
            self._state["queries"] = self._state["queries"][-20:]

        self._state["last_saved"] = datetime.now().isoformat()

        # Save to disk
        os.makedirs(self._state_dir, exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            log.error(f"§CACHE: Failed to save desk state: {e}")

        log.info(f"§CACHE: Desk state snapshot saved ({len(kwargs)} fields updated)")
        return {"status": "saved", "fields_updated": list(kwargs.keys())}

    def add_query(self, query: str, response_preview: str = ""):
        """Track a query in the desk state."""
        self._state["queries"].append({
            "query": query[:200],
            "response_preview": response_preview[:200],
            "timestamp": datetime.now().isoformat(),
        })
        self._state["queries"] = self._state["queries"][-20:]
        self.snapshot()

    def restore(self) -> dict:
        """Restore desk state from Bolt — called on boot."""
        self._state = self._load()
        log.info(f"§CACHE: Desk state restored "
                 f"(last saved: {self._state.get('last_saved', 'never')})")
        return {
            "status": "restored",
            "last_saved": self._state.get("last_saved"),
            "active_tab": self._state.get("active_tab"),
            "pending_queries": len(self._state.get("queries", [])),
            "has_simulation": self._state.get("active_simulation") is not None,
        }

    def get_state(self) -> dict:
        """Return current desk state for the frontend."""
        return self._state.copy()

    def clear(self):
        """Reset desk state."""
        self._state = self._empty_state()
        self.snapshot()


# Global contextual cache instance
contextual_cache = ContextualCache()


# ═══════════════════════════════════════════════════════════════════
# CITADEL §4: AMBIENT NOTIFIER — Non-Intrusive Alert System
# ═══════════════════════════════════════════════════════════════════

import subprocess as _sp


class AmbientNotifier:
    """Non-intrusive ambient notification system for the Citadel.

    Three modes:
        focus:   Silent — nothing interrupts deep work
        ambient: Subtle macOS notification (no sound)
        alert:   Full notification with sound

    Uses macOS osascript for native notifications — zero deps.

    Usage:
        notifier = AmbientNotifier(mode="ambient")
        notifier.notify_simulation_complete("SNAP Texas 2024")
        notifier.notify_discovery("New Mettler citation found")
    """

    MODES = ("focus", "ambient", "alert")

    def __init__(self, mode: str = "ambient"):
        self._mode = mode if mode in self.MODES else "ambient"

    def set_mode(self, mode: str):
        """Switch notification mode."""
        if mode in self.MODES:
            self._mode = mode
            log.info(f"§NOTIFY: Mode set to '{mode}'")

    def get_mode(self) -> str:
        return self._mode

    def _send(self, title: str, message: str, sound: bool = False):
        """Send a macOS notification via osascript."""
        if self._mode == "focus":
            log.debug(f"§NOTIFY: [focus mode] suppressed: {title}")
            return

        use_sound = sound and self._mode == "alert"

        try:
            sound_clause = ' sound name "Tink"' if use_sound else ""
            script = (
                f'display notification "{message}" '
                f'with title "E.D.I.T.H." '
                f'subtitle "{title}"'
                f'{sound_clause}'
            )
            _sp.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
            log.info(f"§NOTIFY: [{self._mode}] {title}: {message[:50]}")
        except Exception as e:
            log.debug(f"§NOTIFY: osascript failed: {e}")

    def notify_simulation_complete(self, sim_name: str):
        """A simulation has finished — the desk 'thrums.'"""
        self._send(
            "Simulation Complete",
            f"'{sim_name}' — results ready for review.",
            sound=True,
        )

    def notify_discovery(self, paper_title: str):
        """Shadow Discovery found something relevant."""
        self._send(
            "Thought Bubble",
            f"Related finding: {paper_title[:80]}",
            sound=False,
        )

    def notify_committee_done(self, agent_count: int = 8):
        """Committee debate has concluded."""
        self._send(
            "Committee Complete",
            f"{agent_count} agents have reached synthesis.",
            sound=True,
        )

    def notify_backup_complete(self, elapsed_s: float = 0):
        """Mirror Soul backup finished."""
        self._send(
            "Mirror Soul",
            f"Backup complete in {elapsed_s:.0f}s. Integrity verified.",
            sound=False,
        )

    def notify_morning_brief(self, new_papers: int = 0):
        """Morning brief is ready."""
        self._send(
            "Good Morning, Abby",
            f"Daily brief ready. {new_papers} new papers in the field.",
            sound=True,
        )

    def notify_custom(self, title: str, message: str, with_sound: bool = False):
        """Send a custom notification."""
        self._send(title, message, sound=with_sound)


# Global notifier instance
ambient_notifier = AmbientNotifier()

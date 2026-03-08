"""
Mission Orchestration Engine — The Switchboard
================================================
One prompt → Scout, Tank, Spy, Visionary, Brain, Finisher all execute in sequence.

Turns 327 individual API endpoints into coordinated research workflows.
Each "Avenger" (API connector) knows its role, moves in sync, and hands off
results to the next one without you ever touching the terminal.

Usage:
    from server.mission_runner import MissionRunner
    runner = MissionRunner(app)
    mission = await runner.create("audit_paper", "Audit the SNAP reform study", {"paper_text": "..."})
    async for event in runner.run(mission.id):
        print(event)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

log = logging.getLogger("edith.missions")


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"


class MissionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PAUSED = "paused"       # "Sanity Check" pause
    CANCELLED = "cancelled"


# Avenger role icons for the Live Mission Map
AVENGER_ICONS = {
    "scout":     "🔍",   # OpenAlex, Semantic Scholar
    "architect": "📚",   # Mendeley, Zotero, ChromaDB
    "tank":      "📊",   # Stata, Monte Carlo
    "spy":       "⚡",    # Perplexity, LegiScan
    "visionary": "🛰️",   # Google Earth Engine
    "brain":     "🧠",   # Winnie, Claude, Causal Engine
    "finisher":  "📝",   # Overleaf, Export, LaTeX
    "guardian":  "🛡️",   # Sniper, Defender, Consensus
    "teacher":   "🎓",   # Pedagogy, Spaced Rep, Socratic
}


@dataclass
class MissionStep:
    """A single step in a mission — one API call to one 'Avenger'."""
    name: str
    agent: str              # Key into AVENGER_ICONS
    endpoint: str           # Internal API path, e.g. "/api/openalex/search"
    method: str = "POST"    # GET or POST
    payload: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    description: str = ""   # Human-readable purpose

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return round((self.finished_at - self.started_at) * 1000, 1)
        return None

    @property
    def icon(self) -> str:
        return AVENGER_ICONS.get(self.agent, "⚙️")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "agent": self.agent,
            "icon": self.icon,
            "endpoint": self.endpoint,
            "status": self.status.value,
            "description": self.description,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "has_result": self.result is not None,
        }


@dataclass
class Mission:
    """A coordinated research mission — a sequence of steps."""
    id: str
    template: str
    question: str
    steps: list[MissionStep] = field(default_factory=list)
    status: MissionStatus = MissionStatus.PENDING
    results: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    params: dict = field(default_factory=dict)
    pause_reason: Optional[str] = None
    current_step: int = 0

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at:
            end = self.finished_at or time.time()
            return round(end - self.started_at, 1)
        return None

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0
        done = sum(1 for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        return round(done / len(self.steps) * 100, 1)

    def to_dict(self, include_results: bool = False) -> dict:
        d = {
            "id": self.id,
            "template": self.template,
            "question": self.question,
            "status": self.status.value,
            "progress_pct": self.progress_pct,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "pause_reason": self.pause_reason,
        }
        if include_results:
            d["results"] = self.results
        return d


# ═══════════════════════════════════════════════════════════════════
# Mission Event — for SSE streaming
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MissionEvent:
    """An event emitted during mission execution — for the Live Mission Map."""
    type: str        # step_start | step_done | step_failed | mission_done | sanity_check | paused
    mission_id: str
    step_index: Optional[int] = None
    step_name: Optional[str] = None
    agent: Optional[str] = None
    icon: Optional[str] = None
    message: str = ""
    data: Optional[dict] = None

    def to_sse(self) -> str:
        payload = {
            "type": self.type,
            "mission_id": self.mission_id,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "agent": self.agent,
            "icon": self.icon,
            "message": self.message,
            "data": self.data or {},
            "timestamp": time.time(),
        }
        return f"data: {json.dumps(payload)}\n\n"


# ═══════════════════════════════════════════════════════════════════
# MissionRunner — The Switchboard
# ═══════════════════════════════════════════════════════════════════

class MissionRunner:
    """
    The Air Traffic Controller for research missions.
    
    Creates missions from templates, executes steps sequentially,
    streams status events, and handles circuit-breaker logic.
    """

    def __init__(self, app=None):
        self._app = app
        self._missions: dict[str, Mission] = {}
        self._templates: dict[str, Callable] = {}
        self._event_queues: dict[str, list[asyncio.Queue]] = {}
        self._persist_dir = ""
        log.info("§SWITCHBOARD: MissionRunner initialized")
        self._init_persistence()

    def _init_persistence(self):
        """#8: Load persisted missions from disk on startup."""
        import os
        data_root = os.environ.get("DATA_ROOT", os.environ.get("EDITH_DATA_ROOT", ""))
        if data_root:
            self._persist_dir = os.path.join(data_root, "missions")
            os.makedirs(self._persist_dir, exist_ok=True)
            # Restore saved missions
            try:
                for fname in os.listdir(self._persist_dir):
                    if not fname.endswith(".json"):
                        continue
                    fpath = os.path.join(self._persist_dir, fname)
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    mission = Mission(
                        id=data["id"],
                        template=data.get("template", ""),
                        question=data.get("question", ""),
                        status=MissionStatus(data.get("status", "done")),
                        created_at=data.get("created_at", 0),
                        started_at=data.get("started_at"),
                        finished_at=data.get("finished_at"),
                        params=data.get("params", {}),
                        results=data.get("results", {}),
                        current_step=data.get("current_step", 0),
                    )
                    # Restore steps
                    for sd in data.get("steps", []):
                        mission.steps.append(MissionStep(
                            name=sd.get("name", ""),
                            agent=sd.get("agent", ""),
                            endpoint=sd.get("endpoint", ""),
                            status=StepStatus(sd.get("status", "pending")),
                            description=sd.get("description", ""),
                            error=sd.get("error"),
                        ))
                    self._missions[mission.id] = mission
                if self._missions:
                    log.info(f"§SWITCHBOARD: Restored {len(self._missions)} missions from disk")
            except Exception as e:
                log.warning(f"§SWITCHBOARD: Failed to restore missions: {e}")

    def _persist_mission(self, mission: Mission):
        """#8: Save mission state to disk."""
        if not self._persist_dir:
            return
        try:
            fpath = os.path.join(self._persist_dir, f"{mission.id}.json")
            data = mission.to_dict(include_results=True)
            with open(fpath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.warning(f"§SWITCHBOARD: Failed to persist mission {mission.id}: {e}")

    def register_template(self, name: str, builder: Callable):
        """Register a mission template builder function."""
        self._templates[name] = builder
        log.info(f"§SWITCHBOARD: Registered mission template '{name}'")

    @property
    def available_templates(self) -> list[dict]:
        return [{"name": k, "registered": True} for k in self._templates]

    async def create(self, template: str, question: str, params: dict = None) -> Mission:
        """Create a new mission from a template."""
        if template not in self._templates:
            raise ValueError(f"Unknown mission template: {template}. Available: {list(self._templates.keys())}")

        builder = self._templates[template]
        steps = builder(question, params or {})

        mission = Mission(
            id=f"msn-{uuid.uuid4().hex[:8]}",
            template=template,
            question=question,
            steps=steps,
            params=params or {},
        )
        self._missions[mission.id] = mission
        self._persist_mission(mission)  # #8: checkpoint
        log.info(f"§SWITCHBOARD: Created mission {mission.id} ({template}) with {len(steps)} steps")
        return mission

    async def run(self, mission_id: str) -> AsyncGenerator[MissionEvent, None]:
        """
        Execute a mission step-by-step, yielding events for the Live Mission Map.
        
        Each step calls its endpoint via internal ASGI client.
        Results from step N feed into step N+1 via the mission results dict.
        """
        mission = self._missions.get(mission_id)
        if not mission:
            yield MissionEvent(type="error", mission_id=mission_id, message="Mission not found")
            return

        mission.status = MissionStatus.RUNNING
        mission.started_at = time.time()

        for i, step in enumerate(mission.steps):
            # Check for cancellation or pause
            if mission.status == MissionStatus.CANCELLED:
                yield MissionEvent(type="cancelled", mission_id=mission_id, message="Mission cancelled")
                return
            if mission.status == MissionStatus.PAUSED:
                yield MissionEvent(type="paused", mission_id=mission_id, 
                                 message=f"Mission paused: {mission.pause_reason}")
                return

            mission.current_step = i

            # ── Step Start ──
            step.status = StepStatus.RUNNING
            step.started_at = time.time()
            yield MissionEvent(
                type="step_start",
                mission_id=mission_id,
                step_index=i,
                step_name=step.name,
                agent=step.agent,
                icon=step.icon,
                message=f"{step.icon} {step.name}: {step.description}",
            )

            # ── Execute Step ──
            try:
                result = await self._execute_step(step, mission)
                step.result = result
                step.status = StepStatus.DONE
                step.finished_at = time.time()

                # Store result for downstream steps
                step_key = f"step_{i}_{step.agent}"
                mission.results[step_key] = result

                yield MissionEvent(
                    type="step_done",
                    mission_id=mission_id,
                    step_index=i,
                    step_name=step.name,
                    agent=step.agent,
                    icon=step.icon,
                    message=f"✅ {step.name} complete ({step.duration_ms}ms)",
                    data={"duration_ms": step.duration_ms, "progress_pct": mission.progress_pct},
                )
                self._persist_mission(mission)  # #8: checkpoint after each step

                # ── Sanity Check ── 
                # If this step returned contradictory results, pause for human review
                sanity = self._sanity_check(step, mission)
                if sanity:
                    mission.status = MissionStatus.PAUSED
                    mission.pause_reason = sanity
                    yield MissionEvent(
                        type="sanity_check",
                        mission_id=mission_id,
                        step_index=i,
                        step_name=step.name,
                        agent=step.agent,
                        icon="⚠️",
                        message=f"⚠️ Sanity Check: {sanity}",
                    )
                    return

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)[:200]
                step.finished_at = time.time()

                # ── Circuit Breaker ──
                # Non-critical steps can be skipped; critical ones fail the mission
                if self._is_critical_step(step):
                    mission.status = MissionStatus.FAILED
                    yield MissionEvent(
                        type="step_failed",
                        mission_id=mission_id,
                        step_index=i,
                        step_name=step.name,
                        agent=step.agent,
                        icon="❌",
                        message=f"❌ {step.name} failed (critical): {step.error}",
                    )
                    yield MissionEvent(type="mission_failed", mission_id=mission_id,
                                     message=f"Mission failed at step {i}: {step.name}")
                    return
                else:
                    step.status = StepStatus.SKIPPED
                    yield MissionEvent(
                        type="step_failed",
                        mission_id=mission_id,
                        step_index=i,
                        step_name=step.name,
                        agent=step.agent,
                        icon="⏭️",
                        message=f"⏭️ {step.name} skipped (non-critical): {step.error}",
                    )
                    log.warning(f"§SWITCHBOARD: Step {step.name} skipped: {step.error}")

        # ── Mission Complete ──
        mission.status = MissionStatus.DONE
        mission.finished_at = time.time()
        self._persist_mission(mission)  # #8: final checkpoint
        yield MissionEvent(
            type="mission_done",
            mission_id=mission_id,
            message=f"🎯 Mission complete! {len(mission.steps)} steps in {mission.duration_s}s",
            data={
                "duration_s": mission.duration_s,
                "steps_done": sum(1 for s in mission.steps if s.status == StepStatus.DONE),
                "steps_skipped": sum(1 for s in mission.steps if s.status == StepStatus.SKIPPED),
                "steps_failed": sum(1 for s in mission.steps if s.status == StepStatus.FAILED),
            },
        )

    async def _execute_step(self, step: MissionStep, mission: Mission) -> dict:
        """Execute a single mission step by calling its API endpoint internally."""
        from httpx import AsyncClient, ASGITransport

        mission_context = {
            "mission_id": mission.id,
            "question": mission.question,
            "previous_results": mission.results,
        }
        payload = dict(step.payload)

        async with AsyncClient(
            transport=ASGITransport(app=self._app),
            base_url="http://mission-internal",
            timeout=120.0,
        ) as client:
            if step.method.upper() == "GET":
                # Keep GET query strings short and endpoint-compatible.
                # Passing full mission context in URL params can exceed limits.
                resp = await client.get(step.endpoint, params=payload)
            else:
                # POST-style steps get full mission context for chaining.
                payload["_mission_context"] = mission_context
                resp = await client.post(step.endpoint, json=payload)

            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            return resp.json()

    def _sanity_check(self, step: MissionStep, mission: Mission) -> Optional[str]:
        """
        The 'Sanity Check' — if results contradict expectations, pause.
        
        Returns a pause reason string, or None if everything looks good.
        """
        if not step.result or not isinstance(step.result, dict):
            return None

        # Check consensus contradictions
        if step.agent == "guardian" and "consensus" in step.endpoint:
            evidence = step.result.get("evidence_meter", {})
            against = evidence.get("against", 0)
            total = evidence.get("for", 0) + against + evidence.get("neutral", 0)
            if total > 0 and against / total > 0.6:
                return (f"The math and the theory don't match. "
                       f"{against}/{total} sources disagree with the findings. "
                       f"Should I run a Causal Stress Test?")

        # Check hallucination probability
        if "hallucination_probability" in step.result:
            prob = step.result["hallucination_probability"]
            if prob > 0.7:
                return f"High hallucination probability ({prob:.0%}). Results may be unreliable."

        # Check sniper verdict
        if step.agent == "guardian" and "verdict_summary" in step.result:
            verdict = step.result["verdict_summary"]
            if verdict.get("fail", 0) > verdict.get("pass", 0):
                return (f"Paper failed more audit stages than it passed "
                       f"({verdict['fail']} failures vs {verdict['pass']} passes). "
                       f"Review findings before continuing.")

        return None

    def _is_critical_step(self, step: MissionStep) -> bool:
        """Determine if a step failure should abort the mission."""
        # First 2 steps (data gathering) are usually critical
        # Later steps (export, flashcards) are non-critical
        critical_agents = {"scout", "tank"}
        return step.agent in critical_agents

    def pause(self, mission_id: str, reason: str = "Manual pause"):
        """Pause a running mission."""
        mission = self._missions.get(mission_id)
        if mission and mission.status == MissionStatus.RUNNING:
            mission.status = MissionStatus.PAUSED
            mission.pause_reason = reason

    def cancel(self, mission_id: str):
        """Cancel a running or paused mission."""
        mission = self._missions.get(mission_id)
        if mission and mission.status in (MissionStatus.RUNNING, MissionStatus.PAUSED, MissionStatus.PENDING):
            mission.status = MissionStatus.CANCELLED

    def get_status(self, mission_id: str) -> Optional[dict]:
        """Get current mission status."""
        mission = self._missions.get(mission_id)
        return mission.to_dict(include_results=True) if mission else None

    def list_missions(self) -> list[dict]:
        """List all missions (active + history)."""
        return [m.to_dict() for m in sorted(
            self._missions.values(), key=lambda m: m.created_at, reverse=True
        )]


# ═══════════════════════════════════════════════════════════════════
# Global instance — initialized when app is available
# ═══════════════════════════════════════════════════════════════════

_runner: Optional[MissionRunner] = None


def get_mission_runner(app=None) -> MissionRunner:
    """Get or create the global MissionRunner instance."""
    global _runner
    if _runner is None:
        _runner = MissionRunner(app)
    elif app and not _runner._app:
        _runner._app = app
    return _runner

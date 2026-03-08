"""
§IX: Shared Pydantic request models for route validation.

These models replace raw body.get() calls in critical endpoints,
catching missing/invalid fields at the FastAPI schema layer instead
of producing runtime errors inside handler code.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ── Jarvis ───────────────────────────────────────────────────────

class JarvisCommentRequest(BaseModel):
    """POST /api/jarvis/comment"""
    text: str = Field(..., min_length=1, description="Comment text")
    file_path: str = Field(default="general", description="File path or 'general'")
    type: str = Field(default="observation", description="Comment type")


class JarvisApproveRequest(BaseModel):
    """POST /api/jarvis/approve"""
    token_id: str = Field(..., min_length=1)
    reason: str = Field(default="")


class JarvisRejectRequest(BaseModel):
    """POST /api/jarvis/reject"""
    token_id: str = Field(..., min_length=1)
    reason: str = Field(default="")


class SandboxQueueRequest(BaseModel):
    """POST /api/jarvis/sandbox/queue"""
    task: str = Field(..., min_length=1, description="Task description")
    priority: str = Field(default="normal")


# ── Oracle ───────────────────────────────────────────────────────

class OracleSynthesisRequest(BaseModel):
    """POST /api/oracle/synthesis"""
    topic_a: str = Field(..., min_length=1)
    topic_b: str = Field(..., min_length=1)


class OracleGapsRequest(BaseModel):
    """POST /api/oracle/gaps"""
    field: str = Field(..., min_length=1)


class OracleAdversarialRequest(BaseModel):
    """POST /api/oracle/adversarial"""
    claim: str = Field(..., min_length=1)
    mechanism: str = Field(default="")


class OracleCommitteeRequest(BaseModel):
    """POST /api/oracle/committee-pushback"""
    proposal: str = Field(..., min_length=1)


# ── Causal ───────────────────────────────────────────────────────

class CausalExtractRequest(BaseModel):
    """POST /api/causal/extract"""
    text: str = Field(..., min_length=1)
    source_id: str = Field(default="")


class CausalCounterfactualRequest(BaseModel):
    """POST /api/causal/counterfactual"""
    scenario: str = Field(..., min_length=1)
    n: int = Field(default=1000, ge=1, le=100000)


class CausalForecastRequest(BaseModel):
    """POST /api/causal/forecast"""
    policy_change: str = Field(..., min_length=1)
    variables: Optional[list] = None
    framework: str = Field(default="submerged_state")


class CausalStressTestRequest(BaseModel):
    """POST /api/causal/stress-test"""
    cause: str = Field(..., min_length=1)
    effect: str = Field(..., min_length=1)
    mechanism: str = Field(default="")


# ── Cognitive ────────────────────────────────────────────────────

class PersonaSwitchRequest(BaseModel):
    """POST /api/cognitive/persona/switch"""
    persona: str = Field(..., min_length=1)


class SocraticQuestionRequest(BaseModel):
    """POST /api/cognitive/socratic/question"""
    topic: str = Field(..., min_length=1)


class SpacedRepAddRequest(BaseModel):
    """POST /api/cognitive/spaced-rep/add"""
    front: str = Field(..., min_length=1, description="Question")
    back: str = Field(..., min_length=1, description="Answer")
    tags: list[str] = Field(default=[])


class PeerReviewRequest(BaseModel):
    """POST /api/cognitive/peer-review"""
    text: str = Field(..., min_length=1)


class CrossLanguageRequest(BaseModel):
    """POST /api/cognitive/cross-language"""
    query: str = Field(..., min_length=1)
    languages: list[str] = Field(default=["en", "es", "fr"])


# ── Orchestration ────────────────────────────────────────────────

class DeepDiveRequest(BaseModel):
    """POST /api/deep-dive/start"""
    question: str = Field(..., min_length=1)


class PeerReviewDraftRequest(BaseModel):
    """POST /api/peer-review"""
    draft: str = Field(..., min_length=1)
    personas: list[str] = Field(default=["mettler", "aldrich", "carsey"])


class TutorRequest(BaseModel):
    """POST /api/tutor"""
    message: str = Field(..., min_length=1)
    topic: str = Field(default="")
    difficulty: str = Field(default="intermediate")
    history: str = Field(default="")


class ExplainTermRequest(BaseModel):
    """POST /api/explain-term"""
    term: str = Field(..., min_length=1)
    context: str = Field(default="")
    difficulty: str = Field(default="intermediate")


class VibeGenerateRequest(BaseModel):
    """POST /api/vibe/generate"""
    directive: str = Field(..., min_length=1)
    language: str = Field(default="python")
    dataset: str = Field(default="")
    analysis_type: str = Field(default="")
    variables: list[str] = Field(default=[])
    auto_execute: bool = Field(default=False)


class VibeExecuteRequest(BaseModel):
    """POST /api/vibe/execute"""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")
    timeout: int = Field(default=60, ge=1, le=300)


class VibeExplainRequest(BaseModel):
    """POST /api/vibe/explain"""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")
    difficulty: str = Field(default="intermediate")


# ── Integrations ─────────────────────────────────────────────────

class WasmExecuteRequest(BaseModel):
    """POST /api/wasm/execute"""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")
    timeout: int = Field(default=30, ge=1, le=120)


class NotionSyncRequest(BaseModel):
    """POST /api/notion/sync"""
    content: str = Field(..., min_length=1)
    title: str = Field(default="E.D.I.T.H. Export")
    tags: list[str] = Field(default=[])
    database_id: str = Field(default="", description="Optional Notion database_id override")


class DesktopNotifyRequest(BaseModel):
    """POST /api/desktop/notify"""
    title: str = Field(default="E.D.I.T.H.")
    message: str = Field(..., min_length=1)


class ConnectorTestRequest(BaseModel):
    """POST /api/connectors/test"""
    connector: str = Field(..., min_length=1)

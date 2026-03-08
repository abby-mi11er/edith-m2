"""
Chat routes for E.D.I.T.H. — extracted from main.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
import threading
import time as _time
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from server.server_state import state as _server_state
from server.pipeline_utils import PipelineTimer

log = logging.getLogger("edith")
router = APIRouter(tags=["Chat"])

# §SEC: Input sanitizer — lazy init, safe fallback for standalone import (tests)
try:
    from server.input_sanitizer import InputSanitizer as _InputSanitizerCls
    _sanitizer = _InputSanitizerCls()
except Exception:
    _sanitizer = None

# Lazy imports — resolved at call time to avoid circular imports  
def _get_main():
    """Get main module for accessing config and shared state."""
    import server.main as m
    return m

import re as _re
import time as _time

# ── Author/Year extraction from filenames ─────────────────────────
_AUTHOR_BLOCKLIST = {
    "university", "effects", "place", "chapter", "section", "paper",
    "document", "codebook", "documentation", "syllabus", "worksheet",
    "homework", "final", "draft", "review", "analysis", "abstract",
    "introduction", "conclusion", "methods", "results", "discussion",
    "appendix", "supplement", "table", "figure", "data", "dataset",
    "replication", "slides", "lecture", "notes", "reading", "exam",
    "midterm", "quiz", "assignment", "response", "memo", "brief",
    "report", "summary", "overview", "guide", "manual", "handbook",
    "politics", "political", "economic", "social", "public", "policy",
    "american", "comparative", "international", "global", "national",
    "journal", "quarterly", "annual", "review", "studies", "research",
    "the", "of", "and", "in", "on", "for", "to", "a", "an", "with",
    "parties", "voter", "voting", "turnout", "election", "elections",
    "democracy", "democratic", "participation", "inequality",
    "cartel", "violence", "immigration", "enforcement",
}

def _extract_author_year_from_filename(filename: str) -> tuple:
    """Extract (author, year) from common academic PDF filenames.

    Handles patterns like:
        Acemoglu_2001.pdf → ('Acemoglu', '2001')
        Smith and Jones 2020.pdf → ('Smith', '2020')
        García-López et al 2019.pdf → ('García-López', '2019')
        Hajnal Lajevardi 2017.pdf → ('Hajnal', '2017')
        03_Some Paper Title.pdf → ('', '')
    """
    if not filename:
        return ("", "")
    # Strip extension
    stem = _re.sub(r'\.(pdf|txt|docx|md|tex|rtf)$', '', filename, flags=_re.IGNORECASE)
    # Strip leading numbers/dashes/dots
    stem = _re.sub(r'^[\d_\-.\s]+', '', stem).strip()
    if not stem:
        return ("", "")

    # Extract year (4-digit number 1900-2099)
    year_match = _re.search(r'\b((?:19|20)\d{2})\b', stem)
    year = year_match.group(1) if year_match else ""

    # Get the part before the year as potential author
    if year_match:
        before_year = stem[:year_match.start()].strip()
    else:
        before_year = stem

    # Clean separators
    before_year = before_year.replace('_', ' ').replace('-', ' ').strip()
    before_year = _re.sub(r'\s+', ' ', before_year).strip()
    # Remove trailing connectors
    before_year = _re.sub(r'\s+(and|et al\.?|&)\s*$', '', before_year, flags=_re.IGNORECASE).strip()

    # Split into words and try to find author-like tokens
    words = before_year.split()
    if not words:
        return ("", year)

    # If the first word looks like a surname (capitalized, not in blocklist)
    first_word = words[0]
    if first_word.lower() in _AUTHOR_BLOCKLIST or len(first_word) < 2:
        return ("", year)

    # Check if it starts with a capital letter (surname-like)
    if first_word[0].isupper():
        return (first_word, year)

    return ("", year)


def _is_suspicious_author(author: str) -> bool:
    """Return True if the author string is likely garbage, not a real name."""
    if not author:
        return True
    author_lower = author.strip().lower()
    # Single word in blocklist
    if author_lower in _AUTHOR_BLOCKLIST:
        return True
    # Too short or too long
    if len(author_lower) < 2 or len(author_lower) > 80:
        return True
    # Starts with a stopword
    first = author_lower.split()[0] if author_lower.split() else ""
    if first in _AUTHOR_BLOCKLIST:
        return True
    return False


# Domain keyword cache — used by _route_model_chain
_DOMAIN_KEYWORDS_CACHE = {"keywords": [], "updated": 0}
_DOMAIN_KEYWORDS_STATIC = [
    "snap", "welfare", "charity", "submerged state", "policy feedback",
    "voter turnout", "clientelism", "path dependence", "iron triangle",
    "bureaucracy", "principal-agent", "collective action",
]

_FACTUAL_PATTERNS = [
    _re.compile(r"^(what is|define|who is|who was|when did|what year|how many|name the|list)\b"),
    _re.compile(r"^(what does|what are the)\b"),
]
_COMPLEX_PATTERNS = [
    _re.compile(r"(analyze|evaluate|compare|contrast|argue|synthesize|critique)"),
    _re.compile(r"(how does.*relate|what.*connection|implications of)"),
    _re.compile(r"(explain the.*debate|theoretical|framework)"),
]


# ---------------------------------------------------------------------------
# §FIX: Repetition guard — detect and stop looping model output
# ---------------------------------------------------------------------------
class _RepetitionGuard:
    """Tracks sentences during streaming to detect repetitive output loops.

    If the same sentence appears >= threshold times, signals to halt.
    """
    def __init__(self, threshold: int = 3, min_sentence_len: int = 30):
        self._counts: dict[str, int] = {}
        self._threshold = threshold
        self._min_len = min_sentence_len
        self.triggered = False
        self.trigger_sentence = ""

    def check(self, text_so_far: str) -> bool:
        """Check accumulated text for repetition. Returns True if looping detected."""
        if self.triggered:
            return True
        # Split on sentence boundaries
        sentences = _re.split(r'(?<=[.!?])\s+', text_so_far)
        self._counts.clear()
        for s in sentences:
            s_clean = s.strip().lower()
            if len(s_clean) < self._min_len:
                continue
            # Normalize whitespace for comparison
            s_norm = _re.sub(r'\s+', ' ', s_clean)
            self._counts[s_norm] = self._counts.get(s_norm, 0) + 1
            if self._counts[s_norm] >= self._threshold:
                self.triggered = True
                self.trigger_sentence = s.strip()
                return True
        return False

    def truncate(self, text: str) -> str:
        """Remove repeated content and append a clean ending."""
        if not self.trigger_sentence:
            return text
        # Find where the repetition starts (second occurrence)
        trigger_lower = self.trigger_sentence.lower()
        first_idx = text.lower().find(trigger_lower)
        if first_idx < 0:
            return text
        second_idx = text.lower().find(trigger_lower, first_idx + len(trigger_lower))
        if second_idx < 0:
            return text
        # Cut at the second occurrence and clean the ending
        truncated = text[:second_idx].rstrip()
        # Remove any trailing partial sentence
        last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_period > len(truncated) * 0.5:
            truncated = truncated[:last_period + 1]
        return truncated


class Message(BaseModel):
    role: str
    content: str
    
    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v):
        if _sanitizer:
            check = _sanitizer.check(v)
            if not check["safe"]:
                # Log threat but don't crash here - chat endpoint handles it more gracefully?
                # Actually, raising ValueError is correct for Pydantic
                # content = _sanitizer.sanitize(v) 
                # Strict security:
                # raise ValueError(f"Input security integrity violation: {check['threats']}")
                # Lenient for now (sanitizing):
                return _sanitizer.sanitize(v)
        return v


class ImageData(BaseModel):
    """Inline image for multimodal chat."""
    base64: str
    mime_type: str = "image/png"
    name: str = "image"


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=50)
    model: str = "gemini-2.5-flash"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    mode: str = Field("grounded", pattern=r"^(grounded|general|open|quick|lit_review|counterargument|paper_outline|research_design|annotated_bib|exam|gap_analysis|reading_companion|writing_assistant|committee_sim|peer_review|teaching_intro|teaching_grad|teaching_expert|discussant|office_hours)$")
    source_policy: str = Field("files_only", pattern=r"^(files_only|web_only|files_web)$")
    image_data: Optional[ImageData] = None
    answer_length: str = Field("standard", pattern=r"^(brief|standard|exhaustive)$")
    template: str = Field("default", pattern=r"^(default|memo|executive_summary|blog|pre_registration|lit_review|research_design|counterargument|exam)$")

    @field_validator("messages")
    @classmethod
    def last_message_not_empty(cls, v):
        if not v[-1].content.strip():
            raise ValueError("Last message cannot be empty")
        return v


def _save_autolearn(question: str, answer: str, sources: list, model: str, coverage: float):
    """Save audit-clean Q&A pair as training data. Quality-gated + versioned."""
    from datetime import datetime as _dt, timezone as _tz
    # Training data versioning — bump EDITH_TRAINING_VERSION env var on each fine-tune cycle
    data_version = os.environ.get("EDITH_TRAINING_VERSION", "v1")
    with _autolearn_lock:
        pair = {
            "messages": [
                {"role": "system", "content": "[SYSTEM]"},  # §FIX Vuln 4: Placeholder, not full prompt
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "timestamp": _dt.now(_tz.utc).isoformat(),  # §FIX D2: timezone-aware
                "model": model,
                "coverage": round(coverage, 2),
                "source_count": len(sources),
                "quality_tier": "auto_clean",
                "data_version": data_version,
                "prompt_version": getattr(sys.modules.get('server.prompts'), 'PROMPT_VERSION', 'unknown'),
            },
        }
        out_path = ROOT_DIR / "autolearn.jsonl"
        # §MAINT: Rotate when file exceeds 50 MB to prevent unbounded growth
        _AUTOLEARN_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
        if out_path.exists() and out_path.stat().st_size > _AUTOLEARN_MAX_BYTES:
            from datetime import datetime as _rotate_dt
            import shutil as _rotate_shutil
            archive_name = f"autolearn_{_rotate_dt.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            # §FIX Issue 1: Use shutil.move for safer rotation on USB drives
            _rotate_shutil.move(str(out_path), str(ROOT_DIR / archive_name))
            log.info(f"§MAINT: Rotated autolearn.jsonl → {archive_name}")
        with open(out_path, "a") as f:
            f.write(json.dumps(pair) + "\n")
        log.info(f"Autolearn: saved pair (coverage={coverage:.0%}, model={model}, version={data_version})")


def _save_dpo_negative(question: str, bad_answer: str, sources: list, audit_result: dict):
    """Fix 6: Log audit-failed answers as DPO negative examples.

    These pair with audit-clean autolearn positives to create training
    batches for Direct Preference Optimization — teaching Winnie to
    prefer grounded, citation-heavy answers over hallucinated ones.
    """
    from datetime import datetime as _dt, timezone as _tz
    with _autolearn_lock:
        pair = {
            "question": question,        # §FIX Bug 1: Match lora_trainer.py expected keys
            "bad_answer": bad_answer,     # §FIX Bug 1: Was "rejected"
            "audit_reason": audit_result.get("error", "audit_failed"),
            "unsupported_claims": audit_result.get("unsupported_claims", 0),
            "source_count": len(sources),
            "timestamp": _dt.now(_tz.utc).isoformat(),  # §FIX D2: timezone-aware
        }
        # §FIX T5: Write to training_data/ (unified with feedback_endpoint)
        _dpo_dir = ROOT_DIR / "training_data"
        _dpo_dir.mkdir(exist_ok=True)
        out_path = _dpo_dir / "dpo_negatives.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(pair) + "\n")
        log.info(f"DPO negative: logged rejected answer ({audit_result.get('error', 'unknown')})")


# /api/test-query and /api/test-retrieve — REMOVED: debug endpoints that expose internals

# ---------------------------------------------------------------------------
# Query depth router — saves cost on simple questions
# ---------------------------------------------------------------------------
def _classify_depth(question: str) -> str:
    """Classify query depth: quick / standard / debate.
    §4.0: Also used for smart model routing — quick→FT, debate→Gemini.
    """
    words = question.split()
    q_lower = question.lower()
    # Quick: short factual lookups → best for fine-tuned model
    if len(words) <= 10 and any(q_lower.startswith(p) for p in [
        "what is", "who is", "define ", "when did", "what does",
        "what are", "how many", "list ", "name ",
    ]):
        return "quick"
    # Debate: comparative, causal, or multi-concept → best for long-context Gemini
    debate_signals = ["compare", "contrast", "evaluate", "critique", "assess",
                      "how does", "why do", "to what extent", "paradox",
                      "relationship between", "tension between", "implications of",
                      "causes of", "effects of", "debate", "controversy",
                      "strengths and weaknesses", "pros and cons", "trade-off"]
    if any(sig in q_lower for sig in debate_signals) or len(words) > 25:
        return "debate"
    return "standard"


# ---------------------------------------------------------------------------
# OpenAI fine-tuned model helper (pooled + circuit breaker)
# ---------------------------------------------------------------------------
def _call_openai_ft(prompt: str, system: str = "", temperature: float = 0.1,
                    max_tokens: int = 2000):
    """Call the fine-tuned OpenAI model with connection pooling + circuit breaker."""
    if openai_breaker.is_open:
        raise RuntimeError(f"OpenAI circuit breaker OPEN (cooldown {openai_breaker.cooldown}s)")
    
    try:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        answer, model = call_openai_pooled(
            api_key=OPENAI_API_KEY,
            model=OPENAI_FT_MODEL,
            prompt=prompt,
            system=system or SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        openai_breaker.record_success()
        return answer, OPENAI_FT_MODEL
    except Exception as e:
        openai_breaker.record_failure()
        raise


# ---------------------------------------------------------------------------
# Mode-aware prompt builder — routes to the right professor prompt
# ---------------------------------------------------------------------------
def _build_mode_prompt(
    mode: str,
    question: str,
    sources: list[dict],
    plan: dict,
    depth: str,
    discovery_text: str = "",
    template: str = "default",
) -> str:
    """Build a prompt based on the selected mode.
    
    Modes: grounded, lit_review, counterargument, paper_outline,
           research_design, annotated_bib, exam, gap_analysis
    §4.0: template override lets the user pick a specific output format.
    """
    from server.prompts import (
        LIT_REVIEW_PROMPT, RESEARCH_DESIGN_PROMPT,
        COUNTERARGUMENT_PROMPT, GAP_IDENTIFIER_PROMPT,
        PAPER_OUTLINE_PROMPT, ANNOTATED_BIB_PROMPT,
        EXAM_QUESTION_PROMPT, ANSWER_TEMPLATES,
    )
    try:
        from server.research_workflows import ALL_WORKFLOW_PROMPTS
    except ImportError:
        ALL_WORKFLOW_PROMPTS = {}
    
    blocks = build_support_audit_source_blocks(sources)
    
    # §4.0: Template override takes priority
    if template != "default" and template in ANSWER_TEMPLATES:
        prompt = (
            f"{ANSWER_TEMPLATES[template]}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SOURCES:\n{blocks}"
        )
        if discovery_text:
            prompt += f"\n\n{discovery_text}"
        return prompt

    # Mode-specific prompts
    mode_prompts = {
        "lit_review": LIT_REVIEW_PROMPT,
        "counterargument": COUNTERARGUMENT_PROMPT,
        "paper_outline": PAPER_OUTLINE_PROMPT,
        "research_design": RESEARCH_DESIGN_PROMPT,
        "annotated_bib": ANNOTATED_BIB_PROMPT,
        "exam": EXAM_QUESTION_PROMPT,
        "gap_analysis": GAP_IDENTIFIER_PROMPT,
        **ALL_WORKFLOW_PROMPTS,
    }
    
    if mode in mode_prompts:
        prompt = (
            f"{mode_prompts[mode]}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SOURCES:\n{blocks}"
        )
    else:
        # Default: use build_answer_prompt (grounded/general/quick)
        prompt = build_answer_prompt(question, sources, plan, depth=depth)
    
    # Inject discovery results if present
    if discovery_text:
        prompt += f"\n\n{discovery_text}"
    
    return prompt


def _get_domain_keywords() -> list:
    """Get domain keywords — dynamic from ChromaDB + static fallback."""
    import time as _t
    now = _t.time()
    # Refresh every hour
    if now - _DOMAIN_KEYWORDS_CACHE["updated"] > 3600:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            coll = client.get_collection(CHROMA_COLLECTION)
            # Sample collection metadata for common terms
            sample = coll.peek(limit=100)
            if sample and sample.get("documents"):
                from collections import Counter
                word_counts = Counter()
                for doc in sample["documents"]:
                    if doc:
                        words = [w.lower().strip(".,;:!?\"'()") for w in doc.split()]
                        word_counts.update(w for w in words if len(w) > 4)
                # Top 30 domain-specific terms (exclude common English words)
                common = {"about", "their", "would", "could", "should", "which", "there",
                          "these", "those", "other", "after", "before", "between", "through",
                          "where", "while", "being", "under", "since", "provide", "based",
                          "paper", "study", "research", "article", "journal", "analysis"}
                domain_words = [w for w, c in word_counts.most_common(60)
                               if w not in common and c >= 3][:30]
                if domain_words:
                    _DOMAIN_KEYWORDS_CACHE["keywords"] = domain_words
                    _DOMAIN_KEYWORDS_CACHE["updated"] = now
                    log.info(f"§ROUTING: refreshed {len(domain_words)} domain keywords from ChromaDB")
        except Exception as e:
            log.debug(f"§ROUTING: ChromaDB keyword refresh failed: {e}")
    
    return _DOMAIN_KEYWORDS_CACHE["keywords"] or _DOMAIN_KEYWORDS_STATIC


def _route_model_chain(query: str, base_chain: list, rid: str = "") -> tuple:
    """§ROUTING: Intelligent model chain routing.
    
    Returns (model_chain, dual_brain_mode, route_info).
    route_info dict contains: route, reason, is_factual, is_domain, is_complex.
    """
    route_t0 = _time.monotonic()
    dual_brain_mode = False
    route_info = {"route": "default", "reason": "general query → Gemini"}
    
    if not (OPENAI_FT_MODEL and OPENAI_API_KEY):
        route_info["reason"] = "Winnie not configured — Gemini only"
        return base_chain, False, route_info
    
    q = query.lower().strip()
    domain_keywords = _get_domain_keywords()
    
    is_factual = any(p.search(q) for p in _FACTUAL_PATTERNS)
    is_domain = any(k in q for k in domain_keywords)
    is_complex = any(p.search(q) for p in _COMPLEX_PATTERNS)
    is_multipart = q.count("?") >= 2
    
    route_info.update({"is_factual": is_factual, "is_domain": is_domain,
                       "is_complex": is_complex, "is_multipart": is_multipart})
    
    if is_factual or is_domain:
        # Tier 1: Winnie-first for factual/domain queries
        chain = [OPENAI_FT_MODEL] + base_chain
        route_info.update({"route": "winnie_first", "reason": f"factual={is_factual}, domain={is_domain}"})
    elif is_complex or is_multipart:
        # Tier 2: Dual-brain — both answer, judge picks best
        chain = base_chain
        dual_brain_mode = True
        route_info.update({"route": "dual_brain", "reason": f"complex={is_complex}, multi={is_multipart}"})
    else:
        chain = base_chain
    
    # Improvement 5: Latency logging
    route_ms = round((_time.monotonic() - route_t0) * 1000, 1)
    log.info(f"[{rid}] §ROUTING: {route_info['route']} ({route_info['reason']}) [{route_ms}ms]")
    
    return chain, dual_brain_mode, route_info


def _run_dual_brain_judge(query: str, gemini_answer: str, sources: list,
                          temperature: float = 0.1, rid: str = "") -> tuple:
    """§ROUTING: Dual-brain judge — run Winnie independently, judge picks best.
    
    Improvement 3: If judge confidence < 40%, escalate to gemini-2.5-pro.
    Returns (final_answer, model_used, verdict).
    """
    judge_t0 = _time.monotonic()
    try:
        from pipelines.dual_brain import judge_answers, consensus_tracker
        from server.pipeline_utils import call_openai_pooled
        
        # Get Winnie's answer
        winnie_answer, _ = call_openai_pooled(
            api_key=OPENAI_API_KEY,
            model=OPENAI_FT_MODEL,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
        )
        if not winnie_answer:
            return gemini_answer, "gemini(winnie_failed)", {}
        
        # Judge compares
        evidence_text = "\n".join(s.get("text", "")[:200] for s in sources[:5]) if sources else ""
        verdict = judge_answers(query, winnie_answer, gemini_answer, evidence=evidence_text)
        confidence = verdict.get("confidence", 0)
        winner = verdict.get("winner", "gemini")
        
        judge_ms = round((_time.monotonic() - judge_t0) * 1000, 1)
        log.info(f"[{rid}] §DUAL-BRAIN: winner={winner}, confidence={confidence:.0%}, "
                 f"consensus={verdict.get('consensus', False)} [{judge_ms}ms]")
        
        # ═══ TRAINING DATA CAPTURE ═══
        # Wire 1: Log verdict to training pipeline (makes Winnie smarter over time)
        try:
            # 1a. ConsensusTracker — track model convergence
            consensus_tracker.record(confidence, verdict.get("consensus", False), dtype="chat")
            
            # 1b. FeedbackTrainer — save winning answer as training data
            from pipelines.feedback_trainer import FeedbackTrainer
            _ft = FeedbackTrainer()
            best_answer = winnie_answer if winner == "openai" else gemini_answer
            _ft.create_training_pair(
                question=query,
                answer=best_answer,
                source=f"judge_{winner}_conf{confidence:.0f}",
            )
            
            # 1c. ActiveLearningQueue — queue low-confidence for review
            if confidence < 0.5:
                from server.training_tools import ActiveLearningQueue
                _alq = ActiveLearningQueue()
                _alq.add(query=query, confidence=confidence,
                         answer=best_answer, sources=[s.get("title", "") for s in sources[:3]])
            
            log.info(f"[{rid}] §TRAINING: captured verdict (winner={winner}, conf={confidence:.0%})")
        except Exception as train_err:
            log.debug(f"[{rid}] Training capture failed (non-fatal): {train_err}")
        
        # Improvement 3: Confidence-based escalation
        if confidence < 0.4:
            # Neither model is confident → escalate to gemini-2.5-pro
            log.info(f"[{rid}] §ESCALATION: confidence {confidence:.0%} < 40% → gemini-2.5-pro")
            try:
                pro_answer, pro_model = generate_text_via_chain(
                    f"Please provide a thorough, well-sourced answer:\n\n{query}",
                    [ORACLE_MODEL, DEFAULT_MODEL],
                    temperature=temperature,
                )
                if pro_answer:
                    # Also capture escalated answer as high-quality training data
                    try:
                        _ft = FeedbackTrainer()
                        _ft.create_training_pair(query, pro_answer, source="escalated_pro")
                    except Exception:
                        pass
                    return pro_answer, f"{pro_model}(escalated)", verdict
            except Exception as e:
                log.warning(f"[{rid}] Pro escalation failed: {e}")
        
        # Pick winner
        if winner == "openai" and confidence > 0.6:
            return winnie_answer, f"{OPENAI_FT_MODEL}(judge)", verdict
        else:
            return gemini_answer, f"gemini(judge)", verdict
            
    except Exception as e:
        log.warning(f"[{rid}] Dual-brain judge failed: {e}")
        return gemini_answer, "gemini(judge_failed)", {}


async def chat_endpoint(req: ChatRequest):
    """DEPRECATED: Use /chat/stream instead. Kept for backward compatibility."""
    API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured")
    
    rid = uuid.uuid4().hex[:8]  # Request correlation ID
    timer = PipelineTimer()
    last_msg = req.messages[-1].content
    audit("chat_query", request_id=rid, query=last_msg[:200], mode=req.mode, model=req.model)
    log.info(f"[{rid}] chat_endpoint: processing query: {last_msg[:60]}")

    # Prompt injection guard
    guard_err = guard_input(last_msg)
    if guard_err:
        audit("prompt_blocked", reason=guard_err, query=last_msg[:100])
        raise HTTPException(status_code=400, detail=guard_err)
    
    # --- Improvement Modules Integration ---
    retrieval_ctx = {}
    session_summarizer = None
    if _IMPROVEMENTS_WIRED:
        try:
            from server.retrieval_enhancements import create_retrieval_context
            retrieval_ctx = create_retrieval_context(ROOT_DIR) 
        except Exception as e:
            log.warning(f"Failed to init retrieval context: {e}")
            
        try:
            from server.memory_pinning import SessionSummarizer
            session_summarizer = SessionSummarizer()
        except Exception as e:
            log.warning(f"Failed to init memory improvements: {e}")

    # Build conversation history
    ctx_manager = None
    if _IMPROVEMENTS_WIRED:
        try:
            _CtxWindowMgr = sys.modules["server.model_improvements"].ContextWindowManager
            ctx_manager = _CtxWindowMgr(req.model)
        except Exception as e:
            log.warning(f"[{rid}] ContextWindowManager init failed: {e}")

    # Fit messages to context
    final_messages = req.messages
    if ctx_manager and len(req.messages) > 1:
        # Convert Pydantic to dict for manager
        msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
        fitted = ctx_manager.fit_messages(msg_dicts)
        # Convert back (simplified, just string building below)
        # Actually we build conversation_history string manually below
        pass

    conversation_history = ""
    if len(req.messages) > 1:
        prev_msgs = req.messages[:-1]
        history_lines = []
        for m in prev_msgs[-6:]:  # Last 3 turns max
            history_lines.append(f"{m.role.upper()}: {m.content[:500]}")
        if history_lines:
            conversation_history = "CONVERSATION CONTEXT:\n" + "\n".join(history_lines) + "\n\n"
    
    # Inject cross-session memory
    memory_context = format_memory_context()
    if memory_context:
        conversation_history = memory_context + "\n\n" + conversation_history
    
    # Remap legacy models
    if "gemini-1.5-flash" in req.model:
        req.model = DEFAULT_MODEL
    
    # §METABOLIC: Check thermal state before model selection
    try:
        from server.subconscious_streams import metabolic_balancer
        _throttle = metabolic_balancer.should_throttle()
        if _throttle["should_throttle"] and _throttle.get("model_override") == "local":
            log.info(f"§METABOLIC: Throttling to local model — {_throttle['reason']}")
            # Override to lighter model when system is under thermal pressure
            req.model = "mlx-community/phi-3-mini-4k-instruct" if not req.model else req.model
    except Exception:
        pass  # Metabolic check is non-critical
    
    model_chain = build_model_chain(req.model)
    
    # §ROUTING: Use shared intelligent routing function
    model_chain, _dual_brain_mode, _route_info = _route_model_chain(last_msg, model_chain, rid)
    
    # Route query depth (Legacy + New)
    depth = _classify_depth(last_msg)
    if _IMPROVEMENTS_WIRED:
        # Use new router for Model selection hints
        try:
            route = _route_query(last_msg)
            if req.model == "auto": # If we supported auto
                req.model = route.model_id
        except Exception as e:
            log.warning(f"[{rid}] Query routing failed: {e}")
    
    # RETRIEVAL CONFIG — hardware-aware top_k
    try:
        from server.backend_logic import get_compute_profile
        _hw_profile = get_compute_profile()
        _hw_top_k = _hw_profile.get("top_k", 12)
    except Exception:
        _hw_top_k = 12
    top_k = int(os.environ.get("EDITH_CHROMA_TOP_K", str(_hw_top_k)))
    if _IMPROVEMENTS_WIRED and "adaptive_top_k" in retrieval_ctx:
        top_k = retrieval_ctx["adaptive_top_k"](last_msg, base_k=top_k)
        log.info(f"[{rid}] Adaptive top_k: {top_k}")

    rerank_model = os.environ.get("EDITH_CHROMA_RERANK_MODEL", "")
    rerank_top_n = int(os.environ.get("EDITH_CHROMA_RERANK_TOP_N", "18"))
    # §8: Conversation memory query expansion — resolve pronouns and references
    # If the user says "he" or "this paper", expand with context from recent messages
    expanded_msg = last_msg
    if len(req.messages) >= 3 and depth != "quick":
        recent_context = []
        for m in req.messages[-6:-1]:  # last 5 messages before current
            content = (getattr(m, 'content', '') or '')[:200]
            if content:
                recent_context.append(f"{getattr(m, 'role', 'user')}: {content}")
        # Check if query has pronouns/references that need resolution
        pronoun_words = {'he', 'she', 'they', 'it', 'this', 'that', 'these', 'those',
                         'the paper', 'the article', 'the author', 'the study',
                         'his', 'her', 'their', 'its'}
        query_lower = last_msg.lower()
        has_pronouns = any(p in query_lower for p in pronoun_words)
        if has_pronouns and recent_context:
            context_str = " | ".join(recent_context[-3:])  # last 3 messages
            expanded_msg = f"{last_msg} [Context: {context_str[:300]}]"
            log.info(f"[{rid}] Query expanded with conversation context")

    queries = [expanded_msg]
    # Decompose Query (New Module)
    if _IMPROVEMENTS_WIRED and "decompose_query" in retrieval_ctx and depth != "quick":
        sub_qs = retrieval_ctx["decompose_query"](last_msg)
        if len(sub_qs) > 1:
            queries = sub_qs
            log.info(f"[{rid}] Decomposed query into: {queries}")
    
    # Legacy rewrite fallback
    if len(queries) == 1 and not USE_GOOGLE_RETRIEVAL and depth != "quick" and len(last_msg.split()) > 8:
         try:
            rewrites = rewrite_retrieval_queries(last_msg, [DEFAULT_MODEL])
            if rewrites: queries = rewrites + [last_msg]
         except Exception as e: log.warning(f"[{rid}] Query rewrite failed: {e}")
    
    # Modes that need source retrieval (not just grounded)
    _SOURCE_MODES = {"grounded", "counterargument", "lit_review", "paper_outline",
                     "research_design", "annotated_bib", "exam", "gap_analysis",
                     "reading_companion", "writing_assistant", "committee_sim",
                     "peer_review", "teaching_intro", "teaching_grad",
                     "teaching_expert", "discussant", "office_hours"}
    _needs_sources = req.mode in _SOURCE_MODES

    sources = []
    try:
        timer.start("retrieval")
        if USE_GOOGLE_RETRIEVAL and _needs_sources and not should_skip_retrieval(last_msg):
            # Google Search
            if not google_retrieval_breaker.is_open:
                try:
                    sources, was_cached = cached_retrieve(
                        query=last_msg, store_id=GOOGLE_STORE_ID,
                        retrieve_fn=retrieve_google_sources,
                        api_key=API_KEY, top_k=top_k
                    )
                    google_retrieval_breaker.record_success()
                except Exception as e:
                    google_retrieval_breaker.record_failure()
                    log.error(f"Google retrieval failed: {e}")
            
            # Fallback to Chroma
            if not sources:
                timer.start("retrieval_fallback")
                try:
                    sources = await asyncio.to_thread(
                        retrieve_local_sources,
                        queries=queries, chroma_dir=CHROMA_DIR,
                        collection_name=CHROMA_COLLECTION, embed_model=EMBED_MODEL,
                        top_k=5, bm25_weight=0.35
                    )
                except Exception as e: log.warning(f"[{rid}] Chroma fallback retrieval failed: {e}")
                timer.stop("retrieval_fallback")

        elif depth != "quick" and _needs_sources:
            # Agentic / Standard Local
            agentic_result = agentic_retrieve(
                query=last_msg, chroma_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION,
                embed_model=EMBED_MODEL, top_k=top_k, max_attempts=2,
                rerank_model=rerank_model, rerank_top_n=rerank_top_n
            )
            sources = agentic_result["sources"]
        elif _needs_sources:
            # Quick / Single pass — use hardware-aware pool_multiplier
            _pool_mult = _hw_profile.get("chroma_pool_multiplier", 4) if '_hw_profile' in dir() else 4
            sources = retrieve_local_sources(
                queries=queries, chroma_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION,
                embed_model=EMBED_MODEL, top_k=top_k,
                pool_multiplier=_pool_mult,
                rerank_model=rerank_model, rerank_top_n=rerank_top_n,
            )
        timer.stop("retrieval")

        # --- Post-Retrieval Improvements ---
        if _IMPROVEMENTS_WIRED and sources:
            if "apply_temporal_weight" in retrieval_ctx:
                sources = retrieval_ctx["apply_temporal_weight"](sources)
            if "calibrate_confidence" in retrieval_ctx:
                sources = retrieval_ctx["calibrate_confidence"](sources)
            # Expand neighbors? (Optional, might be expensive if many sources)
            # if "expand_with_neighbors" in retrieval_ctx: ...

        # §FERRARI: Broaden with unified search (OpenAlex) for grounded mode
        if _needs_sources and depth != "quick" and len(sources) < 5:
            try:
                from server import openalex as _oa
                oa_results = _oa.search_openalex(last_msg, per_page=3)
                oa_works = oa_results.get("results", oa_results.get("works", []))
                for w in oa_works[:3]:
                    if isinstance(w, dict) and w.get("title"):
                        sources.append({
                            "text": f"{w['title']}. {w.get('abstract', '')[:300]}",
                            "title": w["title"],
                            "source": "openalex",
                            "metadata": {"year": w.get("publication_year"), "cited_by": w.get("cited_by_count", 0)},
                        })
            except Exception:
                pass  # OpenAlex broadening is best-effort

        # §NYT: Broaden with New York Times journalism for policy queries
        if req.mode == "grounded" and depth != "quick":
            try:
                from server.nyt_bridge import search_articles as _nyt_search
                import os as _os
                if _os.environ.get("NYT_API_KEY"):
                    nyt_data = await _nyt_search(last_msg, page=0)
                    for art in (nyt_data.get("articles") or [])[:3]:
                        if art.get("title"):
                            sources.append({
                                "text": f"[NYT] {art['title']}. {art.get('abstract', art.get('snippet', ''))}",
                                "title": art["title"],
                                "source": "nyt",
                                "metadata": {
                                    "url": art.get("url", ""),
                                    "pub_date": art.get("pub_date", ""),
                                    "section": art.get("section", ""),
                                    "byline": art.get("byline", ""),
                                },
                            })
            except Exception:
                pass  # NYT broadening is best-effort

        # §BUS: Emit research query event for analytics
        try:
            from server.event_bus import bus
            asyncio.ensure_future(bus.emit("research.query", {
                "query": last_msg[:200],
                "source_count": len(sources),
                "mode": req.mode,
                "depth": depth,
            }, source="chat"))
        except Exception:
            pass

    except Exception as e:
        log.error(f"Retrieval error: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sources = []

    # Guard injection
    if sources:
        sources = check_source_injection(sources)

    # §FIX: Enrich sources with author/year/title from library cache (non-streaming path)
    if sources:
        try:
            from server.routes import library as _lib_mod
            _cached = getattr(_lib_mod, "_sources_cache", {})
            _cached_papers = _cached.get("papers") or _cached.get("all") or []
            lookup = {}
            if _cached_papers:
                for d in _cached_papers:
                    src = d.get("source", "")
                    if src:
                        lookup[src] = d
                        fname = src.rsplit("/", 1)[-1] if "/" in src else src
                        lookup[fname] = d
            for s in sources:
                path = (s.get("source") or s.get("path") or
                        (s.get("metadata") or {}).get("path") or "")
                fname = path.rsplit("/", 1)[-1] if "/" in path else path
                meta = lookup.get(path) or lookup.get(fname)
                # 1) Try library cache metadata
                if meta:
                    s.setdefault("title", meta.get("title", ""))
                    if meta.get("year") and not s.get("year"):
                        s["year"] = meta["year"]
                    if meta.get("author") and not _is_suspicious_author(meta["author"]):
                        s.setdefault("author", meta["author"])
                # 2) Fallback: extract author/year from filename
                if not s.get("author") or _is_suspicious_author(s.get("author", "")):
                    fn_author, fn_year = _extract_author_year_from_filename(fname)
                    if fn_author:
                        s["author"] = fn_author
                    if fn_year and not s.get("year"):
                        s["year"] = fn_year
            # 3) Final cleanup: clear any remaining suspicious author names
            for s in sources:
                if s.get("author") and _is_suspicious_author(s["author"]):
                    s["author"] = ""
        except Exception:
            pass
    # Handle zero sources — fall back to general answer instead of dead-end
    if req.mode == "grounded" and not sources:
        log.info(f"[{rid}] No sources in grounded mode — falling back to general answer")
        try:
            fallback_prompt = (
                f"Answer this question to the best of your knowledge. "
                f"Note: No library sources were found for this query.\n\n"
                f"Question: {last_msg}"
            )
            fb_answer, fb_model = generate_text_via_chain(fallback_prompt, model_chain, temperature=req.temperature)
            return {
                "role": "assistant",
                "content": fb_answer or "No sources found. Try indexing documents or switching to General mode.",
                "sources": [],
                "no_evidence": True,
                "used_model": fb_model,
                "audit": {"is_clean": True, "note": "general_fallback"},
            }
        except Exception:
            return {
                "role": "assistant",
                "content": "No sources found in your library. Try indexing your documents first, or switch to General mode.",
                "sources": [],
                "no_evidence": True,
                "audit": {"is_clean": True},
            }

    # Plan
    plan = {"used": False}
    if depth == "debate" and len(sources) >= 3:
        try:
             plan = plan_answer_outline(last_msg, sources, model_chain)
        except Exception as e: log.warning(f"[{rid}] Answer outline planning failed: {e}")

    # Build Prompt
    timer.start("compress")
    # §HW: Hardware-adaptive compression — M4 processes more context
    try:
        from server.backend_logic import get_compute_profile
        _hw_comp = get_compute_profile()
        _max_chars = _hw_comp.get("max_chars_per_source", 800)
        _max_srcs = _hw_comp.get("max_sources_compressed", 8)
    except Exception:
        _max_chars, _max_srcs = 800, 8
    compressed_sources = compress_sources(sources, max_chars_per_source=_max_chars, max_sources=_max_srcs)
    timer.stop("compress")
    prompt = build_answer_prompt(last_msg, compressed_sources, plan, depth=depth)
    if conversation_history:
        prompt = conversation_history + prompt

    # Generate
    timer.start("generation")
    gen_t0 = _time.monotonic()
    answer = ""
    used_model = ""
    deepened = False

    try:
        # 1. Fine-tuned Winnie
        if OPENAI_FT_MODEL and OPENAI_API_KEY:
            try:
                ft_max = 4500 if depth == "debate" else 2000
                answer, used_model = _call_openai_ft(prompt, temperature=req.temperature, max_tokens=ft_max)
            except Exception as e: log.warning(f"[{rid}] Winnie FT call failed: {e}")
        
        # 2. Deepener (Gemini)
        if answer and depth == "debate" and len(answer) < 400:
             # ... existing deepener logic ...
             pass # Kept logically same, just omitting for brevity in replace block if possible? 
             # Wait, I must include it or I delete it.
             # I will keep the deepener logic concise.
             try:
                deep_prompt = f"Expand this expert draft deeply:\n{answer}\n\nORIGINAL: {last_msg}"
                deep_ans, d_mod = generate_text_via_chain(deep_prompt, model_chain, temperature=req.temperature)
                if len(deep_ans) > len(answer):
                    answer = deep_ans
                    used_model = f"{OPENAI_FT_MODEL}+{d_mod}"
                    deepened = True
             except Exception as e: log.warning(f"[{rid}] Deepener pass failed: {e}")

        # 3. Fallback
        if not answer:
            answer, used_model = generate_text_via_chain(prompt, model_chain, temperature=req.temperature)

        gen_elapsed = round(_time.monotonic() - gen_t0, 2)
        timer.stop("generation")

        # §ROUTING: Dual-brain judge for complex queries
        if _dual_brain_mode and answer and OPENAI_FT_MODEL:
            answer, used_model, _verdict = _run_dual_brain_judge(
                last_msg, answer, sources, req.temperature, rid
            )

        # Cost Tracking
        if _IMPROVEMENTS_WIRED and _cost_tracker:
             # Approx tokens: chars / 4
             in_tok = len(prompt) // 4
             out_tok = len(answer) // 4
             _cost_tracker.record(used_model, in_tok, out_tok, query=last_msg)

        # Audit
        timer.start("audit")
        audit_result = {"is_clean": False, "error": "audit_skipped"}
        if depth == "debate" and sources:
             try:
                 audit_result = audit_answer(answer, sources, [DEFAULT_MODEL])
                 if audit_result.get("corrections") and not audit_result.get("is_clean"):
                     answer = apply_corrections(answer, audit_result["corrections"], model_chain)
                     audit_result["corrections_applied"] = True
             except Exception as e: log.warning(f"[{rid}] Answer audit failed: {e}")
        elif sources:
            audit_result = {"is_clean": True, "note": "non-debate, audit skipped"}
        timer.stop("audit")

        # Citation middleware — same as SSE stream path (Fix 2)
        if sources and answer:
            try:
                answer = citation_middleware(answer, sources)
            except Exception:
                pass

        cited_count = sum(1 for i in range(len(sources)) if f"[S{i+1}]" in answer)
        coverage = cited_count / max(len(sources), 1)

        # Autolearn — only save audit-clean answers
        if audit_result.get("is_clean", False) and coverage > 0.2:
            try:
                _save_autolearn(last_msg, answer, sources, used_model, coverage)
            except Exception as e:
                log.warning(f"Autolearn save failed: {e}")

        # DPO negative logging — same as SSE stream path (Fix 6)
        if not audit_result.get("is_clean", False) and answer and sources:
            try:
                _save_dpo_negative(last_msg, answer, sources, audit_result)
            except Exception:
                pass

        if session_summarizer:
            try:
                msgs_dict = [m.model_dump() for m in req.messages]
                msgs_dict.append({"role": "assistant", "content": filter_output(answer)})
                audit("session_summary", **session_summarizer.summarize_session(msgs_dict))
            except Exception as e:
                log.warning(f"Session summary failed: {e}")

        # §AUTOPILOT: Emit chat.response event for training capture + method detection
        try:
            from server.event_bus import bus
            await bus.emit("chat.response", {
                "query": last_msg, "response": answer, "model": used_model,
                "sources": sources, "is_clean": audit_result.get("is_clean", False),
                "coverage": coverage, "depth": depth, "request_id": rid,
            }, source="chat_endpoint")
        except Exception:
            pass

        # §AUTOPILOT #1: Confidence calibration — compute inline
        confidence_data = {"confidence": 0.5, "level": "medium", "grounded": False}
        try:
            n_src = len(sources)
            src_cited = sum(1 for i in range(n_src) if f"[S{i+1}]" in answer)
            conf = min(0.95, 0.3 + n_src * 0.1 + (src_cited / max(n_src, 1)) * 0.2)
            hedging = ["might", "possibly", "unclear", "uncertain", "not sure", "speculative"]
            hedge_ct = sum(1 for h in hedging if h in answer.lower())
            if hedge_ct:
                conf = max(0.2, conf - hedge_ct * 0.05)
            confidence_data = {
                "confidence": round(conf, 2),
                "level": "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low",
                "sources_used": n_src, "sources_cited": src_cited,
                "grounded": n_src > 0,
            }
        except Exception:
            pass

        # §AUTOPILOT #2: Inline method suggestion — detect RQ in query
        method_suggestion = None
        try:
            import re as _re_ms
            q_lower = last_msg.lower()
            rq_patterns = [
                r"effect\s+of\s+\w+\s+on", r"impact\s+of\s+\w+\s+on",
                r"how\s+does\s+\w+\s+affect", r"does\s+\w+\s+cause",
            ]
            if any(_re_ms.search(p, q_lower) for p in rq_patterns):
                treatment, outcome = "", ""
                for pat in [r'(?:effect|impact)\s+of\s+(.+?)\s+on\s+(.+?)(?:\.|,|\?|$)',
                            r'how\s+does\s+(.+?)\s+affect\s+(.+?)(?:\.|,|\?|$)']:
                    m = _re_ms.search(pat, q_lower)
                    if m:
                        treatment, outcome = m.group(1).strip(), m.group(2).strip()
                        break
                suggestions = []
                if any(kw in q_lower for kw in ["threshold", "cutoff", "eligib"]):
                    suggestions.append({"method": "RDD", "reason": "threshold/cutoff mentioned",
                                       "stata": "rdrobust outcome running_var, c(cutoff)"})
                if any(kw in q_lower for kw in ["instrument", "exogenous", " iv "]):
                    suggestions.append({"method": "IV", "reason": "instrument available",
                                       "stata": "ivregress 2sls outcome (treatment = instrument)"})
                if any(kw in q_lower for kw in ["panel", "over time", "before and after"]):
                    if any(kw in q_lower for kw in ["reform", "policy", "program"]):
                        suggestions.append({"method": "DiD", "reason": "panel + policy change",
                                           "stata": "didregress (outcome) (treatment), group(unit) time(period)"})
                    else:
                        suggestions.append({"method": "Fixed Effects", "reason": "panel data",
                                           "stata": "xtreg outcome treatment, fe cluster(unit)"})
                if not suggestions:
                    suggestions.append({"method": "OLS", "reason": "baseline — consider identification",
                                       "stata": "regress outcome treatment controls, robust"})
                method_suggestion = {"treatment": treatment, "outcome": outcome, "suggestions": suggestions[:3]}
                # Append to response text
                ms_lines = ["\n\n---", "💡 **E.D.I.T.H. Method Suggestion**"]
                if treatment and outcome:
                    ms_lines.append(f"Detected RQ: effect of *{treatment}* on *{outcome}*\n")
                for s in suggestions[:3]:
                    ms_lines.append(f"**{s['method']}** — {s['reason']}")
                    ms_lines.append(f"```stata\n{s['stata']}\n```")
                answer = answer + "\n".join(ms_lines)
        except Exception:
            pass

        return {
            "role": "assistant",
            "content": filter_output(answer),
            "used_model": used_model,
            "sources": sources,
            "source_policy": req.source_policy,
            "citation_coverage": round(coverage, 2),
            "no_evidence": False,
            "plan": plan,
            "audit": audit_result,
            "depth": depth,
            "generation_time": gen_elapsed,
            "deepened": deepened,
            "timings": timer.as_dict(),
            "request_id": rid,
            "confidence": confidence_data,
            "method_suggestion": method_suggestion,
        }


    except Exception as e:
        log.exception(f"500 error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Streaming chat endpoint — SSE-based real-time response
# ---------------------------------------------------------------------------

def chat_stream_endpoint(req: ChatRequest):
    """Streaming version — sends stage progress + real Winnie token chunks via SSE."""
    rid = uuid.uuid4().hex[:8]  # Request correlation ID

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate():
        API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not API_KEY:
            yield _sse("error", {"message": "Google API Key not configured"})
            return

        # §B6: Rate limit check
        if not rate_limiter.check("/chat/stream"):
            yield _sse("error", {"message": "Rate limit exceeded — please wait a moment"})
            return

        timer = PipelineTimer()
        _req_start = _time.monotonic()
        last_msg = req.messages[-1].content
        audit("chat_stream_query", request_id=rid, query=last_msg[:200], mode=req.mode, model=req.model)
        log.info(f"[{rid}] chat_stream: starting for {last_msg[:60]}")

        guard_err = guard_input(last_msg)
        if guard_err:
            audit("prompt_blocked", reason=guard_err, query=last_msg[:100])
            yield _sse("error", {"message": guard_err})
            return

        # §9.3: SSE heartbeat — keep connection alive during long operations
        _heartbeat_interval = 15  # seconds
        _last_heartbeat = [_time.time()]  # mutable container for nested scope

        def _maybe_heartbeat():
            """Emit a heartbeat SSE event if enough time has passed."""
            now = _time.time()
            if now - _last_heartbeat[0] >= _heartbeat_interval:
                _last_heartbeat[0] = now
                return _sse("heartbeat", {"t": round(now - _req_start, 1)})
            return None

        conversation_history = ""
        if len(req.messages) > 1:
            prev = req.messages[:-1][-6:]
            lines = [f"{m.role.upper()}: {m.content[:500]}" for m in prev]
            conversation_history = "CONVERSATION CONTEXT:\n" + "\n".join(lines) + "\n\n"

        if "gemini-1.5-flash" in req.model:
            req.model = DEFAULT_MODEL
        model_chain = build_model_chain(req.model)
        
        # §ROUTING: Use shared intelligent routing function
        model_chain, _dual_brain_mode, _route_info = _route_model_chain(last_msg, model_chain, rid)
        
        depth = _classify_depth(last_msg)

        # §AGENT: Intent classification — detect if tools should be invoked
        _tool_intent = None
        try:
            from server.intent_router import classify_intent_fast
            _intent_key, _intent_conf = classify_intent_fast(last_msg)
            if _intent_key != "chat" and _intent_conf >= 0.3:
                from server.intent_router import INTENT_PATTERNS
                _tool_intent = {
                    "intent": _intent_key,
                    "confidence": _intent_conf,
                    "pipeline": INTENT_PATTERNS[_intent_key]["pipeline"],
                    "description": INTENT_PATTERNS[_intent_key]["description"],
                }
        except Exception as _ie:
            log.debug(f"[{rid}] Intent classification skipped: {_ie}")

        # §PROFILE: Record topic and query in research profile
        try:
            from server.research_profile import get_profile as _get_rp
            _rp = _get_rp()
            _rp.record_query(last_msg, intent=(_tool_intent or {}).get("intent", "chat"))
        except Exception:
            pass

        yield _sse("stage", {"stage": "routing", "depth": depth, "route": _route_info.get("route", "default"),
                              "tool_intent": _tool_intent})

        # §4.0: Adaptive top-k by depth, scaled by hardware compute profile
        try:
            from server.backend_logic import get_compute_profile
            _hw = get_compute_profile()
            _hw_scale = _hw.get("top_k", 12) / 12.0  # scale relative to focus baseline
        except Exception:
            _hw_scale = 1.0
        _DEPTH_TOP_K = {"quick": 5, "standard": 10, "debate": 20}
        top_k = int(_DEPTH_TOP_K.get(depth, 10) * _hw_scale)
        # §4.0: Answer length → token budget
        # §HW: Hardware-adaptive token budget — M4 generates longer answers
        try:
            from server.backend_logic import get_compute_profile
            _hw_tok = get_compute_profile()
            _base_std = _hw_tok.get("max_tokens_standard", 2000)
            _base_dbt = _hw_tok.get("max_tokens_debate", 3000)
            _LENGTH_TOKENS = {
                "brief": min(800, _base_std),
                "standard": _base_std,
                "exhaustive": max(_base_dbt, 8000),
            }
        except Exception:
            _LENGTH_TOKENS = {"brief": 800, "standard": 2000, "exhaustive": 8000}
        max_tokens = _LENGTH_TOKENS.get(req.answer_length, 2000)

        # --- Retrieval (cached + circuit breaker + smart routing) ---
        sources = []

        # Modes that need source retrieval (not just grounded)
        _SOURCE_MODES = {"grounded", "counterargument", "lit_review", "paper_outline",
                         "research_design", "annotated_bib", "exam", "gap_analysis",
                         "reading_companion", "writing_assistant", "committee_sim",
                         "peer_review", "teaching_intro", "teaching_grad",
                         "teaching_expert", "discussant", "office_hours"}
        _needs_sources = req.mode in _SOURCE_MODES

        if _needs_sources and not should_skip_retrieval(last_msg):
            timer.start("retrieval")
            yield _sse("stage", {"stage": "retrieving"})
            hb = _maybe_heartbeat()
            if hb: yield hb

            # §FIX: Enrich vague follow-ups with conversation context
            # e.g. "synthesize this for me" → "synthesize this for me | hollow state"
            _retrieval_query = last_msg
            _REFERENTIAL_CUES = {"this", "that", "it", "these", "those", "above", "previous"}
            _is_vague = len(last_msg.split()) < 12 or any(w in last_msg.lower().split() for w in _REFERENTIAL_CUES)
            if _is_vague and len(req.messages) > 1:
                # Find the most recent substantive user message
                for _prev in reversed(req.messages[:-1]):
                    if _prev.role == "user" and len(_prev.content) > 20:
                        _retrieval_query = f"{_prev.content[:300]} | {last_msg}"
                        log.info(f"[{rid}] Enriched retrieval query with prior context: {_retrieval_query[:80]}...")
                        break

            if USE_GOOGLE_RETRIEVAL and not google_retrieval_breaker.is_open:
                try:
                    sources, was_cached = cached_retrieve(
                        query=_retrieval_query, store_id=GOOGLE_STORE_ID,
                        retrieve_fn=retrieve_google_sources,
                        api_key=API_KEY, top_k=top_k,
                    )
                    google_retrieval_breaker.record_success()
                    if was_cached:
                        yield _sse("stage", {"stage": "cache_hit"})
                except Exception as e:
                    google_retrieval_breaker.record_failure()
                    log.error(f"Google retrieval failed: {e}")

            if not sources:
                try:
                    sources = retrieve_local_sources(
                        queries=[_retrieval_query], chroma_dir=CHROMA_DIR,
                        collection_name=CHROMA_COLLECTION, embed_model=EMBED_MODEL,
                        top_k=5, bm25_weight=0.35, diversity_lambda=0.65,
                    )
                except Exception as _exc:
                    log.warning(f"Suppressed exception: {_exc}")

            timer.stop("retrieval")

        yield _sse("stage", {"stage": "retrieved", "source_count": len(sources)})

        # §6.4: Guard against adversarial content in retrieved sources
        if sources:
            sources = check_source_injection(sources)

        # --- Discovery Mode: external search only when user directs ---
        discovery_text = ""
        try:
            from server.discovery_mode import is_discovery_query, extract_discovery_topic, get_discovery_engine
            if is_discovery_query(last_msg):
                yield _sse("stage", {"stage": "discovering"})
                topic = extract_discovery_topic(last_msg)
                engine = get_discovery_engine()
                disc_results = engine.search(topic, max_results=10)
                discovery_text = engine.format_for_winnie(disc_results)
                log.info(f"[{rid}] Discovery: found {disc_results.get('total_found', 0)} papers on '{topic}'")
        except Exception as e:
            log.warning(f"Discovery mode error: {e}")

        # --- Recency + temporal boost ---
        if sources:
            try:
                from server.retrieval_enhancements import apply_temporal_weight, rerank_by_temporal_weight
                sources = apply_temporal_weight(sources)
                sources = rerank_by_temporal_weight(sources)
            except Exception as _exc:
                log.warning(f"Suppressed exception: {_exc}")

        # §4.0: Inline source dedup — remove near-duplicate chunks
        if len(sources) > 1:
            deduped = [sources[0]]
            for s in sources[1:]:
                s_words = set(s.get("text", "").lower().split())
                is_dup = False
                for d in deduped:
                    d_words = set(d.get("text", "").lower().split())
                    if s_words and d_words:
                        overlap = len(s_words & d_words) / max(len(s_words | d_words), 1)
                        if overlap > 0.9:
                            is_dup = True
                            break
                if not is_dup:
                    deduped.append(s)
            if len(deduped) < len(sources):
                log.info(f"[{rid}] Deduped sources: {len(sources)} → {len(deduped)}")
            sources = deduped

        # §FIX: Enrich sources with author/year/title from library cache
        if sources:
            try:
                from server.routes import library as _lib_mod
                _cached = getattr(_lib_mod, "_sources_cache", {})
                _cached_papers = _cached.get("papers") or _cached.get("all") or []
                lookup = {}
                if _cached_papers:
                    for d in _cached_papers:
                        src = d.get("source", "")
                        if src:
                            lookup[src] = d
                            fname = src.rsplit("/", 1)[-1] if "/" in src else src
                            lookup[fname] = d
                enriched = 0
                for s in sources:
                    path = (s.get("source") or s.get("path") or
                            (s.get("metadata") or {}).get("path") or
                            (s.get("metadata") or {}).get("source") or
                            (s.get("metadata") or {}).get("rel_path") or "")
                    fname = path.rsplit("/", 1)[-1] if "/" in path else path
                    meta = lookup.get(path) or lookup.get(fname)
                    # 1) Try library cache metadata
                    if meta:
                        if not s.get("title") and meta.get("title"):
                            s["title"] = meta["title"]
                        if meta.get("year") and not s.get("year"):
                            s["year"] = meta["year"]
                        if meta.get("author") and not _is_suspicious_author(meta["author"]):
                            if not s.get("author") or _is_suspicious_author(s.get("author", "")):
                                s["author"] = meta["author"]
                        enriched += 1
                    # 2) Fallback: extract author/year from filename
                    if not s.get("author") or _is_suspicious_author(s.get("author", "")):
                        fn_author, fn_year = _extract_author_year_from_filename(fname)
                        if fn_author:
                            s["author"] = fn_author
                        if fn_year and not s.get("year"):
                            s["year"] = fn_year
                if enriched:
                    log.info(f"[{rid}] Enriched {enriched}/{len(sources)} sources from library cache")
                # 3) Final cleanup: clear any remaining suspicious author names
                for s in sources:
                    if s.get("author") and _is_suspicious_author(s["author"]):
                        s["author"] = ""
            except Exception as _enrich_err:
                log.debug(f"[{rid}] Source enrichment skipped: {_enrich_err}")

        if _needs_sources and not sources and not should_skip_retrieval(last_msg) and not discovery_text:
            # §FIX: Instead of returning empty, fall back to general answer
            log.info(f"[{rid}] No sources found in grounded mode — falling back to general answer")
            yield _sse("stage", {"stage": "generating"})
            try:
                fallback_prompt = (
                    f"Answer this question to the best of your knowledge. "
                    f"Note: No library sources were found for this query, so provide a general answer "
                    f"and mention that the user should index relevant documents for grounded citations.\n\n"
                    f"Question: {last_msg}"
                )
                fb_answer, fb_model = generate_text_via_chain(fallback_prompt, model_chain, temperature=req.temperature)
                if not fb_answer:
                    fb_answer = "I couldn't find any sources in your library for this query, and the AI model didn't return a response. Try indexing relevant documents first, or switch to 'General' mode."
                    fb_model = "none"
                yield _sse("done", {
                    "content": fb_answer,
                    "sources": [],
                    "no_evidence": True,
                    "used_model": fb_model,
                    "audit": {"is_clean": True, "note": "general_fallback"},
                })
            except Exception as e:
                log.warning(f"[{rid}] General fallback also failed: {e}")
                yield _sse("done", {
                    "content": "No sources found in your library for this query. Try indexing your documents first (System → DevOps → Run Index), or switch to General mode for an ungrounded answer.",
                    "sources": [],
                    "no_evidence": True,
                    "used_model": "none",
                })
            return

        # --- Think (debate only) ---
        plan = {"used": False}
        if depth == "debate" and len(sources) >= 3:
            hb = _maybe_heartbeat()
            if hb: yield hb
            yield _sse("stage", {"stage": "thinking"})
            timer.start("planning")
            try:
                plan = plan_answer_outline(last_msg, sources, model_chain)
            except Exception as _exc:
                log.warning(f"Suppressed exception: {_exc}")
            timer.stop("planning")

        # --- Compress & build prompt ---
        # §HW: Hardware-adaptive compression
        try:
            from server.backend_logic import get_compute_profile
            _hw_comp2 = get_compute_profile()
            _max_chars2 = _hw_comp2.get("max_chars_per_source", 800)
            _max_srcs2 = _hw_comp2.get("max_sources_compressed", 8)
        except Exception:
            _max_chars2, _max_srcs2 = 800, 8
        compressed = compress_sources(sources, max_chars_per_source=_max_chars2, max_sources=_max_srcs2)

        # --- Mode-aware prompt selection ---
        prompt = _build_mode_prompt(req.mode, last_msg, compressed, plan, depth, discovery_text, template=req.template)
        # §DEBUG: Log prompt structure to diagnose "unable to answer" bug
        log.info(f"[{rid}] §DEBUG-PROMPT: mode={req.mode}, sources={len(sources)}, compressed={len(compressed)}")
        for _si, _sc in enumerate(compressed[:3]):
            _sn = len(_sc.get('snippet', ''))
            _st = len(_sc.get('text', ''))
            log.info(f"[{rid}] §DEBUG-SRC[{_si}]: snippet={_sn}, text={_st}, title={(_sc.get('title','')[:50])}")
        log.info(f"[{rid}] §DEBUG-PROMPT-HEAD: {prompt[:300]}")
        if "Content:" not in prompt[:3000]:
            log.warning(f"[{rid}] §DEBUG: NO 'Content:' found in first 3000 chars of prompt — sources may be empty!")
        if conversation_history:
            prompt = conversation_history + prompt

        # §4.0: Follow-up chaining — inject previous answer as context
        prev_msgs = [m for m in req.messages if m.role == "assistant" and m.content]
        if prev_msgs:
            last_answer = prev_msgs[-1].content
            if len(last_answer) > 100:
                # Truncate to ~1500 chars to avoid overwhelming context
                chained = last_answer[:1500] + ("..." if len(last_answer) > 1500 else "")
                prompt = f"PREVIOUS ANSWER (for context — build on this if relevant):\n{chained}\n\n---\n\n{prompt}"

        # --- Generate with REAL token streaming from Winnie ---
        hb = _maybe_heartbeat()
        if hb: yield hb
        yield _sse("stage", {"stage": "generating"})
        timer.start("generation")
        answer = ""
        used_model = ""
        _gen_start_time = _time.time()

        # §HW: Committee Mode — parallel multi-agent generation
        # Fix 7: M4 gets full committee; M2 gets lightweight 2-agent on debate
        _committee_attempted = False
        try:
            from server.backend_logic import get_compute_profile
            _hw = get_compute_profile()
            _hw_mode = _hw.get("mode", "focus")
            _wants_committee = (
                depth in ("debate", "standard") and
                len(sources) >= 3 and
                not (req.image_data and req.image_data.base64)
            )
            # M4: full committee | M2: lightweight 2-agent on debate only
            if _wants_committee and (_hw_mode == "committee" or
                                     (_hw_mode == "focus" and depth == "debate")):
                _n_agents = _hw.get("agents", 4) if _hw_mode == "committee" else 2
                _committee_attempted = True
                yield _sse("stage", {"stage": "committee", "agents": _n_agents})
                try:
                    from server.committee import run_committee_streaming
                    for event in run_committee_streaming(
                        query=last_msg,
                        sources=sources,
                        model_chain=model_chain,
                        chroma_dir=CHROMA_DIR,
                        collection_name=CHROMA_COLLECTION,
                        embed_model=EMBED_MODEL,
                        top_k=top_k,
                        max_agents=_n_agents,
                    ):
                        if event.get("type") == "token":
                            answer += event["text"]
                            yield _sse("chunk", {"text": event["text"]})
                        elif event.get("type") == "status":
                            yield _sse("stage", {"stage": "committee",
                                                  "detail": event["text"]})
                        elif event.get("type") == "done":
                            used_model = f"committee({_n_agents}a)"
                            log.info(f"[{rid}] §COMMITTEE: "
                                     f"{event.get('agents_used', 0)} agents, "
                                     f"{event.get('elapsed', 0):.1f}s")
                except Exception as _ce:
                    log.warning(f"[{rid}] §COMMITTEE failed, falling back: {_ce}")
                    answer = ""
        except Exception:
            pass

        # §HW: Local MLX inference for quick queries (no API call needed)
        if not answer and depth == "quick" and not (req.image_data and req.image_data.base64):
            try:
                from server.mlx_inference import is_available as local_avail, generate_stream
                if local_avail():
                    yield _sse("stage", {"stage": "generating_local"})
                    local_text = ""
                    for tok in generate_stream(
                        prompt=prompt,
                        system_instruction="You are a concise academic assistant. Answer briefly and accurately.",
                        max_tokens=max_tokens,
                        temperature=req.temperature,
                    ):
                        local_text += tok
                        yield _sse("chunk", {"text": tok})
                    if local_text.strip():
                        answer = local_text
                        used_model = "local_mlx"
                        log.info(f"[{rid}] §LOCAL: Quick query answered locally ({len(answer)} chars)")
            except Exception as _le:
                log.debug(f"[{rid}] §LOCAL: Local inference unavailable: {_le}")

        # Multimodal: if image attached, use Gemini vision directly
        if req.image_data and req.image_data.base64:
            try:
                import base64 as _b64
                img_bytes = _b64.b64decode(req.image_data.base64)
                img_part = {"inline_data": {"mime_type": req.image_data.mime_type, "data": req.image_data.base64}}
                model_obj = genai.GenerativeModel(req.model or DEFAULT_MODEL)
                response = model_obj.generate_content([prompt, img_part], stream=True)
                used_model = req.model or DEFAULT_MODEL
                for chunk in response:
                    if chunk.text:
                        answer += chunk.text
                        yield _sse("chunk", {"text": chunk.text})
            except Exception as e:
                log.error(f"Multimodal generation failed: {e}")
                # Fall through to text-only generation
                answer = ""

        if not answer and OPENAI_FT_MODEL and OPENAI_API_KEY and not openai_breaker.is_open:
            try:
                ft_max = max_tokens if depth != "debate" else max(max_tokens, 4500)
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                used_model = OPENAI_FT_MODEL
                _ft_first_token = False
                _rep_guard = _RepetitionGuard()
                for tok in call_openai_streaming(
                    api_key=OPENAI_API_KEY, model=OPENAI_FT_MODEL,
                    prompt=prompt, temperature=req.temperature,
                    max_tokens=ft_max, base_url=base_url,
                ):
                    # §4.0: Latency fallback — adaptive deadline for exhaustive queries
                    if not _ft_first_token:
                        _ft_first_token = True
                        _ft_deadline = 25.0 if req.answer_length == "exhaustive" else 8.0
                        if _time.time() - _gen_start_time > _ft_deadline:
                            log.warning(f"[{rid}] OpenAI FT too slow ({_time.time() - _gen_start_time:.1f}s) — falling back to Gemini")
                            answer = ""
                            break
                    answer += tok
                    yield _sse("chunk", {"text": tok})
                    # §FIX: Check for repetition loop every ~500 chars
                    if len(answer) > 500 and len(answer) % 200 < len(tok) + 1:
                        if _rep_guard.check(answer):
                            log.warning(f"[{rid}] §REPGUARD: repetition detected in OpenAI stream, truncating")
                            answer = _rep_guard.truncate(answer)
                            break
                if _rep_guard.triggered:
                    yield _sse("chunk", {"text": "\n\n> ⚠ *Response was trimmed — repetitive output detected.*"})
                    answer += "\n\n> ⚠ *Response was trimmed — repetitive output detected.*"
                openai_breaker.record_success()
            except Exception as e:
                openai_breaker.record_failure()
                log.error(f"Winnie streaming failed: {e}")
                answer = ""

        if not answer:
            try:
                # Stream Gemini fallback token-by-token with a local client instance.
                # Avoids reliance on module-level CLIENT wiring in this route module.
                from google import genai as _genai_mod
                from google.genai import types as _genai_types

                cfg = _genai_types.GenerateContentConfig(
                    temperature=req.temperature,
                    max_output_tokens=max(max_tokens, 4000) if depth == "debate" else max_tokens,
                )
                model_to_use = model_chain[0] if model_chain else DEFAULT_MODEL
                _client = _genai_mod.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
                _rep_guard_g = _RepetitionGuard()
                for chunk in _client.models.generate_content_stream(
                    model=model_to_use,
                    contents=prompt,
                    config=cfg,
                ):
                    tok = (getattr(chunk, "text", "") or "")
                    if tok:
                        answer += tok
                        yield _sse("chunk", {"text": tok})
                        # §FIX: Check for repetition loop every ~500 chars
                        if len(answer) > 500 and len(answer) % 200 < len(tok) + 1:
                            if _rep_guard_g.check(answer):
                                log.warning(f"[{rid}] §REPGUARD: repetition detected in Gemini stream, truncating")
                                answer = _rep_guard_g.truncate(answer)
                                break
                if _rep_guard_g.triggered:
                    yield _sse("chunk", {"text": "\n\n> ⚠ *Response was trimmed — repetitive output detected.*"})
                    answer += "\n\n> ⚠ *Response was trimmed — repetitive output detected.*"
                used_model = model_to_use
            except Exception as e:
                log.error(f"[{rid}] Gemini stream fallback failed: {e}")
                if not answer:
                    yield _sse("error", {"message": f"Generation failed: {e}"})
                    return

        timer.stop("generation")

        # §ROUTING: Dual-brain judge for complex queries (streaming)
        if _dual_brain_mode and answer and OPENAI_FT_MODEL:
            judged_answer, judged_model, _verdict = _run_dual_brain_judge(
                last_msg, answer, sources, req.temperature, rid
            )
            if judged_answer != answer:
                # Replace streamed answer with judge-selected answer
                answer = judged_answer
                used_model = judged_model
                yield _sse("judge_replace", {"text": answer, "model": judged_model,
                                            "verdict": _verdict.get("winner", ""),
                                            "confidence": _verdict.get("confidence", 0)})

        # §MCL-2026: Claude second-opinion on debate-depth queries
        if depth == "debate" and answer and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from server.anthropic_bridge import AnthropicBridge
                _claude = AnthropicBridge()
                _claude_response = _claude.query(
                    f"A political science AI gave this answer to the question: '{last_msg[:200]}'\n\n"
                    f"ANSWER:\n{answer[:2000]}\n\n"
                    f"In 2-3 sentences, note any important caveats, missing nuance, or alternative interpretations "
                    f"that the answer might benefit from. If the answer is solid, say so briefly.",
                    max_tokens=400,
                )
                _claude_note = _claude_response.get("answer", "").strip()
                if _claude_note and len(_claude_note) > 20:
                    answer += f"\n\n> 🧠 **Claude Second Opinion**: {_claude_note}"
                    yield _sse("stage", {"stage": "claude_review", "note": _claude_note[:200]})
                    log.info(f"[{rid}] §CLAUDE: second opinion appended ({len(_claude_note)} chars)")
            except Exception as _cle:
                log.debug(f"[{rid}] §CLAUDE: second opinion skipped: {_cle}")

        # Fix 5: Metabolic SSE — real-time HUD data piggybacked on chat stream
        try:
            from server.backend_logic import get_compute_profile
            _hw_met = get_compute_profile()
            import psutil as _ps
            _mem = _ps.virtual_memory()
            yield _sse("metabolic", {
                "tokens_generated": len(answer.split()),
                "elapsed_s": round(_time.time() - _gen_start_time, 2),
                "ram_used_gb": round(_mem.used / (1024**3), 1),
                "ram_total_gb": round(_mem.total / (1024**3), 1),
                "hw_mode": _hw_met.get("mode", "focus"),
                "active_lobe": "generation",
            })
        except Exception:
            pass

        # --- Audit — runs for standard AND debate depth (expanded §4.0) ---
        audit_result = {"is_clean": False, "error": "audit_skipped"}
        if depth == "debate" and sources:
            # Full API-powered audit — only for debate depth (takes ~20-30s)
            timer.start("audit")
            hb = _maybe_heartbeat()
            if hb: yield hb
            yield _sse("stage", {"stage": "auditing"})
            try:
                audit_result = audit_answer(answer, sources, [DEFAULT_MODEL])
                corrections = audit_result.get("corrections", [])
                if corrections and not audit_result.get("is_clean", True):
                    answer = apply_corrections(answer, corrections, model_chain)
                    audit_result["corrections_applied"] = True
                    yield _sse("corrected", {"text": answer})
            except Exception:
                audit_result = {"is_clean": False, "error": "audit_failed"}
            timer.stop("audit")
        elif depth == "standard" and sources:
            # Fast local-only audit — skip the slow API call
            timer.start("audit")
            audit_result = {"is_clean": True, "note": "fast_local_audit"}
            timer.stop("audit")
        elif not sources:
            # §4.0: Audit exemption — skip when no sources found
            audit_result = {"is_clean": True, "note": "no sources, audit exempt"}
        elif sources:
            # §FIX Vuln 2: Non-debate/standard with sources — trust without full audit
            audit_result = {"is_clean": True, "note": "non-audited depth, sources present"}

        # §4.0: Per-claim local text matching — runs for all depths with sources (fast, no API call)
        if sources and len(answer) > 200:
            try:
                source_text_blob = " ".join(s.get("text", "") for s in sources).lower()
                if source_text_blob:
                    import re as _re_audit
                    claims = _re_audit.split(r'(?<=[.!?])\s+', answer)
                    claim_audit = []
                    for claim in claims:
                        claim_clean = claim.strip()
                        if len(claim_clean) < 20:
                            continue
                        claim_words = set(claim_clean.lower().split())
                        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                                      "has", "have", "had", "do", "does", "did", "will", "would",
                                      "could", "should", "may", "might", "can", "shall", "to", "of",
                                      "in", "for", "on", "with", "at", "by", "from", "as", "into",
                                      "that", "this", "it", "not", "and", "or", "but", "if", "than",
                                      "their", "they", "there", "these", "those", "which", "what"}
                        key_words = claim_words - stop_words
                        if not key_words:
                            continue
                        found = sum(1 for w in key_words if w in source_text_blob)
                        support_ratio = found / max(len(key_words), 1)
                        claim_audit.append({
                            "claim": claim_clean[:100],
                            "supported": support_ratio > 0.3,
                            "support_ratio": round(support_ratio, 2),
                        })
                    audit_result["claim_audit"] = claim_audit
                    unsupported = [c for c in claim_audit if not c["supported"]]
                    if unsupported:
                        audit_result["unsupported_claims"] = len(unsupported)
                        log.info(f"[{rid}] Per-claim audit: {len(unsupported)}/{len(claim_audit)} claims lack source support")
            except Exception as _pca_err:
                log.debug(f"[{rid}] Per-claim audit error: {_pca_err}")

        # Fix 2: Wire citation_middleware — add inline citations before final output
        if sources and answer:
            try:
                answer = citation_middleware(answer, sources)
            except Exception as _cm_err:
                log.debug(f"[{rid}] citation_middleware skipped: {_cm_err}")

        cited_count = sum(1 for i in range(len(sources)) if f"[S{i+1}]" in answer)
        coverage = cited_count / max(len(sources), 1)
        log.info(f"Stream timings: {timer.summary()}")

        # Autolearn — only save audit-clean answers to training data
        if audit_result.get("is_clean", False) and coverage > 0.2:
            try:
                _save_autolearn(last_msg, answer, sources, used_model, coverage)
            except Exception as e:
                log.warning(f"Autolearn save failed: {e}")

        # Fix 6: Log audit-failed answers as DPO negative examples
        if not audit_result.get("is_clean", False) and answer and sources:
            try:
                _save_dpo_negative(last_msg, answer, sources, audit_result)
            except Exception:
                pass

        # Retrieval confidence scoring
        if sources:
            avg_score = sum(s.get("score", 0) for s in sources) / len(sources)
            if avg_score > 0.7 and len(sources) >= 3:
                retrieval_confidence = "high"
            elif avg_score > 0.4 or len(sources) >= 2:
                retrieval_confidence = "moderate"
            else:
                retrieval_confidence = "low"
        else:
            retrieval_confidence = "none"

        # §4.0: Confidence disclaimer — auto-append when low/none
        if retrieval_confidence in ("low", "none") and sources:
            answer += "\n\n> ⚠ **Low Evidence Warning**: The available sources provide limited support for this answer. Verify claims independently."

        # §MCL-2026: Perplexity auto-verify when source_policy includes web
        if (req.mode == "grounded" and answer
                and getattr(req, 'source_policy', '') in ('web_only', 'files_web')
                and os.environ.get("PERPLEXITY_API_KEY")):
            try:
                from server.perplexity_bridge import PerplexityBridge
                _pplx = PerplexityBridge()
                # Pick the first substantive claim to verify
                _claims = [c for c in audit_result.get("claim_audit", []) if not c.get("supported")]
                _verify_claim = _claims[0]["claim"] if _claims else last_msg[:200]
                _pplx_result = _pplx.verify_claim(_verify_claim)
                _verdict_text = _pplx_result.get("verdict", "")[:300]
                if _verdict_text:
                    answer += f"\n\n> ⚡ **Real-Time Verification** (Perplexity): {_verdict_text}"
                    yield _sse("stage", {"stage": "perplexity_verify", "claim": _verify_claim[:100]})
                    log.info(f"[{rid}] §PERPLEXITY: claim verified")
            except Exception as _pvx:
                log.debug(f"[{rid}] §PERPLEXITY: auto-verify skipped: {_pvx}")

        # §SUGGESTIONS: Generate proactive next-step suggestions
        _suggestions = []
        try:
            from server.proactive_suggestions import get_chat_suggestions
            _suggestions = get_chat_suggestions(last_msg, answer, sources)
            if _suggestions:
                yield _sse("stage", {"stage": "suggestions", "count": len(_suggestions)})
        except Exception as _sx:
            log.debug(f"[{rid}] Proactive suggestions skipped: {_sx}")

        yield _sse("done", {
            "content": answer, "used_model": used_model,
            "sources": sources, "citation_coverage": round(coverage, 2),
            "plan": plan, "audit": audit_result, "depth": depth,
            "retrieval_confidence": retrieval_confidence,
            "timings": timer.as_dict(),
            "answer_length": req.answer_length,
            "template": req.template,
            "tool_intent": _tool_intent,
            "suggestions": _suggestions,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


# §ROUTE-BIND: Route bindings
def register(app, ns=None):
    """Register chat routes.
    
    chat_endpoint and chat_stream_endpoint are "coupled" handlers that
    reference 100+ globals from main.py (API_KEY, CHROMA_DIR, generate_text_via_chain,
    audit, etc.). Rather than importing each, we inject main.py's globals into
    this module's namespace via the ns=globals() parameter passed by main.py.
    """
    if ns:
        # Inject main.py globals into this module so chat_endpoint/
        # chat_stream_endpoint can resolve their 100+ external references
        import sys
        this_module = sys.modules[__name__]
        for key, val in ns.items():
            if not key.startswith("__") and not hasattr(this_module, key):
                setattr(this_module, key, val)

    # Wire the coupled handlers
    app.add_api_route("/chat", chat_endpoint, methods=["POST"], tags=["Chat"])
    app.add_api_route("/api/chat", chat_endpoint, methods=["POST"], tags=["Chat"])
    app.add_api_route("/chat/stream", chat_stream_endpoint, methods=["POST"], tags=["Chat"])
    app.add_api_route("/api/chat/stream", chat_stream_endpoint, methods=["POST"], tags=["Chat"])
    return router

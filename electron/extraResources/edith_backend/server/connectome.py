"""
Connectome — Logic Audit + Causal Proof Finding
==================================================
The Metacognitive Layer: from reactive processing to predictive
intellectual metabolism.

The Connectome doesn't just link "Note to PDF." It links
"Theory → Mechanism → Proof."

Three functions:
1. AUDIT — Check if a thought contradicts the Bolt's literature
2. PROOF — Find causal evidence (RDD/IV data) to support a claim
3. FUTURE-CAST — If proof > 0.95, archive as a 2027 grant seed

The Logic Auditor runs as a Shadow Argument — if your claim
is weak, the HUD "dims" physically representing doubt.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("edith.connectome")


# ═══════════════════════════════════════════════════════════════════
# Logic Audit — Formal Consistency Checking
# ═══════════════════════════════════════════════════════════════════

LOGICAL_PATTERNS = {
    "affirming_consequent": {
        "pattern": r"(?:because|since)\s+.+?\s+(?:then|therefore|so)\s+.+",
        "description": "Potential affirming the consequent — verify causal direction",
        "severity": "high",
    },
    "hasty_generalization": {
        "signals": ["all", "every", "never", "always", "no one", "everyone"],
        "description": "Hasty generalization — provide sample evidence",
        "severity": "moderate",
    },
    "appeal_to_authority": {
        "signals": ["scholars agree", "experts say", "it is well known",
                     "the literature shows", "research proves"],
        "description": "Appeal to authority — cite specific evidence",
        "severity": "low",
    },
    "causal_confusion": {
        "signals": ["causes", "caused by", "leads to", "results in", "produces"],
        "description": "Causal claim — can you establish mechanism, not just correlation?",
        "severity": "high",
    },
    "unfalsifiable": {
        "signals": ["may", "might", "could", "possibly", "perhaps"],
        "description": "Hedged/unfalsifiable claim — can you be more precise?",
        "severity": "low",
    },
}


@dataclass
class AuditReport:
    """Result of a logic audit."""
    claim: str
    is_consistent: bool
    confidence: float
    issues: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    supporting_evidence: list[dict] = field(default_factory=list)
    conflict_reason: str = ""
    hud_action: str = "none"  # "none", "dim", "glow", "wiggle"

    def to_dict(self) -> dict:
        return {
            "claim": self.claim[:200],
            "consistent": self.is_consistent,
            "confidence": round(self.confidence, 2),
            "issues": self.issues,
            "contradictions_found": len(self.contradictions),
            "supporting_found": len(self.supporting_evidence),
            "hud_action": self.hud_action,
        }


@dataclass
class CausalProof:
    """A piece of causal evidence found on the Bolt."""
    method: str  # "RDD", "IV", "DiD", etc.
    source: str  # file path or paper title
    strength: float  # 0–1
    variables: list[str]
    summary: str

    def to_dict(self) -> dict:
        return {
            "method": self.method, "source": self.source,
            "strength": round(self.strength, 2),
            "variables": self.variables,
            "summary": self.summary[:200],
        }


class Connectome:
    """Theory → Mechanism → Proof connector with logic auditing.

    Usage:
        conn = Connectome()

        # Audit a claim for logical consistency
        report = conn.audit("Charities mask state failure")

        # Find causal proof for a claim
        proof = conn.find_causal_proof("Charity density reduces voter turnout")

        # Full metabolic flow: audit + proof + future-cast
        result = conn.metabolic_flow("My new theoretical claim")
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._audit_history: list[dict] = []
        self._proof_cache: dict[str, CausalProof] = {}
        self._future_projects: list[dict] = []

    def metabolic_flow(self, current_thought: str) -> dict:
        """The brain's pulse: Intake → Audit → Map → Proof → Future-Cast."""
        t0 = time.time()

        # 1. AUDIT: Does this thought contradict the Bolt?
        audit = self.audit(current_thought)

        # 2. PROOF: Find causal evidence
        proofs = self.find_causal_proof(current_thought)

        # 3. FUTURE-CAST: Is this a potential project?
        future = None
        max_strength = max((p.strength for p in proofs), default=0)
        if max_strength > 0.7:
            future = self._future_cast(current_thought, proofs)

        elapsed = time.time() - t0
        return {
            "thought": current_thought[:200],
            "audit": audit.to_dict(),
            "proofs": [p.to_dict() for p in proofs],
            "future_project": future,
            "hud_action": audit.hud_action,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    # ─── Logic Audit ──────────────────────────────────────────────

    def audit(self, claim: str) -> AuditReport:
        """Audit a claim for logical consistency and fallacies."""
        claim_lower = claim.lower()
        issues = []
        confidence = 0.8  # Start optimistic

        # Check for logical patterns
        for pattern_name, pattern_data in LOGICAL_PATTERNS.items():
            detected = False

            if "signals" in pattern_data:
                for signal in pattern_data["signals"]:
                    if signal in claim_lower:
                        detected = True
                        break

            if "pattern" in pattern_data:
                if re.search(pattern_data["pattern"], claim, re.IGNORECASE):
                    detected = True

            if detected:
                severity = pattern_data["severity"]
                penalty = {"high": 0.2, "moderate": 0.1, "low": 0.05}[severity]
                confidence -= penalty

                issues.append({
                    "type": pattern_name,
                    "severity": severity,
                    "description": pattern_data["description"],
                })

        # Check against vault for contradictions
        contradictions = self._check_vault_consistency(claim)

        if contradictions:
            confidence -= len(contradictions) * 0.15

        # Determine HUD action
        hud_action = "none"
        if confidence < 0.3:
            hud_action = "dim"  # Physically dims the text
        elif confidence < 0.5:
            hud_action = "wiggle"  # Atlas wiggles in weak areas
        elif confidence > 0.9 and not issues:
            hud_action = "glow"  # Strong claim — node glows

        confidence = max(0.0, min(1.0, confidence))
        is_consistent = confidence >= 0.5 and len(contradictions) == 0

        report = AuditReport(
            claim=claim,
            is_consistent=is_consistent,
            confidence=confidence,
            issues=issues,
            contradictions=contradictions,
            conflict_reason=(
                f"{len(issues)} logical concerns, {len(contradictions)} vault contradictions"
                if issues or contradictions else ""
            ),
            hud_action=hud_action,
        )

        self._audit_history.append(report.to_dict())
        return report

    def _check_vault_consistency(self, claim: str) -> list[dict]:
        """Check if a claim contradicts anything in the vault."""
        contradictions = []
        vault = Path(self._bolt_path) / "VAULT"

        if not vault.exists():
            return contradictions

        claim_lower = claim.lower()

        # Extract the core assertion terms
        assertion_terms = set(claim_lower.split()) - {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "and", "or", "but", "that", "this",
            "it", "they", "we", "she", "he", "not", "no", "with", "from",
        }

        # Quick scan of existing analyses for contradictions
        forensics_dir = vault / "FORENSICS"
        if forensics_dir.exists():
            for f in forensics_dir.glob("*_audit_*.json"):
                try:
                    data = json.loads(f.read_text())
                    paper_theory = data.get("audit", {}).get("theory", {})
                    primary = paper_theory.get("primary_framework", "").lower()

                    # Check if the claim's terms overlap with known contrary evidence
                    if primary and any(t in primary for t in assertion_terms if len(t) > 4):
                        # Found a potentially relevant paper — check for contradiction
                        paper_title = data.get("audit", {}).get("paper", {}).get("title", f.stem)
                        contradictions.append({
                            "source": paper_title,
                            "type": "theoretical_tension",
                            "detail": f"Paper uses '{primary}' which may tension with your claim",
                        })
                except Exception:
                    continue

        return contradictions[:3]

    # ─── Causal Proof Finding ─────────────────────────────────────

    def find_causal_proof(self, claim: str) -> list[CausalProof]:
        """Find causal evidence on the Bolt to support a claim."""
        proofs = []
        claim_lower = claim.lower()

        # Check what kind of causal method is needed
        relevant_methods = []
        if any(w in claim_lower for w in ["effect", "impact", "cause", "increase", "decrease"]):
            relevant_methods = ["RDD", "IV", "DiD", "OLS"]
        if any(w in claim_lower for w in ["compare", "versus", "difference"]):
            relevant_methods.append("DiD")
        if any(w in claim_lower for w in ["threshold", "cutoff", "boundary"]):
            relevant_methods.append("RDD")

        if not relevant_methods:
            relevant_methods = ["OLS"]  # Default

        # Search vault for regression outputs and annotations
        vault = Path(self._bolt_path) / "VAULT"

        # Check annotations
        annotations_dir = vault / "ANNOTATIONS"
        if annotations_dir.exists():
            for f in annotations_dir.glob("*.md"):
                try:
                    content = f.read_text(errors="ignore").lower()
                    for method in relevant_methods:
                        if method.lower() in content:
                            # Extract key variables
                            var_matches = re.findall(r'(\b\w+)\s+is\s+(?:positively|negatively)\s+associated', content)
                            proofs.append(CausalProof(
                                method=method,
                                source=f.stem,
                                strength=0.7,
                                variables=var_matches[:5],
                                summary=content[:300],
                            ))
                except Exception:
                    continue

        # Check Stata outputs
        stata_dir = vault / "STATA_OUTPUT"
        if stata_dir.exists():
            for f in stata_dir.rglob("*.log"):
                try:
                    content = f.read_text(errors="ignore")
                    for method in relevant_methods:
                        if method.lower() in content.lower():
                            proofs.append(CausalProof(
                                method=method,
                                source=str(f),
                                strength=0.8,
                                variables=[],
                                summary=f"Stata output with {method} found in {f.name}",
                            ))
                except Exception:
                    continue

        # Also generate a "what you need" recommendation if no proof found
        if not proofs:
            for method in relevant_methods:
                proofs.append(CausalProof(
                    method=method,
                    source="NEEDED",
                    strength=0.0,
                    variables=[],
                    summary=f"No {method} evidence found on the Bolt. Consider running this analysis.",
                ))

        return proofs[:5]

    # ─── Future-Casting ───────────────────────────────────────────

    def _future_cast(self, thought: str, proofs: list[CausalProof]) -> dict:
        """If proof strength > 0.7, archive as a Future Project."""
        project = {
            "title": f"Future Project: {thought[:80]}",
            "seed_thought": thought,
            "strongest_proof": proofs[0].to_dict() if proofs else {},
            "grant_tag": "Future Project: 2027 Grant",
            "created": time.strftime("%Y-%m-%d"),
        }

        self._future_projects.append(project)

        # Save to Bolt
        futures_dir = Path(self._bolt_path) / "FUTURE_PROJECTS"
        futures_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M")
        slug = re.sub(r'[^\w]', '_', thought[:40])
        future_file = futures_dir / f"future_{slug}_{ts}.md"
        try:
            future_file.write_text(
                f"# {project['title']}\n\n"
                f"**Tagged**: {project['grant_tag']}\n"
                f"**Seed**: {thought}\n\n"
                f"## Strongest Evidence\n\n"
                f"Method: {proofs[0].method if proofs else 'TBD'}\n"
                f"Source: {proofs[0].source if proofs else 'TBD'}\n"
            )
        except Exception:
            pass

        return project

    @property
    def status(self) -> dict:
        return {
            "audits_performed": len(self._audit_history),
            "future_projects": len(self._future_projects),
            "proof_cache_size": len(self._proof_cache),
        }


# Global instance
connectome = Connectome()

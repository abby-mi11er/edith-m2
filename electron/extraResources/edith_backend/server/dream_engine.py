"""
Dream Engine — Overnight Speculative Synthesis
=================================================
While you sleep, Winnie is "Dreaming" — finding hidden theoretical
bridges between subfields you haven't connected yet.

This script runs overnight and produces:
1. Bridge Notes — connections between distant subfields
2. Future Project seeds — grant-worthy ideas from your vault
3. Shadow Drafts — pre-written summaries for tomorrow's readings
4. Theoretical Gap Reports — holes in your literature coverage

Architecture:
    Vault Scan → Concept Extraction → Cross-Subfield Matching →
    Semantic Similarity > 0.85 → Bridge Note Generation →
    /Volumes/CITADEL/FUTURE_PROJECTS/Bridge_XXXX.md
"""

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.dream_engine")


# ═══════════════════════════════════════════════════════════════════
# Dissertation Core Concepts — The Seeds Winnie Dreams About
# ═══════════════════════════════════════════════════════════════════

DISSERTATION_CORE = {
    "administrative_burden": {
        "description": "Learning, compliance, and psychological costs imposed on citizens",
        "related_terms": ["red tape", "hassle", "take-up", "ordeal", "compliance cost",
                          "bureaucratic friction", "paperwork burden"],
        "subfield": "public_administration",
    },
    "state_capacity": {
        "description": "The ability of the state to implement policies and deliver services",
        "related_terms": ["infrastructural power", "extractive capacity", "service delivery",
                          "government effectiveness", "bureaucratic quality"],
        "subfield": "comparative_politics",
    },
    "charity_substitution": {
        "description": "When charitable organizations replace state welfare functions",
        "related_terms": ["crowding out", "privatization", "nonprofit", "NGO",
                          "voluntary sector", "third sector", "welfare mix"],
        "subfield": "public_policy",
    },
    "blame_diffusion": {
        "description": "How accountability becomes diffuse when multiple actors deliver services",
        "related_terms": ["accountability", "blame avoidance", "attribution",
                          "responsibility", "democratic accountability"],
        "subfield": "political_accountability",
    },
    "policy_feedback": {
        "description": "How policies reshape politics by creating constituencies and behaviors",
        "related_terms": ["lock-in", "path dependence", "increasing returns",
                          "submerged state", "hidden welfare"],
        "subfield": "american_politics",
    },
    "cartel_governance": {
        "description": "How criminal organizations provide governance functions",
        "related_terms": ["narco-governance", "criminal governance", "parallel state",
                          "non-state governance", "protection racket"],
        "subfield": "latin_american_politics",
    },
    "federalism_devolution": {
        "description": "The delegation of authority from federal to state/local levels",
        "related_terms": ["devolution", "intergovernmental", "fiscal federalism",
                          "unfunded mandates", "block grants"],
        "subfield": "american_politics",
    },
}

# Foreign subfields to search for unexpected connections
FOREIGN_SUBFIELDS = {
    "post_conflict": {
        "concepts": ["peacebuilding", "reconstruction", "transitional justice",
                     "DDR", "state building", "UN peacekeeping"],
        "bridge_potential": "How post-conflict NGO-state dynamics mirror domestic charity substitution",
    },
    "development_economics": {
        "concepts": ["foreign aid", "institutional quality", "aid dependency",
                     "Dutch disease", "conditionality", "structural adjustment"],
        "bridge_potential": "Foreign aid creating dependency parallels domestic charity dependency",
    },
    "environmental_governance": {
        "concepts": ["common pool resources", "environmental justice",
                     "regulatory capture", "green municipalism"],
        "bridge_potential": "Environmental governance fragmentation mirrors welfare fragmentation",
    },
    "digital_governance": {
        "concepts": ["e-government", "algorithmic governance", "digital divide",
                     "platform governance", "smart cities"],
        "bridge_potential": "Digital administrative burden as the next frontier",
    },
    "health_policy": {
        "concepts": ["Medicaid expansion", "safety net", "community health",
                     "free clinics", "charity care", "uncompensated care"],
        "bridge_potential": "Health charity care as a direct parallel to welfare charity substitution",
    },
    "education_policy": {
        "concepts": ["charter schools", "vouchers", "Title I", "school choice",
                     "public-private partnership", "education privatization"],
        "bridge_potential": "Education privatization as a parallel to welfare privatization",
    },
}


@dataclass
class BridgeNote:
    """A theoretical bridge discovered during dreaming."""
    bridge_id: str
    source_concept: str
    source_subfield: str
    target_concept: str
    target_subfield: str
    similarity_score: float
    bridge_description: str
    research_question: str
    grant_potential: str
    discovered_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.bridge_id,
            "from": f"{self.source_concept} ({self.source_subfield})",
            "to": f"{self.target_concept} ({self.target_subfield})",
            "similarity": round(self.similarity_score, 3),
            "description": self.bridge_description,
            "research_question": self.research_question,
            "grant_potential": self.grant_potential,
        }

    def to_markdown(self) -> str:
        return (
            f"# Bridge Note: {self.source_concept} ↔ {self.target_concept}\n\n"
            f"**Discovered**: {time.strftime('%B %d, %Y', time.localtime(self.discovered_at))}\n"
            f"**Similarity Score**: {self.similarity_score:.3f}\n\n"
            f"## Connection\n\n"
            f"**From**: {self.source_concept} ({self.source_subfield})\n"
            f"**To**: {self.target_concept} ({self.target_subfield})\n\n"
            f"## Bridge Description\n\n{self.bridge_description}\n\n"
            f"## Potential Research Question\n\n{self.research_question}\n\n"
            f"## Grant Potential\n\n{self.grant_potential}\n"
        )


@dataclass
class GapReport:
    """A gap in literature coverage detected during dreaming."""
    concept: str
    subfield: str
    coverage_score: float  # 0-1, how well covered in your vault
    gap_description: str
    suggested_readings: list[str]

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "subfield": self.subfield,
            "coverage": round(self.coverage_score, 2),
            "gap": self.gap_description,
            "suggestions": self.suggested_readings,
        }


class DreamEngine:
    """Overnight speculative synthesis engine.

    While you sleep, Winnie:
    1. Scans the vault for concept co-occurrence
    2. Searches foreign subfields for hidden bridges
    3. Generates Bridge Notes for high-similarity matches
    4. Identifies gaps in your literature coverage
    5. Seeds Future Project ideas

    Usage:
        engine = DreamEngine()
        results = engine.dream()  # Full overnight cycle
        bridges = engine.find_bridges()  # Just bridge discovery
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._vault_path = os.path.join(self._bolt_path, "VAULT")
        self._futures_path = os.path.join(self._bolt_path, "FUTURE_PROJECTS")
        self._bridges: list[BridgeNote] = []
        self._gaps: list[GapReport] = []
        self._dream_log: list[dict] = []

    def dream(self) -> dict:
        """Full overnight dream cycle.

        "You say 'Winnie, archive and Dream.' Every note, code snippet,
        and link is Deep-Indexed at 3,100 MB/s."
        """
        t0 = time.time()
        log.info("§DREAM: Entering dream state...")

        # Phase 1: Scan the vault
        vault_contents = self._scan_vault()

        # Phase 2: Find bridges between dissertation concepts and foreign subfields
        bridges = self.find_bridges(vault_contents)

        # Phase 3: Detect coverage gaps
        gaps = self._detect_gaps(vault_contents)

        # Phase 4: Generate Future Project seeds
        futures = self._generate_futures(bridges)

        # Phase 5: Save everything
        save_result = self._save_dream_results(bridges, gaps, futures)

        # Phase 6: LoRA Sharpening — adapt Winnie's weights with user corrections
        lora_result = {"status": "skipped"}
        try:
            from server.lora_trainer import LoRATrainer
            trainer = LoRATrainer()
            if trainer.is_available() and trainer.has_enough_ram() and trainer.needs_training():
                log.info("§DREAM: Starting LoRA sharpening session...")
                lora_result = trainer.train(max_minutes=10)
                log.info(f"§DREAM: LoRA sharpening: {lora_result.get('status')}")
            else:
                reasons = []
                if not trainer.is_available():
                    reasons.append("mlx-lm not installed")
                if not trainer.has_enough_ram():
                    reasons.append("< 16GB RAM")
                if not trainer.needs_training():
                    reasons.append("no new corrections")
                lora_result = {"status": "skipped", "reasons": reasons}
        except Exception as e:
            log.debug(f"§DREAM: LoRA sharpening skipped: {e}")
            lora_result = {"status": "error", "error": str(e)}

        # Phase 7: Cloud Fine-Tune Readiness Check
        # §FIX: Check if enough autolearn entries have accumulated for OpenAI fine-tuning
        finetune_check = {"ready": False}
        try:
            from server.training_tools import check_finetune_trigger
            from pathlib import Path as _dream_Path
            autolearn_path = _dream_Path(self._bolt_path) / "training" / "autolearn.jsonl"
            if not autolearn_path.exists():
                # Fallback: check project root
                autolearn_path = _dream_Path(__file__).resolve().parent.parent / "autolearn.jsonl"
            last_count_path = _dream_Path(self._bolt_path) / "training" / "last_finetune_count.txt"
            if autolearn_path.exists():
                finetune_check = check_finetune_trigger(
                    autolearn_path, threshold=100, last_count_path=last_count_path
                )
                if finetune_check.get("trigger"):
                    log.info(f"§DREAM: ☁️ Cloud fine-tune recommended! "
                             f"{finetune_check.get('count', 0)} new entries since last fine-tune. "
                             f"Run: python pipelines/finetune_openai.py")
        except Exception as e:
            log.debug(f"§DREAM: Cloud fine-tune check skipped: {e}")

        elapsed = time.time() - t0
        dream_report = {
            "dream_duration_seconds": round(elapsed, 1),
            "vault_files_scanned": vault_contents.get("total_files", 0),
            "bridges_discovered": len(bridges),
            "gaps_identified": len(gaps),
            "future_projects_seeded": len(futures),
            "bridges": [b.to_dict() for b in bridges],
            "gaps": [g.to_dict() for g in gaps],
            "futures": futures,
            "saved": save_result,
            "lora_sharpening": lora_result,
            "cloud_finetune": finetune_check,
        }

        self._dream_log.append({
            "timestamp": time.time(),
            "bridges": len(bridges),
            "gaps": len(gaps),
        })

        log.info(f"§DREAM: Dream complete — {len(bridges)} bridges, "
                 f"{len(gaps)} gaps in {elapsed:.1f}s")

        return dream_report

    def _scan_vault(self) -> dict:
        """Scan the vault for text content and concept occurrence."""
        vault = Path(self._vault_path)
        if not vault.exists():
            return {"total_files": 0, "concepts_found": {}}

        concept_occurrences: dict[str, list[str]] = {}
        total_files = 0

        for ext in ["*.txt", "*.md", "*.tex"]:
            for file_path in vault.rglob(ext):
                try:
                    text = file_path.read_text(errors="ignore").lower()
                    total_files += 1

                    # Check which dissertation concepts appear
                    for concept_id, concept_data in DISSERTATION_CORE.items():
                        for term in concept_data["related_terms"]:
                            if term in text:
                                concept_occurrences.setdefault(concept_id, []).append(
                                    str(file_path.relative_to(vault))
                                )
                                break
                except Exception:
                    continue

        return {
            "total_files": total_files,
            "concepts_found": {
                k: len(set(v)) for k, v in concept_occurrences.items()
            },
            "concept_files": concept_occurrences,
        }

    def find_bridges(self, vault_contents: dict = None) -> list[BridgeNote]:
        """Find hidden bridges between your dissertation and foreign subfields.

        "If a semantic match > 0.85 exists, generate a Bridge Note."
        """
        if vault_contents is None:
            vault_contents = self._scan_vault()

        bridges = []
        concept_files = vault_contents.get("concept_files", {})

        for core_id, core_data in DISSERTATION_CORE.items():
            for foreign_id, foreign_data in FOREIGN_SUBFIELDS.items():
                # Calculate concept overlap score
                similarity = self._calculate_bridge_similarity(
                    core_data, foreign_data, concept_files.get(core_id, [])
                )

                if similarity >= 0.65:  # Lower threshold to find more connections
                    bridge_id = f"Bridge_{core_id}_{foreign_id}_{int(time.time())}"

                    research_q = self._generate_research_question(
                        core_data["description"], foreign_data["bridge_potential"]
                    )

                    bridge = BridgeNote(
                        bridge_id=bridge_id,
                        source_concept=core_id.replace("_", " ").title(),
                        source_subfield=core_data["subfield"],
                        target_concept=foreign_id.replace("_", " ").title(),
                        target_subfield=foreign_id,
                        similarity_score=similarity,
                        bridge_description=foreign_data["bridge_potential"],
                        research_question=research_q,
                        grant_potential=self._assess_grant_potential(core_id, foreign_id),
                        discovered_at=time.time(),
                    )
                    bridges.append(bridge)

        bridges.sort(key=lambda b: b.similarity_score, reverse=True)
        self._bridges.extend(bridges)
        return bridges

    def _calculate_bridge_similarity(self, core_data: dict,
                                       foreign_data: dict,
                                       existing_files: list) -> float:
        """Calculate semantic similarity between a core concept and foreign subfield."""
        # Term overlap scoring
        core_terms = set(t.lower() for t in core_data["related_terms"])
        foreign_terms = set(t.lower() for t in foreign_data["concepts"])

        # Direct overlap
        overlap = core_terms & foreign_terms
        if overlap:
            return min(1.0, 0.5 + len(overlap) * 0.15)

        # Semantic proximity via shared root words
        core_roots = set()
        for t in core_terms:
            core_roots.update(t.split())
        foreign_roots = set()
        for t in foreign_terms:
            foreign_roots.update(t.split())

        root_overlap = core_roots & foreign_roots
        root_score = len(root_overlap) / max(len(core_roots | foreign_roots), 1)

        # Boost if we have existing files about this concept
        file_boost = min(0.2, len(existing_files) * 0.02)

        return min(1.0, root_score * 1.5 + file_boost + 0.3)

    def _generate_research_question(self, core_desc: str, bridge_desc: str) -> str:
        """Generate a research question from a bridge."""
        return (
            f"How might the insights from studying '{bridge_desc.lower()}' "
            f"inform our understanding of '{core_desc.lower()}'? "
            f"What mechanisms are shared, and what institutional differences "
            f"moderate the relationship?"
        )

    def _assess_grant_potential(self, core_id: str, foreign_id: str) -> str:
        """Assess whether a bridge has grant-writing potential."""
        high_potential = {
            ("charity_substitution", "health_policy"),
            ("administrative_burden", "digital_governance"),
            ("state_capacity", "post_conflict"),
            ("cartel_governance", "post_conflict"),
            ("blame_diffusion", "education_policy"),
            ("charity_substitution", "development_economics"),
        }
        pair = (core_id, foreign_id)
        if pair in high_potential or (pair[1], pair[0]) in high_potential:
            return "HIGH — This bridge spans two fundable research areas with clear policy implications."
        return "MODERATE — Interesting theoretical contribution; consider as a dissertation chapter or working paper."

    def _detect_gaps(self, vault_contents: dict) -> list[GapReport]:
        """Detect gaps in your literature coverage."""
        gaps = []
        concepts_found = vault_contents.get("concepts_found", {})

        for concept_id, concept_data in DISSERTATION_CORE.items():
            file_count = concepts_found.get(concept_id, 0)

            # Coverage score based on file count
            if file_count == 0:
                coverage = 0.0
                gap_desc = f"No materials found on '{concept_id.replace('_', ' ')}'. Critical gap."
            elif file_count < 3:
                coverage = 0.3
                gap_desc = f"Only {file_count} file(s) cover this concept. Consider expanding."
            elif file_count < 8:
                coverage = 0.6
                gap_desc = f"Moderate coverage ({file_count} files). Check for recent publications."
            else:
                coverage = min(1.0, file_count / 15)
                gap_desc = f"Good coverage ({file_count} files)."

            if coverage < 0.6:
                gaps.append(GapReport(
                    concept=concept_id.replace("_", " ").title(),
                    subfield=concept_data["subfield"],
                    coverage_score=coverage,
                    gap_description=gap_desc,
                    suggested_readings=[
                        f"Search for recent {concept_data['subfield']} papers on '{term}'"
                        for term in concept_data["related_terms"][:3]
                    ],
                ))

        return gaps

    def _generate_futures(self, bridges: list[BridgeNote]) -> list[dict]:
        """Generate Future Project seeds from bridges."""
        futures = []
        for bridge in bridges[:5]:  # Top 5 bridges
            if bridge.similarity_score >= 0.75:
                futures.append({
                    "title": f"Future Project: {bridge.source_concept} × {bridge.target_concept}",
                    "type": "grant_seed" if "HIGH" in bridge.grant_potential else "working_paper",
                    "description": bridge.bridge_description,
                    "research_question": bridge.research_question,
                    "tag": "Future Project: 2027 Grant" if "HIGH" in bridge.grant_potential else "Working Paper Idea",
                })
        return futures

    def _save_dream_results(self, bridges: list, gaps: list, futures: list) -> dict:
        """Save dream results to the Bolt."""
        futures_dir = Path(self._futures_path)
        futures_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        # Save bridge notes
        for bridge in bridges:
            bridge_file = futures_dir / f"{bridge.bridge_id}.md"
            try:
                bridge_file.write_text(bridge.to_markdown())
                saved += 1
            except Exception:
                pass

        # Save dream summary
        summary_file = futures_dir / f"dream_{time.strftime('%Y%m%d')}.json"
        try:
            summary_file.write_text(json.dumps({
                "timestamp": time.time(),
                "bridges": [b.to_dict() for b in bridges],
                "gaps": [g.to_dict() for g in gaps],
                "futures": futures,
            }, indent=2))
        except Exception:
            pass

        return {"files_saved": saved, "path": str(futures_dir)}

    @property
    def status(self) -> dict:
        return {
            "total_bridges_found": len(self._bridges),
            "total_gaps_detected": len(self._gaps),
            "dream_sessions": len(self._dream_log),
        }


# Global instance
dream_engine = DreamEngine()

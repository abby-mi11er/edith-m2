"""
Forensic Audit Orchestrator — The Nervous System
===================================================
This is the GLUE that turns individual modules into an automated ecosystem.

When you plug in the Oyen Bolt, this script:
1. Detects new/changed PDFs on the Bolt (08:00 AM — Physical Handshake)
2. Runs Full Forensic Audit on each (10:00 AM — Forensic Deconstruction)
3. Auto-generates Method Lab crash courses (10:00 AM — Methodology Sidebar)
4. Places each paper in the Theoretical Atlas (12:00 PM — Theory Vault)
5. Pre-highlights text based on your research interests
6. Generates Sovereign Notes in LaTeX + Markdown (12:00 PM — Notion Replacement)
7. Pins critical results into M4 Unified Memory (always-on)

Chain:  Bolt Connect → Scan → Audit → Crash Course → Atlas → Notes → Pin

This kills the drudgery. The moment you plug in, Winnie has already
deconstructed your readings, tutored you on the methodology, and placed
every paper in your theoretical universe.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.forensic_audit")


# ═══════════════════════════════════════════════════════════════════
# Research Interest Profile — Pre-Highlighting Calibration
# ═══════════════════════════════════════════════════════════════════

RESEARCH_INTERESTS = {
    "primary_topics": [
        "state capacity", "non-state actors", "charitable organizations",
        "welfare provision", "administrative burden", "blame diffusion",
        "federalism", "devolution", "privatization",
    ],
    "geographic_focus": [
        "potter county", "lubbock", "texas", "mexico", "michoacán",
        "united states", "latin america",
    ],
    "methodological_interests": [
        "instrumental variables", "regression discontinuity",
        "difference-in-differences", "qualitative comparative analysis",
        "mixed methods", "case study", "process tracing",
    ],
    "theoretical_anchors": [
        "principal-agent", "administrative burden", "policy feedback",
        "state capacity", "welfare state", "street-level bureaucracy",
        "institutional choice", "democratic erosion",
    ],
    "key_authors": [
        "mettler", "aldrich", "moynihan", "herd", "pierson",
        "skocpol", "lipsky", "ostrom", "tilly", "mann",
        "esping-andersen", "levitsky", "fukuyama",
    ],
}


@dataclass
class HighlightedPassage:
    """A passage pre-highlighted based on research relevance."""
    text: str
    start_pos: int
    end_pos: int
    relevance_score: float
    reason: str  # Why this passage was highlighted
    category: str  # "topic", "method", "theory", "author", "data"
    color: str  # Highlight color tier

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200],
            "score": round(self.relevance_score, 2),
            "reason": self.reason,
            "category": self.category,
            "color": self.color,
        }


@dataclass
class SovereignNote:
    """A note generated from forensic analysis, ready for the Theory Vault."""
    title: str
    content_markdown: str
    content_latex: str
    tags: list[str]
    linked_papers: list[str]
    linked_chapters: list[str]
    theory_bridge: str  # Which dissertation chapter/argument this connects to
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "markdown": self.content_markdown,
            "latex": self.content_latex,
            "tags": self.tags,
            "links": self.linked_papers,
            "chapters": self.linked_chapters,
            "bridge": self.theory_bridge,
        }


# ═══════════════════════════════════════════════════════════════════
# PDF Text Extraction — Works with or without external libraries
# ═══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file.

    Tries multiple backends in order of preference:
    1. PyMuPDF (fitz) — fastest, best quality
    2. pdfplumber — good with tables
    3. Basic fallback — reads what it can
    """
    # Try PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"PyMuPDF failed: {e}")

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"pdfplumber failed: {e}")

    # Fallback: try to read raw bytes for any text content
    try:
        raw = Path(pdf_path).read_bytes()
        # Extract ASCII-decodable strings
        text_chunks = re.findall(rb'[\x20-\x7E]{20,}', raw)
        return "\n".join(chunk.decode("ascii", errors="ignore") for chunk in text_chunks)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════
# The Forensic Audit Orchestrator — The Main Engine
# ═══════════════════════════════════════════════════════════════════

class ForensicAuditOrchestrator:
    """The nervous system that connects all Forensics Lab modules.

    Usage:
        orchestrator = ForensicAuditOrchestrator()

        # On Bolt connect — auto-process everything new
        results = orchestrator.auto_audit_on_connect()

        # Single paper audit
        result = orchestrator.full_pipeline("path/to/paper.pdf")

        # Overnight dreaming — batch process while you sleep
        results = orchestrator.dream_cycle()
    """

    def __init__(self, bolt_path: str = "", data_root: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._vault_path = os.path.join(self._bolt_path, "VAULT")

        # Track what we've already processed
        self._processed_index = self._load_processed_index()

        # Lazy-import the module engines
        self._deconstructor = None
        self._method_lab = None
        self._lit_locator = None
        self._memory_pinner = None

    def _get_deconstructor(self):
        if not self._deconstructor:
            from server.paper_deconstructor import PaperDeconstructor
            self._deconstructor = PaperDeconstructor(self._bolt_path)
        return self._deconstructor

    def _get_method_lab(self):
        if not self._method_lab:
            from server.method_lab import MethodLab
            self._method_lab = MethodLab()
        return self._method_lab

    def _get_lit_locator(self):
        if not self._lit_locator:
            from server.lit_locator import LitLocator
            self._lit_locator = LitLocator()
        return self._lit_locator

    def _get_memory_pinner(self):
        if not self._memory_pinner:
            from server.memory_pinning import MemoryPinner
            self._memory_pinner = MemoryPinner(self._bolt_path)
        return self._memory_pinner

    # ─── The Full Pipeline ──────────────────────────────────────────

    def full_pipeline(self, pdf_path: str, title: str = "",
                       author: str = "", year: int = 0) -> dict:
        """Run the COMPLETE forensic pipeline on a single paper.

        This is the "Winnie, perform a Full Forensic Audit" command.

        Chain: Extract → Audit → Highlight → Crash Course → Atlas → Notes → Pin
        """
        t0 = time.time()
        result = {"path": pdf_path, "stages": {}}

        # Stage 1: Extract text
        log.info(f"§FORENSIC: Stage 1 — Extracting text from {Path(pdf_path).name}")
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 100:
            return {"error": "Could not extract sufficient text from PDF",
                    "path": pdf_path}
        result["stages"]["extraction"] = {
            "chars": len(text),
            "words": len(text.split()),
        }

        # Stage 2: Full Forensic Audit (deconstructor)
        log.info("§FORENSIC: Stage 2 — Deconstructing into molecular parts")
        deconstructor = self._get_deconstructor()
        audit = deconstructor.full_forensic_audit(
            text, title=title, author=author, year=year
        )
        result["stages"]["audit"] = {
            "citations": audit["citations"]["total"],
            "citations_in_vault": audit["citations"]["in_vault"],
            "datasets": audit["datasets"]["total"],
            "datasets_on_bolt": audit["datasets"]["on_bolt"],
            "primary_estimator": (
                audit["estimators"]["primary"]["label"]
                if audit["estimators"]["primary"] else "none"
            ),
            "theory": audit["theory"]["primary_framework"],
        }
        result["audit"] = audit

        # Stage 3: Pre-Highlight based on research interests
        log.info("§FORENSIC: Stage 3 — Pre-highlighting for research relevance")
        highlights = self._pre_highlight(text)
        result["stages"]["highlights"] = {
            "passages_highlighted": len(highlights),
            "top_passages": [h.to_dict() for h in highlights[:5]],
        }
        result["highlights"] = [h.to_dict() for h in highlights]

        # Stage 4: Method Lab crash course (if estimator identified)
        log.info("§FORENSIC: Stage 4 — Generating methodology crash course")
        estimators = audit["estimators"]["all"]
        crash_courses = []
        if estimators:
            lab = self._get_method_lab()
            for est in estimators[:2]:  # Top 2 estimators
                course = lab.generate_short_course(
                    est["id"],
                    paper_context=text[:500],
                    user_topic="state capacity and charitable organizations",
                )
                if "error" not in course:
                    crash_courses.append({
                        "estimator": est["label"],
                        "course": course,
                    })
        result["stages"]["method_lab"] = {
            "courses_generated": len(crash_courses),
            "estimators": [c["estimator"] for c in crash_courses],
        }
        result["crash_courses"] = crash_courses

        # Stage 5: Theoretical Atlas placement
        log.info("§FORENSIC: Stage 5 — Placing in Theoretical Atlas")
        locator = self._get_lit_locator()
        location = locator.locate_paper(
            text,
            title=title or audit["paper"]["title"],
            citations=audit["citations"]["items"],
            user_dissertation_topic="state capacity, charitable organizations, and welfare provision",
        )
        result["stages"]["atlas"] = {
            "primary_tradition": location["ancestry"]["primary_tradition"]["theory"]
            if location["ancestry"]["primary_tradition"] else "unplaced",
            "traditions": location["ancestry"]["theoretical_density"],
            "bridges": len(location["bridges"]["theory_bridges"]),
            "dissertation_fit": location["dissertation_fit"]["fit_level"],
        }
        result["atlas_location"] = location

        # Stage 6: Generate Sovereign Notes
        log.info("§FORENSIC: Stage 6 — Generating Sovereign Notes")
        notes = self._generate_sovereign_notes(audit, location, highlights)
        result["stages"]["notes"] = {
            "notes_generated": len(notes),
        }
        result["sovereign_notes"] = [n.to_dict() for n in notes]

        # Stage 7: Save to Theory Vault on Bolt
        log.info("§FORENSIC: Stage 7 — Saving to Theory Vault")
        save_result = self._save_to_vault(result, pdf_path)
        result["stages"]["vault"] = save_result

        # Mark as processed
        file_hash = hashlib.sha256(Path(pdf_path).name.encode()).hexdigest()[:16]
        self._processed_index[file_hash] = {
            "path": pdf_path,
            "processed_at": time.time(),
            "title": title or audit["paper"]["title"],
        }
        self._save_processed_index()

        elapsed = time.time() - t0
        result["total_elapsed_seconds"] = round(elapsed, 2)
        result["summary"] = self._generate_greeting(audit, location, crash_courses)

        log.info(f"§FORENSIC: Pipeline complete in {elapsed:.1f}s — "
                 f"{audit['citations']['total']} citations, "
                 f"{len(crash_courses)} crash courses, "
                 f"{len(notes)} sovereign notes")

        return result

    # ─── Pre-Highlighting ──────────────────────────────────────────

    def _pre_highlight(self, text: str) -> list[HighlightedPassage]:
        """Pre-highlight text based on research interests.

        "Winnie has already Pre-Highlighted the text based on your
        research interests (Non-state welfare, federalism)."
        """
        highlights = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for i, sentence in enumerate(sentences):
            if len(sentence) < 20:
                continue

            sentence_lower = sentence.lower()
            best_score = 0
            best_reason = ""
            best_category = ""

            # Check primary topics
            for topic in RESEARCH_INTERESTS["primary_topics"]:
                if topic in sentence_lower:
                    score = 0.9
                    if score > best_score:
                        best_score = score
                        best_reason = f"Core topic: {topic}"
                        best_category = "topic"

            # Check key authors
            for author in RESEARCH_INTERESTS["key_authors"]:
                if author in sentence_lower:
                    score = 0.85
                    if score > best_score:
                        best_score = score
                        best_reason = f"Key author: {author}"
                        best_category = "author"

            # Check methodological terms
            for method in RESEARCH_INTERESTS["methodological_interests"]:
                if method in sentence_lower:
                    score = 0.8
                    if score > best_score:
                        best_score = score
                        best_reason = f"Method: {method}"
                        best_category = "method"

            # Check theoretical anchors
            for theory in RESEARCH_INTERESTS["theoretical_anchors"]:
                if theory in sentence_lower:
                    score = 0.75
                    if score > best_score:
                        best_score = score
                        best_reason = f"Theory: {theory}"
                        best_category = "theory"

            # Check geographic focus
            for geo in RESEARCH_INTERESTS["geographic_focus"]:
                if geo in sentence_lower:
                    score = 0.7
                    if score > best_score:
                        best_score = score
                        best_reason = f"Geography: {geo}"
                        best_category = "geography"

            if best_score >= 0.7:
                color = (
                    "#FFD700" if best_score >= 0.9 else   # Gold — core topic
                    "#87CEEB" if best_score >= 0.85 else   # Sky blue — key author
                    "#90EE90" if best_score >= 0.8 else    # Light green — method
                    "#DDA0DD"                               # Plum — theory/geo
                )
                highlights.append(HighlightedPassage(
                    text=sentence.strip(),
                    start_pos=text.find(sentence),
                    end_pos=text.find(sentence) + len(sentence),
                    relevance_score=best_score,
                    reason=best_reason,
                    category=best_category,
                    color=color,
                ))

        # Sort by relevance
        highlights.sort(key=lambda h: h.relevance_score, reverse=True)
        return highlights[:30]  # Top 30 passages

    # ─── Sovereign Note Generation ─────────────────────────────────

    def _generate_sovereign_notes(self, audit: dict, location: dict,
                                    highlights: list[HighlightedPassage]) -> list[SovereignNote]:
        """Generate Sovereign Notes — formatted for the Theory Vault.

        "These notes are formatted in LaTeX and Markdown, then instantly
        indexed into your permanent Theory Vault on the Bolt."
        """
        notes = []
        paper = audit.get("paper", {})
        title = paper.get("title", "Unknown Paper")
        author = paper.get("author", "")
        year = paper.get("year", "")

        # Note 1: Paper Summary Note
        primary_est = audit["estimators"]["primary"]
        est_label = primary_est["label"] if primary_est else "Not identified"
        theory = audit["theory"]["primary_framework"]
        datasets = [d["name"] for d in audit["datasets"]["items"][:3]]
        fit = location.get("dissertation_fit", {})

        summary_md = (
            f"# {title}\n\n"
            f"**Author**: {author}  \n"
            f"**Year**: {year}  \n"
            f"**Primary Method**: {est_label}  \n"
            f"**Theoretical Framework**: {theory}  \n"
            f"**Datasets**: {', '.join(datasets) if datasets else 'Not identified'}  \n\n"
            f"## Dissertation Relevance\n"
            f"**Fit Level**: {fit.get('fit_level', 'unknown')}  \n"
            f"{fit.get('recommendation', '')}\n\n"
            f"## Key Passages\n"
        )
        for h in highlights[:5]:
            summary_md += f"- *{h.reason}*: \"{h.text[:150]}...\"\n"

        summary_latex = (
            f"\\section{{{title}}}\n"
            f"\\subsection{{Paper Details}}\n"
            f"\\textbf{{Author}}: {author} \\\\\n"
            f"\\textbf{{Year}}: {year} \\\\\n"
            f"\\textbf{{Method}}: {est_label} \\\\\n"
            f"\\textbf{{Theory}}: {theory} \\\\\n\n"
            f"\\subsection{{Dissertation Relevance}}\n"
            f"Fit Level: {fit.get('fit_level', 'unknown')}. "
            f"{fit.get('recommendation', '')}\n"
        )

        # Theory bridge tag
        bridges = location.get("bridges", {}).get("theory_bridges", [])
        bridge_tag = ""
        if bridges:
            bridge_tag = (
                f"Theory Bridge: {bridges[0].get('from', '')} → "
                f"{bridges[0].get('to', '')}"
            )

        # Determine chapter links
        chapter_links = []
        diss_traditions = fit.get("relevant_traditions", [])
        for t in diss_traditions:
            theory_name = t.get("theory", "").lower()
            if "principal" in theory_name or "agent" in theory_name:
                chapter_links.append("Chapter 2: Theoretical Framework")
            if "administrative" in theory_name or "burden" in theory_name:
                chapter_links.append("Chapter 4: Administrative Burden Analysis")
            if "state capacity" in theory_name:
                chapter_links.append("Chapter 3: State Capacity in Potter County")
            if "welfare" in theory_name:
                chapter_links.append("Chapter 5: Welfare Provision Models")
            if "policy feedback" in theory_name:
                chapter_links.append("Chapter 6: Policy Feedback Loops")

        notes.append(SovereignNote(
            title=f"Forensic Note: {title}",
            content_markdown=summary_md,
            content_latex=summary_latex,
            tags=["forensic_audit", theory.lower().replace(" ", "_"),
                  est_label.split("(")[0].strip().lower().replace(" ", "_")],
            linked_papers=[c["author"] + f" ({c['year']})"
                           for c in audit["citations"]["items"][:5]],
            linked_chapters=list(set(chapter_links)),
            theory_bridge=bridge_tag,
            timestamp=time.time(),
        ))

        # Note 2: Methodology Note (if estimator found)
        if primary_est:
            method_md = (
                f"# Methodology Note: {est_label}\n\n"
                f"**Paper**: {title}\n\n"
                f"## Method Details\n"
                f"- **Family**: {primary_est.get('family', '')}\n"
                f"- **Confidence**: {primary_est.get('confidence', 0):.0%}\n\n"
                f"## Assumptions to Check\n"
            )
            for assumption in primary_est.get("assumptions", []):
                method_md += f"- [ ] {assumption}\n"

            # Variables
            variables = audit.get("variables", {})
            if variables.get("dependent"):
                method_md += f"\n## Variables\n"
                method_md += f"- **DV**: {', '.join(variables['dependent'][:3])}\n"
            if variables.get("independent"):
                method_md += f"- **IV**: {', '.join(variables['independent'][:3])}\n"
            if variables.get("instruments"):
                method_md += f"- **Instruments**: {', '.join(variables['instruments'][:3])}\n"

            notes.append(SovereignNote(
                title=f"Method Note: {est_label} in {title[:50]}",
                content_markdown=method_md,
                content_latex="",  # Method notes are primarily markdown
                tags=["methodology", primary_est["id"]],
                linked_papers=[],
                linked_chapters=chapter_links,
                theory_bridge="",
                timestamp=time.time(),
            ))

        return notes

    # ─── Greeting Generation ──────────────────────────────────────

    def _generate_greeting(self, audit: dict, location: dict,
                            crash_courses: list) -> str:
        """Generate the Winnie morning greeting.

        "Good morning. I've ingested the four readings for today.
        I've also pulled the raw replication data for the Mettler paper."
        """
        paper = audit["paper"]
        title = paper.get("title", "this paper")[:60]
        citations = audit["citations"]["total"]
        vault_hits = audit["citations"]["in_vault"]
        datasets = [d["full_name"] for d in audit["datasets"]["items"] if d.get("known")]
        est = audit["estimators"]["primary"]
        fit = location.get("dissertation_fit", {}).get("fit_level", "")

        greeting = f"📋 **Forensic Audit Complete: \"{title}\"**\n\n"

        # Citations summary
        greeting += f"I found **{citations} citations**"
        if vault_hits:
            greeting += f" — {vault_hits} are already in your Vault"
        greeting += ".\n\n"

        # Dataset availability
        if datasets:
            greeting += f"📊 **Datasets detected**: {', '.join(datasets[:3])}. "
            bolt_available = sum(1 for d in audit["datasets"]["items"] if d.get("on_bolt"))
            if bolt_available:
                greeting += f"{bolt_available} are already on your Bolt — ready for replication.\n\n"
            else:
                greeting += "None currently on the Bolt.\n\n"

        # Method crash course
        if crash_courses:
            est_name = crash_courses[0]["estimator"]
            greeting += (
                f"🧪 **Methodology Lab**: The author uses **{est_name}**. "
                f"I've prepared a crash course with Stata and R code you can run right now.\n\n"
            )

        # Dissertation fit
        if fit == "essential":
            greeting += (
                "🎯 **Dissertation Fit: ESSENTIAL** — This paper directly engages "
                "with your theoretical framework. I've linked it to your chapter outline.\n\n"
            )
        elif fit == "relevant":
            greeting += (
                "📌 **Dissertation Fit: RELEVANT** — Useful for your literature review. "
                "I've generated a Theory Bridge note.\n\n"
            )

        # Theory bridges
        bridges = location.get("bridges", {}).get("theory_bridges", [])
        if bridges:
            b = bridges[0]
            greeting += (
                f"🌉 **Theory Bridge Alert**: This paper connects "
                f"**{b.get('from', '')}** with **{b.get('to', '')}** — "
                f"exactly the kind of bridge your dissertation needs.\n"
            )

        return greeting

    # ─── Auto-Audit on Bolt Connect ───────────────────────────────

    def auto_audit_on_connect(self, scan_dirs: list[str] = None) -> dict:
        """Automatically audit all new PDFs when the Bolt is connected.

        "You slide the Bolt into your M4. Winnie has spent the night
        pulling every PDF listed on your Week 7 syllabus."
        """
        bolt = Path(self._bolt_path)
        if not bolt.exists():
            return {"connected": False, "message": "Bolt SSD not detected"}

        # Default scan directories
        if not scan_dirs:
            scan_dirs = [
                "VAULT/READINGS",
                "VAULT/SYLLABI",
                "VAULT/NEW_PAPERS",
                "VAULT/INBOX",
                "VAULT/KEY_PAPERS",
            ]

        new_pdfs = []
        for scan_dir in scan_dirs:
            dir_path = bolt / scan_dir
            if not dir_path.exists():
                continue
            for pdf in dir_path.rglob("*.pdf"):
                file_hash = hashlib.sha256(pdf.name.encode()).hexdigest()[:16]
                if file_hash not in self._processed_index:
                    new_pdfs.append(str(pdf))

        if not new_pdfs:
            return {
                "connected": True,
                "new_papers": 0,
                "message": "All papers already processed. Nothing new to audit.",
            }

        # Process each new PDF
        results = []
        for pdf_path in new_pdfs[:10]:  # Cap at 10 per session
            try:
                result = self.full_pipeline(pdf_path)
                results.append({
                    "path": pdf_path,
                    "title": result.get("audit", {}).get("paper", {}).get("title", ""),
                    "success": "error" not in result,
                    "summary": result.get("summary", ""),
                })
            except Exception as e:
                results.append({
                    "path": pdf_path,
                    "success": False,
                    "error": str(e),
                })

        # Pin critical results into memory
        try:
            pinner = self._get_memory_pinner()
            pinner.auto_pin_on_connect()
        except Exception:
            pass

        successful = sum(1 for r in results if r.get("success"))
        return {
            "connected": True,
            "new_papers": len(new_pdfs),
            "processed": len(results),
            "successful": successful,
            "results": results,
            "greeting": (
                f"Good morning. I've ingested {successful} new readings. "
                f"Forensic audits complete — crash courses and theory bridges are ready."
            ),
        }

    # ─── Dream Cycle — Overnight Batch Processing ─────────────────

    def dream_cycle(self) -> dict:
        """The overnight "Dream" cycle.

        "While you sleep, Winnie is 'Dreaming' — finding hidden
        theoretical bridges you'll wake up to tomorrow."

        Batch processes everything unprocessed and generates a
        morning briefing.
        """
        result = self.auto_audit_on_connect()

        # Also check for theory bridges across ALL processed papers
        bridge_discoveries = []
        processed_papers = list(self._processed_index.values())

        # Look for new connections between recently processed papers
        if len(processed_papers) >= 2:
            bridge_discoveries.append({
                "type": "cross_paper_bridge",
                "message": (
                    f"Found potential theoretical bridges across "
                    f"{len(processed_papers)} papers in your vault."
                ),
            })

        result["dream_discoveries"] = bridge_discoveries

        # Save morning briefing
        briefing_path = Path(self._data_root or ".") / "VAULT" / "BRIEFINGS"
        briefing_path.mkdir(parents=True, exist_ok=True)
        briefing_file = briefing_path / f"morning_{time.strftime('%Y%m%d')}.json"
        try:
            briefing_file.write_text(json.dumps(result, indent=2, default=str))
        except Exception:
            pass

        return result

    # ─── Vault I/O ────────────────────────────────────────────────

    def _save_to_vault(self, result: dict, pdf_path: str) -> dict:
        """Save forensic results to the Theory Vault on the Bolt."""
        vault_dir = Path(self._vault_path) / "FORENSICS"
        vault_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(pdf_path).stem[:50]
        ts = time.strftime("%Y%m%d_%H%M")

        # Save full audit
        audit_file = vault_dir / f"{stem}_audit_{ts}.json"
        try:
            # Strip non-serializable items
            clean = {k: v for k, v in result.items()
                     if k not in {"crash_courses"}}  # courses are large
            audit_file.write_text(json.dumps(clean, indent=2, default=str))
        except Exception as e:
            return {"saved": False, "error": str(e)}

        # Save sovereign notes as markdown
        notes_dir = vault_dir / "NOTES"
        notes_dir.mkdir(parents=True, exist_ok=True)
        notes_saved = 0
        for note in result.get("sovereign_notes", []):
            note_file = notes_dir / f"{stem}_{notes_saved}.md"
            try:
                note_file.write_text(note.get("markdown", ""))
                notes_saved += 1
            except Exception:
                pass

        return {
            "saved": True,
            "audit_path": str(audit_file),
            "notes_dir": str(notes_dir),
            "notes_saved": notes_saved,
        }

    def _load_processed_index(self) -> dict:
        """Load the index of already-processed papers."""
        index_path = Path(self._data_root or ".") / "VAULT" / "FORENSICS" / "_processed.json"
        if index_path.exists():
            try:
                return json.loads(index_path.read_text())
            except Exception:
                pass
        return {}

    def _save_processed_index(self):
        """Persist the processed index."""
        index_dir = Path(self._data_root or ".") / "VAULT" / "FORENSICS"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "_processed.json"
        try:
            index_path.write_text(json.dumps(self._processed_index, indent=2))
        except Exception:
            pass

    @property
    def status(self) -> dict:
        return {
            "bolt_connected": Path(self._bolt_path).exists(),
            "papers_processed": len(self._processed_index),
            "vault_path": self._vault_path,
        }


# ═══════════════════════════════════════════════════════════════════
# Quick API — For direct use from the chat interface
# ═══════════════════════════════════════════════════════════════════

# Global instance
forensic_orchestrator = ForensicAuditOrchestrator()


def forensic_audit(pdf_path: str, **kwargs) -> dict:
    """Quick API: run a full forensic audit on a PDF."""
    return forensic_orchestrator.full_pipeline(pdf_path, **kwargs)


def morning_briefing() -> dict:
    """Quick API: get the morning briefing from overnight processing."""
    return forensic_orchestrator.auto_audit_on_connect()


def dream() -> dict:
    """Quick API: run the overnight dream cycle."""
    return forensic_orchestrator.dream_cycle()

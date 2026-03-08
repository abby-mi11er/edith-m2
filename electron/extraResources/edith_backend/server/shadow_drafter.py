"""
Shadow Drafter — Speculative Background Drafting
===================================================
Force Multiplier 1: You never stare at a blank page again.

As you highlight and read in the morning, Winnie synthesizes those
highlights into a LaTeX/Markdown draft in the background. When you
open your summary at 2 PM, it's already 500 words of structured,
cited first-pass writing. You move from "Writing" to "Editing."

Architecture:
    Highlights → Background Queue → Synthesis Engine →
    Shadow Draft (LaTeX + MD) → Theory Vault on Bolt
"""

import json
import logging
import os
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.shadow_drafter")


@dataclass
class Highlight:
    """A highlight captured during reading."""
    text: str
    source_title: str
    source_author: str = ""
    page: int = 0
    category: str = ""  # "argument", "method", "data", "theory", "critique"
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text[:300],
            "source": f"{self.source_author} — {self.source_title}",
            "category": self.category,
            "page": self.page,
        }


@dataclass
class ShadowDraft:
    """A draft generated from accumulated highlights."""
    title: str
    content_markdown: str
    content_latex: str
    word_count: int
    sources_used: list[str]
    highlights_used: int
    generated_at: float
    status: str = "draft"  # "draft", "reviewed", "finalized"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "words": self.word_count,
            "sources": self.sources_used,
            "highlights": self.highlights_used,
            "status": self.status,
        }


# ═══════════════════════════════════════════════════════════════════
# Highlight Classifier — What kind of highlight is this?
# ═══════════════════════════════════════════════════════════════════

CATEGORY_SIGNALS = {
    "argument": ["argue", "claim", "contend", "suggest", "propose", "assert",
                  "demonstrate", "show that", "conclude", "find that"],
    "method": ["method", "regression", "estimate", "sample", "variable",
               "measure", "operationalize", "survey", "experiment", "design"],
    "data": ["data", "dataset", "N =", "percent", "table", "figure",
             "respondent", "observation", "census", "panel"],
    "theory": ["theory", "framework", "model", "principal", "agent",
               "institution", "mechanism", "causal", "hypothesis"],
    "critique": ["however", "limitation", "challenge", "problem", "gap",
                 "weakness", "fail", "overlook", "neglect", "critique"],
    "definition": ["define", "refers to", "is defined as", "concept of",
                   "meaning of", "understood as"],
}


def classify_highlight(text: str) -> str:
    """Classify a highlighted passage by its function in the paper."""
    text_lower = text.lower()
    scores = {}
    for category, signals in CATEGORY_SIGNALS.items():
        scores[category] = sum(1 for s in signals if s in text_lower)
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


# ═══════════════════════════════════════════════════════════════════
# The Shadow Drafter Engine
# ═══════════════════════════════════════════════════════════════════

class ShadowDrafter:
    """Background draft synthesizer.

    Watches your highlights accumulate and builds a structured
    summary draft in real-time. When you're ready to write at 2 PM,
    the draft is already waiting.

    Usage:
        drafter = ShadowDrafter()
        drafter.add_highlight("State capacity declines when...", "Mettler (2011)")
        drafter.add_highlight("The IV regression shows...", "Mettler (2011)")
        draft = drafter.generate_draft("Week 7 Summary")
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._highlights: list[Highlight] = []
        self._drafts: list[ShadowDraft] = []
        self._auto_draft_threshold = 5  # Generate after 5 highlights
        self._background_thread: Optional[threading.Thread] = None

    def add_highlight(self, text: str, source_title: str,
                       source_author: str = "", page: int = 0,
                       category: str = "") -> dict:
        """Add a highlight from reading. Auto-classifies if no category given."""
        if not category:
            category = classify_highlight(text)

        highlight = Highlight(
            text=text.strip(),
            source_title=source_title,
            source_author=source_author,
            page=page,
            category=category,
            timestamp=time.time(),
        )
        self._highlights.append(highlight)

        # Auto-generate draft if threshold reached
        status = {"added": True, "category": category, "total": len(self._highlights)}
        if len(self._highlights) >= self._auto_draft_threshold:
            if not self._background_thread or not self._background_thread.is_alive():
                status["auto_draft"] = "queued"

        return status

    def generate_draft(self, title: str = "", topic: str = "") -> ShadowDraft:
        """Generate a structured draft from accumulated highlights.

        This is the core "Ghost-Writer" logic. It takes your highlights,
        groups them by category, and synthesizes a coherent first pass.
        """
        if not self._highlights:
            return ShadowDraft(
                title=title or "Empty Draft",
                content_markdown="*No highlights captured yet.*",
                content_latex="\\textit{No highlights captured yet.}",
                word_count=0, sources_used=[], highlights_used=0,
                generated_at=time.time(),
            )

        # Group highlights by category
        by_category: dict[str, list[Highlight]] = {}
        for h in self._highlights:
            by_category.setdefault(h.category, []).append(h)

        # Collect unique sources
        sources = list(set(
            f"{h.source_author} — {h.source_title}" if h.source_author
            else h.source_title
            for h in self._highlights
        ))

        # Build the markdown draft
        draft_title = title or f"Reading Summary: {time.strftime('%B %d, %Y')}"
        if topic:
            draft_title += f" — {topic}"

        md = f"# {draft_title}\n\n"
        md += f"*Auto-generated from {len(self._highlights)} highlights across {len(sources)} source(s).*\n\n"
        md += "---\n\n"

        # Section order matches academic structure
        section_order = [
            ("theory", "Theoretical Framework"),
            ("argument", "Key Arguments"),
            ("method", "Methodology"),
            ("data", "Data & Evidence"),
            ("definition", "Key Definitions"),
            ("critique", "Limitations & Critiques"),
            ("general", "Additional Notes"),
        ]

        for category, section_title in section_order:
            highlights = by_category.get(category, [])
            if not highlights:
                continue

            md += f"## {section_title}\n\n"
            for h in highlights:
                citation = f"({h.source_author}, p. {h.page})" if h.source_author and h.page else f"({h.source_title})"
                md += f"- {h.text} {citation}\n\n"

        # Synthesis paragraph
        md += "## Synthesis\n\n"
        theory_highlights = by_category.get("theory", [])
        argument_highlights = by_category.get("argument", [])

        if theory_highlights and argument_highlights:
            md += (
                f"The readings engage primarily with "
                f"**{theory_highlights[0].text[:80]}** "
                f"({theory_highlights[0].source_title}). "
                f"The central argument is that "
                f"**{argument_highlights[0].text[:100]}** "
                f"({argument_highlights[0].source_title}). "
            )
            critique_highlights = by_category.get("critique", [])
            if critique_highlights:
                md += (
                    f"However, a key limitation is that "
                    f"**{critique_highlights[0].text[:80]}** "
                    f"({critique_highlights[0].source_title})."
                )
            md += "\n\n"

        # Sources section
        md += "## Sources\n\n"
        for s in sources:
            md += f"- {s}\n"

        # Build LaTeX version
        latex = self._markdown_to_latex(md, draft_title, sources)

        word_count = len(md.split())
        draft = ShadowDraft(
            title=draft_title,
            content_markdown=md,
            content_latex=latex,
            word_count=word_count,
            sources_used=sources,
            highlights_used=len(self._highlights),
            generated_at=time.time(),
        )

        self._drafts.append(draft)
        return draft

    def _markdown_to_latex(self, md: str, title: str, sources: list) -> str:
        """Convert the draft to LaTeX format."""
        latex = (
            "\\documentclass[12pt]{article}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\usepackage{setspace}\n"
            "\\doublespacing\n\n"
            f"\\title{{{title}}}\n"
            "\\author{Your Name}\n"
            f"\\date{{{time.strftime('%B %d, %Y')}}}\n\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
        )

        # Convert sections
        lines = md.split("\n")
        for line in lines:
            if line.startswith("## "):
                section = line[3:].strip()
                latex += f"\n\\section{{{section}}}\n\n"
            elif line.startswith("# "):
                continue  # Title already in \title
            elif line.startswith("- "):
                content = line[2:].strip()
                # Convert markdown bold to LaTeX bold
                content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', content)
                latex += f"{content}\n\n"
            elif line.startswith("*") and line.endswith("*"):
                content = line.strip("*")
                latex += f"\\textit{{{content}}}\n\n"
            elif line.strip() and not line.startswith("---"):
                content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', line)
                latex += f"{content}\n\n"

        latex += "\\end{document}\n"
        return latex

    def save_draft(self, draft: ShadowDraft = None) -> dict:
        """Save the latest draft to the Bolt."""
        if not draft and self._drafts:
            draft = self._drafts[-1]
        if not draft:
            return {"saved": False, "error": "No draft to save"}

        save_dir = Path(self._bolt_path) / "VAULT" / "DRAFTS"
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M")
        stem = re.sub(r'[^\w\s-]', '', draft.title)[:40].strip().replace(" ", "_")

        # Save markdown
        md_path = save_dir / f"{stem}_{ts}.md"
        md_path.write_text(draft.content_markdown)

        # Save LaTeX
        tex_path = save_dir / f"{stem}_{ts}.tex"
        tex_path.write_text(draft.content_latex)

        return {
            "saved": True,
            "markdown": str(md_path),
            "latex": str(tex_path),
            "words": draft.word_count,
        }

    def ring_summarize(self, docs: list = None, focus: str = "") -> dict:
        """Generate a ring summary from provided documents.

        This creates a quick summary from a list of docs/text snippets,
        without requiring the full highlight → draft workflow.
        """
        docs = docs or []
        items = []
        for doc in docs:
            text = doc if isinstance(doc, str) else doc.get("text", doc.get("snippet", ""))
            if text:
                items.append(text.strip())

        if not items:
            return {"summary": f"Ring summary for: {focus or 'selected documents'}", "items": 0}

        # Build a concise summary from all provided texts
        combined = "\n\n".join(items)
        word_count = len(combined.split())

        # Group into a structured summary
        summary_md = f"# Ring Summary{f': {focus}' if focus else ''}\n\n"
        summary_md += f"*Synthesized from {len(items)} document(s), {word_count} words total.*\n\n---\n\n"

        for i, item in enumerate(items, 1):
            # Truncate very long passages for the summary
            preview = item[:500] + ("..." if len(item) > 500 else "")
            summary_md += f"### Source {i}\n\n{preview}\n\n"

        return {
            "summary": summary_md,
            "items": len(items),
            "word_count": word_count,
            "focus": focus,
        }

    def clear_highlights(self):
        """Clear highlights for a fresh session."""
        self._highlights = []

    @property
    def status(self) -> dict:
        return {
            "highlights": len(self._highlights),
            "drafts_generated": len(self._drafts),
            "categories": {
                cat: sum(1 for h in self._highlights if h.category == cat)
                for cat in set(h.category for h in self._highlights)
            } if self._highlights else {},
            "latest_draft": self._drafts[-1].to_dict() if self._drafts else None,
        }


# Global instance
shadow_drafter = ShadowDrafter()

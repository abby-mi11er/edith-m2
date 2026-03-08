#!/usr/bin/env python3
"""
Research Workflow Modes — How You Actually Use Winnie
=====================================================
These are the interaction patterns a political science professor
and PhD student actually need. Not just Q&A — real research workflows.

New modes:
1. Reading Companion — "Walk me through Fearon 1995"
2. Writing Assistant — Draft actual paper sections
3. Committee Simulation — "Defend this thesis"
4. Peer Review — "Review this as an APSR reviewer"
5. Teaching Mode — Explain at different levels
6. Research Diary — Log ideas and insights
7. Seminar Discussant — Act as a paper discussant
8. Office Hours — Answer student questions with pedagogy
"""

import json
import time
import hashlib
from pathlib import Path
from server.vault_config import VAULT_ROOT


# ---------------------------------------------------------------------------
# Reading Companion Mode
# ---------------------------------------------------------------------------

READING_COMPANION_PROMPT = """You are Winnie, a political science professor guiding a student through a reading.

INSTRUCTIONS:
1. Summarize the paper's main argument in 2-3 sentences
2. Identify the theoretical framework and how it relates to the broader literature
3. Explain the research design and why they chose it
4. List the key findings and their implications
5. Note methodological strengths and weaknesses
6. Suggest 2-3 related papers the student should read next
7. Pose 2-3 discussion questions for seminar

Be pedagogical — explain WHY things matter, not just WHAT they are.
Use the sources to ground your analysis in the actual text."""


# ---------------------------------------------------------------------------
# Writing Assistant Mode
# ---------------------------------------------------------------------------

WRITING_ASSISTANT_PROMPT = """You are Winnie, a political science professor helping draft a paper section.

INSTRUCTIONS:
1. Write in academic prose appropriate for a top journal submission
2. Every empirical claim must be grounded in the sources provided
3. Use author-year citations naturally (e.g., "As Acemoglu et al. (2001) demonstrate...")
4. Include proper transitions between paragraphs
5. Maintain a clear argument thread throughout
6. Flag where more evidence is needed with [EVIDENCE NEEDED]
7. Suggest specific papers to cite where gaps exist

Do NOT write a generic summary. Write publishable-quality prose that advances an argument.
Structure the output with clear subsection headings."""


# ---------------------------------------------------------------------------
# Committee Simulation Mode
# ---------------------------------------------------------------------------

COMMITTEE_SIMULATION_PROMPT = """You are simulating a PhD dissertation committee of 3 professors reviewing a student's work.

Play three roles:
1. **Chair (supportive but rigorous)**: Identifies the strongest contributions and frames constructive critique
2. **Methodologist (skeptical)**: Challenges the research design, identification strategy, and data choices
3. **Theorist (big-picture)**: Asks how this connects to broader theoretical debates and what the contribution is

For each committee member, provide:
- Their assessment of the work
- 2-3 specific questions they would ask
- Suggested revisions

End with a consensus verdict: pass, revise and resubmit, or major revisions needed."""


# ---------------------------------------------------------------------------
# Peer Review Mode
# ---------------------------------------------------------------------------

PEER_REVIEW_PROMPT = """You are Winnie acting as Reviewer 2 for a top political science journal (APSR/AJPS/JOP).

Write a formal referee report with these sections:

1. **Summary** (2-3 sentences): What the paper argues and finds
2. **Main Contribution**: What is genuinely new and important here?
3. **Major Concerns** (2-3): Issues that must be addressed before publication
   - For each concern, suggest a specific way to address it
4. **Minor Comments** (3-5): Smaller issues, typos, unclear passages
5. **Assessment of Methods**: Is the identification strategy convincing?
6. **Assessment of Theory**: Does the theoretical framework hold together?
7. **Missing Literature**: What papers should the author(s) engage with?
8. **Recommendation**: Accept / R&R / Reject (with explanation)

Be constructive but honest. Good reviewers help papers get better.
Ground your review in the actual text. Cite specific passages."""


# ---------------------------------------------------------------------------
# Teaching Mode (multi-level)
# ---------------------------------------------------------------------------

TEACHING_MODE_INTRO = """You are Winnie explaining a concept to an undergraduate student in Intro to Political Science.

RULES:
- Use simple, accessible language
- Provide concrete real-world examples
- Avoid jargon unless you define it immediately
- Use analogies from everyday life
- Keep it engaging and relatable
- End with a "key takeaway" summary"""

TEACHING_MODE_GRAD = """You are Winnie explaining a concept to a first-year PhD student in political science.

RULES:  
- Assume familiarity with basic concepts
- Focus on theoretical nuance and methodological details
- Reference key debates and where this fits
- Point to seminal readings they should know
- Discuss scope conditions and external validity
- Be precise about causal claims vs correlational findings"""

TEACHING_MODE_EXPERT = """You are Winnie discussing a topic with a fellow political science professor.

RULES:
- Assume deep expertise — skip basics
- Focus on the frontier: what's new, contested, or unresolved
- Engage with the latest methodological innovations
- Discuss ongoing debates at a high level
- Reference working papers and forthcoming pieces
- Be candid about what we don't know"""


# ---------------------------------------------------------------------------
# Seminar Discussant Mode
# ---------------------------------------------------------------------------

SEMINAR_DISCUSSANT_PROMPT = """You are Winnie serving as the discussant for this paper at an academic conference.

As a good discussant:
1. **Summarize** the paper's contribution in your own words (3 sentences)
2. **Praise** what is genuinely strong about the paper (be specific)
3. **Raise 3 questions** that push the paper further:
   - One about theory or framing
   - One about methods or data
   - One about implications or scope
4. **Suggest extensions**: What's the next paper this should inspire?
5. **Connect** to the broader panel/field: How does this advance our understanding?

Be generous but intellectually honest. A good discussant makes the paper better."""


# ---------------------------------------------------------------------------
# Office Hours Mode
# ---------------------------------------------------------------------------

OFFICE_HOURS_PROMPT = """You are Winnie during office hours, helping a student who is struggling with a concept.

APPROACH:
1. First, CHECK UNDERSTANDING: Ask what the student already knows about this topic
2. BUILD ON THEIR KNOWLEDGE: Connect to what they learned in class
3. USE SCAFFOLDING: Break complex ideas into smaller, manageable pieces
4. PROVIDE EXAMPLES: Use concrete cases from the syllabus readings
5. TEST COMPREHENSION: Ask a follow-up question to check they understood
6. ENCOURAGE: Academic work is hard — be supportive while maintaining rigor

If the student seems confused, try a different explanation approach.
Reference specific readings from the course when possible."""


# ---------------------------------------------------------------------------
# Research Diary
# ---------------------------------------------------------------------------

class ResearchDiary:
    """Log ideas, questions, and insights during research sessions."""

    def __init__(self, store_dir: Path = None):
        self.store_dir = Path(store_dir or str(VAULT_ROOT / "Corpus" / "Vault" / "research_diary"))
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def add_entry(self, content: str, category: str = "idea",
                  project: str = "", tags: list = None) -> dict:
        """Add a diary entry.
        
        Categories: idea, question, insight, todo, reference, method_note
        """
        entry = {
            "id": hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "category": category,
            "content": content,
            "project": project,
            "tags": tags or [],
        }

        # Append to daily log
        today = time.strftime("%Y-%m-%d")
        path = self.store_dir / f"diary_{today}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_entries(self, days: int = 7, category: str = None,
                    project: str = None) -> list:
        """Get recent diary entries."""
        import glob
        entries = []
        files = sorted(glob.glob(str(self.store_dir / "diary_*.jsonl")))[-days:]

        for path in files:
            with open(path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if category and entry.get("category") != category:
                            continue
                        if project and entry.get("project") != project:
                            continue
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue

        return entries

    def get_questions(self, project: str = None) -> list:
        """Get open research questions."""
        return self.get_entries(days=30, category="question", project=project)

    def get_ideas(self, project: str = None) -> list:
        """Get research ideas."""
        return self.get_entries(days=30, category="idea", project=project)

    def export_to_markdown(self, days: int = 7) -> str:
        """Export diary as formatted markdown."""
        entries = self.get_entries(days=days)
        if not entries:
            return "No diary entries found."

        md = f"# Research Diary — Last {days} Days\n\n"
        by_date = {}
        for e in entries:
            date = e["timestamp"][:10]
            by_date.setdefault(date, []).append(e)

        icons = {
            "idea": "💡", "question": "❓", "insight": "🔍",
            "todo": "📋", "reference": "📚", "method_note": "🔬",
        }

        for date in sorted(by_date.keys(), reverse=True):
            md += f"\n## {date}\n\n"
            for e in by_date[date]:
                icon = icons.get(e["category"], "📝")
                md += f"- {icon} **{e['category'].title()}**: {e['content']}"
                if e.get("project"):
                    md += f" `[{e['project']}]`"
                if e.get("tags"):
                    md += f" {''.join('#' + t + ' ' for t in e['tags'])}"
                md += "\n"

        return md


# ---------------------------------------------------------------------------
# Quick Prompt Templates — one-click research actions
# ---------------------------------------------------------------------------

QUICK_PROMPTS = [
    {
        "id": "compare",
        "label": "Compare Two Papers",
        "template": "Compare and contrast the arguments, methods, and findings of {paper_a} and {paper_b}. Where do they agree? Disagree? How could their insights be synthesized?",
        "inputs": ["paper_a", "paper_b"],
    },
    {
        "id": "operationalize",
        "label": "Operationalize a Concept",
        "template": "How has '{concept}' been operationalized in the political science literature? What are the main measurement approaches, their strengths, weaknesses, and which datasets support them?",
        "inputs": ["concept"],
    },
    {
        "id": "find_gap",
        "label": "Find Research Gap",
        "template": "Based on what's been studied about '{topic}', what is the most promising research gap? What hasn't been studied, and why would filling this gap matter for the field?",
        "inputs": ["topic"],
    },
    {
        "id": "method_for",
        "label": "Best Method For",
        "template": "What is the best research design to study '{question}'? Consider identification challenges, data availability, and recent methodological innovations. What would a skeptical reviewer want to see?",
        "inputs": ["question"],
    },
    {
        "id": "dataset_for",
        "label": "Find Datasets",
        "template": "What datasets could I use to study '{topic}'? For each, describe the coverage, unit of analysis, key variables, time period, and potential limitations.",
        "inputs": ["topic"],
    },
    {
        "id": "intro_draft",
        "label": "Draft an Introduction",
        "template": "Draft an introduction for a paper with the research question: '{question}'. Include the puzzle, why it matters, a brief literature positioning, the contribution, and a roadmap.",
        "inputs": ["question"],
    },
    {
        "id": "lit_position",
        "label": "Position in Literature",
        "template": "Where does '{my_argument}' sit in the existing literature on '{topic}'? Who would agree? Disagree? What's the biggest objection, and how would I respond?",
        "inputs": ["my_argument", "topic"],
    },
    {
        "id": "teach_concept",
        "label": "Explain for Class",
        "template": "Explain '{concept}' as I would teach it in a {level} political science course. Include key references, examples, and a discussion question.",
        "inputs": ["concept", "level"],
    },
    {
        "id": "this_week",
        "label": "This Week's Readings",
        "template": "Summarize the key arguments and connections across this week's readings on '{topic}'. What's the common thread? Where do the readings disagree? What should students take away?",
        "inputs": ["topic"],
    },
    {
        "id": "reviewer_response",
        "label": "Respond to Reviewer",
        "template": "Help me respond to this reviewer comment: '{comment}'. Draft a point-by-point response that is professional, addresses the concern substantively, and suggests revisions.",
        "inputs": ["comment"],
    },
]


# ---------------------------------------------------------------------------
# All new prompts registered for mode routing
# ---------------------------------------------------------------------------

ALL_WORKFLOW_PROMPTS = {
    "reading_companion": READING_COMPANION_PROMPT,
    "writing_assistant": WRITING_ASSISTANT_PROMPT,
    "committee_sim": COMMITTEE_SIMULATION_PROMPT,
    "peer_review": PEER_REVIEW_PROMPT,
    "teaching_intro": TEACHING_MODE_INTRO,
    "teaching_grad": TEACHING_MODE_GRAD,
    "teaching_expert": TEACHING_MODE_EXPERT,
    "discussant": SEMINAR_DISCUSSANT_PROMPT,
    "office_hours": OFFICE_HOURS_PROMPT,
}

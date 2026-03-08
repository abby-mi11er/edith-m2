"""
Mission Templates — Pre-built Research Workflows
==================================================
Each template returns a list of MissionSteps that chain existing
API endpoints into coordinated "Avenger" missions.

Templates:
  1. audit_paper   — Forensic deconstruct → Sniper 3-stage → Consensus → Defend → Export
  2. lit_review     — OpenAlex → S2 → Dedup → Index → Gap analysis → Reading list → Flashcards
  3. policy_impact  — Perplexity news → GEE satellite → Literature → DAG → Stata → Consensus → Export
  4. class_prep     — Papers → Curate → Key concepts → Flashcards → Study guide → Discussion Qs
"""

from server.mission_runner import MissionStep


# ═══════════════════════════════════════════════════════════════════
# 1. AUDIT THIS PAPER
# ═══════════════════════════════════════════════════════════════════

def build_audit_paper(question: str, params: dict) -> list[MissionStep]:
    """
    Full forensic audit of a research paper.
    
    Scout deconstructs → Guardian runs the Sniper 3-stage →
    Guardian checks consensus → Guardian defends → Finisher exports report.
    """
    paper_text = params.get("paper_text", "")
    pdf_path = params.get("pdf_path", "")

    return [
        MissionStep(
            name="Deconstruct Paper",
            agent="scout",
            endpoint="/api/forensic/deconstruct",
            method="POST",
            payload={"text": paper_text or question, "path": pdf_path},
            description="Extract claims, methods, variables, and hypotheses from the paper",
        ),
        MissionStep(
            name="Extract Causal DAG",
            agent="brain",
            endpoint="/api/sniper/extract-dag",
            method="POST",
            payload={"paper_text": paper_text or question},
            description="Build Directed Acyclic Graph — identify treatment, outcome, and confounders",
        ),
        MissionStep(
            name="Sensitivity Analysis",
            agent="tank",
            endpoint="/api/sniper/sensitivity",
            method="POST",
            payload={"paper_text": paper_text or question},
            description="Run Oster δ, Monte Carlo fragility, and Cook's D — stress-test the math",
        ),
        MissionStep(
            name="Data Integrity Check",
            agent="visionary",
            endpoint="/api/sniper/integrity",
            method="POST",
            payload={"paper_text": paper_text or question},
            description="Cross-reference via Google Earth Engine + citation check",
        ),
        MissionStep(
            name="Consensus Sanity Check",
            agent="guardian",
            endpoint="/api/connectors/consensus/check",
            method="POST",
            payload={"claim": question},
            description="Compare findings against global scientific consensus",
        ),
        MissionStep(
            name="Adversarial Defense",
            agent="guardian",
            endpoint="/api/sniper/defend",
            method="POST",
            payload={"paper_text": paper_text or question, "intensity": "aggressive"},
            description="Self-audit — find weaknesses and generate auto-fix suggestions",
        ),
        MissionStep(
            name="Generate Audit Report",
            agent="finisher",
            endpoint="/api/export/latex",
            method="POST",
            payload={"text": f"Forensic Audit Report: {question}", "template": "critique"},
            description="Export complete audit findings as LaTeX report",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# 2. LIT REVIEW ON X
# ═══════════════════════════════════════════════════════════════════

def build_lit_review(question: str, params: dict) -> list[MissionStep]:
    """
    Automated literature review pipeline.
    
    Scout finds papers → Scout deep-dives citations →
    Architect deduplicates → Architect indexes → Brain finds gaps →
    Finisher generates reading list → Teacher makes flashcards.
    """
    max_papers = params.get("max_papers", 20)

    return [
        MissionStep(
            name="Scout: Academic Search",
            agent="scout",
            endpoint="/api/openalex/search",
            method="GET",
            payload={"q": question, "per_page": min(max_papers * 3, 50)},
            description=f"Search 250M+ academic works for '{question}'",
        ),
        MissionStep(
            name="Scout: Citation Deep-Dive",
            agent="scout",
            endpoint="/api/scholar/search",
            method="GET",
            payload={"q": question, "limit": max_papers},
            description="Find highly-cited papers via Semantic Scholar — sort by influence",
        ),
        MissionStep(
            name="Architect: Deduplicate",
            agent="architect",
            endpoint="/api/sources/dedup",
            method="POST",
            payload={"sources": [], "threshold": 0.85},
            description="Remove near-duplicate papers across search results",
        ),
        MissionStep(
            name="Architect: Index to Vault",
            agent="architect",
            endpoint="/api/index/run",
            method="POST",
            payload={"force": False},
            description=f"Ingest top {max_papers} papers into ChromaDB for Winnie to read",
        ),
        MissionStep(
            name="Brain: Gap Analysis",
            agent="brain",
            endpoint="/api/oracle/gaps",
            method="POST",
            payload={"topic": question},
            description="Detect gaps in literature coverage — what's missing?",
        ),
        MissionStep(
            name="Finisher: Reading List",
            agent="finisher",
            endpoint="/api/reading-list",
            method="POST",
            payload={"topic": question, "max_results": max_papers},
            description="Generate tiered reading list (essential → supplementary → frontier)",
        ),
        MissionStep(
            name="Teacher: Auto-Flashcards",
            agent="teacher",
            endpoint="/api/cognitive/spaced-rep/add",
            method="POST",
            payload={"topic": question, "count": 20},
            description="Generate spaced-repetition flashcards from key findings",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# 3. POLICY IMPACT OF Y
# ═══════════════════════════════════════════════════════════════════

def build_policy_impact(question: str, params: dict) -> list[MissionStep]:
    """
    Policy impact analysis pipeline.
    
    Spy finds recent policy → Visionary pulls satellite data →
    Scout finds academic background → Brain builds causal DAG →
    Tank runs MLE → Guardian checks consensus → Finisher exports.
    """
    location = params.get("location", "")
    year_range = params.get("year_range", "2020-2026")

    return [
        MissionStep(
            name="Spy: Find Recent Policy",
            agent="spy",
            endpoint="/api/connectors/perplexity/verify",
            method="POST",
            payload={"claim": question, "recency_hours": 720},
            description="Search real-time news and government sources for the policy",
        ),
        MissionStep(
            name="Visionary: Satellite Recon",
            agent="visionary",
            endpoint="/api/connectors/earth/land-use",
            method="POST",
            payload={"location": location or question, "years": year_range},
            description="Pull satellite imagery — map physical changes on the ground",
        ),
        MissionStep(
            name="Scout: Academic Background",
            agent="scout",
            endpoint="/api/openalex/search",
            method="GET",
            payload={"q": question, "per_page": 25},
            description="Find the academic literature on this policy area",
        ),
        MissionStep(
            name="Brain: Build Causal DAG",
            agent="brain",
            endpoint="/api/connectors/dagitty/dag",
            method="POST",
            payload={"description": question},
            description="Generate Directed Acyclic Graph — ensure no shadow variables",
        ),
        MissionStep(
            name="Tank: Run MLE Estimation",
            agent="tank",
            endpoint="/api/connectors/stata/execute",
            method="POST",
            payload={"code": f'* Policy Impact: {question}\ndisplay "MLE estimation placeholder"'},
            description="Execute Maximum Likelihood Estimation via Stata — the heavy math",
        ),
        MissionStep(
            name="Guardian: Consensus Check",
            agent="guardian",
            endpoint="/api/connectors/consensus/check",
            method="POST",
            payload={"claim": question},
            description="Sanity check — does this align with global scientific consensus?",
        ),
        MissionStep(
            name="Finisher: Push to Overleaf",
            agent="finisher",
            endpoint="/api/connectors/overleaf/push",
            method="POST",
            payload={"content": f"% Policy Impact Analysis: {question}", "dry_run": True},
            description="Export findings to Overleaf as a Methodological Critique",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# 4. PREPARE FOR CLASS ON Z
# ═══════════════════════════════════════════════════════════════════

def build_class_prep(question: str, params: dict) -> list[MissionStep]:
    """
    Class preparation pipeline.
    
    Scout finds papers → Finisher curates reading list →
    Brain extracts key relationships → Teacher builds flashcards →
    Teacher generates study critique → Teacher creates discussion questions.
    """
    return [
        MissionStep(
            name="Scout: Find Key Papers",
            agent="scout",
            endpoint="/api/openalex/search",
            method="GET",
            payload={"q": question, "per_page": 15},
            description=f"Search academic works on '{question}'",
        ),
        MissionStep(
            name="Finisher: Curate Reading List",
            agent="finisher",
            endpoint="/api/reading-list",
            method="POST",
            payload={"topic": question, "max_results": 10},
            description="Build a tiered reading list sorted by difficulty",
        ),
        MissionStep(
            name="Brain: Extract Key Concepts",
            agent="brain",
            endpoint="/api/causal/extract",
            method="POST",
            payload={"text": question},
            description="Extract key causal relationships and concepts",
        ),
        MissionStep(
            name="Teacher: Build Flashcards",
            agent="teacher",
            endpoint="/api/cognitive/spaced-rep/add",
            method="POST",
            payload={"topic": question, "count": 30},
            description="Generate 30 spaced-repetition flashcards for memorization",
        ),
        MissionStep(
            name="Teacher: Peer Review Guide",
            agent="teacher",
            endpoint="/api/cognitive/peer-review",
            method="POST",
            payload={"text": question, "mode": "teaching"},
            description="Generate a multi-persona critique guide for class discussion",
        ),
        MissionStep(
            name="Teacher: Socratic Questions",
            agent="teacher",
            endpoint="/api/cognitive/socratic/question",
            method="POST",
            payload={"topic": question, "count": 10},
            description="Generate Socratic discussion questions for class",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# Template Registry
# ═══════════════════════════════════════════════════════════════════

MISSION_TEMPLATES = {
    "audit_paper": {
        "builder": build_audit_paper,
        "name": "Audit This Paper",
        "description": "Full forensic audit: deconstruct → Sniper 3-stage → consensus → defend → export",
        "icon": "🔫",
        "agents": ["scout", "brain", "tank", "visionary", "guardian", "finisher"],
        "estimated_time": "2-5 minutes",
        "params": [
            {"name": "paper_text", "type": "text", "required": False, "description": "Full paper text"},
            {"name": "pdf_path", "type": "path", "required": False, "description": "Path to PDF on Bolt"},
        ],
    },
    "lit_review": {
        "builder": build_lit_review,
        "name": "Literature Review",
        "description": "Scout papers → deduplicate → index → gap analysis → reading list → flashcards",
        "icon": "📖",
        "agents": ["scout", "architect", "brain", "finisher", "teacher"],
        "estimated_time": "1-3 minutes",
        "params": [
            {"name": "max_papers", "type": "int", "required": False, "description": "Max papers to review (default 20)"},
        ],
    },
    "policy_impact": {
        "builder": build_policy_impact,
        "name": "Policy Impact Analysis",
        "description": "Find policy → satellite recon → DAG → Stata MLE → consensus → Overleaf export",
        "icon": "🏛️",
        "agents": ["spy", "visionary", "scout", "brain", "tank", "guardian", "finisher"],
        "estimated_time": "3-8 minutes",
        "params": [
            {"name": "location", "type": "text", "required": False, "description": "Geographic focus (e.g. 'Lubbock, TX')"},
            {"name": "year_range", "type": "text", "required": False, "description": "Year range (default '2020-2026')"},
        ],
    },
    "class_prep": {
        "builder": build_class_prep,
        "name": "Class Preparation",
        "description": "Find papers → reading list → key concepts → flashcards → study guide → Socratic questions",
        "icon": "🎓",
        "agents": ["scout", "finisher", "brain", "teacher"],
        "estimated_time": "1-2 minutes",
        "params": [],
    },
}


def register_all_templates(runner):
    """Register all mission templates with the MissionRunner."""
    import logging
    _log = logging.getLogger("edith.missions")
    for name, tmpl in MISSION_TEMPLATES.items():
        runner.register_template(name, tmpl["builder"])

    # §FIX: Validate template endpoints against registered routes at startup
    if runner._app:
        registered_paths = set()
        for route in runner._app.routes:
            if hasattr(route, 'path'):
                registered_paths.add(route.path)
        for name, tmpl in MISSION_TEMPLATES.items():
            test_steps = tmpl["builder"]("test", {})
            for step in test_steps:
                # Normalize path templates like /api/missions/run/{id}
                path = step.endpoint
                if "{" in path:
                    continue  # Skip parameterized paths
                if path not in registered_paths:
                    _log.warning(f"§SWITCHBOARD: Template '{name}' step '{step.name}' "
                                f"references unregistered endpoint: {path}")

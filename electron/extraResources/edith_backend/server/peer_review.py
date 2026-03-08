"""
Synthetic Faculty — Feature #5
================================
Combines three capabilities into one pedagogical engine:

  A. Synthetic Peer Review — simulate academic peer review using
     persona profiles built from indexed works
  B. Socratic Tutor — never gives the answer first; asks guiding
     questions that scaffold understanding
  C. Concept Scaffolding — auto-detects jargon in answers and
     generates inline "pop-up primers"

Exposed as:
  POST /api/peer-review   — submit draft for persona review
  POST /api/tutor         — Socratic teaching interaction
  POST /api/explain-term  — concept scaffolding lookup
"""

import logging
import os
import re
import time
from typing import Optional

log = logging.getLogger("edith.faculty")


# ═══════════════════════════════════════════════════════════════════
# A. Persona Profiles — built from indexed author works
# ═══════════════════════════════════════════════════════════════════

class PersonaProfile:
    """A synthetic academic persona built from an author's indexed works."""

    def __init__(self, name: str, specialization: str = "",
                 methodology_preference: str = "",
                 theoretical_lens: str = "",
                 known_critiques: str = "",
                 custom_instructions: str = ""):
        self.name = name
        self.specialization = specialization
        self.methodology_preference = methodology_preference
        self.theoretical_lens = theoretical_lens
        self.known_critiques = known_critiques
        self.custom_instructions = custom_instructions

    def to_system_prompt(self) -> str:
        """Generate the system prompt for this persona."""
        parts = [
            f"You are roleplaying as Professor {self.name}, a senior political scientist.",
        ]
        if self.specialization:
            parts.append(f"Your specialty is {self.specialization}.")
        if self.methodology_preference:
            parts.append(f"You strongly prefer {self.methodology_preference} methods.")
        if self.theoretical_lens:
            parts.append(f"Your theoretical lens is {self.theoretical_lens}.")
        if self.known_critiques:
            parts.append(f"You are known for critiquing: {self.known_critiques}.")
        if self.custom_instructions:
            parts.append(self.custom_instructions)
        parts.append(
            "\nWhen reviewing a student's draft or argument:\n"
            "1. Start with what is STRONG about the work\n"
            "2. Identify the 3 most important weaknesses\n"
            "3. Ask probing questions that push the student to think deeper\n"
            "4. Suggest specific citations from your body of work that are relevant\n"
            "5. End with a concrete next step the student should take"
        )
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "specialization": self.specialization,
            "methodology": self.methodology_preference,
            "theory": self.theoretical_lens,
            "critiques": self.known_critiques,
        }


# Pre-built faculty personas (user can add more)
FACULTY = {
    "methodologist": PersonaProfile(
        name="The Methodologist",
        specialization="research design and causal inference",
        methodology_preference="quantitative, regression-based",
        theoretical_lens="rational choice institutionalism",
        known_critiques="weak identification strategies, non-representative samples, p-hacking",
    ),
    "theorist": PersonaProfile(
        name="The Theorist",
        specialization="political theory and institutional analysis",
        methodology_preference="comparative historical analysis",
        theoretical_lens="historical institutionalism",
        known_critiques="atheoretical empiricism, missing counterfactuals, ignoring power dynamics",
    ),
    "comparativist": PersonaProfile(
        name="The Comparativist",
        specialization="comparative politics and state-building",
        methodology_preference="mixed methods and case studies",
        theoretical_lens="state capacity and governance",
        known_critiques="selection bias in case studies, overgeneralization from single cases",
    ),
    "americanist": PersonaProfile(
        name="The Americanist",
        specialization="American politics, voting behavior, and welfare policy",
        methodology_preference="survey data and experiments",
        theoretical_lens="behavioral political science",
        known_critiques="ignoring historical context, overreliance on survey data, missing subnational variation",
    ),
    # ─── Named Professor Personas (Dissertation Committee) ────────
    "mettler": PersonaProfile(
        name="Suzanne Mettler",
        specialization="the submerged state, policy feedback effects, and how government "
                        "programs shape (or hide) the state's role in citizens' lives",
        methodology_preference="survey experiments and historical policy analysis",
        theoretical_lens="policy feedback theory — policies reshape politics by altering "
                         "civic participation, group identity, and public perceptions of government",
        known_critiques=(
            "arguments that ignore the visibility dimension of policy, "
            "failure to account for how tax expenditures and loan guarantees "
            "obscure the government's role, overly simplistic models of "
            "state-citizen interaction that miss the 'submerged' dimension"
        ),
        custom_instructions=(
            "You will specifically challenge whether the student has accounted for "
            "policy visibility. If they discuss welfare, SNAP, or social programs, "
            "ask whether the delivery mechanism (direct benefit vs tax break vs "
            "third-party intermediary) changes how citizens perceive government involvement. "
            "Push them to consider how non-state actors like charities create an additional "
            "layer of 'submersion.'"
        ),
    ),
    "aldrich": PersonaProfile(
        name="John Aldrich",
        specialization="political parties, candidate strategy, voter decision-making, "
                        "and the rational calculus of political participation",
        methodology_preference="formal modeling and quantitative analysis with "
                               "rational choice microfoundations",
        theoretical_lens="rational choice theory — political actors are strategic, "
                         "parties exist to solve collective action problems, and "
                         "institutions channel individual incentives into collective outcomes",
        known_critiques=(
            "arguments that lack a clear causal mechanism, "
            "underspecified utility functions, failure to model strategic "
            "interaction between actors, ignoring how parties shape voter "
            "choices rather than merely reflecting them"
        ),
        custom_instructions=(
            "You will push back on any argument that treats political behavior "
            "as irrational or purely emotional. Demand that the student specify "
            "WHO benefits, WHAT incentives drive behavior, and HOW institutions "
            "structure choices. If they discuss voter behavior, ask about the "
            "calculus of voting and whether participation is rational given costs."
        ),
    ),
    "carsey": PersonaProfile(
        name="Tom Carsey",
        specialization="subnational politics, public opinion dynamics, party competition "
                        "in the American states, and research methodology",
        methodology_preference="multilevel regression, panel data analysis, and "
                               "rigorous quantitative methodology with attention to "
                               "unit-of-analysis issues",
        theoretical_lens="empirical political behavior — theory must be testable, "
                         "variables must be operationalized, and claims must survive "
                         "specification checks",
        known_critiques=(
            "heteroscedasticity in panel models, ecological fallacy when "
            "moving between county and individual data, endogeneity in "
            "political participation models, specification sensitivity, and "
            "failure to cluster standard errors at the appropriate level"
        ),
        custom_instructions=(
            "You are ruthless about methodology. If the student uses regression, "
            "immediately ask about: (1) Are residuals iid? (2) What is the unit "
            "of analysis and is there nesting? (3) Have they tested for "
            "heteroscedasticity? (4) Why they chose OLS vs fixed effects vs "
            "random effects. If they discuss rural counties, demand that they "
            "address spatial autocorrelation and the small-N problem."
        ),
    ),
}


def build_persona_from_corpus(
    author_name: str,
    sources: list[dict],
) -> PersonaProfile:
    """Build a persona profile from an author's indexed papers.

    Analyzes their writing style, methods, and theoretical preferences.
    """
    # Collect all text from this author
    author_texts = []
    for s in sources:
        meta = s.get("metadata", {})
        author = meta.get("author", "")
        if author_name.lower() in author.lower():
            text = s.get("text", "") or s.get("content", "")
            author_texts.append(text[:1000])

    if not author_texts:
        return PersonaProfile(
            name=author_name,
            custom_instructions=f"Simulate the academic perspective of {author_name} "
                                f"based on your knowledge of their published work.",
        )

    # Analyze methodology preference
    combined = " ".join(author_texts).lower()
    quant_score = sum(1 for w in ["regression", "data", "variable", "coefficient",
                                   "sample", "n=", "p<", "significant"] if w in combined)
    qual_score = sum(1 for w in ["case study", "interview", "ethnograph", "narrative",
                                  "process tracing", "thick description"] if w in combined)

    method_pref = "quantitative" if quant_score > qual_score else "qualitative"
    if abs(quant_score - qual_score) < 3:
        method_pref = "mixed methods"

    # Detect theoretical lens
    theory_signals = {
        "rational choice": ["rational", "utility", "game theory", "strategic"],
        "institutionalism": ["institution", "rules", "path depend", "historical"],
        "behavioralism": ["behavior", "voting", "opinion", "survey", "attitude"],
        "critical theory": ["power", "inequality", "hegemony", "discourse", "ideology"],
    }
    theory_scores = {}
    for theory, signals in theory_signals.items():
        theory_scores[theory] = sum(1 for s in signals if s in combined)

    best_theory = max(theory_scores, key=theory_scores.get) if theory_scores else ""

    return PersonaProfile(
        name=author_name,
        specialization=f"Based on {len(author_texts)} indexed works",
        methodology_preference=method_pref,
        theoretical_lens=best_theory,
        custom_instructions=(
            f"You have been trained on {len(author_texts)} documents by {author_name}. "
            f"Respond in their analytical style and with their methodological preferences."
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# B. Synthetic Peer Review
# ═══════════════════════════════════════════════════════════════════

def simulate_review(
    draft_text: str,
    persona_key: str = "methodologist",
    custom_persona: Optional[PersonaProfile] = None,
) -> dict:
    """Simulate an academic peer review of a draft.

    Args:
        draft_text: the student's draft to review
        persona_key: key from FACULTY dict
        custom_persona: override with a custom persona

    Returns:
        dict with review text, strengths, weaknesses, questions
    """
    persona = custom_persona or FACULTY.get(persona_key)
    if not persona:
        return {"error": f"Unknown persona: {persona_key}"}

    system_prompt = persona.to_system_prompt()

    review_prompt = (
        f"Review the following draft as Professor {persona.name}.\n\n"
        f"DRAFT:\n{draft_text[:5000]}\n\n"
        f"Provide your review in this structure:\n"
        f"## Strengths\n(What is this draft doing well?)\n\n"
        f"## Critical Weaknesses\n(Top 3 issues that MUST be addressed)\n\n"
        f"## Probing Questions\n(Questions that challenge the student to think deeper)\n\n"
        f"## Missing Citations\n(What sources should be referenced?)\n\n"
        f"## Concrete Next Steps\n(Specific actions to improve the draft)"
    )

    try:
        from server.backend_logic import generate_text_via_chain
        model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
        review_text, used_model = generate_text_via_chain(
            review_prompt, model_chain,
            temperature=0.3,
            system_instruction=system_prompt,
        )
        return {
            "persona": persona.to_dict(),
            "review": review_text,
            "model": used_model,
        }
    except Exception as e:
        # Fallback: generate review using just the system prompt
        try:
            from server.backend_logic import generate_text_via_chain
            model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
            combined_prompt = f"{system_prompt}\n\n{review_prompt}"
            review_text, used_model = generate_text_via_chain(
                combined_prompt, model_chain,
                temperature=0.3,
            )
            return {
                "persona": persona.to_dict(),
                "review": review_text,
                "model": used_model,
            }
        except Exception as e2:
            return {"error": str(e2)}


def run_faculty_review(
    draft_text: str,
    persona_keys: list[str] = None,
) -> dict:
    """Run a full faculty committee review (multiple personas in parallel).

    Returns synthesized review with consensus from all reviewers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not persona_keys:
        persona_keys = ["methodologist", "theorist", "comparativist"]

    reviews = []
    # §SPEEDUP: Run all persona reviews in parallel instead of sequential
    with ThreadPoolExecutor(max_workers=min(len(persona_keys), 5)) as pool:
        futures = {
            pool.submit(simulate_review, draft_text, persona_key=key): key
            for key in persona_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                review = future.result()
                reviews.append(review)
            except Exception as e:
                reviews.append({"persona": key, "error": str(e)})

    return {
        "reviews": reviews,
        "reviewer_count": len(reviews),
        "personas": [FACULTY[k].to_dict() for k in persona_keys if k in FACULTY],
    }


# ═══════════════════════════════════════════════════════════════════
# C. Socratic Tutor — never gives the answer first
# ═══════════════════════════════════════════════════════════════════

_SOCRATIC_SYSTEM = (
    "You are a Socratic Tutor for political science. Your teaching method:\n\n"
    "GOLDEN RULE: NEVER give the direct answer first.\n\n"
    "Instead:\n"
    "1. When the student asks a question, respond with a GUIDING QUESTION "
    "that leads them toward the answer\n"
    "2. Pull a specific excerpt from the source material and ask what "
    "they notice about it\n"
    "3. If the student gives a wrong answer, don't correct them directly — "
    "ask another question that reveals the flaw in their reasoning\n"
    "4. Track the student's understanding level and adjust complexity:\n"
    "   - Struggling → simpler questions, more scaffolding\n"
    "   - Getting it → push to higher-order analysis\n"
    "5. When they successfully connect concepts, acknowledge it and "
    "build on it with a harder question\n"
    "6. Use the Bloom's Taxonomy progression:\n"
    "   Remember → Understand → Apply → Analyze → Evaluate → Create\n\n"
    "You are kind but rigorous. Think of the best professor you ever had."
)

_DIFFICULTY_LEVELS = {
    "intro": {
        "label": "Introduction",
        "max_concepts": 2,
        "question_style": "simple definition/recall",
        "vocabulary": "everyday language with terms defined",
    },
    "intermediate": {
        "label": "Intermediate",
        "max_concepts": 4,
        "question_style": "compare/contrast, application",
        "vocabulary": "field-standard terminology",
    },
    "advanced": {
        "label": "Advanced",
        "max_concepts": 6,
        "question_style": "evaluation, synthesis, critique",
        "vocabulary": "specialized jargon, assumes prior knowledge",
    },
    "doctoral": {
        "label": "Doctoral",
        "max_concepts": 10,
        "question_style": "original analysis, theoretical contribution",
        "vocabulary": "full academic precision, methodology critique",
    },
}


def socratic_query(
    student_message: str,
    topic: str = "",
    difficulty: str = "intermediate",
    conversation_history: str = "",
    sources: list[dict] = None,
) -> dict:
    """Generate a Socratic response — questions, not answers.

    Args:
        student_message: what the student asked or said
        topic: the study topic
        difficulty: intro/intermediate/advanced/doctoral
        conversation_history: prior exchange
        sources: relevant source materials to draw from

    Returns:
        dict with socratic_response, difficulty_assessment, next_question
    """
    level = _DIFFICULTY_LEVELS.get(difficulty, _DIFFICULTY_LEVELS["intermediate"])

    # Build source context
    source_context = ""
    if sources:
        excerpts = []
        for s in sources[:5]:
            text = s.get("text", "") or s.get("content", "")
            author = s.get("metadata", {}).get("author", "")
            excerpts.append(f"[{author}]: {text[:300]}")
        source_context = (
            "\n\nSOURCE EXCERPTS (use these to craft your questions):\n"
            + "\n".join(excerpts)
        )

    prompt = (
        f"DIFFICULTY LEVEL: {level['label']}\n"
        f"Question style: {level['question_style']}\n"
        f"Vocabulary: {level['vocabulary']}\n"
        f"Max concepts per response: {level['max_concepts']}\n\n"
        f"TOPIC: {topic or 'Political Science'}\n\n"
    )

    if conversation_history:
        prompt += f"PRIOR CONVERSATION:\n{conversation_history}\n\n"

    prompt += (
        f"STUDENT SAYS: {student_message}\n\n"
        f"Respond as the Socratic Tutor. Remember: NEVER give the answer directly. "
        f"Ask guiding questions that lead the student to discover the answer."
        f"{source_context}"
    )

    try:
        from server.backend_logic import generate_text_via_chain
        model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

        combined = f"{_SOCRATIC_SYSTEM}\n\n{prompt}"
        response, used_model = generate_text_via_chain(
            combined, model_chain,
            temperature=0.4,
        )

        return {
            "response": response,
            "difficulty": difficulty,
            "level_info": level,
            "model": used_model,
            "mode": "socratic",
        }
    except Exception as e:
        return {"error": str(e), "mode": "socratic"}


# ═══════════════════════════════════════════════════════════════════
# D. Concept Scaffolding — auto-detect and explain jargon
# ═══════════════════════════════════════════════════════════════════

# Political science jargon that should be auto-explained
_POLSCI_JARGON = {
    "clientelism": "A political system where politicians distribute goods/favors to voters in exchange for political support, rather than through universal policy.",
    "causal inference": "Statistical and experimental methods for determining whether X actually causes Y, not just correlates with it.",
    "path dependence": "The idea that earlier decisions constrain later choices — 'history matters' in institutional development.",
    "submerged state": "Government policies (like tax breaks, loan guarantees) that are invisible to citizens, making them unaware of government's role in their lives.",
    "selectorate": "The group of people who have a say in choosing a leader. In democracies, it's voters; in autocracies, it's a small elite.",
    "veto player": "An actor whose agreement is needed to change the status quo — more veto players = harder to change policy.",
    "principal-agent": "A relationship where one party (principal) delegates work to another (agent), creating potential for the agent to act in their own interest.",
    "collective action": "The challenge of getting individuals to cooperate for a shared benefit when they could 'free-ride' on others' efforts.",
    "median voter": "The voter in the exact middle of the political spectrum — theories predict policies will converge to this voter's preferences.",
    "institutional design": "The deliberate creation of rules, procedures, and structures that shape political behavior and outcomes.",
    "endogeneity": "When the 'cause' you're studying is itself influenced by the 'effect,' making it impossible to determine true causation.",
    "heteroscedasticity": "When the variance of errors in a statistical model isn't constant, which can bias standard errors and invalidate significance tests.",
    "instrumental variable": "A technique to estimate causal effects by finding a variable that affects X but only affects Y through X.",
    "difference-in-differences": "A method comparing outcomes before/after a policy change between a treatment and control group.",
    "regression discontinuity": "A method exploiting a cutoff point (e.g., vote threshold) to estimate causal effects by comparing units just above and below the cutoff.",
}


def detect_jargon(text: str) -> list[dict]:
    """Detect political science jargon in text and return explanations.

    Returns list of {term, explanation, position} for each jargon term found.
    """
    found = []
    text_lower = text.lower()

    for term, explanation in _POLSCI_JARGON.items():
        # Find all occurrences
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(text):
            found.append({
                "term": term,
                "explanation": explanation,
                "position": match.start(),
                "matched_text": match.group(),
            })

    # Sort by position in text
    found.sort(key=lambda x: x["position"])

    # Deduplicate (keep first occurrence of each term)
    seen = set()
    unique = []
    for item in found:
        if item["term"] not in seen:
            seen.add(item["term"])
            unique.append(item)

    return unique


def explain_term(
    term: str,
    context: str = "",
    difficulty: str = "intermediate",
) -> dict:
    """Generate a detailed explanation of an academic term.

    Uses indexed sources to provide field-specific examples.
    """
    # Check built-in dictionary first
    builtin = _POLSCI_JARGON.get(term.lower())

    level = _DIFFICULTY_LEVELS.get(difficulty, _DIFFICULTY_LEVELS["intermediate"])

    try:
        from server.backend_logic import generate_text_via_chain
        model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

        prompt = (
            f"Explain the political science concept '{term}' at the "
            f"{level['label']} level.\n\n"
            f"Use {level['vocabulary']}.\n\n"
        )
        if context:
            prompt += f"CONTEXT (where the student encountered this term):\n{context[:500]}\n\n"
        if builtin:
            prompt += f"BASIC DEFINITION: {builtin}\n\n"

        prompt += (
            "Provide:\n"
            "1. A one-sentence definition\n"
            "2. A real-world example from American politics\n"
            "3. Why it matters for research\n"
            "4. Common misconceptions\n"
            "5. Related concepts to explore next"
        )

        explanation, model = generate_text_via_chain(
            prompt, model_chain,
            temperature=0.2,
        )

        return {
            "term": term,
            "explanation": explanation,
            "builtin_definition": builtin,
            "difficulty": difficulty,
            "model": model,
        }
    except Exception as e:
        return {
            "term": term,
            "explanation": builtin or f"Definition not available: {e}",
            "builtin_definition": builtin,
            "difficulty": difficulty,
        }

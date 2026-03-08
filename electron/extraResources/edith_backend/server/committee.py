"""
Committee Mode — Parallel Multi-Agent Retrieval & Generation
=============================================================
When the compute profile detects M4+Thunderbolt (or high-compute),
spawns specialized agents in parallel threads to tackle a query
from multiple angles simultaneously.

Agents:
  1. Methodologist  — focuses on research methods and design
  2. Librarian       — broad semantic retrieval, citation-heavy
  3. Analyst         — data, findings, evidence focus
  4. Judge           — merges all agent outputs into final answer

On M2 (focus mode): committee is bypassed entirely.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

log = logging.getLogger("edith.committee")


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Persona Temperaments — HOW each agent thinks (overlaid on specialty)
# ---------------------------------------------------------------------------

_TEMPERAMENTS = {
    "skeptical_critic": (
        "TEMPERAMENT: You are a Skeptical Critic. Challenge every claim. "
        "Demand evidence for assertions. Flag weak methodology, small sample sizes, "
        "and unsupported generalizations. If a source makes a bold claim without "
        "sufficient backing, say so explicitly. Default to doubt."
    ),
    "optimistic_synthesizer": (
        "TEMPERAMENT: You are an Optimistic Synthesizer. Find connections across sources "
        "that others miss. Build unified narratives from fragments. Identify emerging "
        "patterns, complementary findings, and theoretical synergies. Default to "
        "seeing the bigger picture, but acknowledge limitations honestly."
    ),
    "data_purist": (
        "TEMPERAMENT: You are a Data Purist. Only trust hard numbers, effect sizes, "
        "p-values, and quantifiable evidence. Reject qualitative assertions that lack "
        "empirical grounding. If a source doesn't provide data, note its absence. "
        "Tables and figures are your primary evidence. Default to quantification."
    ),
}

# Agent specialty × temperament matrix
_AGENT_CONFIGS = {
    "methodologist": {
        "role": "Skeptical Methodologist",
        "temperament": "skeptical_critic",
        "system": (
            "You are a research methodology specialist with a Skeptical Critic temperament.\n"
            "Focus on HOW the research was conducted: study design, data collection, "
            "analytical techniques, identification strategies, and methodological limitations.\n"
            "SKEPTICAL LENS: Challenge the validity of each method. Is the design appropriate "
            "for the question? Are there confounding variables? Is the sample representative?\n"
            "Cite specific methods from the sources and rate methodological rigor."
        ),
        "retrieval_filter": {
            "bm25_weight": 0.55,
            "diversity_lambda": 0.5,
        },
    },
    "librarian": {
        "role": "Synthesizing Librarian",
        "temperament": "optimistic_synthesizer",
        "system": (
            "You are a scholarly librarian with an Optimistic Synthesizer temperament.\n"
            "Focus on WHAT the literature says: key arguments, theoretical frameworks, "
            "how authors build on or challenge each other.\n"
            "SYNTHESIS LENS: Find hidden connections between seemingly unrelated sources. "
            "Build a narrative arc across the literature. Note intellectual lineage and "
            "emerging theoretical consensus. Use author-year citations."
        ),
        "retrieval_filter": {
            "bm25_weight": 0.25,
            "diversity_lambda": 0.8,
        },
    },
    "analyst": {
        "role": "Data-Purist Analyst",
        "temperament": "data_purist",
        "system": (
            "You are a data and evidence specialist with a Data Purist temperament.\n"
            "Focus on FINDINGS: empirical results, effect sizes, statistical significance, "
            "datasets used, tables and figures referenced.\n"
            "DATA-PURIST LENS: Only cite claims backed by quantitative evidence. "
            "If a finding is stated without data, flag it as 'unquantified assertion.' "
            "Rank evidence by strength: RCT > quasi-experiment > observational > qualitative."
        ),
        "retrieval_filter": {
            "bm25_weight": 0.5,
            "diversity_lambda": 0.6,
        },
    },
    "devils_advocate": {
        "role": "Devil's Advocate",
        "temperament": "skeptical_critic",
        "system": (
            "You are the Devil's Advocate — your ONLY job is to DESTROY the argument.\n"
            "Find the strongest possible counter-argument to the user's thesis. "
            "Steel-man the opposing view. Cite sources that directly challenge the main claim.\n"
            "RULES:\n"
            "1. Never agree with the thesis\n"
            "2. Find the top 3 counter-arguments, ranked by devastating potential\n"
            "3. For each, provide the citation that most directly undermines the claim\n"
            "4. Suggest what evidence would be needed to survive each attack\n"
            "5. End with: 'The weakest link in this argument is ____.'"
        ),
        "retrieval_filter": {
            "bm25_weight": 0.6,
            "diversity_lambda": 0.9,  # Maximize diversity to find opposing views
        },
    },
    "style_editor": {
        "role": "Style Editor",
        "temperament": "optimistic_synthesizer",
        "system": (
            "You are a Style Editor who matches the user's specific writing voice.\n"
            "You've read their academic prose and know their patterns:\n"
            "- Sentence structure preferences (complex vs. punchy)\n"
            "- Transition words they favor\n"
            "- How they introduce evidence ('As X argues...' vs 'X (2024) shows...')\n"
            "RULES:\n"
            "1. Suggest transitions that sound like THE USER wrote them\n"
            "2. Flag passive voice where the user typically uses active\n"
            "3. Note where the tone shifts from academic to casual (or vice versa)\n"
            "4. Suggest paragraph reorganizations for better flow\n"
            "5. Never change the meaning — only the music of the prose"
        ),
        "retrieval_filter": {
            "bm25_weight": 0.3,
            "diversity_lambda": 0.4,
        },
    },
    "citator": {
        "role": "Citation Tracer",
        "temperament": "data_purist",
        "system": (
            "You are the Citator — a citation specialist who traces every claim to its "
            "ORIGINAL source.\n"
            "RULES:\n"
            "1. For every theory mentioned, find who FIRST proposed it (not who cited it)\n"
            "2. Flag 'recursive citations' — where Author B cites Author C who cites Author A, "
            "and the user cites Author B instead of Author A\n"
            "3. Check for 'citation drift' — where the original claim gets distorted across "
            "generations of citing\n"
            "4. Verify page numbers: does the cited page actually contain the claimed argument?\n"
            "5. Flag 'orphan claims' — assertions with no citation at all\n"
            "6. Output a Citation Audit table: Claim | Cited Source | Original Source | Correct?"
        ),
        "retrieval_filter": {
            "bm25_weight": 0.7,  # Exact term matching for citations
            "diversity_lambda": 0.3,
        },
    },
    "grant_writer": {
        "role": "Grant Writer",
        "temperament": "optimistic_synthesizer",
        "system": (
            "You are a Grant Writer who translates academic jargon into compelling "
            "impact statements for funding bodies (NSF, NIH, SSRC, Ford Foundation).\n"
            "RULES:\n"
            "1. Convert every theoretical claim into a 'so-what' for society\n"
            "2. Frame the research as addressing a CRITICAL GAP in knowledge\n"
            "3. Use funder-friendly language: 'broader impacts,' 'intellectual merit,'\n"
            "   'transformative potential,' 'under-represented communities'\n"
            "4. Quantify everything: 'affects 2.3 million SNAP recipients in rural Texas'\n"
            "5. Suggest specific funding mechanisms (NSF SES, NIH R01, etc.)\n"
            "6. End with a one-paragraph 'Elevator Pitch' version"
        ),
        "retrieval_filter": {
            "bm25_weight": 0.4,
            "diversity_lambda": 0.7,
        },
    },
    "skeptic": {
        "role": "Shadow Variable Skeptic",
        "temperament": "skeptical_critic",
        "system": (
            "You are the Skeptic — check if UNMEASURED variables are driving the results.\n"
            "RULES:\n"
            "1. For every causal claim, list 3 potential confounders NOT in the model\n"
            "2. Check for omitted variable bias: what's missing from the regression?\n"
            "3. Check for selection bias: WHO is in the sample and who's excluded?\n"
            "4. Check for reverse causality: could Y be causing X instead?\n"
            "5. Check for ecological fallacy: are county-level findings applied to individuals?\n"
            "6. Suggest specific robustness checks: IV, RDD, matching, placebo tests\n"
            "7. Rate overall causal credibility: STRONG / MODERATE / WEAK / SPURIOUS"
        ),
        "retrieval_filter": {
            "bm25_weight": 0.5,
            "diversity_lambda": 0.7,
        },
    },
}

# ---------------------------------------------------------------------------
# Consensus Judge — synthesizes + marks confidence levels
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a senior academic Consensus Judge synthesizing three specialist analyses "
    "into one comprehensive, publication-quality answer.\n\n"
    "You have received analyses from:\n"
    "  • A Skeptical Methodologist (challenges methods and validity)\n"
    "  • A Synthesizing Librarian (finds connections and builds narratives)\n"
    "  • A Data-Purist Analyst (only trusts hard numbers)\n\n"
    "YOUR CONSENSUS PROTOCOL:\n"
    "1. For each claim or finding, determine agent agreement:\n"
    "   - ALL 3 AGREE → mark [HIGH CONFIDENCE]\n"
    "   - 2 of 3 AGREE → mark [MODERATE]\n"
    "   - AGENTS DISAGREE → mark [CONTESTED] and explain why\n"
    "2. Merge non-overlapping insights without repetition\n"
    "3. When the Skeptic challenges something the Synthesizer accepts, "
    "present BOTH perspectives with the evidence for each\n"
    "4. When the Data Purist rejects a qualitative finding, note what "
    "quantitative evidence would be needed to settle the question\n"
    "5. Structure the answer with clear sections and confidence markers\n"
    "6. End with a 'Contested Questions' section listing unresolved debates\n"
    "7. Maintain proper citations (author-year) throughout"
)



def _run_single_agent(
    agent_key: str,
    query: str,
    sources: list[dict],
    model_chain: list[str],
    top_k: int,
) -> dict:
    """Run a single committee agent (called in a thread)."""
    config = _AGENT_CONFIGS[agent_key]
    t0 = time.time()

    try:
        from server.backend_logic import (
            build_answer_prompt,
            generate_text_via_chain,
            classify_depth,
        )

        # Build agent-specific prompt
        depth = classify_depth(query)
        prompt = build_answer_prompt(
            question=query,
            sources=sources,
            depth=depth,
        )

        # Prepend agent's system instruction
        agent_text, model_used = generate_text_via_chain(
            prompt_text=prompt,
            model_chain=model_chain,
            system_instruction=config["system"],
            temperature=0.1,
        )

        elapsed = time.time() - t0
        log.info(f"§COMMITTEE: {config['role']} finished in {elapsed:.1f}s "
                 f"({len(agent_text)} chars) via {model_used}")

        return {
            "agent": agent_key,
            "role": config["role"],
            "text": agent_text,
            "model": model_used,
            "elapsed": elapsed,
            "success": True,
        }
    except Exception as e:
        log.error(f"§COMMITTEE: {config['role']} failed: {e}")
        return {
            "agent": agent_key,
            "role": config["role"],
            "text": "",
            "error": str(e),
            "elapsed": time.time() - t0,
            "success": False,
        }


def _run_agent_retrieval(
    agent_key: str,
    query: str,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    top_k: int,
) -> list[dict]:
    """Run agent-specific retrieval with tuned parameters."""
    config = _AGENT_CONFIGS[agent_key]
    filters = config.get("retrieval_filter", {})

    try:
        from server.chroma_backend import retrieve_local_sources
        sources = retrieve_local_sources(
            queries=[query],
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            top_k=top_k,
            bm25_weight=filters.get("bm25_weight", 0.35),
            diversity_lambda=filters.get("diversity_lambda", 0.65),
        )
        return sources
    except Exception as e:
        log.warning(f"§COMMITTEE: {agent_key} retrieval failed: {e}")
        return []


def run_committee(
    query: str,
    sources: list[dict],
    model_chain: list[str],
    chroma_dir: str = "",
    collection_name: str = "",
    embed_model: str = "",
    top_k: int = 50,
    max_agents: int = 4,
) -> dict:
    """Run the full committee: parallel agents + judge synthesis.

    Args:
        query: User question
        sources: Pre-retrieved sources (shared across agents)
        model_chain: Model chain for generation
        chroma_dir: ChromaDB directory (for agent-specific retrieval)
        collection_name: ChromaDB collection name
        embed_model: Embedding model name
        top_k: How many sources per agent
        max_agents: Number of parallel agents (from compute profile)

    Returns:
        dict with keys: answer, agent_results, judge_model, elapsed
    """
    t0 = time.time()
    agent_keys = list(_AGENT_CONFIGS.keys())[:max_agents - 1]  # -1 for judge

    # If we have chroma details, do per-agent retrieval in parallel
    # Otherwise, all agents share the pre-retrieved sources
    do_per_agent_retrieval = bool(chroma_dir and collection_name and embed_model)

    log.info(f"§COMMITTEE: Starting {len(agent_keys)} agents "
             f"(per_agent_retrieval={do_per_agent_retrieval})")

    # --- Phase 1: Parallel agent-specific retrieval (if available) ---
    agent_sources = {}
    if do_per_agent_retrieval:
        with ThreadPoolExecutor(max_workers=len(agent_keys)) as pool:
            retrieval_futures = {
                pool.submit(
                    _run_agent_retrieval, key, query,
                    chroma_dir, collection_name, embed_model,
                    top_k // len(agent_keys),  # Split top_k across agents
                ): key
                for key in agent_keys
            }
            for future in as_completed(retrieval_futures):
                key = retrieval_futures[future]
                try:
                    agent_sources[key] = future.result()
                except Exception:
                    agent_sources[key] = sources  # fallback to shared sources
    else:
        for key in agent_keys:
            agent_sources[key] = sources

    # --- Phase 2: Parallel agent generation ---
    agent_results = []
    with ThreadPoolExecutor(max_workers=len(agent_keys)) as pool:
        gen_futures = {
            pool.submit(
                _run_single_agent, key, query,
                agent_sources.get(key, sources),
                model_chain, top_k,
            ): key
            for key in agent_keys
        }
        for future in as_completed(gen_futures):
            try:
                result = future.result()
                agent_results.append(result)
            except Exception as e:
                log.error(f"§COMMITTEE: Agent thread crashed: {e}")

    # --- Phase 3: Judge synthesis ---
    successful = [r for r in agent_results if r.get("success") and r.get("text")]
    if not successful:
        log.warning("§COMMITTEE: All agents failed — falling back to single-pass")
        return {
            "answer": "",
            "agent_results": agent_results,
            "fallback": True,
            "elapsed": time.time() - t0,
        }

    # Build judge prompt from agent outputs
    judge_prompt = _build_judge_prompt(query, successful)

    try:
        from server.backend_logic import generate_text_via_chain

        judge_answer, judge_model = generate_text_via_chain(
            prompt_text=judge_prompt,
            model_chain=model_chain,
            system_instruction=_JUDGE_SYSTEM,
            temperature=0.1,
        )

        elapsed = time.time() - t0
        log.info(f"§COMMITTEE: Judge completed in {elapsed:.1f}s total — "
                 f"{len(successful)}/{len(agent_keys)} agents contributed")

        return {
            "answer": judge_answer,
            "agent_results": agent_results,
            "judge_model": judge_model,
            "agents_used": len(successful),
            "elapsed": elapsed,
            "fallback": False,
            # §IMP-2.4: Numerical consensus scoring
            "consensus": _analyze_consensus(successful),
        }
    except Exception as e:
        log.error(f"§COMMITTEE: Judge failed: {e}")
        combined = "\n\n".join(r["text"] for r in successful)
        return {
            "answer": combined,
            "agent_results": agent_results,
            "judge_model": "fallback_concat",
            "elapsed": time.time() - t0,
            "fallback": True,
            "consensus": _analyze_consensus(successful),
        }


def _build_judge_prompt(query: str, successful: list[dict]) -> str:
    """Build the Judge prompt from agent results with consensus analysis."""
    agent_sections = []
    for r in successful:
        agent_sections.append(
            f"=== {r['role'].upper()} ANALYSIS ===\n{r['text']}\n"
        )

    # §2.8: Consensus Voting — detect agreement patterns
    consensus = _analyze_consensus(successful)
    consensus_note = ""
    if consensus["agreement_level"] == "unanimous":
        consensus_note = (
            "\n⚡ CONSENSUS NOTE: All agents substantially agree. "
            "Focus on synthesis and depth, not conflict resolution.\n"
        )
    elif consensus["agreement_level"] == "contested":
        contested_items = ", ".join(consensus.get("contested_claims", [])[:3])
        consensus_note = (
            f"\n⚠ CONTESTED: Agents disagree on: {contested_items}. "
            "You MUST present both sides with evidence for each and mark "
            "these as [CONTESTED] in your response.\n"
        )
    elif consensus["agreement_level"] == "split":
        consensus_note = (
            "\n🔀 SPLIT DECISION: Agents are evenly divided. "
            "Weight your synthesis by citation strength, not vote count.\n"
        )

    return (
        f"ORIGINAL QUESTION:\n{query}\n\n"
        f"{''.join(agent_sections)}\n"
        f"{consensus_note}\n"
        "Using the analyses above, produce a comprehensive, unified answer. "
        "Integrate the strongest insights from each specialist."
    )


def _analyze_consensus(agent_results: list[dict]) -> dict:
    """§2.8: Analyze agent outputs for agreement/disagreement.

    Returns agreement_level: unanimous | majority | split | contested
    """
    import re

    texts = [r.get("text", "").lower() for r in agent_results if r.get("text")]
    if len(texts) < 2:
        return {"agreement_level": "unanimous", "contested_claims": []}

    # Disagreement signals
    disagree_patterns = [
        r"however,?\s+(the|this|i|we|it)",
        r"in contrast",
        r"this (contradicts|challenges|disputes|overlooks)",
        r"the (methodologist|librarian|analyst|skeptic) (ignores|misses|overlooks)",
        r"insufficient evidence",
        r"alternative interpretation",
    ]

    # Count cross-agent disagreements
    disagree_count = 0
    contested_claims = []
    for text in texts:
        for pattern in disagree_patterns:
            matches = re.findall(pattern, text)
            disagree_count += len(matches)

        # Extract contested claims (sentences containing challenge language)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if any(re.search(p, sent) for p in disagree_patterns[:4]):
                contested_claims.append(sent[:80].strip())

    # Determine agreement level
    ratio = disagree_count / max(len(texts), 1)
    if ratio == 0:
        level = "unanimous"
    elif ratio < 1.5:
        level = "majority"
    elif ratio < 3:
        level = "split"
    else:
        level = "contested"

    return {
        "agreement_level": level,
        "disagree_signals": disagree_count,
        "contested_claims": contested_claims[:5],
        "agent_count": len(texts),
        # §IMP-2.4: Numerical consensus score (0-100)
        "consensus_score": max(0, min(100, int(100 - ratio * 20))),
        "score_label": f"{max(0, min(100, int(100 - ratio * 20)))}/100 {'(§ unanimous)' if level == 'unanimous' else '(§ ' + level + ')'}",
    }



def run_committee_streaming(
    query: str,
    sources: list[dict],
    model_chain: list[str],
    chroma_dir: str = "",
    collection_name: str = "",
    embed_model: str = "",
    top_k: int = 50,
    max_agents: int = 4,
):
    """Run committee with streaming Judge output.

    Yields dicts: {"type": "status", "text": ...} or {"type": "token", "text": ...}
    The agents run in parallel (non-streaming). The Judge streams token-by-token.
    """
    import os
    t0 = time.time()
    agent_keys = list(_AGENT_CONFIGS.keys())[:max_agents - 1]
    do_per_agent_retrieval = bool(chroma_dir and collection_name and embed_model)

    yield {"type": "status", "text": f"Starting {len(agent_keys)} agents..."}

    # Phase 1: Parallel retrieval
    agent_sources = {}
    if do_per_agent_retrieval:
        with ThreadPoolExecutor(max_workers=len(agent_keys)) as pool:
            retrieval_futures = {
                pool.submit(
                    _run_agent_retrieval, key, query,
                    chroma_dir, collection_name, embed_model,
                    top_k // len(agent_keys),
                ): key
                for key in agent_keys
            }
            for future in as_completed(retrieval_futures):
                key = retrieval_futures[future]
                try:
                    agent_sources[key] = future.result()
                except Exception:
                    agent_sources[key] = sources
    else:
        for key in agent_keys:
            agent_sources[key] = sources

    # Phase 2: Parallel agent generation
    agent_results = []
    with ThreadPoolExecutor(max_workers=len(agent_keys)) as pool:
        gen_futures = {
            pool.submit(
                _run_single_agent, key, query,
                agent_sources.get(key, sources),
                model_chain, top_k,
            ): key
            for key in agent_keys
        }
        for future in as_completed(gen_futures):
            try:
                result = future.result()
                agent_results.append(result)
                if result.get("success"):
                    yield {"type": "status",
                           "text": f"{result['role']} done ({result['elapsed']:.1f}s)"}
            except Exception as e:
                log.error(f"§COMMITTEE: Agent thread crashed: {e}")

    successful = [r for r in agent_results if r.get("success") and r.get("text")]
    if not successful:
        yield {"type": "status", "text": "All agents failed — falling back"}
        return

    yield {"type": "status",
           "text": f"Judge synthesizing {len(successful)} analyses..."}

    # Phase 3: Stream Judge output token-by-token via Gemini
    judge_prompt = _build_judge_prompt(query, successful)

    try:
        from google import genai as _genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        client = _genai.Client(api_key=api_key)
        model_to_use = model_chain[0] if model_chain else "gemini-2.5-flash"
        cfg = types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=_JUDGE_SYSTEM,
            max_output_tokens=5000,
        )

        for chunk in client.models.generate_content_stream(
            model=model_to_use, contents=judge_prompt, config=cfg,
        ):
            tok = chunk.text or ""
            if tok:
                yield {"type": "token", "text": tok}

        elapsed = time.time() - t0
        log.info(f"§COMMITTEE: Streamed Judge in {elapsed:.1f}s — "
                 f"{len(successful)}/{len(agent_keys)} agents contributed")
        yield {"type": "done", "agents_used": len(successful),
               "elapsed": elapsed}
    except Exception as e:
        log.error(f"§COMMITTEE: Judge streaming failed: {e}")
        # Fallback: yield concatenated agent outputs
        combined = "\n\n".join(r["text"] for r in successful)
        yield {"type": "token", "text": combined}
        yield {"type": "done", "agents_used": len(successful),
               "elapsed": time.time() - t0, "fallback": True}


# ═══════════════════════════════════════════════════════════════════
# SCENE 4: COMMITTEE "FIX IT" — Source-Grounded Paragraph Rewrite
# ═══════════════════════════════════════════════════════════════════

def committee_fix_paragraph(
    paragraph: str,
    critique: str,
    critic_persona: str = "",
    model_chain: list[str] = None,
    chroma_dir: str = "",
    collection_name: str = "edith",
    embed_model: str = "",
) -> dict:
    """Scene 4: 'Fix It' — rewrite a paragraph grounded in VAULT sources.

    When a committee agent identifies a flaw in a paragraph, this function:
    1. Retrieves relevant sources from the VAULT matching the critique
    2. Rewrites the paragraph incorporating the fix
    3. Grounds every claim in a real source with proper citations

    Args:
        paragraph: The original paragraph text
        critique: The committee agent's critique/suggestion
        critic_persona: Name of the critic (e.g., 'Suzanne Mettler')
        model_chain: LLM models to use
        chroma_dir: ChromaDB directory
        collection_name: Collection to search
        embed_model: Embedding model

    Returns: dict with original, fixed paragraph, sources used, and changes.
    """
    import os
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
    chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")
    embed_model = embed_model or os.environ.get("EDITH_EMBED_MODEL", "")
    t0 = time.time()

    # Step 1: Retrieve sources relevant to the critique
    sources = []
    try:
        from server.chroma_backend import retrieve_local_sources
        # Search for sources matching the critique + original paragraph context
        search_queries = [
            critique[:200],
            paragraph[:200],
        ]
        if critic_persona:
            search_queries.append(f"{critic_persona} theory argument")

        for q in search_queries:
            results = retrieve_local_sources(
                queries=[q],
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                embed_model=embed_model,
                top_k=5,
            )
            sources.extend(results)

        # Deduplicate
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s.get("text", ""))[:100]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        sources = unique_sources[:8]
    except Exception as e:
        log.warning(f"§COMMITTEE: Source retrieval for fix failed: {e}")

    # Step 2: Build context from retrieved sources
    source_context = ""
    for i, s in enumerate(sources, 1):
        author = s.get("metadata", {}).get("author", "Unknown")
        year = s.get("metadata", {}).get("year", "")
        text = (s.get("text", "") or s.get("document", ""))[:300]
        source_context += f"\n[Source {i}] {author} ({year}): {text}\n"

    # Step 3: LLM rewrite
    persona_note = f"The critic is {critic_persona}. " if critic_persona else ""
    prompt = (
        f"You are E.D.I.T.H., rewriting a paragraph based on committee feedback.\n\n"
        f"ORIGINAL PARAGRAPH:\n{paragraph}\n\n"
        f"CRITIQUE:\n{persona_note}{critique}\n\n"
        f"AVAILABLE SOURCES FROM VAULT:\n{source_context}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Rewrite the paragraph to address the critique\n"
        f"2. Ground every claim in one of the sources above using (Author Year) citations\n"
        f"3. Maintain the same academic tone and approximate length\n"
        f"4. If the critic suggested a specific theoretical adjustment, implement it\n"
        f"5. Mark substantive changes with [CHANGED] at the end of modified sentences\n\n"
        f"Return ONLY the rewritten paragraph."
    )

    fixed_paragraph = paragraph  # Fallback
    model_used = ""
    try:
        from server.backend_logic import generate_text_via_chain
        text, model_used = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                "You are a scholarly writing assistant. Rewrite paragraphs "
                "based on committee critique, grounding all claims in sources."
            ),
            temperature=0.3,
        )
        fixed_paragraph = text.strip()
    except Exception as e:
        log.error(f"§COMMITTEE: Fix-It LLM call failed: {e}")

    # Step 4: Identify changes
    original_sentences = {s.strip() for s in paragraph.split(".") if s.strip()}
    fixed_sentences = {s.strip().replace(" [CHANGED]", "")
                       for s in fixed_paragraph.split(".") if s.strip()}
    changes = fixed_sentences - original_sentences

    elapsed = time.time() - t0
    log.info(f"§COMMITTEE: Fix-It completed in {elapsed:.1f}s "
             f"({len(sources)} sources, {len(changes)} changes)")

    return {
        "original": paragraph,
        "fixed": fixed_paragraph,
        "critique": critique,
        "critic": critic_persona,
        "sources_used": [
            {
                "author": s.get("metadata", {}).get("author", "Unknown"),
                "year": s.get("metadata", {}).get("year", ""),
                "text_preview": (s.get("text", "") or "")[:150],
            }
            for s in sources
        ],
        "changes_made": list(changes)[:10],
        "model": model_used,
        "elapsed_s": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# TITAN §2: AUTONOMOUS PERSONAS — Stanford "Social Simulacra"
# ═══════════════════════════════════════════════════════════════════

import json
from datetime import datetime
from pathlib import Path


class PersonaMemory:
    """Long-term memory for committee personas — Stanford Generative Agents style.

    Each persona remembers:
    - Past critiques they've given
    - Arguments they've approved or rejected
    - Positions they've taken on specific topics
    - Consistency violations detected

    Stored on Bolt at {EDITH_DATA_ROOT}/PERSONAS/{persona_name}.json
    """

    def __init__(self, persona_name: str, data_root: str = ""):
        self.name = persona_name
        self._root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._dir = os.path.join(self._root, "PERSONAS")
        self._path = os.path.join(self._dir, f"{persona_name}.json")
        self._memory = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "name": self.name,
            "created": datetime.now().isoformat(),
            "critiques": [],
            "positions": {},
            "approved": [],
            "rejected": [],
            "consistency_violations": [],
        }

    def _save(self):
        os.makedirs(self._dir, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._memory, f, indent=2, default=str)

    def record_critique(self, topic: str, critique: str, stance: str = ""):
        """Record a critique this persona gave."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "critique": critique[:500],
            "stance": stance,
        }
        self._memory["critiques"].append(entry)
        # Keep last 100
        self._memory["critiques"] = self._memory["critiques"][-100:]
        # Update position on this topic
        if stance:
            self._memory["positions"][topic] = {
                "stance": stance,
                "last_updated": entry["timestamp"],
            }
        self._save()

    def record_approval(self, paragraph: str, reason: str = ""):
        self._memory["approved"].append({
            "timestamp": datetime.now().isoformat(),
            "text": paragraph[:300],
            "reason": reason[:200],
        })
        self._memory["approved"] = self._memory["approved"][-50:]
        self._save()

    def record_rejection(self, paragraph: str, reason: str = ""):
        self._memory["rejected"].append({
            "timestamp": datetime.now().isoformat(),
            "text": paragraph[:300],
            "reason": reason[:200],
        })
        self._memory["rejected"] = self._memory["rejected"][-50:]
        self._save()

    def recall(self, topic: str, limit: int = 5) -> dict:
        """Recall what this persona has said about a topic."""
        relevant = [c for c in self._memory["critiques"]
                     if topic.lower() in c.get("topic", "").lower()
                     or topic.lower() in c.get("critique", "").lower()]
        position = self._memory["positions"].get(topic, {})
        return {
            "persona": self.name,
            "position": position,
            "past_critiques": relevant[-limit:],
            "total_critiques": len(self._memory["critiques"]),
            "approvals": len(self._memory["approved"]),
            "rejections": len(self._memory["rejected"]),
        }

    def check_consistency(self, topic: str, new_stance: str) -> dict:
        """Check if a new stance contradicts this persona's past positions."""
        old = self._memory["positions"].get(topic, {})
        if not old:
            return {"consistent": True, "prior_stance": None}

        prior = old.get("stance", "")
        is_contradiction = (
            ("support" in prior.lower() and "reject" in new_stance.lower()) or
            ("reject" in prior.lower() and "support" in new_stance.lower()) or
            ("agree" in prior.lower() and "disagree" in new_stance.lower()) or
            ("disagree" in prior.lower() and "agree" in new_stance.lower())
        )

        if is_contradiction:
            violation = {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "prior_stance": prior,
                "new_stance": new_stance,
            }
            self._memory["consistency_violations"].append(violation)
            self._save()
            return {"consistent": False, "prior_stance": prior, "violation": violation}

        return {"consistent": True, "prior_stance": prior}


def run_background_debate(
    thesis: str,
    persona_a: str = "devils_advocate",
    persona_b: str = "librarian",
    rounds: int = 3,
    model_chain: list[str] = None,
) -> dict:
    """Stanford-style autonomous background debate between two personas.

    Run while the user is away. Each round:
    1. Persona A critiques the thesis
    2. Persona B counters
    3. Both memories are updated

    Saves disagreement memo to {EDITH_DATA_ROOT}/PERSONAS/memos/
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
    t0 = time.time()

    config_a = _AGENT_CONFIGS.get(persona_a, _AGENT_CONFIGS["devils_advocate"])
    config_b = _AGENT_CONFIGS.get(persona_b, _AGENT_CONFIGS["librarian"])
    mem_a = PersonaMemory(persona_a)
    mem_b = PersonaMemory(persona_b)

    debate_log = []

    for round_num in range(1, rounds + 1):
        # Persona A attacks
        history_a = mem_a.recall(thesis[:50])
        prompt_a = (
            f"THESIS: {thesis}\n\n"
            f"You are {config_a['role']}.\n"
            f"{config_a['system']}\n\n"
        )
        if history_a["past_critiques"]:
            prompt_a += f"YOUR PAST POSITIONS: {json.dumps(history_a['past_critiques'][-2:])}\n\n"
        if round_num > 1:
            prompt_a += f"PREVIOUS COUNTER-ARGUMENT: {debate_log[-1]['text'][:500]}\n\n"
        prompt_a += f"Round {round_num}/{rounds}: Give your critique (2-3 paragraphs)."

        try:
            from server.backend_logic import generate_text_via_chain
            text_a, _ = generate_text_via_chain(
                prompt_a, model_chain,
                system_instruction=config_a["system"],
                temperature=0.4,
            )
        except Exception as e:
            text_a = f"[Debate generation failed: {e}]"

        debate_log.append({"round": round_num, "persona": persona_a,
                           "role": config_a["role"], "text": text_a.strip()})
        mem_a.record_critique(thesis[:50], text_a[:500], stance="critique")

        # Persona B defends
        history_b = mem_b.recall(thesis[:50])
        prompt_b = (
            f"THESIS: {thesis}\n\n"
            f"You are {config_b['role']}.\n"
            f"{config_b['system']}\n\n"
            f"CRITIQUE FROM {config_a['role']}:\n{text_a[:600]}\n\n"
        )
        if history_b["past_critiques"]:
            prompt_b += f"YOUR PAST POSITIONS: {json.dumps(history_b['past_critiques'][-2:])}\n\n"
        prompt_b += f"Round {round_num}/{rounds}: Counter-argue (2-3 paragraphs)."

        try:
            text_b, _ = generate_text_via_chain(
                prompt_b, model_chain,
                system_instruction=config_b["system"],
                temperature=0.4,
            )
        except Exception as e:
            text_b = f"[Counter-argument generation failed: {e}]"

        debate_log.append({"round": round_num, "persona": persona_b,
                           "role": config_b["role"], "text": text_b.strip()})
        mem_b.record_critique(thesis[:50], text_b[:500], stance="defense")

    elapsed = time.time() - t0

    # Generate and save memo
    memo = generate_disagreement_memo(thesis, debate_log, persona_a, persona_b)

    log.info(f"§DEBATE: {persona_a} vs {persona_b}, {rounds} rounds in {elapsed:.1f}s")

    return {
        "thesis": thesis,
        "personas": [persona_a, persona_b],
        "rounds": rounds,
        "debate_log": debate_log,
        "memo": memo,
        "elapsed_s": round(elapsed, 2),
    }


def generate_disagreement_memo(
    thesis: str,
    debate_log: list[dict],
    persona_a: str = "",
    persona_b: str = "",
) -> dict:
    """Format a background debate into a readable Disagreement Memo.

    Saved to {EDITH_DATA_ROOT}/PERSONAS/memos/ for morning brief integration.
    """
    data_root = os.environ.get("EDITH_DATA_ROOT", ".")
    memo_dir = os.path.join(data_root, "PERSONAS", "memos")
    os.makedirs(memo_dir, exist_ok=True)

    # Extract key disagreements
    critiques = [e for e in debate_log if e.get("persona") == persona_a]
    defenses = [e for e in debate_log if e.get("persona") == persona_b]

    memo = {
        "generated": datetime.now().isoformat(),
        "thesis": thesis,
        "debaters": {
            "attacker": {"persona": persona_a, "role": critiques[0]["role"] if critiques else ""},
            "defender": {"persona": persona_b, "role": defenses[0]["role"] if defenses else ""},
        },
        "rounds": len(critiques),
        "key_critiques": [c["text"][:300] for c in critiques],
        "key_defenses": [d["text"][:300] for d in defenses],
        "summary": (
            f"Debate on: '{thesis[:100]}'. "
            f"{persona_a} raised {len(critiques)} challenge(s). "
            f"{persona_b} offered {len(defenses)} defense(s). "
            f"Review the full exchange for follow-up items."
        ),
    }

    # Save to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memo_path = os.path.join(memo_dir, f"debate_{timestamp}.json")
    try:
        with open(memo_path, "w") as f:
            json.dump(memo, f, indent=2, default=str)
        memo["path"] = memo_path
    except Exception as e:
        log.error(f"§DEBATE: Failed to save memo: {e}")

    return memo


# ═══════════════════════════════════════════════════════════════════
# CITADEL §3: SPATIAL AUDIO PERSONA PINNING
# ═══════════════════════════════════════════════════════════════════

class SpatialPersonaConfig:
    """Pin committee personas to physical audio positions.

    Maps each persona to an azimuth (clock position) and elevation
    so responses feel like they come from a person in the room.

    Designed for AirPods Pro spatial audio or spatial speakers.

    Clock positions (facing forward):
        12 o'clock = 0°, 3 o'clock = 90°, 6 o'clock = 180°, 9 o'clock = 270°
    """

    # Default spatial layout — configurable per user
    DEFAULT_POSITIONS = {
        "mettler":         {"azimuth": 60,  "elevation": 0,  "label": "2 o'clock"},
        "aldrich":         {"azimuth": 300, "elevation": 0,  "label": "10 o'clock"},
        "winnie":          {"azimuth": 0,   "elevation": 0,  "label": "Center"},
        "methodologist":   {"azimuth": 330, "elevation": 10, "label": "11 o'clock high"},
        "librarian":       {"azimuth": 30,  "elevation": 10, "label": "1 o'clock high"},
        "analyst":         {"azimuth": 90,  "elevation": -5, "label": "3 o'clock"},
        "devils_advocate": {"azimuth": 270, "elevation": -5, "label": "9 o'clock"},
        "judge":           {"azimuth": 0,   "elevation": 15, "label": "Center high"},
    }

    def __init__(self, custom_positions: dict = None):
        self._positions = {**self.DEFAULT_POSITIONS}
        if custom_positions:
            self._positions.update(custom_positions)

    def get_position(self, persona: str) -> dict:
        """Get the spatial position for a persona."""
        return self._positions.get(persona, {"azimuth": 0, "elevation": 0, "label": "Center"})

    def get_spatial_layout(self) -> dict:
        """Full persona-to-position map for the frontend/audio engine."""
        return {
            "positions": self._positions,
            "total_personas": len(self._positions),
            "coordinate_system": "azimuth_elevation",
            "azimuth_range": "0-360 degrees (clockwise from front)",
            "elevation_range": "-90 to 90 degrees (negative = below ear level)",
        }

    def set_position(self, persona: str, azimuth: int, elevation: int = 0):
        """Repin a persona to a new audio position."""
        self._positions[persona] = {
            "azimuth": azimuth % 360,
            "elevation": max(-90, min(90, elevation)),
            "label": self._azimuth_to_clock(azimuth),
        }

    @staticmethod
    def _azimuth_to_clock(azimuth: int) -> str:
        """Convert azimuth to clock position label."""
        clock = round((azimuth % 360) / 30)
        if clock == 0:
            clock = 12
        return f"{clock} o'clock"


# Global spatial config
spatial_personas = SpatialPersonaConfig()


# ═══════════════════════════════════════════════════════════════════
# GLASS BOX §2: NEURAL WEIGHTING — Source-Grounded Persona Reasoning
# ═══════════════════════════════════════════════════════════════════

def explain_persona_reasoning(
    persona: str,
    critique: str,
    topic: str = "",
    sources: list[dict] = None,
    model_chain: list[str] = None,
) -> dict:
    """Explain WHY a persona prioritized one concept over another.

    When you ask: "Suzanne, why are you prioritizing Traceability
    over Administrative Burden?"

    This traces back to:
    1. Specific VAULT highlights that grounded the persona's prompt
    2. PersonaMemory past positions on the topic
    3. Consistency checks against prior stances

    Returns: source-grounded explanation with citations
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

    # 1. Get persona config
    config = _AGENT_CONFIGS.get(persona, _AGENT_CONFIGS.get("librarian", {}))

    # 2. Pull persona memory — past positions and critiques
    memory = PersonaMemory(persona)
    recall = memory.recall(topic or critique[:50])

    # 3. Get grounded context from VAULT
    vault_passages = []
    try:
        from server.cognitive_engine import _get_grounded_persona_prompt
        grounded = _get_grounded_persona_prompt(persona, topic or critique[:80])
        if grounded:
            vault_passages.append(grounded[:500])
    except Exception:
        pass

    # Add any provided sources
    if sources:
        for src in sources[:3]:
            vault_passages.append(
                f"Source: {src.get('metadata', {}).get('source', 'Unknown')} — "
                f"{src.get('text', '')[:200]}"
            )

    # 4. Generate explanation via LLM
    explanation = ""
    try:
        from server.backend_logic import generate_text_via_chain

        prompt = (
            f"PERSONA: {config.get('role', persona)}\n"
            f"PERSONA SYSTEM: {config.get('system', '')[:200]}\n\n"
            f"CRITIQUE GIVEN: {critique}\n\n"
            f"PAST POSITIONS ON THIS TOPIC:\n"
            f"{json.dumps(recall.get('past_critiques', [])[-3:], indent=2)}\n\n"
            f"VAULT SOURCES GROUNDING THIS PERSONA:\n"
            f"{chr(10).join(vault_passages[:3])}\n\n"
            f"TASK: Explain in first person as {config.get('role', persona)}, "
            f"WHY you gave this specific critique. Reference the specific "
            f"sources from the VAULT that informed your reasoning. Be precise: "
            f"cite page numbers or specific arguments. 3-4 sentences."
        )

        explanation, _ = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                f"You are {config.get('role', persona)}. Explain your "
                f"reasoning by citing the user's own research materials. "
                f"You are reflecting their prioritized research logic."
            ),
            temperature=0.2,
        )
    except Exception as e:
        explanation = (
            f"As {config.get('role', persona)}, my critique was grounded in: "
            f"{', '.join(vault_passages[:2]) if vault_passages else 'general persona expertise'}. "
            f"[LLM explanation unavailable: {e}]"
        )

    # 5. Consistency check
    consistency = memory.check_consistency(
        topic or critique[:50],
        critique[:100],
    )

    return {
        "persona": persona,
        "role": config.get("role", ""),
        "critique": critique,
        "explanation": explanation,
        "vault_sources": vault_passages,
        "past_positions": recall.get("past_critiques", [])[-3:],
        "consistency": consistency,
        "total_critiques": recall.get("total_critiques", 0),
        "glass_box": True,
    }


# ═══════════════════════════════════════════════════════════════════
# §CE-31: Dynamic Persona Selection — Pick the right agents for the query
# ═══════════════════════════════════════════════════════════════════

def select_personas_for_query(query: str, max_agents: int = 4) -> list[str]:
    """Dynamically select the best committee agents for a given query.

    Instead of always running the same 4 agents, we pick the ones
    most relevant to what the user is asking about.

    Query about methodology → Methodologist + Skeptic
    Query about literature → Librarian + Citator
    Query about writing → Style Editor + Grant Writer
    """
    query_lower = query.lower()

    # Score each persona against the query
    persona_scores = {}
    signal_map = {
        "methodologist": ["method", "design", "regression", "experiment", "sample",
                          "validity", "reliability", "measure", "operationalize"],
        "librarian": ["literature", "review", "theory", "framework", "scholars",
                       "published", "journal", "cite", "reference"],
        "analyst": ["data", "evidence", "findings", "results", "significant",
                     "statistics", "table", "figure", "percentage"],
        "devils_advocate": ["argue", "critique", "flaw", "weakness", "challenge",
                            "counter", "problem", "defend", "attack"],
        "style_editor": ["writing", "paragraph", "prose", "draft", "rewrite",
                          "clarity", "tone", "voice", "edit"],
        "citator": ["citation", "cite", "reference", "bibliography", "who said",
                     "source", "original", "attribute"],
        "grant_writer": ["grant", "proposal", "fund", "NSF", "broader impact",
                          "significance", "pitch"],
        "skeptic": ["confound", "variable", "spurious", "causal", "bias",
                     "endogeneity", "selection", "omitted"],
    }

    for persona, signals in signal_map.items():
        score = sum(1 for s in signals if s in query_lower)
        persona_scores[persona] = score

    # Always include at least one skeptical voice
    sorted_personas = sorted(persona_scores, key=persona_scores.get, reverse=True)

    selected = sorted_personas[:max_agents]

    # Ensure diversity: at least one critic if not already present
    critics = {"devils_advocate", "skeptic", "methodologist"}
    if not any(p in critics for p in selected):
        selected[-1] = "devils_advocate"

    return selected


# ═══════════════════════════════════════════════════════════════════
# §CE-32: Debate Scoring — Quantify argument strength
# ═══════════════════════════════════════════════════════════════════

def score_debate(debate_log: list[dict]) -> dict:
    """Score a debate for argument quality on both sides.

    Metrics:
    - Evidence citations per round
    - Logical coherence (fewer contradictions = higher)
    - Specificity (concrete claims vs vague assertions)
    - Engagement (does each round address the previous one?)
    """
    import re as _re

    scores = {"rounds": []}

    for entry in debate_log:
        text = entry.get("text", "")

        # Citation count
        citations = len(_re.findall(r'\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}\)', text))

        # Specificity: numbers, percentages, named theories
        numbers = len(_re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
        named_theories = len(_re.findall(r'[A-Z][a-z]+(?:\'s)?\s+(?:theory|framework|model|hypothesis)', text))

        # Engagement: references to previous argument
        engagement = any(w in text.lower() for w in
                         ["however", "in contrast", "you argued", "the previous", "as noted"])

        word_count = len(text.split())

        round_score = {
            "persona": entry.get("persona", ""),
            "round": entry.get("round", 0),
            "citations": citations,
            "specificity": numbers + named_theories,
            "engagement": engagement,
            "word_count": word_count,
            "quality_score": round(
                min(10, citations * 1.5 + numbers * 0.5 + named_theories * 2 +
                    (2 if engagement else 0)), 1
            ),
        }
        scores["rounds"].append(round_score)

    # Overall scores per persona
    persona_totals = {}
    for r in scores["rounds"]:
        p = r["persona"]
        if p not in persona_totals:
            persona_totals[p] = {"total_score": 0, "rounds": 0, "citations": 0}
        persona_totals[p]["total_score"] += r["quality_score"]
        persona_totals[p]["rounds"] += 1
        persona_totals[p]["citations"] += r["citations"]

    for p, data in persona_totals.items():
        data["avg_score"] = round(data["total_score"] / max(data["rounds"], 1), 1)

    scores["persona_scores"] = persona_totals
    scores["winner"] = max(persona_totals, key=lambda p: persona_totals[p]["avg_score"]) if persona_totals else ""

    return scores


# ═══════════════════════════════════════════════════════════════════
# §CE-33: Evidence Tracker — Track what evidence has been used
# ═══════════════════════════════════════════════════════════════════

class EvidenceTracker:
    """Track which sources have been used in committee responses.

    Prevents over-reliance on a single source and ensures
    diverse evidence across the dissertation.
    """

    def __init__(self):
        self._usage: dict[str, int] = {}  # sha256 → usage count
        self._topics: dict[str, list[str]] = {}  # sha256 → [topics used for]

    def record_usage(self, source_sha: str, topic: str = ""):
        self._usage[source_sha] = self._usage.get(source_sha, 0) + 1
        if topic:
            self._topics.setdefault(source_sha, []).append(topic)

    def get_overused(self, threshold: int = 5) -> list[dict]:
        """Return sources used more than threshold times."""
        return [
            {"sha256": sha, "count": count, "topics": self._topics.get(sha, [])}
            for sha, count in self._usage.items()
            if count >= threshold
        ]

    def get_underused(self, all_shas: list[str]) -> list[str]:
        """Return sources that have never been used."""
        return [sha for sha in all_shas if sha not in self._usage]

    def get_diversity_score(self) -> float:
        """0-1 score. 1.0 = all sources used equally. 0 = one source dominates."""
        if not self._usage:
            return 1.0
        counts = list(self._usage.values())
        total = sum(counts)
        if total == 0:
            return 1.0
        # Normalized entropy
        import math
        proportions = [c / total for c in counts]
        entropy = -sum(p * math.log(p + 1e-10) for p in proportions)
        max_entropy = math.log(len(counts) + 1e-10)
        return round(entropy / max_entropy if max_entropy > 0 else 1.0, 3)

    @property
    def status(self) -> dict:
        return {
            "total_sources_used": len(self._usage),
            "total_citations": sum(self._usage.values()),
            "diversity_score": self.get_diversity_score(),
            "most_cited": sorted(
                self._usage.items(), key=lambda x: -x[1]
            )[:5],
        }


# Global evidence tracker
evidence_tracker = EvidenceTracker()


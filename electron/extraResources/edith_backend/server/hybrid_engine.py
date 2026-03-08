"""
Hybrid Architecture Engine — "Stark Industries" Gold Standard
=============================================================
Fine-Tuning for Tone (The Brain) + RAG for Facts (The Library).

Division of Labor:
    Fine-Tuning: Teaches Winnie HOW you think — vocabulary, argument
                 structure, departmental "vibe."
    Bolt (RAG):  Gives her WHAT to cite — 93,000 papers, syllabi,
                 census data at 3,100 MB/s.

Two-Stage Query:
    Stage 1 (Retrieval): ChromaDB on Bolt → relevant snippets
    Stage 2 (Synthesis): Snippets → fine-tuned model → Abby-style response

Usage:
    # Generate training data from your best work
    gen = ToneTrainingGenerator()
    gen.ingest_sample("/path/to/comp_answer.pdf", "exam_answer")
    gen.ingest_sample("/path/to/thesis_chapter.pdf", "thesis")
    gen.export_jsonl("edith_train.jsonl")

    # Then fine-tune:  python fine_tune_sft.py --train edith_train.jsonl

    # Query the hybrid engine
    engine = HybridEngine()
    result = engine.hybrid_query("How does charity density affect SNAP?")
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("edith.hybrid")


# ═══════════════════════════════════════════════════════════════════
# §1: TONE TRAINING GENERATOR — "Distill Your Academic Voice"
# ═══════════════════════════════════════════════════════════════════

# The system prompt that defines Winnie's fine-tuned persona
WINNIE_SYSTEM_PROMPT = (
    "You are Winnie, a rigorous AI research assistant specializing in "
    "American Political Science, welfare policy, and causal inference. "
    "You speak with academic precision, use author-date citations, "
    "structure arguments with clear topic sentences, and always "
    "anticipate committee objections. Your tone is confident but "
    "appropriately hedged — you never overstate significance. "
    "You think in terms of causal mechanisms, not just correlations."
)


class ToneTrainingGenerator:
    """Extract Q/A pairs from your best academic writing for fine-tuning.

    Takes: comp exam answers, thesis chapters, polished papers
    Produces: OpenAI-format JSONL that teaches Winnie YOUR voice

    The model learns:
        - Your vocabulary (e.g., "submerged state" not "hidden programs")
        - Your argument structure (claim → evidence → caveat → implication)
        - Your department's rigor expectations
        - Your citation patterns (Mettler 2011; Hacker & Pierson 2010)
    """

    def __init__(self):
        self._samples: list[dict] = []
        self._qa_pairs: list[dict] = []

    def ingest_sample(self, filepath: str, doc_type: str = "general") -> dict:
        """Ingest an academic writing sample.

        Args:
            filepath: Path to PDF, txt, or md file
            doc_type: "exam_answer", "thesis", "paper", "memo"
        """
        text = self._read_file(filepath)
        if not text or len(text) < 100:
            return {"status": "too_short", "file": filepath}

        self._samples.append({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "doc_type": doc_type,
            "text": text,
            "word_count": len(text.split()),
            "ingested_at": datetime.now().isoformat(),
        })

        log.info(
            f"§HYBRID: Ingested {os.path.basename(filepath)} "
            f"({len(text.split())} words, type={doc_type})"
        )

        return {
            "status": "ingested",
            "file": os.path.basename(filepath),
            "word_count": len(text.split()),
            "doc_type": doc_type,
        }

    def _read_file(self, filepath: str) -> str:
        """Extract text from various file formats."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext in (".txt", ".md"):
            try:
                with open(filepath, "r", errors="replace") as f:
                    return f.read()
            except Exception:
                return ""

        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except Exception:
                return ""

        return ""

    def generate_qa_pairs(
        self,
        model_chain: list[str] = None,
    ) -> dict:
        """Generate question-answer pairs from ingested samples.

        Uses LLM to extract natural Q/A pairs that capture your voice.
        Falls back to rule-based extraction if LLM unavailable.
        """
        model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
        total_generated = 0

        for sample in self._samples:
            text = sample["text"]
            doc_type = sample["doc_type"]

            # Chunk into manageable sections
            sections = self._split_into_sections(text)

            for section in sections:
                if len(section.split()) < 50:
                    continue

                pairs = self._extract_pairs_from_section(
                    section, doc_type, model_chain,
                )
                self._qa_pairs.extend(pairs)
                total_generated += len(pairs)

        log.info(f"§HYBRID: Generated {total_generated} Q/A pairs from {len(self._samples)} samples")

        return {
            "total_pairs": len(self._qa_pairs),
            "new_pairs": total_generated,
            "samples_processed": len(self._samples),
        }

    def _split_into_sections(self, text: str) -> list[str]:
        """Split text into logical sections (by headers, paragraphs)."""
        # Try splitting by headers first
        header_pattern = re.compile(
            r'\n(?:#{1,3}\s|[A-Z][A-Z\s]{3,}\n|(?:Chapter|Section)\s+\d)',
        )
        sections = header_pattern.split(text)

        if len(sections) < 3:
            # Fall back to paragraph groups (~500 words each)
            words = text.split()
            sections = []
            for i in range(0, len(words), 400):
                chunk = " ".join(words[i:i + 500])
                sections.append(chunk)

        return [s.strip() for s in sections if len(s.strip()) > 50]

    def _extract_pairs_from_section(
        self,
        section: str,
        doc_type: str,
        model_chain: list[str],
    ) -> list[dict]:
        """Extract Q/A pairs from a single section."""
        pairs = []

        # Try LLM extraction
        try:
            from server.backend_logic import generate_text_via_chain

            prompt = (
                f"ACADEMIC TEXT ({doc_type}):\n{section[:1500]}\n\n"
                f"Extract 2-3 question-answer pairs from this text that capture:\n"
                f"1. The SPECIFIC vocabulary and phrasing used\n"
                f"2. The argument structure (how claims are supported)\n"
                f"3. The citation style used\n\n"
                f"Format each pair as:\n"
                f"Q: [A question a researcher might ask]\n"
                f"A: [Answer that mirrors the text's exact tone and rigor]\n\n"
                f"The answers should sound like the ORIGINAL AUTHOR, "
                f"not a generic AI. Preserve their academic voice."
            )

            response, _ = generate_text_via_chain(
                prompt, model_chain,
                system_instruction=(
                    "Extract training pairs that preserve the author's "
                    "unique academic voice, vocabulary, and argument style."
                ),
                temperature=0.3,
            )

            # Parse Q/A pairs from response
            qa_blocks = re.split(r'\nQ:\s*', response)
            for block in qa_blocks[1:]:  # Skip first empty split
                parts = re.split(r'\nA:\s*', block, maxsplit=1)
                if len(parts) == 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                    if len(question) > 10 and len(answer) > 30:
                        pairs.append({
                            "question": question,
                            "answer": answer,
                            "doc_type": doc_type,
                            "source": "llm_extracted",
                        })

        except Exception:
            # Fallback: rule-based extraction
            pairs.extend(self._rule_based_pairs(section, doc_type))

        return pairs

    def _rule_based_pairs(self, section: str, doc_type: str) -> list[dict]:
        """Fallback: extract pairs using patterns."""
        pairs = []
        sentences = re.split(r'(?<=[.!?])\s+', section)

        for i, sent in enumerate(sentences):
            # Find claim sentences (contain author citations)
            if re.search(r'\(\w+\s+\d{4}\)', sent) and len(sent) > 50:
                # Use preceding sentence as context
                context = sentences[i - 1] if i > 0 else ""
                question = f"What does the literature say about {context[:60].lower().rstrip('.')}?"
                pairs.append({
                    "question": question,
                    "answer": sent.strip(),
                    "doc_type": doc_type,
                    "source": "rule_based",
                })

            # Find methodological statements
            if re.search(r'(?:we use|I employ|this paper uses|the model)', sent, re.I):
                pairs.append({
                    "question": "What method would you use to test this?",
                    "answer": sent.strip(),
                    "doc_type": doc_type,
                    "source": "rule_based",
                })

        return pairs[:3]  # Limit per section

    def export_jsonl(self, output_path: str = "") -> dict:
        """Export Q/A pairs as OpenAI fine-tuning JSONL.

        Format: {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
        """
        if not self._qa_pairs:
            return {"status": "empty", "reason": "No Q/A pairs generated. Run generate_qa_pairs() first."}

        if not output_path:
            data_root = os.environ.get("EDITH_DATA_ROOT", ".")
            output_path = os.path.join(data_root, "edith_tone_train.jsonl")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        written = 0
        with open(output_path, "w") as f:
            for pair in self._qa_pairs:
                entry = {
                    "messages": [
                        {"role": "system", "content": WINNIE_SYSTEM_PROMPT},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]},
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

        log.info(f"§HYBRID: Exported {written} training pairs to {output_path}")

        return {
            "status": "exported",
            "path": output_path,
            "pairs": written,
            "next_step": f"python fine_tune_sft.py --train {output_path} --suffix winnie-tone",
        }

    def get_stats(self) -> dict:
        """Get summary of ingested samples and generated pairs."""
        return {
            "samples": len(self._samples),
            "total_words": sum(s["word_count"] for s in self._samples),
            "qa_pairs": len(self._qa_pairs),
            "by_type": {
                doc_type: len([p for p in self._qa_pairs if p["doc_type"] == doc_type])
                for doc_type in set(p["doc_type"] for p in self._qa_pairs)
            } if self._qa_pairs else {},
        }


# ═══════════════════════════════════════════════════════════════════
# §2: TWO-STAGE HYBRID ENGINE — RAG + Fine-Tuned Synthesis
# ═══════════════════════════════════════════════════════════════════

class HybridEngine:
    """The "Stark Industries" two-stage query engine.

    Stage 1 (Retrieval):
        Query ChromaDB on Bolt at 3,100 MB/s.
        Pull snippets from VAULT + PEDAGOGY ancestral nodes.

    Stage 2 (Synthesis):
        Pass snippets through fine-tuned model as grounding context.
        Response sounds like Abby on her best day, backed by owned facts.

    Usage:
        engine = HybridEngine()
        result = engine.hybrid_query("How does charity density affect SNAP?")
    """

    def __init__(
        self,
        fine_tuned_model: str = "",
        base_model: str = "",
        chroma_dir: str = "",
    ):
        self._ft_model = fine_tuned_model or os.environ.get(
            "EDITH_FT_MODEL", "ft:gpt-4o-mini-2024-07-18:personal:winnie-v1:D9xqwC8p",
        )
        self._base_model = base_model or os.environ.get(
            "EDITH_MODEL", "gemini-2.5-flash",
        )
        self._chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")

    def hybrid_query(
        self,
        question: str,
        top_k: int = 8,
        include_pedagogy: bool = True,
        model_override: str = "",
    ) -> dict:
        """Two-stage hybrid query — the main entry point.

        Stage 1: Retrieve from Bolt (facts + citations)
        Stage 2: Synthesize through fine-tuned model (tone + logic)
        """
        start_time = time.time()

        # ── Stage 1: RETRIEVAL (The Library) ──────────────────────
        retrieval = self._stage_1_retrieve(question, top_k, include_pedagogy)

        retrieval_ms = round((time.time() - start_time) * 1000, 1)

        # ── Stage 2: SYNTHESIS (The Brain) ────────────────────────
        synthesis = self._stage_2_synthesize(
            question,
            retrieval["snippets"],
            retrieval.get("ancestral_context", ""),
            model_override,
        )

        total_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "question": question,
            "response": synthesis["response"],
            "citations": retrieval["citations"],
            "model_used": synthesis["model"],
            "stage_1_retrieval_ms": retrieval_ms,
            "stage_2_synthesis_ms": total_ms - retrieval_ms,
            "total_ms": total_ms,
            "snippets_retrieved": len(retrieval["snippets"]),
            "ancestral_grounding": retrieval.get("ancestral_found", False),
            "hybrid": True,
        }

    def _stage_1_retrieve(
        self,
        question: str,
        top_k: int = 8,
        include_pedagogy: bool = True,
    ) -> dict:
        """Stage 1: Retrieve relevant snippets from the Bolt."""
        snippets = []
        citations = []
        ancestral_context = ""
        ancestral_found = False

        # 1. Search main VAULT collections
        try:
            from server.infrastructure import parallel_retrieve
            vault_results = parallel_retrieve(
                query=question,
                chroma_dir=self._chroma_dir,
                top_k=top_k,
            )

            for result in vault_results.get("results", []):
                snippets.append({
                    "text": result.get("text", ""),
                    "source": result.get("metadata", {}).get("source", "Unknown"),
                    "score": result.get("score", 0),
                    "collection": result.get("collection", ""),
                })
                # Build citation
                meta = result.get("metadata", {})
                citations.append({
                    "source": meta.get("source", ""),
                    "page": meta.get("page", ""),
                    "collection": result.get("collection", ""),
                })
        except Exception as e:
            log.debug(f"§HYBRID: VAULT retrieval failed: {e}")

        # 2. Search PEDAGOGY ancestral nodes (if enabled)
        if include_pedagogy:
            try:
                from server.chroma_backend import _get_client
                from server.index_pedagogy import PedagogicalIndexer

                if self._chroma_dir:
                    client = _get_client(self._chroma_dir)
                    try:
                        collection = client.get_collection(
                            PedagogicalIndexer.COLLECTION_NAME,
                        )
                        results = collection.query(
                            query_texts=[question],
                            n_results=3,
                        )

                        if results and results["documents"]:
                            ancestral_found = True
                            for doc, meta in zip(
                                results["documents"][0],
                                results["metadatas"][0],
                            ):
                                ancestral_context += f"\n[ANCESTRAL — {meta.get('filename', '')}]: {doc[:300]}"
                                snippets.append({
                                    "text": doc,
                                    "source": meta.get("source", ""),
                                    "score": 1.0,  # Ancestral = high priority
                                    "collection": "pedagogy_ancestral",
                                    "ancestral": True,
                                })
                    except Exception:
                        pass
            except Exception:
                pass

        return {
            "snippets": snippets,
            "citations": citations,
            "ancestral_context": ancestral_context,
            "ancestral_found": ancestral_found,
        }

    def _stage_2_synthesize(
        self,
        question: str,
        snippets: list[dict],
        ancestral_context: str = "",
        model_override: str = "",
    ) -> dict:
        """Stage 2: Synthesize through fine-tuned model."""
        # Build grounding context from retrieved snippets
        grounding = []
        for i, snippet in enumerate(snippets[:6]):  # Top 6 snippets
            source = snippet.get("source", "Unknown")
            text = snippet.get("text", "")[:400]
            is_ancestral = snippet.get("ancestral", False)
            tag = "[ANCESTRAL]" if is_ancestral else f"[SOURCE {i+1}]"
            grounding.append(f"{tag} {os.path.basename(str(source))}: {text}")

        grounding_text = "\n\n".join(grounding)

        # Determine model — prefer fine-tuned for tone
        model = model_override or self._ft_model

        # Try fine-tuned OpenAI model first
        try:
            from pipelines.dual_brain import query_openai
            response = query_openai(
                prompt=(
                    f"RETRIEVED SOURCES:\n{grounding_text}\n\n"
                    f"{ancestral_context}\n\n"
                    f"QUESTION: {question}\n\n"
                    f"Using ONLY the sources above, provide a rigorous academic "
                    f"answer. Cite specific sources by name. If ancestral "
                    f"(syllabus/exam) sources are present, calibrate your rigor "
                    f"to match that academic standard."
                ),
                system=WINNIE_SYSTEM_PROMPT,
            )

            if response and not response.startswith("Error"):
                return {"response": response, "model": model}
        except Exception:
            pass

        # Fallback: Gemini base model
        try:
            from server.backend_logic import generate_text_via_chain
            response, used_model = generate_text_via_chain(
                (
                    f"RETRIEVED SOURCES:\n{grounding_text}\n\n"
                    f"{ancestral_context}\n\n"
                    f"QUESTION: {question}\n\n"
                    f"Using ONLY the sources above, provide a rigorous academic "
                    f"answer. Cite specific sources by name."
                ),
                [self._base_model],
                system_instruction=WINNIE_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return {"response": response, "model": used_model}
        except Exception as e:
            return {
                "response": (
                    f"[Hybrid synthesis unavailable: {e}]\n\n"
                    f"Retrieved {len(snippets)} source snippets. "
                    f"Sources:\n" + "\n".join(
                        f"  - {s.get('source', 'Unknown')}" for s in snippets[:5]
                    )
                ),
                "model": "fallback",
            }

    def get_status(self) -> dict:
        """Report hybrid engine configuration."""
        return {
            "fine_tuned_model": self._ft_model,
            "base_model": self._base_model,
            "chroma_dir": self._chroma_dir or "(not set)",
            "pedagogy_available": bool(self._chroma_dir),
            "speculative_decoding": os.environ.get("USE_SPECULATIVE_DECODING", "false").lower() == "true",
            "mode": "hybrid",
        }


# ═══════════════════════════════════════════════════════════════════
# §M4-2: Speculative Decoding — Draft Locally, Verify via API
# ═══════════════════════════════════════════════════════════════════

_USE_SPECULATIVE = os.environ.get("USE_SPECULATIVE_DECODING", "false").lower() == "true"


def speculative_generate(
    question: str,
    system_instruction: str = "",
    max_tokens: int = 1024,
    draft_tokens: int = 64,
) -> dict:
    """§M4-2: Speculative decoding for the M4 Pro.

    Strategy:
        1. Local MLX model (phi-3 or Qwen) generates a fast draft
        2. Gemini API verifies/corrects the draft
        3. Returns the verified response

    This achieves 1.5-2x perceived speedup because:
        - Draft generation is ~60 tok/s on M4 (instant)
        - Verification prompt is smaller than full generation prompt
        - API call is "fix this" not "write from scratch"

    Falls back to standard generation if:
        - USE_SPECULATIVE_DECODING != true
        - Local model unavailable
        - Draft generation fails
    """
    if not _USE_SPECULATIVE:
        return {"method": "standard", "response": None}

    t0 = time.time()

    # Step 1: Fast local draft
    try:
        from server import mlx_inference
        if not mlx_inference.is_available():
            return {"method": "standard", "response": None}

        draft = mlx_inference.generate(
            prompt=question,
            system_instruction=system_instruction or WINNIE_SYSTEM_PROMPT,
            max_tokens=draft_tokens,
            temperature=0.2,
        )

        if not draft or len(draft.strip()) < 20:
            return {"method": "standard", "response": None}

        draft_ms = round((time.time() - t0) * 1000, 1)
        log.info(f"§M4-2: Draft generated in {draft_ms}ms ({len(draft.split())} words)")

    except Exception as e:
        log.debug(f"§M4-2: Draft failed: {e}")
        return {"method": "standard", "response": None}

    # Step 2: API verification — "correct this draft"
    try:
        from server.backend_logic import generate_text_via_chain

        verify_prompt = (
            f"A fast local model drafted this response to the question below. "
            f"Verify the draft for accuracy, improve its academic rigor, and "
            f"expand where needed. Keep the draft's structure if it's sound.\n\n"
            f"QUESTION: {question}\n\n"
            f"DRAFT ANSWER:\n{draft}\n\n"
            f"VERIFIED ANSWER:"
        )

        base_model = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
        response, model_used = generate_text_via_chain(
            verify_prompt,
            [base_model],
            system_instruction=system_instruction or WINNIE_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=max_tokens,
        )

        total_ms = round((time.time() - t0) * 1000, 1)
        log.info(f"§M4-2: Speculative decode complete in {total_ms}ms "
                 f"(draft={draft_ms}ms, verify={total_ms - draft_ms}ms)")

        return {
            "method": "speculative",
            "response": response,
            "model_draft": mlx_inference.get_model_info().get("model", "local"),
            "model_verify": model_used,
            "draft_ms": draft_ms,
            "verify_ms": total_ms - draft_ms,
            "total_ms": total_ms,
        }
    except Exception as e:
        log.warning(f"§M4-2: Verification failed, using draft: {e}")
        return {
            "method": "speculative_draft_only",
            "response": draft,
            "total_ms": round((time.time() - t0) * 1000, 1),
        }


# Global instances
tone_generator = ToneTrainingGenerator()
hybrid_engine = HybridEngine()

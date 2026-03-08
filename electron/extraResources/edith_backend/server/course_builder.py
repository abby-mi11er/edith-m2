"""
Course Builder -- Syllabus Parser and Course Construction
=========================================================
Accepts uploaded syllabus files (PDF/DOCX/TXT), extracts text,
uses the LLM to parse readings and topics, and builds a structured
course that E.D.I.T.H. can teach from.
"""
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.course_builder")

# ---------------------------------------------------------------
# Text extraction (reuses existing patterns from forensic_audit)
# ---------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT files."""
    ext = Path(file_path).suffix.lower()

    if ext == ".txt" or ext == ".md":
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        return _extract_pdf(file_path)

    if ext in (".docx", ".doc"):
        return _extract_docx(file_path)

    # Fallback: try reading as text
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz
        doc = fitz.open(path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        log.warning("PyMuPDF not installed -- cannot parse PDF syllabus")
        return ""
    except Exception as e:
        log.error(f"PDF extraction failed: {e}")
        return ""


def _extract_docx(path: str) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        log.warning("python-docx not installed -- cannot parse DOCX syllabus")
        return ""
    except Exception as e:
        log.error(f"DOCX extraction failed: {e}")
        return ""


# ---------------------------------------------------------------
# LLM-based syllabus parsing
# ---------------------------------------------------------------

SYLLABUS_PARSE_PROMPT = """You are an academic course parser. Given the raw text of a course syllabus, extract ALL information into the following JSON structure. Be thorough -- capture every reading, every topic, every week.

Return ONLY valid JSON, no markdown fences, no commentary.

{
  "course_title": "string",
  "instructor": "string or null",
  "term": "string or null (e.g. Spring 2026)",
  "institution": "string or null",
  "description": "one-paragraph course description",
  "weeks": [
    {
      "week": 1,
      "title": "Week topic title",
      "topics": ["topic1", "topic2"],
      "readings": [
        {
          "id": "r1_1",
          "author": "Last, First",
          "title": "Full title of the reading",
          "year": 2020,
          "type": "required",
          "source": "journal/book name or null",
          "chapter": "chapter number or pages if specified, or null",
          "notes": "any instructor notes about this reading or null"
        }
      ]
    }
  ],
  "assignments": [
    {
      "title": "Assignment name",
      "due_week": 5,
      "type": "paper/exam/presentation/other",
      "description": "brief description"
    }
  ],
  "policies": "brief summary of grading/attendance policies or null"
}

Rules:
- Number weeks sequentially starting from 1
- If the syllabus uses dates instead of week numbers, convert to week numbers
- Mark readings as "required" or "recommended" based on syllabus language
- Include ALL readings, even if repeated across weeks
- If a reading has no author, use "Unknown"
- Generate a unique id for each reading (e.g. r1_1 for week 1, reading 1)
"""


def parse_syllabus(raw_text: str, model_chain: Optional[list[str]] = None) -> dict:
    """Parse syllabus text using the LLM. Returns structured course data."""
    if not raw_text.strip():
        return {"error": "Empty syllabus text"}

    # Truncate very long syllabi to fit context window
    max_chars = 50_000
    text = raw_text[:max_chars]
    if len(raw_text) > max_chars:
        text += "\n\n[TRUNCATED -- syllabus exceeds 50K characters]"

    try:
        from server.backend_logic import generate_text_via_chain

        chain = model_chain or _default_chain()
        prompt = f"Parse this syllabus:\n\n{text}"

        answer, model_used = generate_text_via_chain(
            prompt_text=prompt,
            model_chain=chain,
            system_instruction=SYLLABUS_PARSE_PROMPT,
            temperature=0.05,
        )

        # Strip markdown fences if the LLM included them
        answer = answer.strip()
        if answer.startswith("```"):
            answer = re.sub(r"^```(?:json)?\s*\n?", "", answer)
            answer = re.sub(r"\n?```\s*$", "", answer)

        course_data = json.loads(answer)
        course_data["_model"] = model_used
        course_data["_char_count"] = len(raw_text)

        # Validate structure
        if "weeks" not in course_data:
            course_data["weeks"] = []
        if "course_title" not in course_data:
            course_data["course_title"] = "Untitled Course"

        # Count total readings
        total_readings = sum(len(w.get("readings", [])) for w in course_data["weeks"])
        course_data["_total_readings"] = total_readings
        course_data["_total_weeks"] = len(course_data["weeks"])

        log.info(
            f"Parsed syllabus: {course_data['course_title']} -- "
            f"{total_readings} readings across {len(course_data['weeks'])} weeks "
            f"(model: {model_used})"
        )
        return course_data

    except json.JSONDecodeError as e:
        log.error(f"LLM returned invalid JSON: {e}")
        return {"error": f"Failed to parse LLM output as JSON: {str(e)}"}
    except Exception as e:
        log.error(f"Syllabus parsing failed: {e}")
        return {"error": str(e)}


def match_readings_to_library(course_data: dict) -> dict:
    """Match each reading in the course against the indexed library.

    Returns the course data with each reading annotated with:
    - matched: bool
    - sha256: string or null (if matched)
    - library_title: string or null (matched title from library)
    """
    try:
        from server.chroma_backend import retrieve_local_sources
    except ImportError:
        log.warning("chroma_backend not available -- skipping library matching")
        return course_data

    matched_count = 0
    total = 0

    for week in course_data.get("weeks", []):
        for reading in week.get("readings", []):
            total += 1
            query = f"{reading.get('author', '')} {reading.get('title', '')} {reading.get('year', '')}"
            query = query.strip()

            if not query:
                reading["matched"] = False
                reading["sha256"] = None
                continue

            try:
                results = retrieve_local_sources(query, top_k=3)
                # Check if any result is a close title match
                best_match = None
                reading_title_lower = reading.get("title", "").lower()

                for r in results:
                    r_title = (r.get("title") or r.get("filename") or "").lower()
                    # Fuzzy title match: check if significant portion overlaps
                    if reading_title_lower and r_title:
                        words = set(reading_title_lower.split())
                        r_words = set(r_title.split())
                        overlap = len(words & r_words) / max(len(words), 1)
                        if overlap > 0.5:
                            best_match = r
                            break

                if best_match:
                    reading["matched"] = True
                    reading["sha256"] = best_match.get("sha256") or best_match.get("id")
                    reading["library_title"] = best_match.get("title")
                    matched_count += 1
                else:
                    reading["matched"] = False
                    reading["sha256"] = None

            except Exception as e:
                log.debug(f"Library match failed for '{query}': {e}")
                reading["matched"] = False
                reading["sha256"] = None

    course_data["_matched_readings"] = matched_count
    course_data["_unmatched_readings"] = total - matched_count
    log.info(f"Library matching: {matched_count}/{total} readings matched")
    return course_data


# ---------------------------------------------------------------
# Course persistence (JSON file in vault)
# ---------------------------------------------------------------

def _courses_dir() -> Path:
    """Get or create the courses directory."""
    vault = os.environ.get("VAULT_ROOT") or os.environ.get("EDITH_DATA_ROOT", ".")
    d = Path(vault) / ".edith" / "courses"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_course(course_data: dict, filename: str = "active_course.json") -> str:
    """Save course data to disk. Returns the file path."""
    path = _courses_dir() / filename
    path.write_text(json.dumps(course_data, indent=2, default=str), encoding="utf-8")
    log.info(f"Course saved to {path}")
    return str(path)


def load_active_course() -> Optional[dict]:
    """Load the active course from disk."""
    path = _courses_dir() / "active_course.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Failed to load active course: {e}")
        return None


def list_courses() -> list[dict]:
    """List all saved courses."""
    courses = []
    for f in _courses_dir().glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            courses.append({
                "filename": f.name,
                "title": data.get("course_title", "Untitled"),
                "term": data.get("term"),
                "weeks": data.get("_total_weeks", 0),
                "readings": data.get("_total_readings", 0),
            })
        except Exception:
            pass
    return courses


def _default_chain() -> list[str]:
    """Get the default model chain."""
    default = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
    fallbacks = os.environ.get("EDITH_MODEL_FALLBACKS", "gemini-2.5-flash,gemini-2.0-flash")
    chain = [default] + [m.strip() for m in fallbacks.split(",") if m.strip() and m.strip() != default]
    return chain

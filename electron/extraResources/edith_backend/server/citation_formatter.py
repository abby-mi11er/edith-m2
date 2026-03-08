"""
Citation Formatter — Converts [S1]/[S2] markers into proper academic citations.

Supports APA and BibTeX styles. Parses author/year/title from filenames
and chunk metadata.
"""

import re
import json
from pathlib import Path


# Pattern to extract author/year from common academic filename formats:
#   "Smith_2019_Electoral_Systems.pdf" → ("Smith", "2019", "Electoral Systems")
#   "Smith & Jones (2020) - Title.pdf" → ("Smith & Jones", "2020", "Title")
#   "doe2021.pdf" → ("Doe", "2021", "")
FILENAME_PATTERNS = [
    # Author_Year_Title.pdf
    re.compile(r"^(?P<author>[A-Z][a-z]+(?:_[A-Z][a-z]+)*)_(?P<year>\d{4})_(?P<title>.+?)\.pdf$", re.I),
    # Author & Author (Year) - Title.pdf
    re.compile(r"^(?P<author>.+?)\s*\((?P<year>\d{4})\)\s*[-–—]\s*(?P<title>.+?)\.pdf$", re.I),
    # Author (Year).pdf
    re.compile(r"^(?P<author>.+?)\s*\((?P<year>\d{4})\)\.pdf$", re.I),
    # AuthorYear.pdf
    re.compile(r"^(?P<author>[A-Z][a-z]+)(?P<year>\d{4})\.pdf$", re.I),
]


def parse_citation_from_filename(filename: str) -> dict:
    """Extract author, year, title from a filename."""
    name = Path(filename).stem
    for pat in FILENAME_PATTERNS:
        m = pat.match(filename)
        if m:
            author = m.group("author").replace("_", " ").strip()
            year = m.group("year")
            title = m.group("title").replace("_", " ").strip() if "title" in m.groupdict() else ""
            return {"author": author, "year": year, "title": title}

    # Fallback: just use the filename
    return {"author": name, "year": "", "title": ""}


def format_apa(citation: dict) -> str:
    """Format as APA: Author (Year). Title."""
    author = citation.get("author", "Unknown")
    year = citation.get("year", "n.d.")
    title = citation.get("title", "")
    if title:
        return f"{author} ({year}). {title}."
    return f"{author} ({year})."


def format_bibtex(citation: dict, key: str = "ref") -> str:
    """Format as BibTeX entry."""
    author = citation.get("author", "Unknown")
    year = citation.get("year", "")
    title = citation.get("title", "Untitled")
    bib_key = re.sub(r"[^a-zA-Z0-9]", "", f"{author}{year}")
    return (
        f"@article{{{bib_key},\n"
        f"  author = {{{author}}},\n"
        f"  year = {{{year}}},\n"
        f"  title = {{{title}}}\n"
        f"}}"
    )


def format_inline_apa(citation: dict) -> str:
    """Format as inline APA: (Author, Year)."""
    author = citation.get("author", "Unknown")
    year = citation.get("year", "n.d.")
    return f"({author}, {year})"


def replace_source_markers(text: str, sources: list, style: str = "apa") -> str:
    """Replace [S1], [S2] etc. with proper citations.

    Args:
        text: The answer text with [S1], [S2] markers
        sources: List of source dicts with 'filename' or 'source' keys
        style: 'apa' or 'bibtex'

    Returns:
        Text with markers replaced by formatted citations
    """
    if not sources:
        return text

    citation_map = {}
    for i, src in enumerate(sources, start=1):
        filename = ""
        if isinstance(src, dict):
            filename = src.get("filename") or src.get("source") or ""
        if not filename:
            continue
        citation = parse_citation_from_filename(Path(filename).name)
        if style == "apa":
            citation_map[f"[S{i}]"] = format_inline_apa(citation)
        else:
            citation_map[f"[S{i}]"] = format_inline_apa(citation)

    for marker, replacement in citation_map.items():
        text = text.replace(marker, replacement)

    return text


def generate_bibliography(sources: list, style: str = "apa") -> str:
    """Generate a formatted bibliography from source list.

    Returns:
        Formatted bibliography string
    """
    if not sources:
        return ""

    seen = set()
    entries = []
    for src in sources:
        filename = ""
        if isinstance(src, dict):
            filename = src.get("filename") or src.get("source") or ""
        if not filename or filename in seen:
            continue
        seen.add(filename)
        citation = parse_citation_from_filename(Path(filename).name)
        if style == "apa":
            entries.append(format_apa(citation))
        elif style == "bibtex":
            entries.append(format_bibtex(citation))

    if style == "bibtex":
        return "\n\n".join(sorted(entries))
    else:
        entries.sort()
        return "\n".join(f"- {e}" for e in entries)

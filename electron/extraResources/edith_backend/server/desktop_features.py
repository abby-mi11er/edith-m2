#!/usr/bin/env python3
"""
Desktop App Enhancements — Reading List, Source Preview, Research Notebook
============================================================================
App #1: Reading list panel — sidebar showing papers in current conversation
App #2: Source preview drawer — click [S1] to see full paragraph
App #7: Research notebook — persistent notes organized by topic
App #4: Literature map view — connections between cited papers

These are backend API endpoints that the Electron app calls.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional
from collections import defaultdict
from server.vault_config import VAULT_ROOT


# ---------------------------------------------------------------------------
# App #1: Reading List — Papers referenced in conversation
# ---------------------------------------------------------------------------

def extract_reading_list(messages: list[dict], sources_by_msg: dict = None) -> list:
    """Extract all papers referenced across a conversation.
    
    Returns a deduplicated list of sources with frequency counts.
    """
    papers = {}  # title → info

    for i, msg in enumerate(messages):
        # Check embedded sources
        msg_sources = (sources_by_msg or {}).get(str(i), [])
        for s in msg_sources:
            title = s.get("title", s.get("file_name", s.get("filename", "")))
            if not title:
                continue
            key = title.lower().strip()
            if key not in papers:
                papers[key] = {
                    "title": title,
                    "author": s.get("author", ""),
                    "year": s.get("year", ""),
                    "times_cited": 0,
                    "first_mentioned_msg": i,
                    "metadata": {k: v for k, v in s.items()
                                if k in ("doc_type", "academic_topic", "project")},
                }
            papers[key]["times_cited"] += 1

    return sorted(papers.values(), key=lambda x: -x["times_cited"])


# ---------------------------------------------------------------------------
# App #2: Source Preview — Click [S1] to see full context
# ---------------------------------------------------------------------------

def get_source_preview(source_id: str, chroma_dir: str, collection: str,
                       context_chars: int = 2000) -> dict:
    """Get expanded context around a source chunk.
    
    Returns the source chunk plus surrounding text from the same document.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection(collection)

        # Get the source document
        result = col.get(ids=[source_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return {"error": "Source not found"}

        doc = result["documents"][0]
        meta = result["metadatas"][0]

        # Get surrounding chunks from the same file
        file_name = meta.get("file_name", "")
        if file_name:
            # Query for adjacent chunks
            neighbors = col.get(
                where={"file_name": file_name},
                include=["documents", "metadatas"],
                limit=10,
            )

            # Sort by chunk number
            chunks = sorted(
                zip(neighbors["documents"], neighbors["metadatas"]),
                key=lambda x: x[1].get("chunk", 0),
            )

            full_context = "\n\n---\n\n".join(c[0] for c in chunks if c[0])
        else:
            full_context = doc

        return {
            "source_id": source_id,
            "text": doc,
            "full_context": full_context[:context_chars],
            "metadata": meta,
            "file_name": file_name,
            "total_chunks": len(chunks) if file_name else 1,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# App #7: Research Notebook — Persistent notes by topic
# ---------------------------------------------------------------------------

class ResearchNotebook:
    """Persistent research notes organized by project and topic."""

    def __init__(self, store_dir: Path = None):
        self.store_dir = Path(store_dir or str(VAULT_ROOT / "Corpus" / "notebooks"))
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _notebook_path(self, project: str) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_"
                           for c in project).strip().replace(" ", "_")
        return self.store_dir / f"{safe_name}.json"

    def create_notebook(self, project: str, description: str = "") -> dict:
        """Create a new research notebook for a project."""
        path = self._notebook_path(project)
        notebook = {
            "project": project,
            "description": description,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sections": {},
            "notes": [],
        }
        path.write_text(json.dumps(notebook, indent=2))
        return notebook

    def add_note(self, project: str, content: str, section: str = "General",
                 tags: list = None, source: str = "") -> dict:
        """Add a note to a project notebook."""
        path = self._notebook_path(project)
        if not path.exists():
            self.create_notebook(project)

        notebook = json.loads(path.read_text())

        note = {
            "id": hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12],
            "content": content,
            "section": section,
            "tags": tags or [],
            "source": source,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        notebook["notes"].append(note)
        notebook["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Update section index
        if section not in notebook["sections"]:
            notebook["sections"][section] = []
        notebook["sections"][section].append(note["id"])

        path.write_text(json.dumps(notebook, indent=2))
        return note

    def get_notes(self, project: str, section: str = None) -> list:
        """Get notes, optionally filtered by section."""
        path = self._notebook_path(project)
        if not path.exists():
            return []
        notebook = json.loads(path.read_text())
        notes = notebook.get("notes", [])
        if section:
            notes = [n for n in notes if n.get("section") == section]
        return notes

    def list_notebooks(self) -> list:
        """List all project notebooks."""
        notebooks = []
        for path in self.store_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                notebooks.append({
                    "project": data.get("project", path.stem),
                    "description": data.get("description", ""),
                    "note_count": len(data.get("notes", [])),
                    "sections": list(data.get("sections", {}).keys()),
                    "updated": data.get("updated", ""),
                })
            except Exception:
                continue
        return notebooks

    def export_notebook(self, project: str) -> str:
        """Export a notebook as markdown."""
        path = self._notebook_path(project)
        if not path.exists():
            return "Notebook not found."
        notebook = json.loads(path.read_text())

        md = f"# {notebook['project']}\n\n"
        if notebook.get("description"):
            md += f"*{notebook['description']}*\n\n"
        md += f"Updated: {notebook.get('updated', 'N/A')}\n\n---\n\n"

        # Group by section
        by_section = defaultdict(list)
        for note in notebook.get("notes", []):
            by_section[note.get("section", "General")].append(note)

        for section, notes in by_section.items():
            md += f"## {section}\n\n"
            for note in notes:
                md += f"- {note['content']}"
                if note.get("source"):
                    md += f" _{note['source']}_"
                if note.get("tags"):
                    md += f" `{'`, `'.join(note['tags'])}`"
                md += "\n"
            md += "\n"

        return md

    def notify(self, title: str = "E.D.I.T.H.", message: str = "") -> None:
        """Send a native macOS desktop notification via osascript."""
        import subprocess
        safe_title = title.replace('"', '\\"')
        safe_msg = message.replace('"', '\\"')
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{safe_msg}" with title "{safe_title}"'],
                timeout=5, capture_output=True,
            )
        except Exception:
            pass  # Notification is best-effort

    def list_features(self) -> list[str]:
        """List available desktop features."""
        return ["notebooks", "notifications", "reading_list", "source_preview", "literature_map"]




# ---------------------------------------------------------------------------
# App #4: Literature Map — Connections between cited papers
# ---------------------------------------------------------------------------

def build_literature_map(sources: list[dict]) -> dict:
    """Build a connection map between papers cited in a conversation.
    
    Returns a network structure suitable for visualization.
    """
    nodes = []
    edges = []
    author_index = defaultdict(list)
    topic_index = defaultdict(list)

    for i, s in enumerate(sources):
        title = s.get("title", s.get("file_name", f"Source {i+1}"))
        author = s.get("author", "")
        topic = s.get("academic_topic", "")
        year = s.get("year", "")

        node = {
            "id": f"s{i}",
            "title": title,
            "author": author,
            "year": year,
            "topic": topic,
        }
        nodes.append(node)

        if author:
            author_index[author.lower()].append(f"s{i}")
        if topic:
            topic_index[topic.lower()].append(f"s{i}")

    # Create edges for shared authors
    for author, node_ids in author_index.items():
        if len(node_ids) > 1:
            for j in range(len(node_ids)):
                for k in range(j + 1, len(node_ids)):
                    edges.append({
                        "source": node_ids[j],
                        "target": node_ids[k],
                        "type": "same_author",
                        "label": author,
                    })

    # Create edges for shared topics
    for topic, node_ids in topic_index.items():
        if len(node_ids) > 1:
            for j in range(len(node_ids)):
                for k in range(j + 1, len(node_ids)):
                    edges.append({
                        "source": node_ids[j],
                        "target": node_ids[k],
                        "type": "same_topic",
                        "label": topic,
                    })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_papers": len(nodes),
            "unique_authors": len(author_index),
            "unique_topics": len(topic_index),
        },
    }

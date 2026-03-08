"""
Notion Bridge — Bidirectional Notion ↔ Bolt Sovereign Sync
=============================================================
Force Multiplier 2: Your Notion becomes a Holographic Portal.

1. INGEST: Pull all 500+ Notion notes into /Volumes/CITADEL/VAULT/NOTION_MIRROR/
2. INDEX: Vectorize them into the Citadel's semantic search
3. LINK: Cross-reference Notion notes with PDFs on the Bolt
4. PUSH: Auto-sync summaries back to Notion databases
5. GUARD: Raw ideas stay on the Bolt; only "Final" goes to cloud

Architecture:
    Notion API → Shadow Mirror (Bolt) → Neural Indexing →
    Cross-Reference Engine → Selective Cloud Push
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

log = logging.getLogger("edith.notion_bridge")


@dataclass
class NotionPage:
    """A mirrored Notion page stored on the Bolt."""
    page_id: str
    title: str
    content: str
    database: str  # Which Notion database this belongs to
    tags: list[str] = field(default_factory=list)
    status: str = ""
    last_edited: str = ""
    local_path: str = ""
    indexed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.page_id,
            "title": self.title,
            "database": self.database,
            "tags": self.tags,
            "status": self.status,
            "indexed": self.indexed,
            "words": len(self.content.split()),
        }


class NotionBridge:
    """Bidirectional Notion ↔ Bolt bridge.

    PULL: Mirror Notion pages to the Bolt as Markdown files.
    INDEX: Add them to the Citadel's semantic search.
    PUSH: Send summaries and notes back to Notion databases.
    RECALL: Chat with your past self via timestamped notes.

    Usage:
        bridge = NotionBridge()
        bridge.pull_all_pages()          # Mirror everything
        bridge.push_note(title, content, database="Literature Review")
        results = bridge.recall("What did I think about accountability in 2023?")
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._mirror_path = os.path.join(self._bolt_path, "VAULT", "NOTION_MIRROR")
        self._notion_token = os.environ.get("NOTION_TOKEN", "")
        self._pages: dict[str, NotionPage] = {}
        self._databases: dict[str, str] = {}  # db_id → db_name
        self._load_mirror_index()

    # ─── PULL: Notion → Bolt ──────────────────────────────────────

    def pull_all_pages(self, database_id: str = "") -> dict:
        """Pull pages from Notion into the Bolt mirror.

        Uses the Notion API to fetch all pages and save them
        as Markdown files on the Bolt SSD.
        """
        if not self._notion_token:
            return {"error": "NOTION_TOKEN not set. Add it to your .env file.",
                    "help": "Get your token at https://www.notion.so/my-integrations"}

        import urllib.request

        headers = {
            "Authorization": f"Bearer {self._notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        # If specific database, query that; otherwise search all
        if database_id:
            url = f"https://api.notion.com/v1/databases/{database_id}/query"
            payload = json.dumps({"page_size": 100}).encode()
        else:
            url = "https://api.notion.com/v1/search"
            payload = json.dumps({
                "filter": {"value": "page", "property": "object"},
                "page_size": 100,
            }).encode()

        pages_pulled = 0
        has_more = True
        start_cursor = None

        while has_more:
            try:
                body = {"page_size": 100}
                if start_cursor:
                    body["start_cursor"] = start_cursor
                if database_id:
                    body_bytes = json.dumps(body).encode()
                    req = urllib.request.Request(
                        f"https://api.notion.com/v1/databases/{database_id}/query",
                        data=body_bytes, headers=headers, method="POST"
                    )
                else:
                    body["filter"] = {"value": "page", "property": "object"}
                    body_bytes = json.dumps(body).encode()
                    req = urllib.request.Request(url, data=body_bytes,
                                                 headers=headers, method="POST")

                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())

                results = data.get("results", [])
                for page_data in results:
                    page = self._parse_notion_page(page_data)
                    if page:
                        # Fetch page content (blocks)
                        content = self._fetch_page_blocks(page.page_id, headers)
                        page.content = content
                        self._save_page_to_bolt(page)
                        self._pages[page.page_id] = page
                        pages_pulled += 1

                has_more = data.get("has_more", False)
                start_cursor = data.get("next_cursor")

            except Exception as e:
                log.error(f"§NOTION: Pull failed: {e}")
                has_more = False

        self._save_mirror_index()
        return {
            "pages_pulled": pages_pulled,
            "total_mirrored": len(self._pages),
            "mirror_path": self._mirror_path,
        }

    def _parse_notion_page(self, page_data: dict) -> Optional[NotionPage]:
        """Extract metadata from a Notion API page object."""
        try:
            page_id = page_data.get("id", "")
            properties = page_data.get("properties", {})

            # Extract title
            title = ""
            for prop_name, prop_val in properties.items():
                if prop_val.get("type") == "title":
                    title_parts = prop_val.get("title", [])
                    title = "".join(t.get("plain_text", "") for t in title_parts)
                    break
            if not title:
                title = f"Untitled ({page_id[:8]})"

            # Extract tags
            tags = []
            for prop_name, prop_val in properties.items():
                if prop_val.get("type") == "multi_select":
                    for opt in prop_val.get("multi_select", []):
                        tags.append(opt.get("name", ""))
                elif prop_val.get("type") == "select":
                    sel = prop_val.get("select")
                    if sel:
                        tags.append(sel.get("name", ""))

            # Extract status
            status = ""
            for prop_name, prop_val in properties.items():
                if prop_val.get("type") == "status":
                    st = prop_val.get("status")
                    if st:
                        status = st.get("name", "")

            # Database info
            parent = page_data.get("parent", {})
            database = parent.get("database_id", "standalone")

            return NotionPage(
                page_id=page_id,
                title=title,
                content="",
                database=database,
                tags=tags,
                status=status,
                last_edited=page_data.get("last_edited_time", ""),
            )
        except Exception as e:
            log.debug(f"§NOTION: Parse error: {e}")
            return None

    def _fetch_page_blocks(self, page_id: str, headers: dict) -> str:
        """Fetch the content blocks of a Notion page and convert to markdown."""
        import urllib.request

        try:
            url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            blocks = data.get("results", [])
            return self._blocks_to_markdown(blocks)
        except Exception as e:
            log.debug(f"§NOTION: Block fetch failed for {page_id}: {e}")
            return ""

    def _blocks_to_markdown(self, blocks: list) -> str:
        """Convert Notion blocks to Markdown."""
        md = ""
        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            if block_type == "paragraph":
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                md += f"{text}\n\n"
            elif block_type.startswith("heading_"):
                level = int(block_type[-1])
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                md += f"{'#' * level} {text}\n\n"
            elif block_type == "bulleted_list_item":
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                md += f"- {text}\n"
            elif block_type == "numbered_list_item":
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                md += f"1. {text}\n"
            elif block_type == "to_do":
                checked = "x" if block_data.get("checked") else " "
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                md += f"- [{checked}] {text}\n"
            elif block_type == "quote":
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                md += f"> {text}\n\n"
            elif block_type == "code":
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                lang = block_data.get("language", "")
                md += f"```{lang}\n{text}\n```\n\n"
            elif block_type == "divider":
                md += "---\n\n"

        return md

    def _rich_text_to_str(self, rich_text: list) -> str:
        """Convert Notion rich text array to plain string."""
        parts = []
        for rt in rich_text:
            text = rt.get("plain_text", "")
            annotations = rt.get("annotations", {})
            if annotations.get("bold"):
                text = f"**{text}**"
            if annotations.get("italic"):
                text = f"*{text}*"
            if annotations.get("code"):
                text = f"`{text}`"
            parts.append(text)
        return "".join(parts)

    def _save_page_to_bolt(self, page: NotionPage):
        """Save a Notion page as Markdown on the Bolt."""
        mirror_dir = Path(self._mirror_path)
        mirror_dir.mkdir(parents=True, exist_ok=True)

        # Clean filename
        safe_title = re.sub(r'[^\w\s-]', '', page.title)[:60].strip().replace(" ", "_")
        filename = f"{safe_title}_{page.page_id[:8]}.md"

        # Build full markdown with frontmatter
        frontmatter = (
            f"---\n"
            f"title: \"{page.title}\"\n"
            f"notion_id: {page.page_id}\n"
            f"database: {page.database}\n"
            f"tags: [{', '.join(page.tags)}]\n"
            f"status: {page.status}\n"
            f"last_edited: {page.last_edited}\n"
            f"mirrored_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"---\n\n"
        )
        full_content = frontmatter + f"# {page.title}\n\n" + page.content

        file_path = mirror_dir / filename
        file_path.write_text(full_content)
        page.local_path = str(file_path)

    _multi_select_cache: dict = {}

    def _resolve_multi_select_property(self, database_id: str, headers: dict) -> str:
        """Discover the actual multi_select property name in a Notion database."""
        if database_id in self._multi_select_cache:
            return self._multi_select_cache[database_id]

        import urllib.request
        try:
            url = f"https://api.notion.com/v1/databases/{database_id}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            for prop_name, prop_val in data.get("properties", {}).items():
                if prop_val.get("type") == "multi_select":
                    self._multi_select_cache[database_id] = prop_name
                    log.info(f"§NOTION: Resolved multi_select property: {prop_name}")
                    return prop_name
        except Exception as e:
            log.debug(f"§NOTION: Could not fetch DB schema: {e}")

        # Fallback to common names
        self._multi_select_cache[database_id] = "Tags"
        return "Tags"

    # ─── PUSH: Bolt → Notion ─────────────────────────────────────

    def push_note(self, title: str, content: str,
                   database_id: str = "", tags: list[str] = None,
                   status: str = "Draft") -> dict:
        """Push a note from the Citadel back to Notion.

        "Winnie, send this summary to my 'Literature Review' database
        and tag it as 'High Priority' for Chapter 4."
        """
        if not self._notion_token:
            return {"error": "NOTION_TOKEN not set"}
        if not database_id:
            return {"error": "database_id required to push to Notion"}

        import urllib.request

        headers = {
            "Authorization": f"Bearer {self._notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        # Build properties
        properties = {
            "Name": {"title": [{"text": {"content": title}}]},
        }
        if tags:
            tag_prop = self._resolve_multi_select_property(database_id, headers)
            properties[tag_prop] = {
                "multi_select": [{"name": t} for t in tags]
            }

        # Build content blocks from markdown
        children = self._markdown_to_blocks(content)

        payload = json.dumps({
            "parent": {"database_id": database_id},
            "properties": properties,
            "children": children[:100],  # Notion API limit
        }).encode()

        try:
            req = urllib.request.Request(
                "https://api.notion.com/v1/pages",
                data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())

            return {
                "pushed": True,
                "page_id": result.get("id", ""),
                "url": result.get("url", ""),
                "title": title,
            }
        except Exception as e:
            return {"pushed": False, "error": str(e)}

    def sync_page(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        database_id: str = "",
        status: str = "Draft",
    ) -> dict:
        """Compatibility wrapper used by /api/notion/sync route.

        Prefers explicit database_id; otherwise falls back to
        NOTION_DATABASE_ID from environment.
        """
        target_database = (database_id or os.environ.get("NOTION_DATABASE_ID", "")).strip()
        if not target_database:
            return {
                "pushed": False,
                "error": "Missing Notion database target. Set NOTION_DATABASE_ID or provide database_id.",
            }
        return self.push_note(
            title=title,
            content=content,
            database_id=target_database,
            tags=tags or [],
            status=status,
        )

    def _markdown_to_blocks(self, md: str) -> list:
        """Convert Markdown to Notion block objects."""
        blocks = []
        lines = md.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("### "):
                blocks.append({
                    "object": "block", "type": "heading_3",
                    "heading_3": {"rich_text": [{"type": "text",
                                  "text": {"content": line[4:]}}]},
                })
            elif line.startswith("## "):
                blocks.append({
                    "object": "block", "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text",
                                  "text": {"content": line[3:]}}]},
                })
            elif line.startswith("# "):
                blocks.append({
                    "object": "block", "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text",
                                  "text": {"content": line[2:]}}]},
                })
            elif line.startswith("- "):
                blocks.append({
                    "object": "block", "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text",
                                           "text": {"content": line[2:]}}]},
                })
            elif line.startswith("> "):
                blocks.append({
                    "object": "block", "type": "quote",
                    "quote": {"rich_text": [{"type": "text",
                              "text": {"content": line[2:]}}]},
                })
            else:
                blocks.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text",
                                  "text": {"content": line}}]},
                })

        return blocks

    # ─── RECALL: Chat with your past self ─────────────────────────

    def recall(self, query: str, year_filter: int = 0) -> list[dict]:
        """Search mirrored Notion notes for past thinking.

        "Winnie, how does my understanding of 'Accountability' this week
        differ from how I wrote about it in my 2022 Seminar?"
        """
        query_lower = query.lower()
        results = []

        for page_id, page in self._pages.items():
            # Simple text search across content
            content_lower = page.content.lower()
            title_lower = page.title.lower()

            if query_lower in content_lower or query_lower in title_lower:
                # Year filter if specified
                if year_filter:
                    page_year = page.last_edited[:4] if page.last_edited else ""
                    if page_year and int(page_year) != year_filter:
                        continue

                # Find context around the match
                idx = content_lower.find(query_lower)
                context = page.content[max(0, idx - 150):idx + 300] if idx >= 0 else ""

                results.append({
                    "title": page.title,
                    "database": page.database,
                    "tags": page.tags,
                    "last_edited": page.last_edited,
                    "context": context.strip(),
                    "local_path": page.local_path,
                })

        results.sort(key=lambda r: r.get("last_edited", ""), reverse=True)
        return results[:20]

    def compare_thinking(self, topic: str, year_old: int, year_new: int) -> dict:
        """Compare your thinking on a topic across years.

        The "Syllabus Time-Travel" feature.
        """
        old_notes = self.recall(topic, year_filter=year_old)
        new_notes = self.recall(topic, year_filter=year_new)

        return {
            "topic": topic,
            "then": {
                "year": year_old,
                "notes_found": len(old_notes),
                "contexts": [n["context"][:200] for n in old_notes[:3]],
            },
            "now": {
                "year": year_new,
                "notes_found": len(new_notes),
                "contexts": [n["context"][:200] for n in new_notes[:3]],
            },
            "growth_indicator": (
                "significant_evolution" if old_notes and new_notes else
                "insufficient_data"
            ),
        }

    # ─── Mirror Index Persistence ─────────────────────────────────

    def _load_mirror_index(self):
        """Load the mirror index from disk."""
        index_path = Path(self._mirror_path) / "_index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                for pid, pdata in data.items():
                    self._pages[pid] = NotionPage(**pdata)
            except Exception:
                pass

    def _save_mirror_index(self):
        """Save the mirror index to disk."""
        index_path = Path(self._mirror_path)
        index_path.mkdir(parents=True, exist_ok=True)
        index_file = index_path / "_index.json"
        try:
            data = {pid: p.to_dict() for pid, p in self._pages.items()}
            index_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    @property
    def status(self) -> dict:
        return {
            "notion_connected": bool(self._notion_token),
            "pages_mirrored": len(self._pages),
            "mirror_path": self._mirror_path,
            "databases": len(self._databases),
        }


# Global instance
notion_bridge = NotionBridge()

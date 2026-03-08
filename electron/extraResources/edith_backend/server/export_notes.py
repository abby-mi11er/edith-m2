"""
Edith Export — Push notes to Notion or Obsidian.

Supports:
- Notion: Creates pages via Notion API (requires NOTION_TOKEN)
- Obsidian: Writes .md files to local vault directory (requires EDITH_OBSIDIAN_VAULT)
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime


def export_to_notion(title: str, content: str, tags: list = None) -> dict:
    """Create a Notion page with the given content.

    Requires NOTION_TOKEN and NOTION_DATABASE_ID environment variables.

    Returns: {"ok": bool, "url": str or "", "error": str or ""}
    """
    token = os.environ.get("NOTION_TOKEN", "").strip()
    db_id = os.environ.get("NOTION_DATABASE_ID", "").strip()

    if not token:
        return {"ok": False, "url": "", "error": "NOTION_TOKEN not set. Add it to .env"}
    if not db_id:
        return {"ok": False, "url": "", "error": "NOTION_DATABASE_ID not set. Add it to .env"}

    try:
        import requests
    except ImportError:
        return {"ok": False, "url": "", "error": "requests library not installed"}

    # Build page properties
    properties = {
        "Name": {"title": [{"text": {"content": title[:100]}}]},
    }
    if tags:
        properties["Tags"] = {
            "multi_select": [{"name": t[:50]} for t in tags[:5]]
        }

    # Convert content to Notion blocks (simplified)
    blocks = []
    for para in content.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if para.startswith("# "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": para[2:][:2000]}}]},
            })
        elif para.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": para[3:][:2000]}}]},
            })
        elif para.startswith("- "):
            for item in para.split("\n"):
                if item.startswith("- "):
                    blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item[2:][:2000]}}]},
                    })
        else:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": para[:2000]}}]},
            })

    payload = {
        "parent": {"database_id": db_id},
        "properties": properties,
        "children": blocks[:100],  # Notion limit
    }

    try:
        resp = requests.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            return {"ok": True, "url": data.get("url", ""), "error": ""}
        else:
            return {"ok": False, "url": "", "error": f"Notion API {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"ok": False, "url": "", "error": str(e)}


def export_to_obsidian(title: str, content: str, tags: list = None, subfolder: str = "Edith") -> dict:
    """Write a markdown file to the Obsidian vault.

    Requires EDITH_OBSIDIAN_VAULT environment variable.

    Returns: {"ok": bool, "path": str or "", "error": str or ""}
    """
    vault = os.environ.get("EDITH_OBSIDIAN_VAULT", "").strip()
    if not vault:
        return {"ok": False, "path": "", "error": "EDITH_OBSIDIAN_VAULT not set. Add vault path to .env"}

    vault_path = Path(vault).expanduser().resolve()
    if not vault_path.exists():
        return {"ok": False, "path": "", "error": f"Vault not found: {vault_path}"}

    # Create subfolder
    target_dir = vault_path / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_title = re.sub(r"[^a-zA-Z0-9\s\-_]", "", title)[:80].strip() or "Edith Note"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{safe_title}_{timestamp}.md"
    filepath = target_dir / filename

    # Build frontmatter
    frontmatter = "---\n"
    frontmatter += f"title: \"{title}\"\n"
    frontmatter += f"created: {datetime.now().isoformat()}\n"
    frontmatter += f"source: edith\n"
    if tags:
        frontmatter += f"tags: [{', '.join(tags)}]\n"
    frontmatter += "---\n\n"

    try:
        filepath.write_text(frontmatter + content, encoding="utf-8")
        return {"ok": True, "path": str(filepath), "error": ""}
    except Exception as e:
        return {"ok": False, "path": "", "error": str(e)}

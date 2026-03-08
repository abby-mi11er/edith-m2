"""
Notes Router — Extracted from main.py (CH1)
============================================
CRUD operations for research notes stored as JSON files.
"""
import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body, Query

router = APIRouter(prefix="/api/notes", tags=["Notes"])

DATA_ROOT = os.environ.get("EDITH_DATA_ROOT", "")


def _get_notes_dir() -> Path | None:
    """Return the notes directory based on current DATA_ROOT."""
    root = os.environ.get("EDITH_DATA_ROOT", "")
    if root:
        return Path(root) / "notes"
    return None


def _versions_path(notes_dir: Path) -> Path:
    return notes_dir / "_versions.json"


@router.get("")
async def list_notes():
    """List all saved notes."""
    notes_dir = _get_notes_dir()
    if not notes_dir or not notes_dir.exists():
        return {"notes": []}
    notes = []
    for f in sorted(notes_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            notes.append(data)
        except Exception:
            continue
    return {"notes": notes}


@router.post("/draft")
async def save_note_draft(body: dict = Body(...)):
    """Save or update a note draft."""
    notes_dir = _get_notes_dir()
    if not notes_dir:
        raise HTTPException(status_code=500, detail="DATA_ROOT not set")
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_id = body.get("id") or uuid.uuid4().hex[:12]
    note = {
        "id": note_id,
        "title": body.get("title", "Untitled Note"),
        "content": body.get("content", ""),
        "status": "draft",
        "linked_doc": body.get("linked_doc"),
        "linked_chat": body.get("linked_chat"),
        "tags": body.get("tags", []),
        "created": body.get("created") or datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    (notes_dir / f"{note_id}.json").write_text(
        json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return note


@router.post("")
async def create_note(body: dict = Body(...)):
    """Create a note — alias for POST /api/notes/draft."""
    return await save_note_draft(body)


@router.post("/commit")
async def commit_note(body: dict = Body(...)):
    """Mark a note as committed."""
    notes_dir = _get_notes_dir()
    if not notes_dir:
        raise HTTPException(status_code=500, detail="DATA_ROOT not set")
    note_id = body.get("id", "")
    note_file = notes_dir / f"{note_id}.json"
    if not note_file.exists():
        raise HTTPException(status_code=404, detail="Note not found")
    note = json.loads(note_file.read_text(encoding="utf-8"))
    note["status"] = "committed"
    note["updated"] = datetime.now(timezone.utc).isoformat()
    note_file.write_text(json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8")
    return note


@router.post("/version-save")
async def save_note_version(body: dict = Body(...)):
    """Compatibility endpoint for NotesPanel server snapshots."""
    notes_dir = _get_notes_dir()
    if not notes_dir:
        raise HTTPException(status_code=500, detail="DATA_ROOT not set")
    notes_dir.mkdir(parents=True, exist_ok=True)

    text = str(body.get("text", ""))
    if not text.strip():
        raise HTTPException(status_code=422, detail="text is required")

    versions_file = _versions_path(notes_dir)
    versions: list[dict] = []
    if versions_file.exists():
        try:
            loaded = json.loads(versions_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                versions = loaded
        except Exception:
            versions = []

    now = datetime.now(timezone.utc)
    entry = {
        "id": uuid.uuid4().hex[:12],
        "label": str(body.get("label", "untitled")),
        "text": text,
        "timestamp": int(now.timestamp()),
        "date": now.isoformat(),
    }
    versions.append(entry)
    versions = versions[-200:]
    versions_file.write_text(json.dumps(versions, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "version": entry}


@router.get("/version-history")
async def note_version_history(limit: int = Query(default=50, ge=1, le=200)):
    """Compatibility endpoint for NotesPanel revision drawer."""
    notes_dir = _get_notes_dir()
    if not notes_dir or not notes_dir.exists():
        return {"versions": []}

    versions_file = _versions_path(notes_dir)
    if not versions_file.exists():
        return {"versions": []}

    try:
        loaded = json.loads(versions_file.read_text(encoding="utf-8"))
        versions = loaded if isinstance(loaded, list) else []
    except Exception:
        versions = []
    return {"versions": versions[-limit:]}


@router.delete("/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    notes_dir = _get_notes_dir()
    if not notes_dir:
        raise HTTPException(status_code=500, detail="DATA_ROOT not set")
    note_file = notes_dir / f"{note_id}.json"
    if note_file.exists():
        note_file.unlink()
    return {"ok": True}

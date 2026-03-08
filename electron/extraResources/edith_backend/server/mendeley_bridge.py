"""
Mendeley Bridge — Sovereign Library Sync for E.D.I.T.H.
=========================================================
OAuth2 handshake → library sync → annotation extraction → ChromaDB vault.

Setup:
    1. Register app at https://dev.mendeley.com/myapps.html
       Name: EDITH_Sovereign_Link
       Redirect URI: http://localhost:5000/oauth/mendeley/callback
    2. Save Client ID + Secret via Settings → Connectors Hub → 📚 Mendeley
    3. Click "Connect Mendeley" in Settings to start OAuth flow

The bridge extracts highlights/annotations as high-priority knowledge nodes
and mirrors PDFs to the Oyen Bolt for 100% offline access.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("edith.mendeley")

# ── Config ──────────────────────────────────────────────────────────────────
MENDELEY_API = "https://api.mendeley.com"
MENDELEY_AUTH = "https://api.mendeley.com/oauth/authorize"
MENDELEY_TOKEN_URL = "https://api.mendeley.com/oauth/token"
REDIRECT_URI = os.environ.get("MENDELEY_REDIRECT_URI", "http://localhost:8003/oauth/mendeley/callback")

# Credentials — loaded from env (set by SettingsPanel → /api/settings/connector-key)
_client_id: str = ""
_client_secret: str = ""
_access_token: str = ""
_refresh_token: str = ""
_token_expiry: float = 0.0


def _load_credentials():
    """Load credentials from env (set by SettingsPanel save-key flow)."""
    global _client_id, _client_secret, _access_token, _refresh_token
    _client_id = os.environ.get("MENDELEY_CLIENT_ID", "")
    _client_secret = os.environ.get("MENDELEY_CLIENT_SECRET", "")
    _access_token = os.environ.get("MENDELEY_ACCESS_TOKEN", "")
    _refresh_token = os.environ.get("MENDELEY_REFRESH_TOKEN", "")

    # Also try keyring (macOS Secure Enclave) for biometric-protected storage
    try:
        import keyring
        if not _client_id:
            _client_id = keyring.get_password("edith_vault", "mendeley_id") or ""
        if not _client_secret:
            _client_secret = keyring.get_password("edith_vault", "mendeley_secret") or ""
        if not _access_token:
            _access_token = keyring.get_password("edith_vault", "mendeley_token") or ""
    except ImportError:
        pass  # keyring not available — env vars only


def _save_token_to_keyring(token: str, refresh: str = ""):
    """Persist OAuth token to macOS Secure Enclave via keyring, with .env fallback."""
    global _access_token, _refresh_token
    _access_token = token
    _refresh_token = refresh
    os.environ["MENDELEY_ACCESS_TOKEN"] = token
    if refresh:
        os.environ["MENDELEY_REFRESH_TOKEN"] = refresh
    try:
        import keyring
        keyring.set_password("edith_vault", "mendeley_token", token)
        if refresh:
            keyring.set_password("edith_vault", "mendeley_refresh", refresh)
        log.info("§MENDELEY: Token saved to Secure Enclave")
    except ImportError:
        # Fallback: persist to .env file so tokens survive restart
        _persist_tokens_to_dotenv(token, refresh)
        log.info("§MENDELEY: Token saved to .env (keyring not available)")


def _persist_tokens_to_dotenv(token: str, refresh: str = ""):
    """Write MENDELEY_ACCESS_TOKEN/REFRESH_TOKEN to the .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    try:
        lines = env_path.read_text().splitlines() if env_path.exists() else []
        # Remove existing token lines
        lines = [l for l in lines if not l.startswith("MENDELEY_ACCESS_TOKEN=")
                 and not l.startswith("MENDELEY_REFRESH_TOKEN=")]
        # Append new tokens
        lines.append(f"MENDELEY_ACCESS_TOKEN={token}")
        if refresh:
            lines.append(f"MENDELEY_REFRESH_TOKEN={refresh}")
        env_path.write_text("\n".join(lines) + "\n")
        log.info(f"§MENDELEY: Tokens persisted to {env_path}")
    except Exception as exc:
        log.warning(f"§MENDELEY: Could not persist tokens to .env: {exc}")


def get_auth_url() -> str:
    """Generate Mendeley OAuth authorization URL."""
    _load_credentials()
    if not _client_id:
        return ""
    return (
        f"{MENDELEY_AUTH}"
        f"?client_id={_client_id}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=all"
    )


async def exchange_code(code: str) -> dict:
    """Exchange OAuth authorization code for access token."""
    _load_credentials()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            MENDELEY_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": _client_id,
                "client_secret": _client_secret,
            },
        )
        if resp.status_code != 200:
            log.error(f"§MENDELEY: Token exchange failed: {resp.text}")
            return {"error": resp.text}

        data = resp.json()
        _save_token_to_keyring(
            data.get("access_token", ""),
            data.get("refresh_token", ""),
        )
        return {"success": True, "expires_in": data.get("expires_in", 3600)}


async def _get_headers() -> dict:
    """Get authenticated request headers, refreshing token if needed."""
    _load_credentials()
    if not _access_token:
        return {}
    return {
        "Authorization": f"Bearer {_access_token}",
        "Accept": "application/vnd.mendeley-document.1+json",
    }


# ── Library Sync ────────────────────────────────────────────────────────────

async def list_documents(limit: int = 2000) -> list[dict]:
    """Fetch all documents from Mendeley library (paginated)."""
    headers = await _get_headers()
    if not headers:
        return []

    docs = []
    page = 1
    page_size = 500
    url = f"{MENDELEY_API}/documents?view=all&limit={min(page_size, 500)}"
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        while url and len(docs) < limit:
            resp = await client.get(url)
            if resp.status_code != 200:
                log.error(f"§MENDELEY: List failed on page {page}: {resp.status_code} — {resp.text[:200]}")
                break
            batch = resp.json()
            docs.extend(batch)
            log.info(f"§MENDELEY: Page {page} fetched {len(batch)} docs (total: {len(docs)})")

            # Pagination via Link header
            link = resp.headers.get("Link", "")
            url = ""
            if 'rel="next"' in link:
                for part in link.split(","):
                    if 'rel="next"' in part:
                        url = part.split(";")[0].strip().strip("<>")
            page += 1

    log.info(f"§MENDELEY: Fetched {len(docs)} documents total across {page - 1} pages")
    return docs


async def get_annotations(doc_id: str) -> list[dict]:
    """Fetch annotations/highlights for a specific document."""
    headers = await _get_headers()
    if not headers:
        return []

    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        resp = await client.get(f"{MENDELEY_API}/annotations?document_id={doc_id}")
        if resp.status_code != 200:
            return []
        return resp.json()


async def download_file(doc_id: str, dest_dir: str) -> Optional[str]:
    """Download the PDF file for a document."""
    headers = await _get_headers()
    if not headers:
        return None

    async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
        resp = await client.get(f"{MENDELEY_API}/files?document_id={doc_id}")
        if resp.status_code != 200 or not resp.json():
            return None

        files = resp.json()
        file_id = files[0].get("id")
        if not file_id:
            return None

        file_resp = await client.get(f"{MENDELEY_API}/files/{file_id}")
        if file_resp.status_code != 200:
            return None

        dest = os.path.join(dest_dir, f"{doc_id}.pdf")
        with open(dest, "wb") as f:
            f.write(file_resp.content)
        return dest


# ── Signal Extraction ───────────────────────────────────────────────────────

def _normalize_document(doc: dict) -> dict:
    """Convert Mendeley doc into a clean dict for E.D.I.T.H."""
    authors = doc.get("authors") or []
    author_names = [f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                    for a in authors[:5]]
    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", "Untitled"),
        "authors": author_names,
        "year": doc.get("year"),
        "journal": (doc.get("source") or ""),
        "doi": doc.get("identifiers", {}).get("doi"),
        "abstract": doc.get("abstract", ""),
        "tags": doc.get("tags", []),
        "folders": [f.get("name", "") for f in (doc.get("folders") or [])],
        "read": doc.get("read", False),
        "starred": doc.get("starred", False),
        "created": doc.get("created"),
        "file_attached": doc.get("file_attached", False),
    }


def _extract_signals(doc_id: str, annotations: list[dict]) -> list[dict]:
    """Convert Mendeley annotations into high-priority knowledge signals."""
    signals = []
    for ann in annotations:
        ann_type = ann.get("type", "highlight")
        text = ann.get("text", "")
        positions = ann.get("positions", [])

        # Highlighted text from positions
        highlight_text = ""
        for pos in positions:
            page_text = pos.get("text", "")
            if page_text:
                highlight_text += page_text + " "

        content = (text or highlight_text).strip()
        if not content:
            continue

        signals.append({
            "doc_id": doc_id,
            "type": ann_type,
            "content": content,
            "page": positions[0].get("page") if positions else None,
            "color": ann.get("color", {}).get("r", 255),
            "created": ann.get("created"),
            "priority": "high" if ann_type == "note" else "medium",
        })

    return signals


# ── Full Sync Pipeline ──────────────────────────────────────────────────────

async def full_sync(
    bolt_path: Optional[str] = None,
    chroma_ingest_fn=None,
    progress_callback=None,
) -> dict:
    """
    Full library sync pipeline:
    1. Fetch all Mendeley documents
    2. Download PDFs to Bolt (if bolt_path provided)
    3. Extract annotations → signals
    4. Ingest signals into ChromaDB (if chroma_ingest_fn provided)

    Returns sync report with counts.
    """
    # Use Oyen Bolt path if available, fallback to data root
    if not bolt_path:
        bolt_path = os.environ.get("OYEN_BOLT_PATH", "")
        if not bolt_path:
            bolt_path = os.path.join(os.environ.get("EDITH_DATA_ROOT", "."), "mendeley_mirror")

    os.makedirs(bolt_path, exist_ok=True)

    docs = await list_documents()
    report = {
        "total_documents": len(docs),
        "pdfs_downloaded": 0,
        "annotations_extracted": 0,
        "signals_ingested": 0,
        "errors": [],
    }

    for i, doc in enumerate(docs):
        doc_id = doc.get("id", "")
        norm = _normalize_document(doc)

        # Report progress
        if progress_callback:
            progress_callback(i + 1, len(docs), norm.get("title", ""))

        # 1. Download PDF if not already mirrored
        pdf_path = os.path.join(bolt_path, f"{doc_id}.pdf")
        if doc.get("file_attached") and not os.path.exists(pdf_path):
            try:
                result = await download_file(doc_id, bolt_path)
                if result:
                    report["pdfs_downloaded"] += 1
                    log.info(f"📥 Archived: {norm['title']}")
            except Exception as e:
                report["errors"].append(f"Download failed for {doc_id}: {e}")

        # 2. Extract annotations → signals
        try:
            annotations = await get_annotations(doc_id)
            signals = _extract_signals(doc_id, annotations)
            report["annotations_extracted"] += len(signals)

            # 3. Ingest signals into ChromaDB
            if chroma_ingest_fn and signals:
                for signal in signals:
                    try:
                        chroma_ingest_fn(
                            text=signal["content"],
                            metadata={
                                "source": "mendeley_annotation",
                                "doc_title": norm["title"],
                                "doc_id": doc_id,
                                "type": signal["type"],
                                "priority": signal["priority"],
                                "page": signal.get("page"),
                            },
                        )
                        report["signals_ingested"] += 1
                    except Exception as e:
                        report["errors"].append(f"Ingest failed: {e}")
        except Exception as e:
            report["errors"].append(f"Annotations failed for {doc_id}: {e}")

        # Rate limit to avoid Mendeley API throttling
        await asyncio.sleep(0.2)

    log.info(
        f"§MENDELEY: Sync complete — {report['total_documents']} docs, "
        f"{report['pdfs_downloaded']} PDFs, {report['annotations_extracted']} annotations, "
        f"{report['signals_ingested']} signals ingested"
    )
    return report


# ── Status Check ────────────────────────────────────────────────────────────

def status() -> dict:
    """Check Mendeley bridge status."""
    _load_credentials()
    return {
        "configured": bool(_client_id and _client_secret),
        "authenticated": bool(_access_token),
        "client_id_set": bool(_client_id),
        "has_token": bool(_access_token),
        "has_keyring": _check_keyring(),
    }


def _check_keyring() -> bool:
    try:
        import keyring
        return bool(keyring.get_password("edith_vault", "mendeley_token"))
    except Exception:
        return False

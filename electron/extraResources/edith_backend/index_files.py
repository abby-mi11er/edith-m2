import os
import time
import csv
import json
import re
import sqlite3
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

dotenv_override = os.environ.get("EDITH_DOTENV_PATH")
if dotenv_override:
    load_dotenv(dotenv_path=Path(dotenv_override).expanduser(), override=False)
else:
    load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
store_id = os.environ.get("EDITH_STORE_ID")
root = os.environ.get("EDITH_DATA_ROOT")
max_mb = int(os.environ.get("EDITH_MAX_FILE_MB", "50"))
try:
    chunk_size_tokens = int(os.environ.get("EDITH_CHUNK_SIZE_TOKENS", "250"))
except ValueError:
    chunk_size_tokens = 250
try:
    chunk_overlap_tokens = int(os.environ.get("EDITH_CHUNK_OVERLAP_TOKENS", "30"))
except ValueError:
    chunk_overlap_tokens = 30
INDEX_PRUNE_MANIFEST = os.environ.get("EDITH_INDEX_PRUNE_MANIFEST", "true").lower() == "true"

if not api_key:
    raise SystemExit("GOOGLE_API_KEY is missing. Set it in .env.")
if not store_id:
    raise SystemExit("EDITH_STORE_ID is missing. Set it in .env.")
if not root:
    raise SystemExit("EDITH_DATA_ROOT is missing. Set it in .env.")
if max_mb < 1:
    max_mb = 1
if chunk_size_tokens < 50:
    chunk_size_tokens = 50
if chunk_overlap_tokens < 0:
    chunk_overlap_tokens = 0
if chunk_overlap_tokens >= chunk_size_tokens:
    chunk_overlap_tokens = max(0, chunk_size_tokens // 5)

if not store_id.startswith("fileSearchStores/"):
    store_id = f"fileSearchStores/{store_id}"

root_path = Path(root).expanduser().resolve()
if not root_path.exists() or not root_path.is_dir():
    raise SystemExit(f"EDITH_DATA_ROOT not found or not a directory: {root_path}")

VALID_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".tex",
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl",
    ".py", ".ipynb", ".js", ".ts", ".sql", ".r", ".R",
}

IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
IGNORE_FILES = {".ds_store", "thumbs.db"}

CHUNK = {"max_tokens_per_chunk": chunk_size_tokens, "max_overlap_tokens": chunk_overlap_tokens}
STATE_DIR = Path(os.environ.get("EDITH_APP_DATA_DIR", ".")).expanduser()
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STATE_DIR / "edith_index.sqlite3"
REPORT_PATH = STATE_DIR / "edith_index_report.csv"
MAX_BYTES = max_mb * 1024 * 1024

client = genai.Client(api_key=api_key)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_op(op, poll_s=3):
    while not getattr(op, "done", False):
        time.sleep(poll_s)
        op = client.operations.get(op)
    return op


def upload_to_store(path: Path, config: dict):
    cfg = dict(config or {})
    custom_metadata = cfg.get("custom_metadata")
    if isinstance(custom_metadata, dict):
        cfg["custom_metadata"] = [
            {"key": str(k), "string_value": str(v or "").strip()}
            for k, v in custom_metadata.items()
            if str(v or "").strip()
        ]

    # Newer SDK signatures use `file=` and reject legacy kwargs.
    try:
        return client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_id,
            file=str(path),
            config=cfg,
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e).lower():
            raise

    # Backward-compatible fallback across SDK revisions.
    for key in ("local_file_path", "file_path"):
        try:
            return client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_id,
                config=cfg,
                **{key: str(path)},
            )
        except TypeError as e:
            if key == "file_path":
                break
            if "unexpected keyword argument" not in str(e).lower():
                raise

    # Legacy API fallback.
    try:
        return client.file_search_stores.import_file(
            file_search_store_name=store_id,
            file_name=str(path),
            config=cfg,
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e).lower():
            raise
    return client.file_search_stores.import_file(
        file_search_store_name=store_id,
        file_path=str(path),
        config=cfg,
    )


def db_init():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY, rel_path TEXT, indexed_at INTEGER)"
    )
    con.commit()
    return con


def prune_manifest(con, live_rel_paths):
    if not INDEX_PRUNE_MANIFEST:
        return 0
    live = set(live_rel_paths or [])
    stale_hashes = []
    cur = con.cursor()
    cur.execute("SELECT sha256, rel_path FROM blobs")
    for sha, rel in cur.fetchall():
        if (rel or "") not in live:
            stale_hashes.append(sha)
    if not stale_hashes:
        return 0
    cur.executemany("DELETE FROM blobs WHERE sha256=?", [(h,) for h in stale_hashes])
    con.commit()
    return len(stale_hashes)


def iter_files():
    for root_dir, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.lower() in IGNORE_FILES or fn.startswith("."):
                continue
            p = Path(root_dir) / fn
            if p.is_symlink():
                continue
            if p.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
            except OSError:
                continue
            yield p


def infer_project(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return parts[0]
    return ""


def infer_tag(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"#([A-Za-z0-9_-]+)", stem)
    if m:
        return m.group(1)
    m = re.search(r"\[([A-Za-z0-9_-]+)\]", stem)
    if m:
        return m.group(1)
    return ""


def infer_citation(filename: str):
    stem = Path(filename).stem
    cleaned = re.sub(r"[_]+|[-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    year_match = re.search(r"(19|20)\d{2}", cleaned)
    year = year_match.group(0) if year_match else ""

    author = ""
    title = cleaned
    if year_match:
        before, _, after = cleaned.partition(year)
        before = before.strip(" -_()")
        after = after.strip(" -_()")
        if before:
            author = before.split(",")[0].strip()
        if after:
            title = after

    return title, author, year


def clean_text(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("\x00", "")
    return value


def infer_citation_from_pdf(path: Path):
    if PdfReader is None:
        return "", "", "", "filename"
    try:
        reader = PdfReader(str(path))
    except Exception:
        return "", "", "", "filename"

    title = ""
    author = ""
    year = ""
    source = "filename"

    meta = reader.metadata or {}
    title = clean_text(str(meta.get("/Title", "") or meta.get("title", "") or ""))
    author = clean_text(str(meta.get("/Author", "") or meta.get("author", "") or ""))
    date_candidates = [
        str(meta.get("/CreationDate", "") or ""),
        str(meta.get("/ModDate", "") or ""),
    ]
    for raw in date_candidates:
        m = re.search(r"(19|20)\d{2}", raw)
        if m:
            year = m.group(0)
            break

    if title or author or year:
        source = "pdf_metadata"

    # Lightweight first-page fallback if metadata is incomplete.
    first_page_text = ""
    try:
        if reader.pages:
            first_page_text = clean_text(reader.pages[0].extract_text() or "")
    except Exception:
        first_page_text = ""

    if first_page_text:
        if not year:
            m = re.search(r"\b(19|20)\d{2}\b", first_page_text)
            if m:
                year = m.group(0)
        lines = [clean_text(x) for x in first_page_text.split("  ") if clean_text(x)]
        if not title:
            for line in lines[:12]:
                if len(line) < 15:
                    continue
                lower = line.lower()
                if "abstract" in lower or "introduction" in lower:
                    continue
                if re.search(r"https?://", lower):
                    continue
                title = line[:220]
                source = "pdf_first_page"
                break
        if not author:
            author_match = re.search(
                r"\b([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,3})\b",
                first_page_text,
            )
            if author_match:
                author = author_match.group(1)
                if source == "filename":
                    source = "pdf_first_page"

    return title, author, year, source


def load_vault_file_manifest():
    override = (os.environ.get("EDITH_VAULT_FILE_MANIFEST") or "").strip()
    if override:
        path = Path(override).expanduser()
    else:
        path = root_path / "vault_sync" / "_manifests" / "vault_file_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def join_manifest_field(value):
    if isinstance(value, list):
        out = []
        for item in value:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
        return ",".join(out)
    return str(value or "").strip()


def main():
    con = db_init()
    cur = con.cursor()
    vault_manifest = load_vault_file_manifest()

    files = list(iter_files())
    if not files:
        print("No matching files found to index.")
        return

    print(f"Found {len(files)} files under {root_path}")

    indexed = 0
    skipped = 0
    replaced = 0
    report_rows = []
    live_rel_paths = set()

    for p in tqdm(files, desc="Indexing"):
        rel = str(p.relative_to(root_path))
        live_rel_paths.add(rel)
        sha = sha256_file(p)
        project = infer_project(rel)
        tag = infer_tag(p.name)
        title_guess, author_guess, year_guess = infer_citation(p.name)
        citation_source = "filename"
        if p.suffix.lower() == ".pdf":
            pdf_title, pdf_author, pdf_year, pdf_source = infer_citation_from_pdf(p)
            if pdf_title:
                title_guess = pdf_title
            if pdf_author:
                author_guess = pdf_author
            if pdf_year:
                year_guess = pdf_year
            if pdf_source != "filename":
                citation_source = pdf_source

        report_rows.append(
            {
                "file_name": p.name,
                "rel_path": rel,
                "project": project,
                "tag": tag,
                "title_guess": title_guess,
                "author_guess": author_guess,
                "year_guess": year_guess,
                "citation_source": citation_source,
                "vault_export_id": join_manifest_field((vault_manifest.get(rel) or {}).get("vault_export_id")),
                "vault_export_date": join_manifest_field((vault_manifest.get(rel) or {}).get("vault_export_date")),
                "vault_custodian": join_manifest_field((vault_manifest.get(rel) or {}).get("vault_custodian")),
                "vault_matter_name": join_manifest_field((vault_manifest.get(rel) or {}).get("vault_matter_name")),
            }
        )

        cur.execute("SELECT 1 FROM blobs WHERE sha256=?", (sha,))
        if cur.fetchone():
            skipped += 1
            continue

        cur.execute("SELECT COUNT(1) FROM blobs WHERE rel_path=? AND sha256<>?", (rel, sha))
        existing_rel_rows = int(cur.fetchone()[0] or 0)

        try:
            vault_meta = vault_manifest.get(rel) or {}
            op = upload_to_store(
                p,
                {
                    "display_name": p.name,
                    "chunking_config": {"white_space_config": CHUNK},
                    "custom_metadata": {
                        "sha256": sha,
                        "rel_path": rel,
                        "project": project,
                        "tag": tag,
                        "title": title_guess,
                        "author": author_guess,
                        "year": year_guess,
                        "citation_source": citation_source,
                        "vault_export_id": join_manifest_field(vault_meta.get("vault_export_id")),
                        "vault_export_ids": join_manifest_field(vault_meta.get("vault_export_ids")),
                        "vault_export_date": join_manifest_field(vault_meta.get("vault_export_date")),
                        "vault_custodian": join_manifest_field(vault_meta.get("vault_custodian")),
                        "vault_custodians": join_manifest_field(vault_meta.get("vault_custodians")),
                        "vault_matter_name": join_manifest_field(vault_meta.get("vault_matter_name")),
                        "vault_matter_names": join_manifest_field(vault_meta.get("vault_matter_names")),
                    },
                },
            )
            wait_op(op)
            if existing_rel_rows > 0:
                cur.execute("DELETE FROM blobs WHERE rel_path=? AND sha256<>?", (rel, sha))
                replaced += existing_rel_rows
            cur.execute(
                "INSERT INTO blobs(sha256, rel_path, indexed_at) VALUES(?,?,?)",
                (sha, rel, int(time.time())),
            )
            con.commit()
            indexed += 1
        except Exception as e:
            print(f"SKIP {rel}: {e}")

    stale_pruned = prune_manifest(con, live_rel_paths)
    con.close()
    try:
        with REPORT_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_name",
                    "rel_path",
                    "project",
                    "tag",
                    "title_guess",
                    "author_guess",
                "year_guess",
                "citation_source",
                "vault_export_id",
                "vault_export_date",
                "vault_custodian",
                "vault_matter_name",
                ],
            )
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Index report: {REPORT_PATH}")
    except Exception as e:
        print(f"Report write failed: {e}")
    print(f"\nIndexed new: {indexed}")
    print(f"Skipped duplicates: {skipped}")
    print(f"Replaced changed files in manifest: {replaced}")
    if INDEX_PRUNE_MANIFEST:
        print(f"Pruned removed files from manifest: {stale_pruned}")
        if stale_pruned > 0:
            print("Note: Google File Search store may still contain stale uploaded copies. Create a fresh store for a hard reset.")


if __name__ == "__main__":
    main()

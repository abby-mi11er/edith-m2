#!/usr/bin/env python3
"""
Nightly Edith maintenance:
- reindex changed files (google/chroma backend aware)
- run lightweight health checks
- persist a startup-readable status report
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_template_env_file(path: Path) -> bool:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return (
        "GOOGLE_API_KEY=your_key_here" in sample
        and "EDITH_STORE_ID=fileSearchStores/your_store_id_here" in sample
        and "EDITH_DATA_ROOT=/Users/yourname/Documents/YourFiles" in sample
    )


def clean_runtime_env_value(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    low = v.lower()
    if low in {
        "your_key_here",
        "your_store_id_here",
        "your_vault_id_here",
        "filesearchstores/your_store_id_here",
        "filesearchstores/your_vault_id_here",
        "/users/yourname/documents/yourfiles",
    }:
        return ""
    if "yourname" in low or "your_store_id" in low or "your_vault_id" in low:
        return ""
    return v


def normalize_store_id(value: str) -> str:
    s = clean_runtime_env_value(value)
    if not s:
        return ""
    if s.startswith("fileSearchStores/"):
        return s
    return f"fileSearchStores/{s}"


def load_env(project_root: Path):
    if not load_dotenv:
        return
    app_home = Path.home() / "Library" / "Application Support" / "Edith"
    candidates = []
    override = os.environ.get("EDITH_DOTENV_PATH")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            project_root / ".env",
            Path.cwd() / ".env",
            app_home / ".env",
        ]
    )
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            try:
                if is_template_env_file(p):
                    continue
                load_dotenv(dotenv_path=p, override=False)
                os.environ.setdefault("EDITH_DOTENV_PATH", str(p))
            except Exception:
                continue


def load_desktop_config_defaults() -> dict:
    cfg_paths = [
        Path.home() / "Library" / "Application Support" / "Edith" / "config.json",
        Path.home() / "Library" / "Application Support" / "edith-desktop-shell" / "config.json",
    ]
    defaults = {}
    for cfg in cfg_paths:
        if not cfg.exists():
            continue
        try:
            payload = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not defaults.get("EDITH_RETRIEVAL_BACKEND"):
            defaults["EDITH_RETRIEVAL_BACKEND"] = (
                payload.get("retrieval_backend") or payload.get("retrievalBackend") or ""
            )
        if not defaults.get("EDITH_DATA_ROOT"):
            defaults["EDITH_DATA_ROOT"] = payload.get("data_root") or payload.get("dataRoot") or ""
        if not defaults.get("EDITH_STORE_ID"):
            defaults["EDITH_STORE_ID"] = (
                payload.get("store_id")
                or payload.get("storeId")
                or payload.get("vault_id")
                or payload.get("vaultId")
                or ""
            )
    return defaults


def apply_runtime_defaults():
    defaults = load_desktop_config_defaults()
    for key, value in defaults.items():
        if key not in os.environ or not clean_runtime_env_value(os.environ.get(key, "")):
            cleaned = clean_runtime_env_value(str(value))
            if cleaned:
                os.environ[key] = cleaned


def app_state_dir() -> Path:
    raw = clean_runtime_env_value(os.getenv("EDITH_APP_DATA_DIR") or "")
    if raw:
        p = Path(raw).expanduser().resolve()
    else:
        p = (Path.home() / "Library" / "Application Support" / "Edith").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd(cmd: list[str], cwd: Path, timeout_s: int = 1800, env: dict | None = None):
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        out = "\n".join([x for x in [proc.stdout, proc.stderr] if x]).strip()
        return proc.returncode, out
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = f"Timed out after {timeout_s}s."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 124, msg
    except Exception as e:
        return 1, str(e)


def health_checks(backend: str):
    checks = []
    api_key = clean_runtime_env_value(os.getenv("GOOGLE_API_KEY") or "")
    data_root = clean_runtime_env_value(os.getenv("EDITH_DATA_ROOT") or "")

    if backend == "google":
        checks.append({"name": "api_key", "ok": bool(api_key), "detail": "configured" if api_key else "missing"})
    else:
        checks.append({"name": "api_key", "ok": True, "detail": "not required for chroma maintenance"})

    if data_root:
        p = Path(data_root).expanduser()
        checks.append({"name": "data_root", "ok": p.exists() and p.is_dir(), "detail": str(p)})
    else:
        checks.append({"name": "data_root", "ok": False, "detail": "missing"})

    if backend == "google":
        store_id = normalize_store_id(
            os.getenv("EDITH_STORE_ID")
            or os.getenv("EDITH_STORE_MAIN")
            or os.getenv("EDITH_VAULT_ID")
            or ""
        )
        checks.append({"name": "store_id", "ok": bool(store_id), "detail": store_id or "missing"})
        if api_key and store_id:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                client.file_search_stores.get(name=store_id)
                checks.append({"name": "store_connectivity", "ok": True, "detail": "ok"})
            except Exception as e:
                checks.append({"name": "store_connectivity", "ok": False, "detail": str(e)[:240]})
        else:
            checks.append({"name": "store_connectivity", "ok": False, "detail": "missing API key or store id"})
    else:
        checks.append({"name": "backend", "ok": True, "detail": "chroma"})

    return checks


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_env(project_root)
    apply_runtime_defaults()

    state = app_state_dir()
    report_path = state / "nightly_maintenance_status.json"
    timeout_s = max(300, int(os.getenv("EDITH_REINDEX_TIMEOUT_SECONDS", "1800")))

    backend = clean_runtime_env_value(os.getenv("EDITH_RETRIEVAL_BACKEND") or "chroma").lower() or "chroma"
    if backend not in {"google", "chroma"}:
        backend = "chroma"
    index_script = "index_files.py" if backend == "google" else "chroma_index.py"
    index_path = project_root / index_script

    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(state))
    env.setdefault("EDITH_DOTENV_PATH", str(Path.home() / "Library" / "Application Support" / "Edith" / ".env"))

    if index_path.exists():
        index_code, index_output = run_cmd([sys.executable, str(index_path)], cwd=project_root, timeout_s=timeout_s, env=env)
    else:
        index_code, index_output = 1, f"Indexer script missing: {index_path}"

    if index_output:
        ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        cleaned = []
        skip_tokens = (
            "Loading weights:",
            "Materializing param=",
            "BertModel LOAD REPORT",
            "embeddings.position_ids",
            "UNEXPECTED",
            "Notes:",
            "HF Hub",
            "| Status",
            "+------------",
        )
        for raw_line in str(index_output).splitlines():
            line = ansi.sub("", raw_line).strip()
            if not line:
                continue
            if any(token in line for token in skip_tokens):
                continue
            cleaned.append(line)
        index_output = "\n".join(cleaned).strip()

    checks = health_checks(backend)
    ok = (index_code == 0) and all(bool(c.get("ok")) for c in checks)

    report = {
        "last_run_at": utc_now(),
        "ok": ok,
        "backend": backend,
        "index_script": index_script,
        "index_code": int(index_code),
        "index_output": (index_output or "")[:4000],
        "checks": checks,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def load_env():
    candidates = []
    override = os.environ.get("EDITH_DOTENV_PATH")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend([Path(__file__).parent / ".env", Path.cwd() / ".env"])
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def is_preview(name: str) -> bool:
    n = name.lower()
    return "preview" in n or "exp" in n


def model_sort_key(name: str):
    n = name.lower()
    m = re.search(r"gemini-(\d+)(?:[.-](\d+))?", n)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if m and m.group(2) else 0
    family = 0
    if "pro" in n:
        family = 3
    elif "flash" in n and "lite" not in n:
        family = 2
    elif "lite" in n:
        family = 1
    preview = 1 if is_preview(n) else 0
    return (major, minor, preview, family, n)


def main():
    load_env()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY missing in environment.")

    client = genai.Client(api_key=api_key)

    models = []
    try:
        for m in client.models.list():
            methods = (
                getattr(m, "supported_generation_methods", None)
                or getattr(m, "supportedGenerationMethods", None)
                or getattr(m, "supported_actions", None)
                or []
            )
            if methods and "generateContent" not in methods:
                continue
            name = (getattr(m, "name", "") or "").replace("models/", "")
            if name:
                models.append(name)
    except Exception as e:
        raise SystemExit(f"Model listing failed: {e}")

    models = sorted(set(models), key=model_sort_key, reverse=True)
    if not models:
        print("No generateContent models returned for this key.")
        return

    print(f"Total models: {len(models)}")
    print("Top candidates:")
    for name in models[:20]:
        print(f"- {name}")

    stable = [m for m in models if not is_preview(m)]
    print("\nRecommended defaults:")
    print(f"- Latest (may include preview): {models[0]}")
    if stable:
        print(f"- Stable only: {stable[0]}")


if __name__ == "__main__":
    main()

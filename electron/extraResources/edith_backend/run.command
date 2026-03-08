#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

if ! python - <<'PY'
import importlib
mods = [
    "streamlit",
    "dotenv",
    "google.genai",
    "tqdm",
    "watchdog",
    "reportlab",
    "cryptography",
    "pypdf",
    "chromadb",
    "sentence_transformers",
    "docx",
]
for m in mods:
    importlib.import_module(m)
PY
then
  echo "Installing dependencies..."
  python -m pip install -r requirements.txt
fi

echo "Launching Edith..."
streamlit run app.py

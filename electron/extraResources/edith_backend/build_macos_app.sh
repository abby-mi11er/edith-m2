#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv-build" ]; then
  python3 -m venv .venv-build
fi

source .venv-build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-packaging.txt

python make_icon.py

pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "Edith" \
  --icon "assets/edith.icns" \
  --add-data "app.py:." \
  --add-data ".env.example:." \
  --add-data ".streamlit:.streamlit" \
  --add-data "index_files.py:." \
  --add-data "chroma_index.py:." \
  --add-data "chroma_backend.py:." \
  --add-data "watch_files.py:." \
  desktop_launcher.py

echo "Built: dist/Edith.app"

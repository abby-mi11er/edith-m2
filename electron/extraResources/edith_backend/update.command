#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt

echo
echo "Dependency update complete."
echo "Checking available Gemini models for this API key..."
python check_models.py || true

echo
echo "Done. Relaunch with ./run.command"

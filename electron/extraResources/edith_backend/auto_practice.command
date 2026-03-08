#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install -r requirements.txt

python scripts/run_practice_loop.py \
  --mode "Files only" \
  --backend chroma \
  --generate-cases \
  --export-sft

echo
echo "Automation run complete."

#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/electron/extraResources/edith_backend"

declare -a CANDIDATES=()

if [[ -n "${EDITH_M2_BACKEND_ROOT:-}" ]]; then
  CANDIDATES+=("$EDITH_M2_BACKEND_ROOT")
fi

CANDIDATES+=(
  "/Applications/Edith.app/Contents/Resources/edith_backend"
  "/Volumes/Edith Bolt/Edith_M4/electron/extraResources/edith_backend"
  "/Volumes/Edith Bolt/Edith_M4/dist/edith_backend"
  "/Volumes/Edith Bolt/Edith_M4/build/edith_backend"
)

SOURCE_DIR=""
for candidate in "${CANDIDATES[@]}"; do
  if [[ -d "$candidate" && ( -f "$candidate/desktop_launcher.py" || -f "$candidate/server/main.py" ) ]]; then
    SOURCE_DIR="$candidate"
    break
  fi
done

if [[ -z "$SOURCE_DIR" ]]; then
  if [[ -f "$TARGET_DIR/desktop_launcher.py" || -f "$TARGET_DIR/server/main.py" ]]; then
    echo "No backend source found; keeping existing bundled backend at $TARGET_DIR"
    exit 0
  fi
  echo "No valid backend source found for packaging."
  echo "Set EDITH_M2_BACKEND_ROOT to a backend folder containing desktop_launcher.py or server/main.py."
  exit 1
fi

mkdir -p "$TARGET_DIR"
rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$SOURCE_DIR/" "$TARGET_DIR/"

# Prune dangling symlinks that break electron-builder packaging.
while IFS= read -r broken_link; do
  rm -f "$broken_link"
  echo "Removed broken symlink: $broken_link"
done < <(find -L "$TARGET_DIR" -type l)

echo "Bundled backend prepared from: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

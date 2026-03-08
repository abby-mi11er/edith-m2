#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ELECTRON_DIR="$ROOT_DIR/electron"
NPM_CACHE_DIR="$ROOT_DIR/.npm-cache"
EB_CACHE_DIR="$ROOT_DIR/.electron-builder-cache"
ICON_SRC="$ROOT_DIR/assets/edith.icns"
ICON_PNG_SRC="$ROOT_DIR/assets/edith_icon.png"
ICON_BUILD_DIR="$ELECTRON_DIR/build"
ICON_DEST="$ICON_BUILD_DIR/icon.icns"
ICON_PNG_DEST="$ICON_BUILD_DIR/icon.png"
VERIFY_SCRIPT="$ELECTRON_DIR/scripts/verify_signature.sh"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Install Node.js first."
  exit 1
fi

cd "$ELECTRON_DIR"
mkdir -p "$NPM_CACHE_DIR"
mkdir -p "$ICON_BUILD_DIR"
mkdir -p "$EB_CACHE_DIR"
export NPM_CONFIG_CACHE="$NPM_CACHE_DIR"
export ELECTRON_BUILDER_CACHE="$EB_CACHE_DIR"

if [ ! -f "$ICON_SRC" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/make_icon.py" || true
  else
    python3 "$ROOT_DIR/make_icon.py" || true
  fi
fi
if [ -f "$ICON_SRC" ]; then
  cp "$ICON_SRC" "$ICON_DEST"
fi
if [ -f "$ICON_PNG_SRC" ]; then
  cp "$ICON_PNG_SRC" "$ICON_PNG_DEST"
fi

SIGNED_BUILD=false
if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]; then
  SIGNED_BUILD=true
fi

if [ "$SIGNED_BUILD" = false ]; then
  export CSC_IDENTITY_AUTO_DISCOVERY=false
fi

npm install
if [ "$SIGNED_BUILD" = true ]; then
  npm run dist:mac
else
  npm run pack:mac
fi

echo
echo "Build complete (app bundle):"
echo "  $ELECTRON_DIR/dist/mac-arm64/Edith.app"
echo
if [ "$SIGNED_BUILD" = true ]; then
  if [ -x "$VERIFY_SCRIPT" ]; then
    "$VERIFY_SCRIPT" "$ELECTRON_DIR/dist/mac-arm64/Edith.app" "${APPLE_TEAM_ID:-}"
  fi
  echo "Signed/notarized build path completed."
else
  echo "Local build is unsigned by default."
fi

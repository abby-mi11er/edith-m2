#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_SRC="$ROOT_DIR/electron/dist/mac-arm64/Edith.app"
USER_APPS_DIR="$HOME/Applications"
APP_DEST="$USER_APPS_DIR/Edith.app"

"$ROOT_DIR/build_electron_macos.sh"

if [ ! -d "$APP_SRC" ]; then
  echo "Build completed, but app bundle not found at: $APP_SRC"
  exit 1
fi

mkdir -p "$USER_APPS_DIR"
rm -rf "$APP_DEST"
if cp -R "$APP_SRC" "$APP_DEST"; then
  xattr -cr "$APP_DEST" || true
  echo
  echo "Installed: $APP_DEST"
  echo "You can launch Edith from Finder > Applications."
  open "$USER_APPS_DIR" || true
  exit 0
fi

FALLBACK_DEST="$ROOT_DIR/Edith.app"
rm -rf "$FALLBACK_DEST"
cp -R "$APP_SRC" "$FALLBACK_DEST"
xattr -cr "$FALLBACK_DEST" || true

echo
echo "Could not write to $USER_APPS_DIR. Installed fallback app here:"
echo "  $FALLBACK_DEST"
echo "Move it to Applications manually if needed."

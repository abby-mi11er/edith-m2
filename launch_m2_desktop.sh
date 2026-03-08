#!/bin/bash
set -euo pipefail

M2_ROOT="/Volumes/Edith Bolt/Edith_M2"
M4_ROOT="/Volumes/Edith Bolt/Edith_M4"
M2_ENV_FALLBACK="$M2_ROOT/.env"
if [[ ! -f "$M2_ENV_FALLBACK" && -f "$M4_ROOT/.env" ]]; then
  M2_ENV_FALLBACK="$M4_ROOT/.env"
fi
SHARED_CHROMA_DEFAULT="$M4_ROOT/ChromaDB"
DEFAULT_BACKEND_ROOT="/Applications/Edith.app/Contents/Resources/edith_backend"
LOCAL_BUNDLED_BACKEND="$M2_ROOT/electron/extraResources/edith_backend"

if [[ ! -d "$M2_ROOT" ]]; then
  echo "M2 root not found: $M2_ROOT"
  exit 1
fi
if [[ ! -d "$M4_ROOT" ]]; then
  echo "M4 root not found: $M4_ROOT"
  exit 1
fi
if [[ ! -f "$M2_ROOT/package.json" ]]; then
  echo "Missing package.json at $M2_ROOT"
  exit 1
fi

export EDITH_M2_BACKEND_PORT="${EDITH_M2_BACKEND_PORT:-8003}"
export EDITH_M2_FRONTEND_PORT="${EDITH_M2_FRONTEND_PORT:-5176}"
export EDITH_M2_BACKEND_URL="${EDITH_M2_BACKEND_URL:-http://127.0.0.1:$EDITH_M2_BACKEND_PORT}"
export EDITH_BACKEND_URL="${EDITH_BACKEND_URL:-$EDITH_M2_BACKEND_URL}"
export EDITH_M2_USER_DATA_DIR="${EDITH_M2_USER_DATA_DIR:-$HOME/Library/Application Support/Edith_M2}"
export EDITH_M2_SESSION_DATA_DIR="${EDITH_M2_SESSION_DATA_DIR:-$EDITH_M2_USER_DATA_DIR/SessionData}"
export EDITH_M2_APP_DATA_DIR="${EDITH_M2_APP_DATA_DIR:-$EDITH_M2_USER_DATA_DIR}"
export EDITH_M2_DATA_ROOT="${EDITH_M2_DATA_ROOT:-$M2_ROOT}"
export EDITH_SHARED_DATA_ROOT="${EDITH_SHARED_DATA_ROOT:-$M4_ROOT}"
export EDITH_SHARED_CHROMA_DIR="${EDITH_SHARED_CHROMA_DIR:-$SHARED_CHROMA_DEFAULT}"
export EDITH_SHARED_CHROMA_COLLECTION="${EDITH_SHARED_CHROMA_COLLECTION:-edith_docs_pdf}"
export EDITH_M2_CHROMA_COLLECTION="${EDITH_M2_CHROMA_COLLECTION:-$EDITH_SHARED_CHROMA_COLLECTION}"
export EDITH_M2_CHROMA_COLLECTION_STRICT="${EDITH_M2_CHROMA_COLLECTION_STRICT:-true}"
export EDITH_M2_REQUIRE_SHARED_INDEX="${EDITH_M2_REQUIRE_SHARED_INDEX:-true}"
export EDITH_M2_DOTENV_PATH="${EDITH_M2_DOTENV_PATH:-$M2_ENV_FALLBACK}"

if [[ -d "$LOCAL_BUNDLED_BACKEND" ]]; then
  export EDITH_M2_BACKEND_ROOT="${EDITH_M2_BACKEND_ROOT:-$LOCAL_BUNDLED_BACKEND}"
elif [[ -d "$DEFAULT_BACKEND_ROOT" ]]; then
  export EDITH_M2_BACKEND_ROOT="${EDITH_M2_BACKEND_ROOT:-$DEFAULT_BACKEND_ROOT}"
fi

if [[ -n "${EDITH_M2_DOTENV_PATH:-}" && ! -f "$EDITH_M2_DOTENV_PATH" ]]; then
  echo "Warning: dotenv path not found, running without EDITH_DOTENV_PATH: $EDITH_M2_DOTENV_PATH"
  export EDITH_M2_DOTENV_PATH=""
fi

if [[ ! -d "$EDITH_SHARED_CHROMA_DIR" ]]; then
  echo "Shared Chroma directory missing: $EDITH_SHARED_CHROMA_DIR"
  exit 1
fi
if [[ "$EDITH_M2_REQUIRE_SHARED_INDEX" == "true" && ! -f "$EDITH_SHARED_CHROMA_DIR/chroma.sqlite3" ]]; then
  echo "Shared Chroma sqlite missing: $EDITH_SHARED_CHROMA_DIR/chroma.sqlite3"
  echo "Set EDITH_M2_REQUIRE_SHARED_INDEX=false to bypass this guard."
  exit 1
fi

echo "Launching E.D.I.T.H. M2"
echo "  m2 root: $M2_ROOT"
echo "  m4 root: $M4_ROOT"
echo "  frontend: $EDITH_M2_FRONTEND_PORT"
echo "  backend: $EDITH_M2_BACKEND_PORT"
echo "  backend url: $EDITH_M2_BACKEND_URL"
echo "  backend root: ${EDITH_M2_BACKEND_ROOT:-<auto-detect>}"
echo "  shared chroma: $EDITH_SHARED_CHROMA_DIR"
echo "  collection: $EDITH_M2_CHROMA_COLLECTION (strict=$EDITH_M2_CHROMA_COLLECTION_STRICT)"
echo "  dotenv: ${EDITH_M2_DOTENV_PATH:-<none>}"
echo "  app data: $EDITH_M2_APP_DATA_DIR"

if [[ "${EDITH_DRY_RUN:-false}" == "true" ]]; then
  echo "Dry run complete. Skipping electron launch."
  exit 0
fi

mkdir -p \
  "$EDITH_M2_USER_DATA_DIR" \
  "$EDITH_M2_SESSION_DATA_DIR" \
  "$EDITH_M2_APP_DATA_DIR" \
  "$M2_ROOT/logs"

cd "$M2_ROOT"
npm run electron

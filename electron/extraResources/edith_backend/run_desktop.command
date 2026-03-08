#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ELECTRON_DIR="$ROOT_DIR/electron"
NPM_CACHE_DIR="$ROOT_DIR/.npm-cache"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Install Node.js (https://nodejs.org) first."
  exit 1
fi

cd "$ELECTRON_DIR"
mkdir -p "$NPM_CACHE_DIR"
export NPM_CONFIG_CACHE="$NPM_CACHE_DIR"

if [ ! -d node_modules ]; then
  npm install
fi

npm run dev

#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$PROJECT_DIR/packaging/ai-privacy-check/app/server"
APP_DATA_DIR="${APP_DATA_DIR:-/tmp/ai-privacy-check-dev}"
APP_PORT="${APP_PORT:-8976}"

mkdir -p "$APP_DATA_DIR"
cd "$SERVER_DIR"
exec env APP_DATA_DIR="$APP_DATA_DIR" APP_PORT="$APP_PORT" python3 server.py

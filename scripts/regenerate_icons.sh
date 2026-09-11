#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_ICON="$PROJECT_DIR/assets/icon.svg"
PACKAGE_DIR="$PROJECT_DIR/packaging/ai-privacy-check"

if ! command -v rsvg-convert >/dev/null 2>&1; then
  echo "需要 rsvg-convert 才能重新生成 PNG 图标。" >&2
  exit 1
fi

rsvg-convert -w 64 -h 64 "$SOURCE_ICON" -o "$PACKAGE_DIR/ICON.PNG"
rsvg-convert -w 256 -h 256 "$SOURCE_ICON" -o "$PACKAGE_DIR/ICON_256.PNG"
rsvg-convert -w 64 -h 64 "$SOURCE_ICON" -o "$PACKAGE_DIR/app/ui/images/icon_64.png"
rsvg-convert -w 256 -h 256 "$SOURCE_ICON" -o "$PACKAGE_DIR/app/ui/images/icon_256.png"

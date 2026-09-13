#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD="uv run python3"
else
  PYTHON_CMD="python3"
fi

$PYTHON_CMD -m unittest discover -s tests -v
$PYTHON_CMD scripts/benchmark.py
env PYTHONPYCACHEPREFIX=/tmp/ai-privacy-check-pycache $PYTHON_CMD -m py_compile \
  packaging/ai-privacy-check/app/server/server.py \
  packaging/ai-privacy-check/app/server/model_installer.py \
  packaging/ai-privacy-check/app/server/privacy/*.py \
  packaging/ai-privacy-check/app/server/privacy/workers/*.py
node --check packaging/ai-privacy-check/app/server/web/app.js
$PYTHON_CMD -m json.tool packaging/ai-privacy-check/config/privilege >/dev/null
$PYTHON_CMD -m json.tool packaging/ai-privacy-check/config/resource >/dev/null
$PYTHON_CMD -m json.tool packaging/ai-privacy-check/app/ui/config >/dev/null
$PYTHON_CMD -m json.tool packaging/ai-privacy-check/wizard/uninstall >/dev/null
grep -Eq '^install_dep_apps[[:space:]]*=[[:space:]]*python312$' packaging/ai-privacy-check/manifest
if [ -d packaging/ai-privacy-check/app/docker ]; then
  echo "检测到遗留的 Docker 打包目录" >&2
  exit 1
fi
for script in packaging/ai-privacy-check/cmd/*; do
  bash -n "$script"
done

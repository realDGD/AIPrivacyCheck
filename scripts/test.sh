#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

python3 -m unittest discover -s tests -v
env PYTHONPYCACHEPREFIX=/tmp/ai-privacy-check-pycache python3 -m py_compile \
  packaging/ai-privacy-check/app/server/server.py \
  packaging/ai-privacy-check/app/server/model_installer.py \
  packaging/ai-privacy-check/app/server/privacy/*.py
node --check packaging/ai-privacy-check/app/server/web/app.js
python3 -m json.tool packaging/ai-privacy-check/config/privilege >/dev/null
python3 -m json.tool packaging/ai-privacy-check/config/resource >/dev/null
python3 -m json.tool packaging/ai-privacy-check/app/ui/config >/dev/null
grep -Eq '^install_dep_apps[[:space:]]*=[[:space:]]*python312$' packaging/ai-privacy-check/manifest
if [ -d packaging/ai-privacy-check/app/docker ]; then
  echo "检测到遗留的 Docker 打包目录" >&2
  exit 1
fi
for script in packaging/ai-privacy-check/cmd/*; do
  bash -n "$script"
done

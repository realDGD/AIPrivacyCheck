#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_DIR="$PROJECT_DIR/packaging/ai-privacy-check"
DIST_DIR="$PROJECT_DIR/dist"
FNPACK_BIN="${FNPACK_BIN:-}"

if [ -z "$FNPACK_BIN" ]; then
  if command -v fnpack >/dev/null 2>&1; then
    FNPACK_BIN="$(command -v fnpack)"
  else
    # 尝试在项目上级目录寻找 fnpack 可执行文件
    for candidate in "$PROJECT_DIR/.."/*fnpack*; do
      if [ -x "$candidate" ]; then
        FNPACK_BIN="$candidate"
        break
      fi
    done
  fi
fi

if [ -z "$FNPACK_BIN" ] || [ ! -x "$FNPACK_BIN" ]; then
  echo "未找到可执行的 fnpack 工具。" >&2
  echo "请设置环境变量 FNPACK_BIN（例如：export FNPACK_BIN=/path/to/fnpack）或将 fnpack 加入系统 PATH。" >&2
  exit 1
fi

"$PROJECT_DIR/scripts/test.sh"
mkdir -p "$DIST_DIR"
chmod +x "$PACKAGE_DIR"/cmd/*

export COPYFILE_DISABLE=1
cd "$DIST_DIR"
"$FNPACK_BIN" build --directory "$PACKAGE_DIR"
VERSION="$(grep -E '^version[[:space:]]*=' "$PACKAGE_DIR/manifest" | awk -F'=' '{print $2}' | tr -d '[:space:]')"
mv -f ai-privacy-check.fpk "ai-privacy-check_${VERSION}_all.fpk"
echo "已生成：$DIST_DIR/ai-privacy-check_${VERSION}_all.fpk"

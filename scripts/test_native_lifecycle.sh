#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/ai-privacy-check-native.XXXXXX)"
PACKAGE_APP="$PROJECT_DIR/packaging/ai-privacy-check/app"
LIFECYCLE="$PROJECT_DIR/packaging/ai-privacy-check/cmd/main"

cleanup() {
  TRIM_APPDEST="$TEST_ROOT/app" \
    TRIM_PKGVAR="$TEST_ROOT/var" \
    TRIM_TEMP_LOGFILE="$TEST_ROOT/lifecycle-error.log" \
    AI_PRIVACY_PYTHON_BIN="$(command -v python3)" \
    "$LIFECYCLE" stop >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

cp -R "$PACKAGE_APP" "$TEST_ROOT/app"
mkdir -p "$TEST_ROOT/var"

export TRIM_APPDEST="$TEST_ROOT/app"
export TRIM_PKGVAR="$TEST_ROOT/var"
export TRIM_TEMP_LOGFILE="$TEST_ROOT/lifecycle-error.log"
export AI_PRIVACY_PYTHON_BIN="$(command -v python3)"

"$LIFECYCLE" start
"$LIFECYCLE" status
health="$(curl --silent --show-error --fail \
  --unix-socket "$TRIM_APPDEST/ai-privacy-check.sock" \
  http://localhost/app/ai-privacy-check/api/health)"
printf '%s' "$health" | python3 -c 'import json, sys; data=json.load(sys.stdin); assert data["ok"] is True; assert data["version"] == "0.3.0"'
"$LIFECYCLE" stop

if [ -e "$TRIM_APPDEST/ai-privacy-check.sock" ]; then
  echo "原生生命周期测试失败：Socket 未清理" >&2
  exit 1
fi

echo "fnOS Native 生命周期冒烟测试通过"

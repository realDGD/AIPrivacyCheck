#!/bin/bash
# ==============================================================================
# AI Privacy Check (AI 脱敏器) - fnOS GPU & Isolated Runtime Test Script
# Tests NVIDIA GPU hardware probe and isolated PyTorch CUDA runtime environment.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$PROJECT_DIR/packaging/ai-privacy-check/app/server"

echo "=== 1. Checking Host NVIDIA Hardware Probe ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[OK] Found nvidia-smi on PATH"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
else
    echo "[INFO] nvidia-smi not found on PATH. CPU mode will be used by default."
fi

echo ""
echo "=== 2. Testing DeviceManager Diagnostic Telemetry ==="
uv run python3 - << PYEOF
import sys
from pathlib import Path
sys.path.insert(0, "$SERVER_DIR")
from privacy.device import DEVICE_MANAGER
diag = DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
print("Requested Device :", diag.get("requested_device"))
print("Actual Device    :", diag.get("actual_device"))
print("NVIDIA Available :", diag.get("hardware", {}).get("nvidia_available"))
print("CUDA Available   :", diag.get("cuda_available"))
print("GPU Count        :", diag.get("cuda_device_count"))
if diag.get("warnings"):
    print("Warnings         :", diag.get("warnings"))
print("Model Devices    :", diag.get("model_devices"))
PYEOF

echo ""
echo "=== 3. Model Scope & Catalog Readiness ==="
uv run python3 - << PYEOF
import sys
from pathlib import Path
sys.path.insert(0, "$SERVER_DIR")
from privacy.model_catalog import list_all_models
models = list_all_models()
print(f"Total catalog models: {len(models)}")
for m in models:
    print(f"- [{m.slot}] {m.id} ({m.display_name}) -> Repo: {m.repo_id} (Rev: {m.revision})")
PYEOF

echo ""
echo "=== GPU & Runtime diagnostics test completed successfully ==="

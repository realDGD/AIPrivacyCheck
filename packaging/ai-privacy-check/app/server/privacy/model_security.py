"""Pre-load security gate for model directories.

Refuses remote-code execution shipped inside model directories: strips
`allow_remote` / `plugins` declarations from configuration.json BEFORE any
worker loads the model, and rejects when the directory cannot be made safe.

The gate runs at worker-load time (not only at install time) so weights
imported or downloaded by older app versions (v0.6.3 and earlier) are
neutralized on first use after upgrade. The production worker does not
opt into trust_remote_code and blocks/sanitizes the currently supported
ModelScope remote-code configuration paths (e.g. `allow_remote` and `plugins`
declarations in configuration.json, which would otherwise trigger pip install
of requirements.txt or execution of repository Python modules).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("ai_privacy.model_security")

REMOTE_CODE_KEYS = ("allow_remote", "plugins")


class ModelSecurityGateError(RuntimeError):
    """Raised when a model directory cannot be made safe for worker load."""


def gate_model_security(model_dir: Path) -> Dict[str, Any]:
    """Ensures the model directory carries no remote-code declarations.

    Returns {"status": "safe"|"sanitized", "removed": [...], "detail": str}.
    Raises ModelSecurityGateError when the gate cannot verify safety
    (unreadable/corrupt configuration that previously declared remote code,
    or an unwritable directory), which callers must treat as load refusal.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "configuration.json"
    if not config_path.is_file():
        return {"status": "safe", "removed": [], "detail": "no configuration.json"}

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelSecurityGateError(
            f"模型 configuration.json 无法解析，拒绝加载: {config_path} ({exc})"
        ) from exc

    if not isinstance(config, dict):
        return {"status": "safe", "removed": [], "detail": "configuration.json is not an object"}

    removed = [key for key in REMOTE_CODE_KEYS if key in config]
    if not removed:
        return {"status": "safe", "removed": [], "detail": "already clean"}

    for key in removed:
        config.pop(key, None)

    temporary = config_path.with_name("configuration.json.gate.tmp")
    try:
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, config_path)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise ModelSecurityGateError(
            f"模型目录携带远程代码声明且无法净化，拒绝加载: {config_path} ({exc})"
        ) from exc

    logger.warning(
        "安全门已在加载前净化模型配置 (移除 %s): %s", ", ".join(removed), config_path
    )
    return {"status": "sanitized", "removed": removed, "detail": f"removed {', '.join(removed)}"}

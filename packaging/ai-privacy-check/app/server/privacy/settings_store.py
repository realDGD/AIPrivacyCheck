"""Persistent Settings Store for AI Privacy Check.

Manages ${DATA_DIR}/settings.json with thread-safe atomic file transactions.
Prevents loss of requested_device, slot enable/disable states, and active model selections across restarts.
Safely falls back to defaults if settings file is absent or corrupted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("ai_privacy.settings")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "schema_version": 1,
    "requested_device": "auto",
    "slots": {
        "built_in": True,
        "chinese_ie": True,
        "general_pii": False,
        "semantic_privacy": False,
    },
    "active_models": {
        "chinese_ie": "siamese-uie",
        "general_pii": "gliner-pii-edge",
        "semantic_privacy": "memprivacy-1.7b-rl",
    },
}


class SettingsStore:
    """Thread-safe persistent settings storage."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.settings_file = data_dir / "settings.json"
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.settings_file.is_file():
                self._cache = json.loads(json.dumps(DEFAULT_SETTINGS))
                return

            try:
                with self.settings_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("schema_version") == 1:
                    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
                    if "requested_device" in data:
                        merged["requested_device"] = str(data["requested_device"])
                    if isinstance(data.get("slots"), dict):
                        merged["slots"].update(data["slots"])
                    if isinstance(data.get("active_models"), dict):
                        merged["active_models"].update(data["active_models"])
                    self._cache = merged
                else:
                    logger.warning("settings.json 格式或版本不兼容，回退至默认配置")
                    self._cache = json.loads(json.dumps(DEFAULT_SETTINGS))
            except Exception as exc:
                logger.warning(f"读取 settings.json 失败 ({exc})，使用默认配置")
                self._cache = json.loads(json.dumps(DEFAULT_SETTINGS))

    def _save(self) -> None:
        """Atomic write using temporary file and os.replace."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = self.data_dir / "settings.json.tmp"
        payload = json.dumps(self._cache, indent=2, ensure_ascii=False)
        try:
            with tmp_file.open("w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.settings_file)
        except Exception as exc:
            logger.error(f"持久化 settings.json 失败: {exc}")
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    def get_requested_device(self) -> str:
        with self._lock:
            return self._cache.get("requested_device", "auto")

    def set_requested_device(self, device: str) -> None:
        valid = ("auto", "cpu", "cuda")
        dev = device.lower() if device.lower() in valid else "auto"
        with self._lock:
            if self._cache.get("requested_device") != dev:
                self._cache["requested_device"] = dev
                self._save()

    def get_slot_enabled(self, slot: str, default: bool = False) -> bool:
        with self._lock:
            slots = self._cache.get("slots", {})
            return bool(slots.get(slot, default))

    def set_slot_enabled(self, slot: str, enabled: bool) -> None:
        with self._lock:
            slots = self._cache.setdefault("slots", {})
            if slots.get(slot) != enabled:
                slots[slot] = bool(enabled)
                self._save()

    def get_active_model(self, slot: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            models = self._cache.get("active_models", {})
            return models.get(slot, default)

    def set_active_model(self, slot: str, model_id: str) -> None:
        with self._lock:
            models = self._cache.setdefault("active_models", {})
            if models.get(slot) != model_id:
                models[slot] = str(model_id)
                self._save()

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._cache))


_GLOBAL_SETTINGS_STORE: Optional[SettingsStore] = None


def get_settings_store(data_dir: Optional[Path] = None) -> SettingsStore:
    global _GLOBAL_SETTINGS_STORE
    if _GLOBAL_SETTINGS_STORE is None:
        target_dir = data_dir or Path("/tmp")
        _GLOBAL_SETTINGS_STORE = SettingsStore(target_dir)
    elif data_dir is not None and _GLOBAL_SETTINGS_STORE.data_dir != data_dir:
        _GLOBAL_SETTINGS_STORE = SettingsStore(data_dir)
    return _GLOBAL_SETTINGS_STORE

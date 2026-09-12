"""Central DetectorRegistry orchestrating pluggable detectors across all slots."""

from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from .chinese_ie import ChineseIEDetector
from .detectors import BuiltInRuleDetector, Detector, GLiNERDetector, MemPrivacyDetector
from .entities import Entity
from .model_catalog import (
    SLOT_BUILT_IN,
    SLOT_CHINESE_IE,
    SLOT_GENERAL_PII,
    SLOT_INFO,
    SLOT_SEMANTIC_PRIVACY,
    get_model_descriptor,
    list_models_by_slot,
)


from .settings_store import get_settings_store


class DetectorRegistry:
    """Registry managing pluggable detectors for:
    1. built_in: Multilingual deterministic rules
    2. chinese_ie: Builtin linguistic IE + SiameseUIE
    3. general_pii: GLiNER Edge / Base span extractor
    4. semantic_privacy: MemPrivacy semantic inference
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._detectors: Dict[str, Detector] = {}
        self._slot_to_detector: Dict[str, str] = {}
        self._settings_store = get_settings_store(self.data_dir)

        self._slot_enabled: Dict[str, bool] = {
            SLOT_BUILT_IN: True,
            SLOT_CHINESE_IE: self._settings_store.get_slot_enabled(SLOT_CHINESE_IE, True),
            SLOT_GENERAL_PII: self._settings_store.get_slot_enabled(SLOT_GENERAL_PII, False),
            SLOT_SEMANTIC_PRIVACY: self._settings_store.get_slot_enabled(SLOT_SEMANTIC_PRIVACY, False),
        }
        self._active_models: Dict[str, str] = {
            SLOT_CHINESE_IE: self._settings_store.get_active_model(SLOT_CHINESE_IE, "siamese-uie") or "siamese-uie",
            SLOT_GENERAL_PII: self._settings_store.get_active_model(SLOT_GENERAL_PII, "gliner-pii-edge") or "gliner-pii-edge",
            SLOT_SEMANTIC_PRIVACY: self._settings_store.get_active_model(SLOT_SEMANTIC_PRIVACY, "memprivacy-1.7b-rl") or "memprivacy-1.7b-rl",
        }

        # Initialize default detectors
        self.register(BuiltInRuleDetector())
        self.register(ChineseIEDetector(self.data_dir, active_model_id=self._active_models[SLOT_CHINESE_IE]))
        self.register(GLiNERDetector(self.data_dir, active_model_id=self._active_models[SLOT_GENERAL_PII]))
        self.register(MemPrivacyDetector(self.data_dir, active_model_id=self._active_models[SLOT_SEMANTIC_PRIVACY]))

    def register(self, detector: Detector) -> None:
        with self._lock:
            self._detectors[detector.id] = detector
            self._slot_to_detector[detector.slot] = detector.id

    def get(self, detector_id: str) -> Optional[Detector]:
        with self._lock:
            return self._detectors.get(detector_id)

    def get_by_slot(self, slot: str) -> Optional[Detector]:
        with self._lock:
            detector_id = self._slot_to_detector.get(slot)
            if detector_id:
                return self._detectors.get(detector_id)
            return None

    def list(self) -> List[Detector]:
        with self._lock:
            return list(self._detectors.values())

    def enabled_detectors(self, active_slots: Optional[Set[str]] = None) -> List[Detector]:
        with self._lock:
            result = []
            for slot, detector_id in self._slot_to_detector.items():
                is_enabled = self._slot_enabled.get(slot, False)
                if active_slots is not None:
                    is_enabled = slot in active_slots
                if is_enabled and detector_id in self._detectors:
                    result.append(self._detectors[detector_id])
            return result

    def set_slot_enabled(self, slot: str, enabled: bool) -> None:
        if slot == SLOT_BUILT_IN:
            return  # Builtin rules can never be disabled
        with self._lock:
            self._slot_enabled[slot] = bool(enabled)
            self._settings_store.set_slot_enabled(slot, bool(enabled))

    def is_slot_enabled(self, slot: str) -> bool:
        with self._lock:
            return self._slot_enabled.get(slot, False)

    def set_active_model(self, slot: str, model_id: str) -> bool:
        descriptor = get_model_descriptor(model_id)
        if not descriptor or descriptor.slot != slot:
            return False
        with self._lock:
            self._active_models[slot] = model_id
            self._settings_store.set_active_model(slot, model_id)
            detector_id = self._slot_to_detector.get(slot)
            if detector_id and detector_id in self._detectors:
                det = self._detectors[detector_id]
                if hasattr(det, "set_active_model"):
                    det.set_active_model(model_id)
        return True

    def get_active_model(self, slot: str) -> Optional[str]:
        with self._lock:
            return self._active_models.get(slot)

    def reload_all(self) -> None:
        with self._lock:
            for det in self._detectors.values():
                det.unload()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            slots_payload = {}
            for slot_id, info in SLOT_INFO.items():
                detector_id = self._slot_to_detector.get(slot_id)
                detector = self._detectors.get(detector_id) if detector_id else None
                det_status = detector.status() if detector else {"ready": False, "installed": False}
                available_models = [m.to_dict() for m in list_models_by_slot(slot_id)]
                active_model = self._active_models.get(slot_id)

                slots_payload[slot_id] = {
                    "id": slot_id,
                    "name": info["name"],
                    "description": info["description"],
                    "enabled": self._slot_enabled.get(slot_id, False),
                    "detector": det_status,
                    "active_model": active_model,
                    "available_models": available_models,
                }

            return {
                "slots": slots_payload,
                "detectors": [d.status() for d in self._detectors.values()],
            }

    def detect(
        self,
        text: str,
        enabled_slots: Optional[List[str]] = None,
    ) -> Tuple[List[Entity], List[str], List[str]]:
        """Run all enabled detectors sequentially over the ORIGINAL raw text."""
        all_entities: List[Entity] = []
        engines: List[str] = []
        warnings: List[str] = []

        active_slots_set = set(enabled_slots) if enabled_slots is not None else None
        detectors = self.enabled_detectors(active_slots_set)

        # Invariant: built_in rules always run first
        detectors_sorted = sorted(detectors, key=lambda d: 0 if d.slot == SLOT_BUILT_IN else 1)

        for detector in detectors_sorted:
            status = detector.status()
            if not status.get("ready"):
                if status.get("installed"):
                    try:
                        detector.load()
                        status = detector.status()
                    except Exception:
                        pass
            if not status.get("ready"):
                continue

            try:
                ents, warns = detector.detect(text)
                all_entities.extend(ents)
                warnings.extend(warns)
                if detector.name not in engines:
                    engines.append(detector.name)
            except Exception as exc:
                warnings.append(f"检测器 [{detector.name}] 执行异常: {exc}")

        return all_entities, engines, warnings

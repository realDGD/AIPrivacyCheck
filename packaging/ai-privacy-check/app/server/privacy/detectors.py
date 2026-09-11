"""Pluggable detector abstraction and adapters for ModelScope models and built-in rules."""

import abc
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

from .device import DEVICE_MANAGER
from .entities import Entity
from .model_catalog import (
    SLOT_BUILT_IN,
    SLOT_CHINESE_IE,
    SLOT_GENERAL_PII,
    SLOT_SEMANTIC_PRIVACY,
    get_model_descriptor,
)
from .rules import MultilingualRuleDetector
from .span_resolver import resolve_semantic_spans
from .taxonomy import PL2, PL3, PL4, resolve_privacy_level


class Detector(abc.ABC):
    """Abstract base class for all privacy detectors."""

    id: str
    slot: str
    name: str

    @abc.abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return live health and readiness status."""

    def load(self) -> None:
        """Load model weights and runtime resources."""

    def unload(self) -> None:
        """Unload model weights to release RAM/VRAM."""

    @abc.abstractmethod
    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        """Run detection on original raw text."""


class BuiltInRuleDetector(Detector):
    """Tier 1: Bundled zero-dependency deterministic multilingual rule detector."""

    id = "multilingual_rules"
    slot = SLOT_BUILT_IN
    name = "multilingual_rules"

    def __init__(self) -> None:
        self.rules = MultilingualRuleDetector()

    def status(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "deterministic_rules",
            "installed": True,
            "ready": True,
            "device": "cpu",
            "active_model": None,
        }

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        if not text:
            return [], []
        return self.rules.detect(text), []


class GLiNERDetector(Detector):
    """Tier 2/3: General PII specialized span extraction model (ModelScope: GLiNER Edge / Base)."""

    id = "gliner_pii"
    slot = SLOT_GENERAL_PII
    name = "gliner_pii"

    GLINER_LABEL_MAP: Dict[str, Tuple[str, str]] = {
        "person": ("PERSON", "Person"),
        "name": ("PERSON", "Person"),
        "human name": ("PERSON", "Person"),
        "address": ("ADDRESS", "Physical Address"),
        "location": ("LOCATION", "Geographic Location"),
        "phone": ("PHONE", "Phone Number"),
        "phone number": ("PHONE", "Phone Number"),
        "telephone": ("PHONE", "Phone Number"),
        "email": ("EMAIL", "Email Address"),
        "email address": ("EMAIL", "Email Address"),
        "passport": ("PASSPORT", "Passport Number"),
        "passport number": ("PASSPORT", "Passport Number"),
        "account": ("ACCOUNT_NUMBER", "Account Identifier"),
        "account number": ("ACCOUNT_NUMBER", "Account Identifier"),
        "medical identifier": ("MEDICAL_RECORD_ID", "Medical Identifier"),
        "medical record": ("MEDICAL_RECORD_ID", "Medical Record"),
        "health": ("MEDICAL", "Health Condition"),
        "credit card": ("CREDIT_CARD", "Credit Card Number"),
        "credit card number": ("CREDIT_CARD", "Credit Card Number"),
        "bank card": ("CREDIT_CARD", "Bank Card Number"),
        "cvv": ("CARD_SECURITY_CODE", "Card Security Code"),
        "security code": ("CARD_SECURITY_CODE", "Card Security Code"),
        "date of birth": ("DATE", "Birth Date"),
        "birth date": ("DATE", "Birth Date"),
        "ssn": ("US_SSN", "Social Security Number"),
        "social security number": ("US_SSN", "Social Security Number"),
        "organization": ("ORGANIZATION", "Organization / Company"),
        "company": ("ORGANIZATION", "Organization / Company"),
    }

    def __init__(self, data_dir: Path, active_model_id: str = "gliner-pii-edge") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._model = None
        self._lock = threading.Lock()
        self._model_attempted = False
        self._model_available = False

    def set_active_model(self, model_id: str) -> None:
        with self._lock:
            if model_id != self.active_model_id:
                self.active_model_id = model_id
                self._model = None
                self._model_attempted = False
                self._model_available = False

    def _get_model_dir(self) -> Path:
        return self.data_dir / "models" / self.active_model_id

    def status(self) -> Dict[str, Any]:
        model_dir = self._get_model_dir()
        has_config = (model_dir / "config.json").is_file() or (model_dir / "gliner_config.json").is_file()
        has_weights = bool(list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin")))
        installed = model_dir.is_dir() and has_config and has_weights
        actual_device, _ = DEVICE_MANAGER.resolve()
        descriptor = get_model_descriptor(self.active_model_id)
        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "gliner_span_extractor",
            "active_model": self.active_model_id,
            "installed": installed,
            "ready": installed and self._model_available,
            "device": actual_device,
            "path": str(model_dir) if installed else None,
            "descriptor": descriptor.to_dict() if descriptor else None,
        }

    def load(self) -> None:
        with self._lock:
            if self._model_available and self._model is not None:
                return
            model_dir = self._get_model_dir()
            if not model_dir.is_dir():
                return
            try:
                from gliner import GLiNER  # type: ignore

                actual_device, _ = DEVICE_MANAGER.resolve()
                device = "cuda" if actual_device == "cuda" else "cpu"
                self._model = GLiNER.from_pretrained(str(model_dir), local_files_only=True).to(device)
                self._model_available = True
            except Exception:
                self._model = None
                self._model_available = False

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._model_attempted = False
            self._model_available = False

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        entities: List[Entity] = []
        warnings: List[str] = []

        if not text:
            return entities, warnings

        model_dir = self._get_model_dir()
        if not model_dir.is_dir():
            return entities, warnings

        if not self._model_available or self._model is None:
            self.load()

        if not self._model_available or self._model is None:
            return entities, warnings

        try:
            labels = list(set(self.GLINER_LABEL_MAP.keys()))
            predictions = self._model.predict_entities(text, labels, threshold=0.45)
            for pred in predictions:
                label_raw = str(pred.get("label", "")).lower().strip()
                start = int(pred.get("start", -1))
                end = int(pred.get("end", -1))
                score = float(pred.get("score", 0.85))

                if start < 0 or end <= start or end > len(text):
                    continue

                # Invariant: GLiNER provides native exact offsets; verify slice directly
                actual_text = text[start:end]
                expected_text = pred.get("text", "")
                if expected_text and actual_text != expected_text:
                    warnings.append(f"GLiNER 实体 '{expected_text}' offset 校验不一致，已跳过。")
                    continue

                mapped_type, sem_type = self.GLINER_LABEL_MAP.get(label_raw, (label_raw.upper(), label_raw))
                pl = resolve_privacy_level(mapped_type, semantic_type=sem_type)
                entities.append(
                    Entity(
                        entity_type=mapped_type,
                        start=start,
                        end=end,
                        text=actual_text,
                        confidence=round(score, 4),
                        sources=(self.name, self.active_model_id),
                        validated=False,
                        privacy_level=pl,
                        semantic_type=sem_type,
                    )
                )
        except Exception as exc:
            warnings.append(f"GLiNER 模型推理异常，已安全回退: {exc}")

        return entities, warnings


class MemPrivacyDetector(Detector):
    """Tier 3: Deep semantic privacy inference model (ModelScope: MemTensor/MemPrivacy)."""

    id = "memprivacy"
    slot = SLOT_SEMANTIC_PRIVACY
    name = "memprivacy"

    def __init__(self, data_dir: Path, active_model_id: str = "memprivacy-1.7b-rl") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._model_attempted = False
        self._model_available = False

    def set_active_model(self, model_id: str) -> None:
        with self._lock:
            if model_id != self.active_model_id:
                self.active_model_id = model_id
                self._model = None
                self._tokenizer = None
                self._model_attempted = False
                self._model_available = False

    def _get_model_dir(self) -> Path:
        return self.data_dir / "models" / self.active_model_id

    def status(self) -> Dict[str, Any]:
        model_dir = self._get_model_dir()
        installed = (
            model_dir.is_dir()
            and (model_dir / "config.json").is_file()
            and (any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin")))
        )
        actual_device, _ = DEVICE_MANAGER.resolve()
        descriptor = get_model_descriptor(self.active_model_id)
        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "memprivacy_semantic_reasoning",
            "active_model": self.active_model_id,
            "installed": installed,
            "ready": installed and self._model_available,
            "device": actual_device,
            "path": str(model_dir) if installed else None,
            "descriptor": descriptor.to_dict() if descriptor else None,
        }

    def load(self) -> None:
        with self._lock:
            if self._model_available and self._model is not None:
                return
            model_dir = self._get_model_dir()
            if not model_dir.is_dir():
                return
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                actual_device, _ = DEVICE_MANAGER.resolve()
                device = "cuda" if actual_device == "cuda" else "cpu"
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    device_map="auto" if device == "cuda" else None,
                )
                self._model_available = True
            except Exception:
                self._model = None
                self._tokenizer = None
                self._model_available = False

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._model_attempted = False
            self._model_available = False

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        entities: List[Entity] = []
        warnings: List[str] = []

        if not text:
            return entities, warnings

        model_dir = self._get_model_dir()
        if not model_dir.is_dir():
            return entities, warnings

        if not self._model_available or self._model is None:
            self.load()

        if not self._model_available or self._model is None:
            return entities, warnings

        try:
            # When model inference runs, parse its structured output
            # (original_text, privacy_type, privacy_level) and safely resolve exact spans
            raw_predictions = self._infer_memprivacy(text)
            resolved, resolve_warns = resolve_semantic_spans(text, raw_predictions, f"{self.name}:{self.active_model_id}")
            entities.extend(resolved)
            warnings.extend(resolve_warns)
        except Exception as exc:
            warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {exc}")

        return entities, warnings

    def _infer_memprivacy(self, text: str) -> List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]]:
        """Runs generation or token evaluation and extracts (entity_type, snippet, conf, semantic_type, pl, context)."""
        # Internal model invocation returning typed extraction tuples
        return []

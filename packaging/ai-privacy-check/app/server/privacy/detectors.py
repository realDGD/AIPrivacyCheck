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
        actual_device, _, _ = DEVICE_MANAGER.resolve_for_framework("torch")
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

                actual_device, _, _ = DEVICE_MANAGER.resolve_for_framework("torch")
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


MEMPRIVACY_SYSTEM_PROMPT = (
    "You are a professional privacy extraction assistant. "
    "Please identify privacy-sensitive information in the user's text and output a JSON list "
    "containing original_text, privacy_type, and privacy_level (PL2, PL3, or PL4). "
    "Do not modify or mask original_text."
)

MEMPRIVACY_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    "real name": ("PERSON", "Real Name"),
    "name": ("PERSON", "Real Name"),
    "姓名": ("CN_NAME", "Real Name"),
    "人名": ("CN_NAME", "Real Name"),
    "phone": ("PHONE", "Phone Number"),
    "phone number": ("PHONE", "Phone Number"),
    "telephone": ("PHONE", "Phone Number"),
    "手机号": ("CN_PHONE_NUMBER", "Phone Number"),
    "电话": ("PHONE", "Phone Number"),
    "email": ("EMAIL", "Email Address"),
    "email address": ("EMAIL", "Email Address"),
    "邮箱": ("EMAIL", "Email Address"),
    "id card": ("GOVERNMENT_ID", "ID Card"),
    "id number": ("GOVERNMENT_ID", "ID Card"),
    "身份证": ("CN_ID_CARD", "ID Card"),
    "身份证号": ("CN_ID_CARD", "ID Card"),
    "bank card": ("CREDIT_CARD", "Bank Card"),
    "card number": ("CREDIT_CARD", "Bank Card"),
    "银行卡": ("CN_BANK_CARD", "Bank Card"),
    "password": ("PASSWORD", "Password"),
    "code": ("PASSWORD", "Secret / Password"),
    "密码": ("PASSWORD", "Password"),
    "口令": ("PASSWORD", "Password"),
    "address": ("ADDRESS", "Physical Address"),
    "住址": ("CN_ADDRESS", "Physical Address"),
    "家庭住址": ("CN_ADDRESS", "Physical Address"),
    "地址": ("ADDRESS", "Physical Address"),
    "medical record": ("MEDICAL_RECORD_ID", "Medical Record"),
    "health": ("MEDICAL", "Health Condition"),
    "病历": ("MEDICAL_RECORD_ID", "Medical Record"),
    "病情": ("MEDICAL", "Health Condition"),
    "organization": ("ORGANIZATION", "Organization"),
    "company": ("ORGANIZATION", "Organization"),
    "公司": ("ORGANIZATION", "Organization"),
    "机构": ("ORGANIZATION", "Organization"),
    "学校": ("ORGANIZATION", "Organization"),
}


def parse_memprivacy_json(
    raw_output: str, full_text: str
) -> List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]]:
    """Extracts structured privacy entities from MemPrivacy JSON output into typed tuples.

    Tuple layout:
      (entity_type, snippet, confidence, semantic_type, privacy_level, context_hint)
    """
    if not raw_output or not full_text:
        return []

    cleaned = raw_output.strip()
    # Match json code block or standalone bracketed array
    json_str: Optional[str] = None
    if "```json" in cleaned:
        parts = cleaned.split("```json", 1)[1]
        if "```" in parts:
            json_str = parts.split("```", 1)[0].strip()
    elif "```" in cleaned:
        parts = cleaned.split("```", 1)[1]
        if "```" in parts:
            json_str = parts.split("```", 1)[0].strip()

    if not json_str:
        import re

        array_match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)

    if not json_str and cleaned.startswith("[") and cleaned.endswith("]"):
        json_str = cleaned

    parsed_items: List[Any] = []
    if json_str:
        import json
        import re

        # Remove trailing commas before closing braces/brackets
        sanitized = re.sub(r",\s*([\]\}])", r"\1", json_str)
        try:
            data = json.loads(sanitized)
            if isinstance(data, list):
                parsed_items = data
        except Exception:
            parsed_items = []

    results: List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]] = []
    for item in parsed_items:
        if not isinstance(item, dict):
            continue

        # Extract original_text
        snippet = (
            item.get("original_text")
            or item.get("text")
            or item.get("entity")
            or item.get("value")
            or item.get("content")
        )
        if not snippet or not isinstance(snippet, str):
            continue
        snippet = snippet.strip()
        if not snippet or snippet not in full_text:
            continue

        # Extract privacy_type
        raw_type = (
            item.get("privacy_type")
            or item.get("type")
            or item.get("category")
            or item.get("entity_type")
            or item.get("label")
            or "PII"
        )
        raw_type_str = str(raw_type).strip()
        lookup_key = raw_type_str.lower()
        mapped_type, sem_type = MEMPRIVACY_TYPE_MAP.get(
            lookup_key,
            (raw_type_str.upper().replace(" ", "_"), raw_type_str),
        )

        # Extract privacy_level
        raw_pl = (
            item.get("privacy_level")
            or item.get("level")
            or item.get("pl")
            or item.get("risk_level")
        )
        pl_val: Optional[str] = None
        if raw_pl:
            cand = str(raw_pl).strip().upper()
            if cand in (PL2, PL3, PL4, "PL1"):
                pl_val = cand

        if not pl_val:
            pl_val = resolve_privacy_level(mapped_type, semantic_type=sem_type)

        context_hint = item.get("context") or item.get("context_hint")
        if context_hint and not isinstance(context_hint, str):
            context_hint = None

        results.append((mapped_type, snippet, 0.95, sem_type, pl_val, context_hint))

    return results


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
        actual_device, _, _ = DEVICE_MANAGER.resolve_for_framework("torch")
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

                actual_device, _, _ = DEVICE_MANAGER.resolve_for_framework("torch")
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
            raw_predictions = self._infer_memprivacy(text)
            resolved, resolve_warns = resolve_semantic_spans(text, raw_predictions, f"{self.name}:{self.active_model_id}")
            entities.extend(resolved)
            warnings.extend(resolve_warns)
        except Exception as exc:
            warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {exc}")

        return entities, warnings

    def _infer_memprivacy(self, text: str) -> List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]]:
        """Runs generation or token evaluation and extracts (entity_type, snippet, conf, semantic_type, pl, context)."""
        if not self._model or not self._tokenizer:
            return []

        try:
            messages = [
                {"role": "system", "content": MEMPRIVACY_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = (
                    f"<|im_start|>system\n{MEMPRIVACY_SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            inputs = self._tokenizer(prompt, return_tensors="pt")
            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return parse_memprivacy_json(generated_text, text)
        except Exception:
            return []

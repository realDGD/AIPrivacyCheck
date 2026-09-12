"""Pluggable detector abstraction and adapters for ModelScope models and built-in rules.

Control-plane process NEVER directly imports torch, transformers, or gliner.
All neural inference is delegated to isolated worker subprocesses running within
dedicated virtual environments via pure JSONL IPC over stdin/stdout.
"""

from __future__ import annotations

import abc
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .device import DEVICE_MANAGER
from .entities import Entity
from .model_catalog import (
    SLOT_BUILT_IN,
    SLOT_CHINESE_IE,
    SLOT_GENERAL_PII,
    SLOT_SEMANTIC_PRIVACY,
    check_model_integrity,
    get_model_descriptor,
)
from .rules import MultilingualRuleDetector
from .span_resolver import resolve_semantic_spans
from .taxonomy import PL2, PL3, PL4, resolve_privacy_level
from .worker_client import get_worker_client

logger = logging.getLogger("ai_privacy.detectors")


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
    """Tier 3: General PII specialized span extraction model (ModelScope: GLiNER Edge / Base)."""

    id = "gliner_pii"
    slot = SLOT_GENERAL_PII
    name = "gliner_pii"

    GLINER_LABEL_MAP: Dict[str, Tuple[str, str]] = {
        "person": ("PERSON", "Person"),
        "people": ("PERSON", "Person"),
        "name": ("PERSON", "Person"),
        "organization": ("ORGANIZATION", "Organization"),
        "company": ("ORGANIZATION", "Organization"),
        "phone number": ("PHONE", "Phone Number"),
        "phone": ("PHONE", "Phone Number"),
        "mobile phone": ("CN_PHONE_NUMBER", "Phone Number"),
        "email": ("EMAIL", "Email Address"),
        "email address": ("EMAIL", "Email Address"),
        "passport number": ("PASSPORT", "Passport"),
        "passport": ("PASSPORT", "Passport"),
        "driver license": ("GOVERNMENT_ID", "Driver License"),
        "driving license": ("GOVERNMENT_ID", "Driver License"),
        "social security number": ("US_SSN", "Social Security Number"),
        "ssn": ("US_SSN", "Social Security Number"),
        "credit card number": ("CREDIT_CARD", "Credit Card"),
        "credit card": ("CREDIT_CARD", "Credit Card"),
        "bank account": ("ACCOUNT_NUMBER", "Bank Account"),
        "bank account number": ("ACCOUNT_NUMBER", "Bank Account"),
        "address": ("ADDRESS", "Physical Address"),
        "street address": ("ADDRESS", "Physical Address"),
        "location": ("LOCATION", "Location"),
        "city": ("LOCATION", "Location"),
        "country": ("LOCATION", "Location"),
        "username": ("USERNAME", "Username"),
        "user name": ("USERNAME", "Username"),
        "date of birth": ("DATE", "Date of Birth"),
        "birth date": ("DATE", "Date of Birth"),
        "ip address": ("IP_ADDRESS", "IP Address"),
        "ipv4": ("IP_ADDRESS", "IP Address"),
        "ipv6": ("IPV6_ADDRESS", "IPv6 Address"),
        "mac address": ("MAC_ADDRESS", "MAC Address"),
    }

    def __init__(self, data_dir: Path, active_model_id: str = "gliner-pii-edge") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._lock = threading.Lock()

    def set_active_model(self, model_id: str) -> None:
        with self._lock:
            if model_id != self.active_model_id:
                self.active_model_id = model_id
                get_worker_client(self.data_dir).stop_worker_for_model(self.active_model_id)

    def _get_model_dir(self) -> Path:
        return self.data_dir / "models" / self.active_model_id

    def status(self) -> Dict[str, Any]:
        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        actual_device = model_res.get("actual_device", "cpu")
        model_ready = installed and bool(model_res.get("ready", False))
        descriptor = get_model_descriptor(self.active_model_id)

        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "gliner_span_extractor",
            "active_model": self.active_model_id,
            "installed": installed,
            "ready": model_ready,
            "device": actual_device,
            "path": str(model_dir) if installed else None,
            "descriptor": descriptor.to_dict() if descriptor else None,
        }

    def load(self) -> None:
        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        if not installed:
            return
        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        if not model_res.get("ready"):
            return
        profile = model_res.get("runtime_profile")
        dev = model_res.get("actual_device", "cpu")
        if profile:
            try:
                worker = get_worker_client(self.data_dir).get_worker(
                    self.active_model_id, profile, device=dev
                )
                worker.query({"action": "load", "model_path": str(model_dir)})
            except Exception as exc:
                logger.warning(f"预热 GLiNER worker 失败: {exc}")

    def unload(self) -> None:
        with self._lock:
            get_worker_client(self.data_dir).stop_worker_for_model(self.active_model_id)

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        entities: List[Entity] = []
        warnings: List[str] = []

        if not text:
            return entities, warnings

        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        if not installed:
            return entities, warnings

        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        if not model_res.get("ready"):
            return entities, warnings

        profile = model_res.get("runtime_profile")
        dev = model_res.get("actual_device", "cpu")
        if not profile:
            return entities, warnings

        try:
            worker_client = get_worker_client(self.data_dir)
            worker = worker_client.get_worker(self.active_model_id, profile, device=dev)
            _, infer_timeout = worker_client.get_timeout_for_model(self.active_model_id)

            labels = list(set(self.GLINER_LABEL_MAP.keys()))
            res = worker.query({
                "action": "detect",
                "model_path": str(model_dir),
                "text": text,
                "labels": labels,
                "threshold": 0.40,
            }, timeout=infer_timeout)

            if not res.get("ok"):
                err_msg = res.get("error") or "Worker returned ok=False"
                warnings.append(f"GLiNER 推理未完成: {err_msg}")
                return entities, warnings

            for pred in res.get("entities", []):
                label_raw = str(pred.get("label", "")).lower().strip()
                start = int(pred.get("start", -1))
                end = int(pred.get("end", -1))
                score = float(pred.get("score", 0.85))

                if start < 0 or end <= start or end > len(text):
                    continue

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
    "detailed address": ("ADDRESS", "Detailed Address"),
    "address": ("ADDRESS", "Detailed Address"),
    "地址": ("CN_ADDRESS", "Detailed Address"),
    "detailed_address": ("ADDRESS", "Detailed Address"),
    "id number": ("GOVERNMENT_ID", "ID Number"),
    "id card": ("CN_ID_CARD", "ID Number"),
    "身份证": ("CN_ID_CARD", "ID Number"),
    "financial account": ("ACCOUNT_NUMBER", "Financial Account"),
    "bank card": ("CREDIT_CARD", "Bank Card"),
    "银行卡": ("CN_BANK_CARD", "Bank Card"),
    "transaction record": ("TRADE_RECORD", "Transaction Record"),
    "assets/income": ("FINANCIAL", "Assets/Income"),
    "medical health": ("MEDICAL", "Medical Health"),
    "medical": ("MEDICAL", "Medical Health"),
    "health": ("MEDICAL", "Medical Health"),
    "precise location": ("LOCATION_TRAJECTORY", "Precise Location"),
    "itinerary/trajectory": ("LOCATION_TRAJECTORY", "Itinerary/Trajectory"),
    "biometrics": ("BIOMETRIC", "Biometrics"),
    "communication content": ("COMMUNICATION", "Communication Content"),
    "sensitive identity": ("GOVERNMENT_ID", "Sensitive Identity"),
    "judicial record": ("JUDICIAL", "Judicial Record"),
    "password": ("PASSWORD", "Password"),
    "密码": ("PASSWORD", "Password"),
    "verification code": ("SECRET", "Verification Code"),
    "验证码": ("SECRET", "Verification Code"),
    "token": ("API_TOKEN", "Token"),
    "key": ("SECRET", "Key"),
    "private key": ("PRIVATE_KEY", "Private Key"),
    "payment security code": ("CARD_SECURITY_CODE", "Payment Security Code"),
    "database connection string": ("DATABASE_URI", "Database Connection String"),
    "vulnerability details": ("COMMERCIAL_SECRET", "Vulnerability Details"),
    "business secret": ("COMMERCIAL_SECRET", "Business Secret"),
    "account id/username": ("USERNAME", "Account ID/Username"),
    "username": ("USERNAME", "Account ID/Username"),
    "network identifier": ("IP_ADDRESS", "Network Identifier"),
    "identity background": ("IDENTITY_BACKGROUND", "Identity Background"),
    "relationship info": ("RELATIONSHIP", "Relationship Info"),
}


def parse_memprivacy_json(
    raw_output: str,
    original_text: str,
) -> List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]]:
    """Extracts entity array from MemPrivacy generated text and validates against document."""
    cleaned = raw_output.strip()

    # Strip Qwen reasoning think blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>")[0]

    # Find JSON array
    fence_m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence_m:
        cand = fence_m.group(1).strip()
    else:
        arr_m = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if arr_m:
            cand = arr_m.group(0).strip()
        else:
            cand = cleaned

    cand = re.sub(r",\s*([\]\}])", r"\1", cand)

    try:
        data = json.loads(cand)
    except Exception:
        data = []
        for obj_m in re.finditer(r"\{[^{}]*\}", cand):
            try:
                item = json.loads(re.sub(r",\s*\}", "}", obj_m.group(0)))
                data.append(item)
            except Exception:
                continue

    if not isinstance(data, list):
        return []

    results: List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        snippet = (
            item.get("original_text")
            or item.get("text")
            or item.get("entity")
            or ""
        )
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        snippet = snippet.strip()
        if snippet not in original_text:
            continue

        raw_type = (
            item.get("privacy_type")
            or item.get("type")
            or item.get("label")
            or "PII"
        )
        raw_type_str = str(raw_type).strip()
        lookup_key = raw_type_str.lower()
        mapped_type, sem_type = MEMPRIVACY_TYPE_MAP.get(
            lookup_key,
            (raw_type_str.upper().replace(" ", "_"), raw_type_str),
        )

        raw_pl = (
            item.get("privacy_level")
            or item.get("level")
            or item.get("pl")
            or item.get("risk_level")
        )
        pl_val: Optional[str] = None
        if raw_pl:
            cand_pl = str(raw_pl).strip().upper()
            if cand_pl in (PL2, PL3, PL4, "PL1"):
                pl_val = cand_pl

        if not pl_val:
            pl_val = resolve_privacy_level(mapped_type, semantic_type=sem_type)

        context_hint = item.get("context") or item.get("context_hint")
        if context_hint and not isinstance(context_hint, str):
            context_hint = None

        results.append((mapped_type, snippet, 0.95, sem_type, pl_val, context_hint))

    return results


class MemPrivacyDetector(Detector):
    """Tier 4: Deep semantic privacy inference model (ModelScope: MemTensor/MemPrivacy)."""

    id = "memprivacy"
    slot = SLOT_SEMANTIC_PRIVACY
    name = "memprivacy"

    def __init__(self, data_dir: Path, active_model_id: str = "memprivacy-1.7b-rl") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._lock = threading.Lock()

    def set_active_model(self, model_id: str) -> None:
        with self._lock:
            if model_id != self.active_model_id:
                self.active_model_id = model_id
                get_worker_client(self.data_dir).stop_worker_for_model(self.active_model_id)

    def _get_model_dir(self) -> Path:
        return self.data_dir / "models" / self.active_model_id

    def status(self) -> Dict[str, Any]:
        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        actual_device = model_res.get("actual_device", "cpu")
        model_ready = installed and bool(model_res.get("ready", False))
        descriptor = get_model_descriptor(self.active_model_id)

        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "memprivacy_semantic_reasoning",
            "active_model": self.active_model_id,
            "installed": installed,
            "ready": model_ready,
            "device": actual_device,
            "path": str(model_dir) if installed else None,
            "descriptor": descriptor.to_dict() if descriptor else None,
        }

    def load(self) -> None:
        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        if not installed:
            return
        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        if not model_res.get("ready"):
            return
        profile = model_res.get("runtime_profile")
        dev = model_res.get("actual_device", "cpu")
        if profile:
            try:
                worker = get_worker_client(self.data_dir).get_worker(
                    self.active_model_id, profile, device=dev
                )
                worker.query({"action": "load", "model_path": str(model_dir)})
            except Exception as exc:
                logger.warning(f"预热 MemPrivacy worker 失败: {exc}")

    def unload(self) -> None:
        with self._lock:
            get_worker_client(self.data_dir).stop_worker_for_model(self.active_model_id)

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        entities: List[Entity] = []
        warnings: List[str] = []

        if not text:
            return entities, warnings

        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        if not installed:
            return entities, warnings

        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        if not model_res.get("ready"):
            return entities, warnings

        profile = model_res.get("runtime_profile")
        dev = model_res.get("actual_device", "cpu")
        if not profile:
            return entities, warnings

        try:
            worker_client = get_worker_client(self.data_dir)
            worker = worker_client.get_worker(self.active_model_id, profile, device=dev)
            _, infer_timeout = worker_client.get_timeout_for_model(self.active_model_id)

            res = worker.query({
                "action": "detect",
                "model_path": str(model_dir),
                "text": text,
                "real_name": "unknown",
                "max_new_tokens": 2048,
            }, timeout=infer_timeout)

            if not res.get("ok"):
                err_msg = res.get("error") or "Worker returned ok=False"
                warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {err_msg}")
                return entities, warnings

            if res.get("truncated"):
                warnings.append("MemPrivacy 达到最大生成长度上限，长文本可能存在截断。")

            extracted_items = []
            for item in res.get("entities", []):
                snippet = item.get("original_text", "").strip()
                if not snippet or snippet not in text:
                    continue
                raw_type = item.get("privacy_type", "PII")
                mapped_type, sem_type = MEMPRIVACY_TYPE_MAP.get(
                    str(raw_type).lower(),
                    (str(raw_type).upper().replace(" ", "_"), str(raw_type))
                )
                pl_val = item.get("privacy_level") or resolve_privacy_level(mapped_type, semantic_type=sem_type)
                ctx = item.get("context")
                extracted_items.append((mapped_type, snippet, 0.95, sem_type, pl_val, ctx))

            resolved, resolve_warns = resolve_semantic_spans(
                text, extracted_items, f"{self.name}:{self.active_model_id}"
            )
            entities.extend(resolved)
            warnings.extend(resolve_warns)
        except Exception as exc:
            warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {exc}")

        return entities, warnings

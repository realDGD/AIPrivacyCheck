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
import time
from typing import Any, Dict, List, Optional, Tuple

from .device import DEVICE_MANAGER
from .entities import Entity
from .model_security import gate_model_security
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
from .worker_client import (
    WorkerRetiredError,
    get_worker_client,
    is_cpu_oom_exception,
    is_cpu_oom_response,
    is_cuda_oom_exception,
    is_cuda_oom_response,
)

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

    # Single source of truth for the prediction threshold. Production
    # detectors and the benchmark harness MUST both import this value so the
    # benchmark measures exactly what production runs.
    # Re-swept on the 100-doc layered corpus (0.35..0.65): Detection F1 peaks
    # at 0.50 (P 55.2 / R 30.9 / F1 39.6, model-only), PII-free FPR 3/9 and
    # USERNAME->PERSON confusion 4 both flat vs 0.55; 0.60+ trades 9-19 F1
    # points for no FPR gain.
    GLINER_DEFAULT_THRESHOLD = 0.50

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
                old_model_id = self.active_model_id
                self.active_model_id = model_id
                get_worker_client(self.data_dir).stop_worker_for_model(old_model_id)

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
                gate_model_security(model_dir)
                worker = get_worker_client(self.data_dir).get_worker(
                    self.active_model_id, profile, device=dev
                )
                worker.query({"action": "load", "model_path": str(model_dir)})
            except Exception as exc:
                logger.warning(f"预热 GLiNER worker 失败: {exc}")

    def unload(self) -> None:
        with self._lock:
            get_worker_client(self.data_dir).stop_worker_for_model(self.active_model_id)

    @staticmethod
    def _is_plausible_username(
        text: str,
        start: int,
        end: int,
        entity_text: str,
    ) -> bool:
        """Plausibility filter for GLiNER USERNAME predictions to eliminate Chinese false positives."""
        if not entity_text:
            return False
        val = entity_text.strip()
        if len(val) < 2 or len(val) > 40:
            return False

        # Sentence/clause-breaking punctuation is unacceptable in any username
        clause_punctuations = ("，", "。", "！", "？", "；", "…", "\n", "\r", ",", ";", "!", "?")
        if any(ch in val for ch in clause_punctuations):
            return False

        # Check for CJK characters
        has_cjk = any(
            "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" or "\u20000" <= ch <= "\u2ceaf"
            for ch in val
        )

        if not has_cjk:
            # Case A: ASCII / Latin token-like username (allowed without explicit context)
            if re.search(r"\s", val):
                return False
            if not re.search(r"[a-zA-Z0-9]", val):
                return False
            clean_token = val.rstrip(".")
            if len(clean_token) >= 2 and re.fullmatch(r"[a-zA-Z0-9_@.-]+", clean_token):
                return True
            return False

        # Case B: CJK / Chinese / CJK-heavy username
        # Must have explicit account context in nearby local window (preceding ~20-30 chars, following ~10-20 chars)
        pre_window = text[max(0, start - 30):start]
        post_window = text[end:min(len(text), end + 20)]

        context_indicators = (
            "用户名",
            "用户名称",
            "登录账号",
            "登录帐号",
            "登录id",
            "登录ID",
            "账号",
            "账户名",
            "帐号",
            "username",
            "user name",
            "login id",
            "account name",
        )
        pre_lower = pre_window.lower()
        post_lower = post_window.lower()

        has_explicit_context = any(ind in pre_lower or ind in post_lower for ind in context_indicators)
        if not has_explicit_context:
            return False

        # Even with context, CJK span must not exceed reasonable handle length
        if len(val) > 20:
            return False

        # Must not contain excessive whitespace
        if len(val.split()) > 2:
            return False

        # Narrative words indicate natural language sentence rather than a username
        narrative_words = (
            "今天", "明天", "昨天", "去了", "来了", "然后", "打电话", "联系了",
            "上班", "下班", "吃饭", "回家", "发生", "看到", "听到", "觉得",
            "因为", "所以", "如果", "但是", "而且", "不过", "由于", "不仅",
            "公司", "会议", "工作", "报告", "讨论", "协商", "安排",
        )
        if any(w in val for w in narrative_words):
            return False

        return True

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
            gate_model_security(model_dir)
            worker_client = get_worker_client(self.data_dir)
            _, infer_timeout = worker_client.get_timeout_for_model(self.active_model_id, device=dev)
            with worker_client.cuda_execution_session(device=dev, timeout=float(infer_timeout)):
                worker = worker_client.get_worker(self.active_model_id, profile, device=dev)
                labels = list(set(self.GLINER_LABEL_MAP.keys()))
                res = worker.query({
                    "action": "detect",
                    "model_path": str(model_dir),
                    "text": text,
                    "labels": labels,
                    "threshold": self.GLINER_DEFAULT_THRESHOLD,
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

                if mapped_type == "USERNAME" and not self._is_plausible_username(text, start, end, actual_text):
                    continue
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


def choose_memprivacy_generation_budget(text: str) -> int:
    """Computes a bounded max_new_tokens generation budget for MemPrivacy based on text length.
    Short text (<= 1000 chars): 256 tokens.
    Medium text (1001 ~ 4000 chars): 384 tokens.
    Long text (> 4000 chars): 512 tokens.
    Hard upper bound: <= 512 tokens (avoids KV cache explosion and timeout).
    """
    length = len(text)
    if length <= 1000:
        budget = 256
    elif length <= 4000:
        budget = 384
    else:
        budget = 512
    return min(budget, 512)


def chunk_memprivacy_text(
    text: str,
    max_chunk_chars: int = 3000,
    overlap_chars: int = 200,
) -> List[Tuple[int, int, str]]:
    """Splits long text into bounded overlapping chunks for semantic inference.
    Returns a list of (start_offset, end_offset, chunk_text).
    Texts <= 3500 chars are returned as a single chunk to preserve global context.
    """
    if not text:
        return []
    n = len(text)
    if n <= 3500:
        return [(0, n, text)]

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    delimiters = ("\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";")

    while start < n:
        ideal_end = min(n, start + max_chunk_chars)
        if ideal_end >= n:
            chunks.append((start, n, text[start:n]))
            break

        search_start = max(start + 100, ideal_end - overlap_chars)
        split_pos = -1
        for delim in delimiters:
            pos = text.rfind(delim, search_start, ideal_end)
            if pos != -1:
                split_pos = pos + len(delim)
                break

        if split_pos == -1 or split_pos <= start:
            chunk_end = ideal_end
        else:
            chunk_end = split_pos

        chunks.append((start, chunk_end, text[start:chunk_end]))
        next_start = max(start + 1, chunk_end - overlap_chars)
        if next_start >= n or next_start <= start:
            break
        start = next_start

    return chunks


class MemPrivacyDetector(Detector):
    """Tier 4: Deep semantic privacy inference model (ModelScope: MemTensor/MemPrivacy)."""

    id = "memprivacy"
    slot = SLOT_SEMANTIC_PRIVACY
    name = "memprivacy"

    def __init__(self, data_dir: Path, active_model_id: str = "memprivacy-1.7b-rl") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def set_active_model(self, model_id: str) -> None:
        with self._lock:
            if model_id != self.active_model_id:
                old_model_id = self.active_model_id
                self.active_model_id = model_id
                get_worker_client(self.data_dir).stop_worker_for_model(old_model_id)

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
                gate_model_security(model_dir)
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
        if not text:
            return [], []

        model_dir = self._get_model_dir()
        installed, _ = check_model_integrity(self.active_model_id, model_dir)
        if not installed:
            return [], []

        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        if not model_res.get("ready"):
            return [], []

        profile = model_res.get("runtime_profile")
        dev = model_res.get("actual_device", "cpu")
        if not profile:
            return [], []

        is_cuda = (dev == "cuda")
        started_at = time.monotonic()
        total_budget = 240.0 if is_cuda else 480.0
        deadline = started_at + total_budget

        acquire_timeout = max(0.1, deadline - time.monotonic())
        acquired_lock = self._inference_lock.acquire(blocking=True, timeout=acquire_timeout)
        if not acquired_lock:
            return [], ["MemPrivacy 语义推理繁忙，本次未在时间预算内执行。"]

        try:
            worker_client = get_worker_client(self.data_dir)
            cuda_timeout = max(0.1, deadline - time.monotonic())
            try:
                with worker_client.cuda_execution_session(device=dev, timeout=cuda_timeout):
                    return self._detect_session_locked(text, model_dir, profile, dev, deadline)
            except TimeoutError:
                return [], ["MemPrivacy 语义推理繁忙，本次未在时间预算内执行。"]
        finally:
            self._inference_lock.release()

    def _detect_session_locked(
        self,
        text: str,
        model_dir: Path,
        profile: str,
        dev: str,
        deadline: float,
    ) -> Tuple[List[Entity], List[str]]:
        entities: List[Entity] = []
        warnings: List[str] = []
        worker_client = get_worker_client(self.data_dir)
        is_cuda = (dev == "cuda")

        try:
            if is_cuda:
                worker_client.stop_other_cuda_workers(keep_model_id=self.active_model_id)

            worker = worker_client.get_worker(self.active_model_id, profile, device=dev)
            _, infer_timeout = worker_client.get_timeout_for_model(self.active_model_id, device=dev)

            chunks = chunk_memprivacy_text(text)
            total_chunks = len(chunks)
            completed_chunks = 0
            seen_keys = set()
            truncated_chunk_count = 0
            timed_out_budget = False

            gate_model_security(model_dir)
            for chunk_start, chunk_end, chunk_text in chunks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out_budget = True
                    break

                chunk_timeout = max(1.0, min(float(infer_timeout), remaining))
                chunk_budget = choose_memprivacy_generation_budget(chunk_text)
                req_payload = {
                    "action": "detect",
                    "model_path": str(model_dir),
                    "text": chunk_text,
                    "real_name": "unknown",
                    "max_new_tokens": chunk_budget,
                }

                try:
                    res = worker.query(req_payload, timeout=int(chunk_timeout))
                except WorkerRetiredError:
                    try:
                        worker = worker_client.get_worker(self.active_model_id, profile, device=dev)
                        remaining_retry = deadline - time.monotonic()
                        if remaining_retry <= 0:
                            timed_out_budget = True
                            break
                        retry_timeout = max(1.0, min(float(infer_timeout), remaining_retry))
                        res = worker.query(req_payload, timeout=int(retry_timeout))
                    except Exception as retry_exc:
                        if is_cuda_oom_exception(retry_exc, device=dev):
                            warnings.append("MemPrivacy 可用显存不足，已终止语义模型并释放显存，其他检测结果不受影响。")
                            try:
                                worker_client.stop_worker_for_model(self.active_model_id)
                            except Exception:
                                pass
                        elif is_cpu_oom_exception(retry_exc, device=dev):
                            warnings.append("MemPrivacy 内存不足，已安全回退。")
                            try:
                                worker_client.stop_worker_for_model(self.active_model_id)
                            except Exception:
                                pass
                        elif "timed out" in str(retry_exc).lower():
                            warnings.append(f"MemPrivacy 语义推理单块超时，已保留已完成结果: {retry_exc}")
                        else:
                            warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {retry_exc}")
                        break
                except Exception as query_exc:
                    if is_cuda_oom_exception(query_exc, device=dev):
                        warnings.append("MemPrivacy 可用显存不足，已终止语义模型并释放显存，其他检测结果不受影响。")
                        try:
                            worker_client.stop_worker_for_model(self.active_model_id)
                        except Exception:
                            pass
                    elif is_cpu_oom_exception(query_exc, device=dev):
                        warnings.append("MemPrivacy 内存不足，已安全回退。")
                        try:
                            worker_client.stop_worker_for_model(self.active_model_id)
                        except Exception:
                            pass
                    elif "timed out" in str(query_exc).lower():
                        warnings.append(f"MemPrivacy 语义推理单块超时，已保留已完成结果: {query_exc}")
                    else:
                        warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {query_exc}")
                    break

                if is_cuda and is_cuda_oom_response(res):
                    warnings.append("MemPrivacy 可用显存不足，已终止语义模型并释放显存，其他检测结果不受影响。")
                    try:
                        worker_client.stop_worker_for_model(self.active_model_id)
                    except Exception:
                        pass
                    break
                elif not is_cuda and is_cpu_oom_response(res):
                    warnings.append("MemPrivacy 内存不足，已安全回退。")
                    try:
                        worker_client.stop_worker_for_model(self.active_model_id)
                    except Exception:
                        pass
                    break

                if not res.get("ok"):
                    err_msg = res.get("error") or "Worker returned ok=False"
                    if is_cuda and is_cuda_oom_response(res):
                        warnings.append("MemPrivacy 可用显存不足，已终止语义模型并释放显存，其他检测结果不受影响。")
                    else:
                        warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {err_msg}")
                    break

                if res.get("truncated"):
                    truncated_chunk_count += 1

                extracted_items = []
                for item in res.get("entities", []):
                    snippet = item.get("original_text", "").strip()
                    if not snippet or snippet not in chunk_text:
                        continue
                    raw_type = item.get("privacy_type", "PII")
                    mapped_type, sem_type = MEMPRIVACY_TYPE_MAP.get(
                        str(raw_type).lower(),
                        (str(raw_type).upper().replace(" ", "_"), str(raw_type))
                    )
                    pl_val = item.get("privacy_level") or resolve_privacy_level(mapped_type, semantic_type=sem_type)
                    ctx = item.get("context")
                    extracted_items.append((mapped_type, snippet, 0.95, sem_type, pl_val, ctx))

                chunk_resolved, chunk_warns = resolve_semantic_spans(
                    chunk_text, extracted_items, f"{self.name}:{self.active_model_id}"
                )
                warnings.extend(chunk_warns)

                for cent in chunk_resolved:
                    g_start = chunk_start + cent.start
                    g_end = chunk_start + cent.end
                    if g_start < 0 or g_end > len(text) or text[g_start:g_end] != cent.text:
                        continue
                    dedup_key = (cent.entity_type, g_start, g_end, cent.text)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)
                    entities.append(
                        Entity(
                            entity_type=cent.entity_type,
                            start=g_start,
                            end=g_end,
                            text=cent.text,
                            confidence=cent.confidence,
                            sources=cent.sources,
                            validated=cent.validated,
                            privacy_level=cent.privacy_level,
                            semantic_type=cent.semantic_type,
                        )
                    )
                completed_chunks += 1

            if timed_out_budget and completed_chunks == 0:
                warnings.append("MemPrivacy 语义推理时间预算耗尽，未完成分块处理。")

            if truncated_chunk_count > 0:
                warnings.append(f"MemPrivacy 有 {truncated_chunk_count} 个分块达到生成上限，部分超长上下文可能存在截断。")

            if 0 < completed_chunks < total_chunks:
                warnings.append(f"MemPrivacy 仅完成 {completed_chunks}/{total_chunks} 个语义分块，结果可能不完整。")

        except Exception as exc:
            warnings.append(f"MemPrivacy 语义推理异常，已安全回退: {exc}")
        finally:
            if is_cuda:
                try:
                    worker_client.stop_worker_for_model(self.active_model_id)
                except Exception as exc:
                    logger.warning(f"释放 MemPrivacy CUDA worker 显存异常: {exc}")

        return entities, warnings

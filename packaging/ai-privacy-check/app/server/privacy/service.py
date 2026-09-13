"""Privacy detection orchestration uniting deterministic rules and pluggable ModelScope detectors."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from .device import DEVICE_MANAGER
from .merge import merge_entities
from .model_catalog import (
    SLOT_BUILT_IN,
    SLOT_CHINESE_IE,
    SLOT_GENERAL_PII,
    SLOT_SEMANTIC_PRIVACY,
    list_all_models,
)
from .registry import DetectorRegistry
from .rules import supported_rule_types


MAX_TEXT_CHARS = 500_000


class PrivacyService:
    """Orchestration service combining:
    1. Built-in Deterministic Rules (built_in slot)
    2. Chinese Information Extraction (chinese_ie slot)
    3. General PII Span Extraction (general_pii slot: GLiNER Edge/Base)
    4. Deep Semantic Privacy Reasoning (semantic_privacy slot: MemPrivacy)
    -> Span arbitration via merge_entities()
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.registry = DetectorRegistry(self.data_dir)

    @property
    def rules(self):
        det = self.registry.get("multilingual_rules")
        return det.rules if det else None

    @property
    def chinese_ie(self):
        return self.registry.get("chinese_ie")

    def detect(
        self,
        text: str,
        use_model: bool = False,
        slots: Optional[List[str]] = None,
        policy_level: Optional[str] = None,
    ) -> Dict[str, object]:
        if not isinstance(text, str):
            raise ValueError("text 必须是字符串")
        if not text.strip():
            raise ValueError("请输入需要检测的文本")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("单次文本不能超过 500,000 个字符")

        started = time.perf_counter()

        # Determine enabled slots: built_in and chinese_ie are always included
        active_slots = [SLOT_BUILT_IN, SLOT_CHINESE_IE]
        if slots is not None:
            active_slots = list(set(slots).union({SLOT_BUILT_IN, SLOT_CHINESE_IE}))
        elif use_model:
            # Legacy use_model=True without explicit slots only enables general_pii (GLiNER).
            # MemPrivacy requires explicit opt-in via slots.
            active_slots.append(SLOT_GENERAL_PII)

        # All detectors run sequentially on the ORIGINAL raw text.
        # Rules run first, models read original text (never mask before model inference).
        entities, engines, warnings = self.registry.detect(text, enabled_slots=active_slots)

        # Check if optional models were requested but unready
        optional_requested = (SLOT_GENERAL_PII in active_slots) or (SLOT_SEMANTIC_PRIVACY in active_slots)
        if optional_requested and not any(e in engines for e in ("gliner_pii", "memprivacy")):
            # Emit graceful status notice
            gliner_status = self.registry.get("gliner_pii").status() if self.registry.get("gliner_pii") else {}
            memprivacy_status = self.registry.get("memprivacy").status() if self.registry.get("memprivacy") else {}
            if not gliner_status.get("ready") and not memprivacy_status.get("ready"):
                warnings.append("增强模型尚未就绪，已使用基础规则与中文语义引擎完成检测。")

        # Merge entities using priority arbitration:
        # validated deterministic rule > strict deterministic rule > specialized span > semantic generative
        merged = merge_entities(entities)

        # Privacy Policy filtering (PL1 - PL4)
        pl_ranks = {"PL1": 1, "PL2": 2, "PL3": 3, "PL4": 4}
        effective_policy = policy_level.upper() if policy_level and policy_level.upper() in pl_ranks else "PL2"
        if policy_level and policy_level.upper() in pl_ranks:
            req_rank = pl_ranks[policy_level.upper()]
            merged = [e for e in merged if pl_ranks.get(e.resolved_privacy_level, 2) >= req_rank]

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        counts: Dict[str, int] = {}
        for entity in merged:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

        return {
            "entities": [entity.to_dict(text=text) for entity in merged],
            "counts": counts,
            "engines": engines,
            "warnings": warnings,
            "policy_level": effective_policy,
            "processing_ms": elapsed_ms,
            "text_length": len(text),
        }

    def reset_models(self) -> None:
        self.registry.reload_all()

    def capabilities(self) -> Dict[str, object]:
        return {
            "rule_types": list(supported_rule_types()),
            "registry": self.registry.status(),
            "models": [m.to_dict() for m in list_all_models()],
            "device": DEVICE_MANAGER.probe_diagnostics(),
            "max_text_chars": MAX_TEXT_CHARS,
        }

"""Privacy detection orchestration uniting deterministic rules, Chinese IE, and deep models."""

from pathlib import Path
import time
from typing import Dict, List

from .chinese_ie import ChineseIEDetector
from .device import DEVICE_MANAGER
from .merge import merge_entities
from .model import OpenAIModelDetector
from .rules import MultilingualRuleDetector, supported_rule_types


MAX_TEXT_CHARS = 500_000


class PrivacyService:
    """Three-tier privacy detection service:
    1. DeterministicRuleDetector (rules.py)
    2. ChineseIEDetector (chinese_ie.py)
    3. OpenAIPrivacyFilterDetector (model.py)
    -> merge_entities()
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.rules = MultilingualRuleDetector()
        self.chinese_ie = ChineseIEDetector(self.data_dir)
        self.model = OpenAIModelDetector(self.data_dir)

    def detect(self, text: str, use_model: bool = False) -> Dict[str, object]:
        if not isinstance(text, str):
            raise ValueError("text 必须是字符串")
        if not text.strip():
            raise ValueError("请输入需要检测的文本")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("单次文本不能超过 500,000 个字符")

        started = time.perf_counter()
        entities = self.rules.detect(text)
        engines = [self.rules.name]
        warnings: List[str] = []

        # 2. Chinese IE semantic detection (zero network, local-first)
        ie_entities, ie_warnings = self.chinese_ie.detect(text)
        entities.extend(ie_entities)
        if self.chinese_ie.name not in engines:
            engines.append(self.chinese_ie.name)
        warnings.extend(ie_warnings)

        # 3. Optional deep model enhancement
        if use_model:
            if self.model.status()["ready"]:
                try:
                    model_entities, model_warnings = self.model.detect(text)
                    entities.extend(model_entities)
                    warnings.extend(model_warnings)
                    engines.append(self.model.name)
                except Exception as exc:
                    warnings.append(f"OpenAI 模型检测失败，已使用规则与中文语义引擎: {exc}")
            else:
                warnings.append("OpenAI Privacy Filter 尚未就绪，已使用规则与中文语义引擎。")

        merged = merge_entities(entities)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        counts: Dict[str, int] = {}
        for entity in merged:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

        return {
            "entities": [entity.to_dict() for entity in merged],
            "counts": counts,
            "engines": engines,
            "warnings": warnings,
            "processing_ms": elapsed_ms,
            "text_length": len(text),
        }

    def reset_models(self) -> None:
        self.model.reset()
        self.chinese_ie = ChineseIEDetector(self.data_dir)

    def capabilities(self) -> Dict[str, object]:
        return {
            "rule_types": list(supported_rule_types()),
            "chinese_ie": self.chinese_ie.status(),
            "model_types": [
                "ACCOUNT_NUMBER",
                "PRIVATE_ADDRESS",
                "EMAIL",
                "PRIVATE_PERSON",
                "PHONE",
                "PRIVATE_URL",
                "PRIVATE_DATE",
                "SECRET",
            ],
            "device": DEVICE_MANAGER.probe_diagnostics(),
            "max_text_chars": MAX_TEXT_CHARS,
        }

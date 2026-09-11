"""Privacy detection orchestration."""

from pathlib import Path
import time
from typing import Dict, List

from .merge import merge_entities
from .model import OpenAIModelDetector
from .rules import MultilingualRuleDetector, supported_rule_types


MAX_TEXT_CHARS = 500_000


class PrivacyService:
    """Combine multilingual local rules with the optional OpenAI model."""

    def __init__(self, data_dir: Path) -> None:
        self.rules = MultilingualRuleDetector()
        self.model = OpenAIModelDetector(data_dir)

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

        if use_model:
            if self.model.status()["ready"]:
                try:
                    model_entities, model_warnings = self.model.detect(text)
                    entities.extend(model_entities)
                    warnings.extend(model_warnings)
                    engines.append(self.model.name)
                except Exception as exc:
                    warnings.append("模型检测失败，已仅使用中文规则：{}".format(exc))
            else:
                warnings.append("OpenAI Privacy Filter 尚未就绪，已仅使用中文规则。")

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

    def capabilities(self) -> Dict[str, object]:
        return {
            "rule_types": list(supported_rule_types()),
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
            "max_text_chars": MAX_TEXT_CHARS,
        }

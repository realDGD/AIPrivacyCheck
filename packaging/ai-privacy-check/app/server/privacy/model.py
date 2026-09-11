"""Optional OpenAI Privacy Filter adapter.

The heavy runtime and checkpoint live in the fnOS application data directory and
are imported only when the user enables model-assisted detection.
"""

import importlib
import os
from pathlib import Path
import sys
import threading
from typing import List, Tuple

from .entities import Entity


OPF_SOURCE_REVISION = "f7f00ca7fb869683eb732c010299d901457f19c3"
OPF_MODEL_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"

MODEL_LABEL_MAP = {
    "account_number": "ACCOUNT_NUMBER",
    "private_address": "PRIVATE_ADDRESS",
    "private_email": "EMAIL",
    "private_person": "PRIVATE_PERSON",
    "private_phone": "PHONE",
    "private_url": "PRIVATE_URL",
    "private_date": "PRIVATE_DATE",
    "secret": "SECRET",
    "redacted": "SECRET",
}


class OpenAIModelDetector:
    """Lazy, process-local wrapper around the public ``opf.OPF`` API."""

    name = "openai_privacy_filter"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.package_dir = self.data_dir / "python-packages"
        self.checkpoint_dir = self.data_dir / "models" / "privacy-filter"
        self._model = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        package_ready = (self.package_dir / ".opf-revision").is_file() and (self.package_dir / "opf").is_dir()
        model_ready = (
            (self.checkpoint_dir / "config.json").is_file()
            and any(self.checkpoint_dir.glob("*.safetensors"))
        )
        if package_ready and model_ready:
            state = "ready"
        elif package_ready:
            state = "model_missing"
        else:
            state = "packages_missing"
        return {
            "state": state,
            "ready": package_ready and model_ready,
            "package_ready": package_ready,
            "model_ready": model_ready,
            "source_revision": OPF_SOURCE_REVISION,
            "model_revision": OPF_MODEL_REVISION,
            "checkpoint": str(self.checkpoint_dir),
        }

    def reset(self) -> None:
        with self._lock:
            self._model = None
            importlib.invalidate_caches()

    def _load(self):
        if not self.status()["ready"]:
            raise RuntimeError("OpenAI Privacy Filter 尚未安装完成")
        package_path = str(self.package_dir)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        from opf import OPF

        device = os.environ.get("OPF_DEVICE", "cpu").strip().lower()
        if device not in ("cpu", "cuda"):
            device = "cpu"
        return OPF(
            model=str(self.checkpoint_dir),
            device=device,
            output_mode="typed",
            decode_mode="viterbi",
        )

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        warnings: List[str] = []
        with self._lock:
            if self._model is None:
                self._model = self._load()
            result = self._model.redact(text)

        if getattr(result, "warning", None):
            warnings.append(str(result.warning))

        entities: List[Entity] = []
        for span in result.detected_spans:
            start = int(span.start)
            end = int(span.end)
            span_text = str(span.text)
            if text[start:end] != span_text:
                occurrences = [index for index in range(len(text)) if text.startswith(span_text, index)]
                if len(occurrences) != 1:
                    warnings.append("模型返回了无法安全映射到原文的片段，已跳过一项。")
                    continue
                start = occurrences[0]
                end = start + len(span_text)
            entity_type = MODEL_LABEL_MAP.get(str(span.label), str(span.label).upper())
            entities.append(
                Entity(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    text=text[start:end],
                    confidence=0.78,
                    sources=(self.name,),
                    validated=False,
                )
            )
        return entities, warnings

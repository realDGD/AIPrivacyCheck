"""Centralized Model Catalog for AI Privacy Check.

All officially supported downloadable models originate exclusively from ModelScope (魔搭社区).
Centralizes model metadata, slot association, licensing, hardware compatibility,
and prevents scattered hardcoding across server, UI, installer, and detectors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SLOT_BUILT_IN = "built_in"
SLOT_CHINESE_IE = "chinese_ie"
SLOT_GENERAL_PII = "general_pii"
SLOT_SEMANTIC_PRIVACY = "semantic_privacy"

SLOT_INFO: Dict[str, Dict[str, str]] = {
    SLOT_BUILT_IN: {
        "id": SLOT_BUILT_IN,
        "name": "基础规则",
        "description": "多语言强规则与中文本地语言学启发式引擎，0 依赖开箱即用，始终启用。",
    },
    SLOT_CHINESE_IE: {
        "id": SLOT_CHINESE_IE,
        "name": "中文信息抽取",
        "description": "专用于复杂中文长难句的姓名、地址、机构、学校、职位等多 Schema 抽取。",
    },
    SLOT_GENERAL_PII: {
        "id": SLOT_GENERAL_PII,
        "name": "通用 PII",
        "description": "专精于国际通用个人可标识信息（PII / PHI / PCI）轻量级跨语言边界识别。",
    },
    SLOT_SEMANTIC_PRIVACY: {
        "id": SLOT_SEMANTIC_PRIVACY,
        "name": "语义隐私",
        "description": "基于长文本推理的医疗健康、财务资产、敏感行踪及人际关系等深层隐私识别与 PL 风险分级。",
    },
}


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    display_name: str
    slot: str
    provider: str
    repo_id: str
    revision: str
    license: str
    approx_size: str
    architectures: Tuple[str, ...]
    runtime: str  # "paddle", "torch", "gliner"
    supports_cpu: bool
    supports_cuda: bool
    recommended: bool
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "slot": self.slot,
            "provider": self.provider,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "license": self.license,
            "approx_size": self.approx_size,
            "architectures": list(self.architectures),
            "runtime": self.runtime,
            "supports_cpu": self.supports_cpu,
            "supports_cuda": self.supports_cuda,
            "recommended": self.recommended,
            "description": self.description,
        }


MODEL_CATALOG: Dict[str, ModelDescriptor] = {
    "siamese-uie": ModelDescriptor(
        id="siamese-uie",
        display_name="SiameseUIE Chinese Base",
        slot=SLOT_CHINESE_IE,
        provider="modelscope",
        repo_id="iic/nlp_structbert_siamese-uie_chinese-base",
        revision="v1.0.0",
        license="Apache-2.0",
        approx_size="420 MB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="paddle",
        supports_cpu=True,
        supports_cuda=True,
        recommended=True,
        description="面向中文信息抽取的通用 UIE 模型，针对人名、地址、机构、职位等复杂上下文实体具有高泛化能力。",
    ),
    "gliner-pii-edge": ModelDescriptor(
        id="gliner-pii-edge",
        display_name="GLiNER PII Edge",
        slot=SLOT_GENERAL_PII,
        provider="modelscope",
        repo_id="knowledgator/gliner-pii-edge-v1.0",
        revision="v1.0.0",
        license="Apache-2.0",
        approx_size="310 MB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=True,
        supports_cuda=True,
        recommended=True,
        description="专用通用 PII/PHI 轻量级抽取模型，资源占用极低，响应迅速，专为 NAS 本地环境优化。",
    ),
    "gliner-pii-base": ModelDescriptor(
        id="gliner-pii-base",
        display_name="GLiNER PII Base",
        slot=SLOT_GENERAL_PII,
        provider="modelscope",
        repo_id="knowledgator/gliner-multitask-v1.0",
        revision="v1.0.0",
        license="Apache-2.0",
        approx_size="850 MB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=True,
        supports_cuda=True,
        recommended=False,
        description="更高容量的多任务通用 PII 抽取模型，提供更精细的多语言实体边界判定。",
    ),
    "memprivacy-1.7b-rl": ModelDescriptor(
        id="memprivacy-1.7b-rl",
        display_name="MemPrivacy 1.7B RL",
        slot=SLOT_SEMANTIC_PRIVACY,
        provider="modelscope",
        repo_id="MemTensor/MemPrivacy-1.7B-RL",
        revision="v1.0.0",
        license="CC BY-NC-ND 4.0",
        approx_size="3.4 GB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=True,
        supports_cuda=True,
        recommended=True,
        description="基于强化学习对齐的语义级隐私理解模型，专精于医疗、财务、行踪、社会关系等长难句深层隐私提取与分级。",
    ),
    "memprivacy-4b-rl": ModelDescriptor(
        id="memprivacy-4b-rl",
        display_name="MemPrivacy 4B RL",
        slot=SLOT_SEMANTIC_PRIVACY,
        provider="modelscope",
        repo_id="MemTensor/MemPrivacy-4B-RL",
        revision="v1.0.0",
        license="CC BY-NC-ND 4.0",
        approx_size="7.8 GB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=False,
        supports_cuda=True,
        recommended=False,
        description="高参数量语义隐私理解大模型，需要具备充足专用显存（>=8GB）的 GPU 环境运行。",
    ),
}


def get_model_descriptor(model_id: str) -> Optional[ModelDescriptor]:
    return MODEL_CATALOG.get(model_id)


def list_models_by_slot(slot: str) -> List[ModelDescriptor]:
    return [m for m in MODEL_CATALOG.values() if m.slot == slot]


def list_all_models() -> List[ModelDescriptor]:
    return list(MODEL_CATALOG.values())

"""Centralized Model Catalog for AI Privacy Check.

All officially supported downloadable models originate from ModelScope (魔搭社区).
Centralizes model metadata, slot association, licensing, hardware compatibility,
and model integrity validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    runtime: str  # "torch"
    supports_cpu: bool
    supports_cuda: bool
    recommended: bool
    description: str
    # Model-specific pip requirements installed into the shared runtime venv
    # BEFORE download/smoke verification. Only declare packages empirically
    # proven to be required by the model's actual inference import chain
    # (verify against upstream metadata and a real pipeline load; never guess).
    runtime_dependencies: Tuple[str, ...] = ()

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
            "runtime_dependencies": list(self.runtime_dependencies),
        }


MODEL_CATALOG: Dict[str, ModelDescriptor] = {
    "siamese-uie": ModelDescriptor(
        id="siamese-uie",
        display_name="SiameseUIE Chinese Base",
        slot=SLOT_CHINESE_IE,
        provider="modelscope",
        repo_id="iic/nlp_structbert_siamese-uie_chinese-base",
        revision="master",
        license="Apache-2.0",
        approx_size="420 MB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=True,
        supports_cuda=True,
        recommended=True,
        description="面向中文信息抽取的通用 UIE 深度模型，针对人名、地址、机构、学校、职位等实体具有高泛化跨度抽取能力。",
        # Empirically verified against modelscope 1.40: loading the
        # siamese-uie pipeline imports modelscope.utils.config (addict),
        # modelscope.msdatasets (datasets), scipy.special.softmax,
        # modelscope.pipeline_inputs (PIL), and the pipeline builder chain
        # (simplejson, sortedcontainers). None of these are core deps of a
        # bare `modelscope` install - they only ship in its extras.
        runtime_dependencies=(
            "addict",
            "datasets",
            "scipy",
            "Pillow",
            "simplejson",
            "sortedcontainers",
        ),
    ),
    "gliner-pii-edge": ModelDescriptor(
        id="gliner-pii-edge",
        display_name="GLiNER PII Edge",
        slot=SLOT_GENERAL_PII,
        provider="modelscope",
        repo_id="knowledgator/gliner-pii-edge-v1.0",
        revision="master",
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
        repo_id="knowledgator/gliner-pii-base-v1.0",
        revision="master",
        license="Apache-2.0",
        approx_size="850 MB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=True,
        supports_cuda=True,
        recommended=False,
        description="更高容量的专用通用 PII 抽取模型，基于 DeBERTa 提供更精细的多语言实体边界判定。",
    ),
    "memprivacy-1.7b-rl": ModelDescriptor(
        id="memprivacy-1.7b-rl",
        display_name="MemPrivacy 1.7B RL",
        slot=SLOT_SEMANTIC_PRIVACY,
        provider="modelscope",
        repo_id="MemTensor/MemPrivacy-1.7B-RL",
        revision="master",
        license="CC BY-NC-ND 4.0",
        approx_size="3.4 GB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=True,
        supports_cuda=True,
        recommended=False,
        # Provenance: ModelScope hosts this checkpoint under the MemTensor org
        # (the download channel this catalog pins). The author's HuggingFace
        # namespace for the same weights is IAAR-Shanghai/MemPrivacy-1.7B-RL
        # (per the model card inside the snapshot). The ModelScope metadata
        # carries NO license field; the license statement (CC BY-NC-ND 4.0,
        # non-commercial) is only on the HF model card - verified 2026-09-13.
        description="可选的深度语义隐私模型（深度扫描模式，用户主动开启）。Benchmark v2 证实其为唯一具备语义级召回的候选（语义切片跨度覆盖召回 90%，小参数 Qwen 挑战者要么高误报要么漏检），但 3.4GB 权重 + CPU 约 9GB RAM / 数百秒延迟 + 受限显存设备 CUDA OOM 压力 + CC BY-NC-ND 非商业许可（依据作者 HF 模型卡 IAAR-Shanghai/MemPrivacy-1.7B-RL；ModelScope 元数据未携带 license 字段），不适合作为 NAS 默认推荐；生产默认检测由内置规则与 NER 承担。支持 CUDA（建议显存 >=6GB）或 CPU 模式（CPU 推理耗时较长且内存占用约 8-10GB，单请求有时间预算限制）。",
    ),
    "memprivacy-4b-rl": ModelDescriptor(
        id="memprivacy-4b-rl",
        display_name="MemPrivacy 4B RL",
        slot=SLOT_SEMANTIC_PRIVACY,
        provider="modelscope",
        # Same provenance as the 1.7B: ModelScope=MemTensor org, HF mirror=IAAR-Shanghai, license only on the HF card (CC BY-NC-ND 4.0).
        repo_id="MemTensor/MemPrivacy-4B-RL",
        revision="master",
        license="CC BY-NC-ND 4.0",
        approx_size="7.8 GB",
        architectures=("x86_64", "arm64", "aarch64"),
        runtime="torch",
        supports_cpu=False,
        supports_cuda=True,
        recommended=False,
        description="高参数量语义隐私理解大模型，需要具备充足专用显存（最低 >=12GB，推荐 16GB+）的 NVIDIA CUDA GPU 环境运行。强烈不建议在 <=8GB 显存设备（如 Tesla P4）上运行，存在极高 CUDA OOM 风险。",
    ),
}


def get_model_descriptor(model_id: str) -> Optional[ModelDescriptor]:
    return MODEL_CATALOG.get(model_id)


def list_models_by_slot(slot: str) -> List[ModelDescriptor]:
    return [m for m in MODEL_CATALOG.values() if m.slot == slot]


def list_all_models() -> List[ModelDescriptor]:
    return list(MODEL_CATALOG.values())


def check_model_integrity(model_id: str, model_dir: Path) -> Tuple[bool, Optional[str]]:
    """Strictly validates model directory contents according to model architecture contract."""
    if not isinstance(model_id, str):
        return False, f"model_id 类型无效: {type(model_id).__name__}"
    if not isinstance(model_dir, Path):
        try:
            model_dir = Path(model_dir)
        except Exception:
            return False, f"model_dir 类型无效: {type(model_dir).__name__}"

    if not model_dir.is_dir():
        return False, f"模型目录不存在: {model_dir}"

    descriptor = get_model_descriptor(model_id)
    if not descriptor:
        return False, f"未知模型 ID: {model_id}"

    if model_id == "siamese-uie":
        # PyTorch StructBERT SiameseUIE requires:
        # configuration.json, config.json, pytorch_model.bin, vocab.txt
        req_files = ["configuration.json", "config.json", "vocab.txt"]
        for rf in req_files:
            if not (model_dir / rf).is_file():
                return False, f"SiameseUIE 缺少必要配置文件: {rf}"
        has_weights = (model_dir / "pytorch_model.bin").is_file() or (model_dir / "model.safetensors").is_file()
        if not has_weights:
            return False, "SiameseUIE 缺少权重文件 (pytorch_model.bin 或 model.safetensors)"
        return True, None

    elif model_id in ("gliner-pii-edge", "gliner-pii-base"):
        # GLiNER requires config, weights, and tokenizer files
        has_config = (model_dir / "gliner_config.json").is_file() or (model_dir / "config.json").is_file()
        if not has_config:
            return False, "GLiNER 缺少 gliner_config.json 或 config.json"
        has_weights = (
            (model_dir / "pytorch_model.bin").is_file()
            or (model_dir / "model.safetensors").is_file()
        )
        if not has_weights:
            return False, "GLiNER 缺少权重文件 (pytorch_model.bin 或 model.safetensors)"
        has_tokenizer = (
            (model_dir / "tokenizer.json").is_file()
            or (model_dir / "tokenizer_config.json").is_file()
            or (model_dir / "spm.model").is_file()
        )
        if not has_tokenizer:
            return False, "GLiNER 缺少分词器文件 (tokenizer.json 或 tokenizer_config.json)"
        return True, None

    elif model_id in ("memprivacy-1.7b-rl", "memprivacy-4b-rl"):
        # MemPrivacy requires config.json, weights (safetensors or sharded), and tokenizer
        if not (model_dir / "config.json").is_file():
            return False, "MemPrivacy 缺少 config.json"
        has_weights = (
            (model_dir / "model.safetensors").is_file()
            or (model_dir / "model.safetensors.index.json").is_file()
            or bool(list(model_dir.glob("model-*.safetensors")))
            or (model_dir / "pytorch_model.bin").is_file()
        )
        if not has_weights:
            return False, "MemPrivacy 缺少权重文件 (*.safetensors)"
        has_tokenizer = (
            (model_dir / "tokenizer.json").is_file()
            or (model_dir / "vocab.json").is_file()
        )
        if not has_tokenizer:
            return False, "MemPrivacy 缺少分词器文件 (tokenizer.json 或 vocab.json)"
        return True, None

    # Fallback generic check
    has_cfg = any(model_dir.glob("*.json"))
    has_wt = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    if not (has_cfg and has_wt):
        return False, "模型文件校验失败：缺少配置文件或权重文件"
    return True, None


def resolve_for_model(
    model_id: str,
    requested_device: str,
    runtime_manager: Any,
    hardware_nvidia_available: bool,
) -> Dict[str, Any]:
    """Resolves runtime profile and execution device for a specific model, strictly enforcing capabilities.

    For example, MemPrivacy 4B has supports_cpu=False; if CUDA is unavailable, it will NOT fallback to CPU.
    """
    descriptor = get_model_descriptor(model_id)
    if not descriptor:
        return {
            "model_id": model_id,
            "requested_device": requested_device,
            "runtime_profile": None,
            "actual_device": "none",
            "ready": False,
            "fallback": False,
            "reason": f"未在目录中找到模型: {model_id}",
        }

    req = requested_device.lower()
    can_use_cuda = hardware_nvidia_available and descriptor.supports_cuda

    # Check CUDA runtime
    cuda_profile = f"{descriptor.runtime}-cuda"
    cuda_status = runtime_manager.probe_profile(cuda_profile) if runtime_manager else {}
    cuda_ready = bool(
        can_use_cuda
        and cuda_status.get("installed")
        and cuda_status.get("verified")
        and cuda_status.get("cuda_available")
    )

    # Check CPU runtime
    cpu_profile = f"{descriptor.runtime}-cpu"
    cpu_status = runtime_manager.probe_profile(cpu_profile) if runtime_manager else {}
    cpu_ready = bool(
        descriptor.supports_cpu
        and cpu_status.get("installed")
        and cpu_status.get("verified")
    )

    if req == "cuda":
        if cuda_ready:
            return {
                "model_id": model_id,
                "requested_device": "cuda",
                "runtime_profile": cuda_profile,
                "actual_device": "cuda",
                "ready": True,
                "fallback": False,
                "reason": None,
            }
        else:
            reason = "请求了 CUDA 运行，但未检测到就绪的 NVIDIA CUDA 驱动或隔离环境。"
            if not descriptor.supports_cuda:
                reason = "该模型不支持 CUDA 设备加速。"
            return {
                "model_id": model_id,
                "requested_device": "cuda",
                "runtime_profile": None,
                "actual_device": "none",
                "ready": False,
                "fallback": False,
                "reason": reason,
            }

    elif req == "cpu":
        if not descriptor.supports_cpu:
            return {
                "model_id": model_id,
                "requested_device": "cpu",
                "runtime_profile": None,
                "actual_device": "none",
                "ready": False,
                "fallback": False,
                "reason": f"模型 {descriptor.display_name} 限制仅支持 CUDA GPU 运行，不支持 CPU 模式。",
            }
        if cpu_ready:
            return {
                "model_id": model_id,
                "requested_device": "cpu",
                "runtime_profile": cpu_profile,
                "actual_device": "cpu",
                "ready": True,
                "fallback": False,
                "reason": None,
            }
        else:
            return {
                "model_id": model_id,
                "requested_device": "cpu",
                "runtime_profile": None,
                "actual_device": "none",
                "ready": False,
                "fallback": False,
                "reason": "CPU 隔离运行时未就绪或未安装。",
            }

    else:  # "auto"
        if cuda_ready:
            return {
                "model_id": model_id,
                "requested_device": "auto",
                "runtime_profile": cuda_profile,
                "actual_device": "cuda",
                "ready": True,
                "fallback": False,
                "reason": None,
            }
        elif cpu_ready:
            fallback = hardware_nvidia_available and descriptor.supports_cuda
            return {
                "model_id": model_id,
                "requested_device": "auto",
                "runtime_profile": cpu_profile,
                "actual_device": "cpu",
                "ready": True,
                "fallback": fallback,
                "reason": "已安全降级至 CPU 隔离环境运行。" if fallback else None,
            }
        else:
            reason = "无可用隔离运行时。"
            if not descriptor.supports_cpu and not cuda_ready:
                reason = f"模型 {descriptor.display_name} 需要 NVIDIA CUDA 加速环境，但当前 CUDA 驱动或运行时未就绪。"
            return {
                "model_id": model_id,
                "requested_device": "auto",
                "runtime_profile": None,
                "actual_device": "none",
                "ready": False,
                "fallback": False,
                "reason": reason,
            }

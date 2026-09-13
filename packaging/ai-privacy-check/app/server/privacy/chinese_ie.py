"""Chinese Information Extraction (IE) detector for semantic entities (Name, Address).

Combines built-in high-precision deterministic linguistic segmentation with
optional SiameseUIE deep-learning model enhancement.
Ensures zero-network, local-first inference with exact half-open character offsets.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .device import DEVICE_MANAGER
from .model_security import gate_model_security
from .entities import Entity
from .worker_client import get_worker_client

# Chinese Common Surnames (Top 100+ high frequency)
COMMON_SURNAMES: Set[str] = {
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周",
    "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "郑",
    "谢", "罗", "梁", "宋", "唐", "许", "韩", "冯", "邓", "曹",
    "彭", "曾", "肖", "田", "董", "袁", "潘", "于", "蒋", "蔡",
    "余", "杜", "叶", "程", "苏", "魏", "吕", "丁", "任", "沈",
    "姚", "卢", "姜", "崔", "钟", "谭", "陆", "汪", "范", "金",
    "石", "廖", "贾", "夏", "韦", "付", "方", "白", "邹", "孟",
    "熊", "秦", "邱", "江", "尹", "薛", "闫", "段", "雷", "侯",
    "龙", "史", "陶", "黎", "贺", "顾", "毛", "郝", "龚", "邵",
    "万", "钱", "严", "覃", "武", "戴", "莫", "孔", "向", "汤",
    "欧阳", "司马", "诸葛", "东方", "上官", "皇甫", "令狐",
}

# Exclusion dictionary to prevent false positives for common non-human words
EXCLUDE_NAMES: Set[str] = {
    "中国", "中华", "北京", "上海", "广州", "深圳", "天津", "重庆",
    "开发", "测试", "发布", "项目", "系统", "服务", "产品", "功能",
    "用户", "客户", "管理员", "张贴", "高大", "马路", "白云", "青山",
    "黄河", "长江", "东方", "黄金", "方针", "严厉", "金钱", "周报",
    "徐徐", "武力", "王道", "李子", "杨树", "林木", "蔡司", "夏天",
}

# Administrative division units for Chinese addresses
ADDR_PROVINCES = r"(?:河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|北京|天津|上海|重庆|香港|澳门)"
ADDR_ADMIN = r"(?:省|自治区|特别行政区|市|地区|盟|自治州|区|县|县级市|旗|海域)"
ADDR_STREET = r"(?:街道|镇|乡|苏木|路|街|道|巷|弄|胡同|里|堤|段|桥|大道)"
ADDR_DETAIL = r"(?:[0-9一二三四五六七八九十百]+号|[0-9一二三四五六七八九十百A-Za-z_-]+(?:室|号楼|幢|栋|单元|层|楼)|[\u4e00-\u9fa50-9A-Za-z_-]{2,20}(?:大厦|广场|中心|小区|园区|花园|公寓|家园|庄园|弄))"


@dataclass
class SemanticMatch:
    entity_type: str
    text: str
    confidence: float
    start: Optional[int] = None
    end: Optional[int] = None


class BuiltinChineseIE:
    """Zero-dependency deterministic Chinese linguistic information extractor."""

    def __init__(self) -> None:
        self.address_pattern = re.compile(
            rf"(?:(?:住在|寄往|寄到|送往|送至|发往|地址(?:是|为|：|:)?)\s*)?"
            rf"({ADDR_PROVINCES}(?:{ADDR_ADMIN})?"
            rf"(?:[^\s，。；;！？\n]{{1,25}}(?:市|区|县|镇|街道|路|街|道|弄|巷|胡同))+"
            rf"(?:[^\s，。；;！？\n]{{0,40}}(?:{ADDR_DETAIL}))*)",
            re.IGNORECASE,
        )

        # 1. 字段模式：必须包含显式冒号/等号分隔符，前置限定高置信字段词（排除宽泛的"用户"）
        self.person_field_pattern = re.compile(
            r"(?<![\u4e00-\u9fa5])"
            r"(?:姓名|联系人|收件人|发件人|负责人|经办人|候选人|当事人|作者|客户)"
            r"\s*[：:=]\s*"
            r"([\u4e00-\u9fa5]{2,4}?)"
            r"(?:先生|女士|老师|同学|医生|教授|经理|主任)?"
            r"(?=[^\u4e00-\u9fa5]|$)"
        )
        self.person_context_pattern = self.person_field_pattern

        # 2. 自然语言谓词模式：连接词（是/为）必须显式存在，排除可选匹配
        self.person_predicate_pattern = re.compile(
            r"(?<![\u4e00-\u9fa5])"
            r"(?:负责人|联系人|作者|经办人|当事人|候选人)"
            r"\s*(?:是|为)\s*"
            r"([\u4e00-\u9fa5]{2,4}?)"
            r"(?:先生|女士|老师|同学|医生|教授|经理|主任)?"
            r"(?=[^\u4e00-\u9fa5]|$)"
        )

        # 3. 自我介绍模式
        self.person_self_intro_pattern = re.compile(
            r"(?<![\u4e00-\u9fa5])"
            r"(?:我叫|我是)\s*"
            r"([\u4e00-\u9fa5]{2,4})"
            r"(?=[^\u4e00-\u9fa5]|$)"
        )

        # 4. 后置称谓模式：将先生/女士/老师/医生/教授/经理/主任作为后置称谓，而非前置 anchor
        self.person_post_title_pattern = re.compile(
            r"(?<![\u4e00-\u9fa5])"
            r"([\u4e00-\u9fa5]{2,3})"
            r"(?:先生|女士|老师|同学|医生|教授|经理|主任)"
            r"(?![\u4e00-\u9fa5])"
        )

        # 5. 由 X 负责/经办/承办 专有人行动作短语
        self.person_action_by_pattern = re.compile(
            r"(?<![\u4e00-\u9fa5])"
            r"由\s*([\u4e00-\u9fa5]{2,3})(?:先生|女士|老师|同学|医生|教授|经理|主任)?\s*"
            r"(?:负责|经办|承办)"
        )

        # 6. 高置信人行动作对象短语（移除和/同/与/跟及裸"给"等宽泛介词/连词）
        self.person_action_target_pattern = re.compile(
            r"(?<![\u4e00-\u9fa5])"
            r"(?:请?通知|请?联系|找|转交给|拜访|采访|陪同)\s*"
            r"([\u4e00-\u9fa5]{2,3})(?:先生|女士|老师|同学|医生|教授|经理|主任)?"
            r"(?=[，。；;！？\s、\n]|$)"
        )

    def extract_names(self, text: str) -> List[Tuple[int, int, str, float]]:
        results: List[Tuple[int, int, str, float]] = []

        patterns: List[Tuple[re.Pattern, float]] = [
            (self.person_field_pattern, 0.92),
            (self.person_predicate_pattern, 0.92),
            (self.person_self_intro_pattern, 0.92),
            (self.person_post_title_pattern, 0.90),
            (self.person_action_by_pattern, 0.88),
            (self.person_action_target_pattern, 0.85),
        ]

        for pattern, conf in patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                start, end = match.span(1)
                if self._is_valid_chinese_name(name):
                    # Ensure no overlap with existing
                    if not any(r[0] <= start < r[1] or r[0] < end <= r[1] for r in results):
                        results.append((start, end, name, conf))

        results.sort(key=lambda x: x[0])
        return results

    def extract_addresses(self, text: str) -> List[Tuple[int, int, str, float]]:
        results: List[Tuple[int, int, str, float]] = []
        for match in self.address_pattern.finditer(text):
            addr = match.group(1)
            start, end = match.span(1)
            addr_clean = addr.rstrip("，。；;！？ \t\r\n")
            end = start + len(addr_clean)
            if len(addr_clean) >= 6:
                results.append((start, end, addr_clean, 0.90))
        return results

    def _is_valid_chinese_name(self, name: str) -> bool:
        if len(name) < 2 or len(name) > 4:
            return False
        if name in EXCLUDE_NAMES:
            return False
        # Check surname
        if len(name) >= 3 and name[:2] in COMMON_SURNAMES:
            return True
        if name[0] in COMMON_SURNAMES:
            return True
        return False


class ChineseIEDetector:
    """Unified Chinese IE detector: supports builtin rules and optional deep SiameseUIE / UIE models."""

    id = "chinese_ie"
    slot = "chinese_ie"
    name = "chinese_ie"

    def __init__(self, data_dir: Optional[Path] = None, active_model_id: str = "siamese-uie") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._builtin = BuiltinChineseIE()
        self._uie_model = None
        self._model_lock = threading.Lock()
        self._model_attempted = False
        self._model_available = False

    def _get_model_dir(self) -> Optional[Path]:
        if not self.data_dir:
            return None
        # Support both new siamese-uie and legacy chinese-ie directories
        p1 = self.data_dir / "models" / self.active_model_id
        if p1.is_dir():
            return p1
        p2 = self.data_dir / "models" / "siamese-uie"
        if p2.is_dir():
            return p2
        p3 = self.data_dir / "models" / "chinese-ie"
        if p3.is_dir():
            return p3
        return p1

CHINESE_IE_LABEL_MAP: Dict[str, str] = {
    "姓名": "CN_NAME",
    "人名": "CN_NAME",
    "人物": "CN_NAME",
    "地址": "CN_ADDRESS",
    "地理位置": "CN_ADDRESS",
    "机构": "ORGANIZATION",
    "组织机构": "ORGANIZATION",
    "学校": "ORGANIZATION",
    "医院": "ORGANIZATION",
    "职位": "JOB_TITLE",
    "头衔": "JOB_TITLE",
    "身份背景": "IDENTITY_BACKGROUND",
}


class ChineseIEDetector:
    """Tier 2: High-precision Chinese Information Extraction (IE) detector."""

    id = "chinese_ie"
    slot = "chinese_ie"
    name = "chinese_ie"

    def __init__(self, data_dir: Path, active_model_id: str = "siamese-uie") -> None:
        self.data_dir = data_dir
        self.active_model_id = active_model_id
        self._builtin = BuiltinChineseIE()
        self._model_lock = threading.Lock()

    def set_active_model(self, model_id: str) -> None:
        with self._model_lock:
            if model_id != self.active_model_id:
                old_model_id = self.active_model_id
                self.active_model_id = model_id
                get_worker_client(self.data_dir).stop_worker_for_model(old_model_id)

    def _get_model_dir(self) -> Path:
        return self.data_dir / "models" / self.active_model_id

    def status(self) -> Dict[str, object]:
        model_dir = self._get_model_dir()
        from .model_catalog import check_model_integrity, get_model_descriptor
        installed, _ = check_model_integrity(self.active_model_id, model_dir)

        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        actual_device = model_res.get("actual_device", "cpu")
        base_runtime_ready = bool(model_res.get("ready", False))
        profile = model_res.get("runtime_profile")
        descriptor = get_model_descriptor(self.active_model_id)

        target_profile = profile
        if not target_profile and descriptor:
            req = DEVICE_MANAGER.get_requested_device()
            diag = DEVICE_MANAGER.probe_diagnostics()
            has_nv = bool(diag.get("hardware", {}).get("nvidia_available", False))
            if req == "cuda" or (req == "auto" and has_nv and descriptor.supports_cuda):
                target_profile = "torch-cuda"
            else:
                target_profile = "torch-cpu"

        rt_manager = DEVICE_MANAGER.get_runtime_manager()
        rt_probe = rt_manager.probe_profile(target_profile) if (rt_manager and target_profile) else {}
        if base_runtime_ready and not rt_probe.get("installed", False):
            python_runtime_ready = True
            python_runtime_source = "mock"
            python_runtime_version = "3.12"
            missing_python_capabilities = []
            runtime_rebuild_required = False
            base_packages_ready = True
        else:
            python_runtime_ready = bool(rt_probe.get("python_runtime_ready", True))
            python_runtime_source = str(rt_probe.get("python_runtime_source", "none"))
            python_runtime_version = rt_probe.get("python_runtime_version")
            missing_python_capabilities = list(rt_probe.get("missing_python_capabilities", []))
            runtime_rebuild_required = bool(rt_probe.get("runtime_rebuild_required", False))
            base_packages_ready = bool(rt_probe.get("base_packages_ready", base_runtime_ready))

        model_dependencies_ready = True
        missing_dependencies: List[str] = []
        if installed and base_runtime_ready and descriptor and descriptor.runtime_dependencies and profile:
            try:
                from model_installer import probe_model_runtime_dependencies
                probe_res = probe_model_runtime_dependencies(self.data_dir, self.active_model_id, profile)
                model_dependencies_ready = bool(probe_res.get("satisfied", False))
                missing_dependencies = list(probe_res.get("missing", []))
            except Exception:
                model_dependencies_ready = False
                missing_dependencies = list(descriptor.runtime_dependencies)

        model_ready = (
            installed
            and base_runtime_ready
            and python_runtime_ready
            and not runtime_rebuild_required
            and base_packages_ready
            and model_dependencies_ready
        )
        repairable = (
            installed
            and (
                (base_runtime_ready and not model_dependencies_ready)
                or runtime_rebuild_required
                or not python_runtime_ready
            )
        )

        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "siamese_uie" if model_ready else "builtin_linguistic_ie",
            "active_model": self.active_model_id,
            "installed": installed,
            "ready": True,  # Built-in is always ready
            "model_ready": model_ready,
            "base_runtime_ready": base_runtime_ready,
            "python_runtime_ready": python_runtime_ready,
            "python_runtime_source": python_runtime_source,
            "python_runtime_version": python_runtime_version,
            "missing_python_capabilities": missing_python_capabilities,
            "runtime_rebuild_required": runtime_rebuild_required,
            "base_packages_ready": base_packages_ready,
            "model_dependencies_ready": model_dependencies_ready,
            "missing_dependencies": missing_dependencies,
            "repairable": repairable,
            "device": actual_device,
            "path": str(model_dir) if installed else None,
        }

    def load(self) -> None:
        # Pre-warm worker if model is installed and fully ready
        st = self.status()
        if not st.get("model_ready"):
            return

        model_dir = self._get_model_dir()
        model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
        profile = model_res.get("runtime_profile")
        actual_dev = model_res.get("actual_device", "cpu")
        if profile:
            from .worker_client import get_worker_client
            try:
                gate_model_security(model_dir)
                worker = get_worker_client(self.data_dir).get_worker(
                    self.active_model_id, profile, device=actual_dev
                )
                worker.query({"action": "load", "model_path": str(model_dir)})
            except Exception:
                pass

    def unload(self) -> None:
        with self._model_lock:
            from .worker_client import get_worker_client
            get_worker_client(self.data_dir).stop_worker_for_model(self.active_model_id)

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        """Detect Chinese semantic entities via Built-in rules and optional isolated SiameseUIE worker."""
        entities: List[Entity] = []
        warnings: List[str] = []

        if not text:
            return entities, warnings

        # 1. First run built-in high-precision zero-dependency semantic extractor
        for start, end, val, conf in self._builtin.extract_names(text):
            entities.append(
                Entity(
                    entity_type="CN_NAME",
                    start=start,
                    end=end,
                    text=val,
                    confidence=conf,
                    sources=(self.name,),
                    validated=False,
                )
            )

        for start, end, val, conf in self._builtin.extract_addresses(text):
            entities.append(
                Entity(
                    entity_type="CN_ADDRESS",
                    start=start,
                    end=end,
                    text=val,
                    confidence=conf,
                    sources=(self.name,),
                    validated=False,
                )
            )

        # 2. Query isolated SiameseUIE worker if installed and ready
        st = self.status()
        if st.get("model_ready"):
            model_dir = self._get_model_dir()
            model_res = DEVICE_MANAGER.resolve_for_model(self.active_model_id)
            profile = model_res.get("runtime_profile")
            device = model_res.get("actual_device", "cpu")
            if profile:
                    try:
                        from .worker_client import get_worker_client
                        worker_client = get_worker_client(self.data_dir)
                        _, infer_timeout = worker_client.get_timeout_for_model(self.active_model_id, device=device)
                        with worker_client.cuda_execution_session(device=device, timeout=float(infer_timeout)):
                            gate_model_security(model_dir)
                            worker = worker_client.get_worker(
                                self.active_model_id, profile, device=device
                            )
                            res = worker.query({
                                "action": "detect",
                                "model_path": str(model_dir),
                                "text": text,
                            }, timeout=infer_timeout)
                        if res.get("ok"):
                            for ent in res.get("entities", []):
                                s = int(ent.get("start", 0))
                                e = int(ent.get("end", 0))
                                ent_text = ent.get("text", "")
                                raw_lbl = ent.get("label", "")
                                score = float(ent.get("score", 0.9))

                                # Map label accurately
                                mapped_type = CHINESE_IE_LABEL_MAP.get(
                                    raw_lbl,
                                    "ORGANIZATION" if any(k in raw_lbl for k in ("机构", "公司", "学校", "院"))
                                    else ("CN_NAME" if "名" in raw_lbl else "CN_ADDRESS")
                                )

                                if 0 <= s < e <= len(text) and text[s:e] == ent_text:
                                    entities.append(
                                        Entity(
                                            entity_type=mapped_type,
                                            start=s,
                                            end=e,
                                            text=ent_text,
                                            confidence=score,
                                            sources=(self.name, "siamese-uie"),
                                            validated=False,
                                        )
                                    )
                        else:
                            err_msg = res.get("error") or "Worker query failed"
                            warnings.append(f"SiameseUIE worker 推理未完成，保持内置规则抽取: {err_msg}")
                    except Exception as exc:
                        warnings.append(f"SiameseUIE worker 通信异常: {exc}")

        return entities, warnings


def safe_sequential_span_alignment(
    full_text: str,
    predictions: Iterable[Tuple[str, str, float]],
) -> Tuple[List[Entity], List[str]]:
    """Align entity strings back to full text using cursor-based sequential scanning.

    Never uses full_text.find() naively. Prevents assigning the first occurrence
    when the same entity appears multiple times.
    """
    entities: List[Entity] = []
    warnings: List[str] = []
    cursor = 0

    for entity_type, target_text, confidence in predictions:
        if not target_text:
            continue
        idx = full_text.find(target_text, cursor)
        if idx != -1:
            end = idx + len(target_text)
            entities.append(
                Entity(
                    entity_type=entity_type,
                    start=idx,
                    end=end,
                    text=target_text,
                    confidence=confidence,
                    sources=("semantic_alignment",),
                    validated=False,
                )
            )
            cursor = end
        else:
            # Fallback search from start if text appears before current cursor
            alt_idx = full_text.find(target_text)
            if alt_idx != -1:
                warnings.append(f"实体 '{target_text}' 乱序匹配，跳过非单调候选以避免破坏 span。")

    return entities, warnings

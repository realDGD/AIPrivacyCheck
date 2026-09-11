"""Chinese Information Extraction (IE) detector for semantic entities (Name, Address).

Combines built-in high-precision deterministic linguistic segmentation with
optional PaddleNLP UIE deep-learning model enhancement.
Ensures zero-network, local-first inference with exact half-open character offsets.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .device import DEVICE_MANAGER
from .entities import Entity

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

    def status(self) -> Dict[str, object]:
        installed = False
        model_path = None
        model_dir = self._get_model_dir()
        if model_dir and model_dir.is_dir():
            has_weights = (
                (model_dir / "model_state.pdparams").is_file()
                or (model_dir / "config.json").is_file()
                or any(model_dir.glob("*.safetensors"))
                or any(model_dir.glob("*.pdparams"))
            )
            if has_weights:
                installed = True
                model_path = str(model_dir)

        actual_device, _ = DEVICE_MANAGER.resolve()
        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "engine": "siamese_uie" if installed and self._model_available else "builtin_semantic_ie",
            "active_model": self.active_model_id,
            "installed": installed,
            "ready": True,  # Built-in is always ready
            "model_ready": self._model_available,
            "device": actual_device,
            "path": model_path,
        }

    def load(self) -> None:
        self._try_init_uie()

    def unload(self) -> None:
        with self._model_lock:
            self._uie_model = None
            self._model_attempted = False
            self._model_available = False

    def _try_init_uie(self) -> None:
        with self._model_lock:
            if self._model_attempted:
                return
            self._model_attempted = True
            model_dir = self._get_model_dir()
            if not model_dir or not model_dir.is_dir():
                return

            try:
                from paddlenlp import Taskflow  # type: ignore

                device, _ = DEVICE_MANAGER.resolve()
                use_gpu = device == "cuda"
                self._uie_model = Taskflow(
                    "information_extraction",
                    schema=["姓名", "地址", "机构", "学校", "职位", "医院"],
                    task_path=str(model_dir),
                    device_id=0 if use_gpu else -1,
                )
                self._model_available = True
            except Exception:
                self._uie_model = None
                self._model_available = False

    def detect(self, text: str) -> Tuple[List[Entity], List[str]]:
        """Detect Chinese semantic entities with chunking and safe exact span mapping."""
        entities: List[Entity] = []
        warnings: List[str] = []

        if not text:
            return entities, warnings

        # 1. First run built-in high-precision semantic extractor
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

        # 2. Check if optional SiameseUIE / UIE is ready
        self._try_init_uie()
        if self._model_available and self._uie_model is not None:
            try:
                uie_entities, uie_warn = self._infer_uie(text)
                entities.extend(uie_entities)
                warnings.extend(uie_warn)
            except Exception as exc:
                warnings.append(f"Chinese IE 模型推理失败，已使用内置语义引擎回退: {exc}")

        return entities, warnings

    def _infer_uie(self, text: str, chunk_size: int = 512, overlap: int = 64) -> Tuple[List[Entity], List[str]]:
        """Chunked execution with exact global span offset mapping."""
        results: List[Entity] = []
        warnings: List[str] = []

        # Split text into chunks if long
        chunks: List[Tuple[int, str]] = []
        if len(text) <= chunk_size:
            chunks.append((0, text))
        else:
            idx = 0
            while idx < len(text):
                end = min(len(text), idx + chunk_size)
                chunks.append((idx, text[idx:end]))
                if end == len(text):
                    break
                idx += chunk_size - overlap

        for chunk_offset, chunk_text in chunks:
            raw_predictions = self._uie_model(chunk_text)
            for pred in raw_predictions:
                if not isinstance(pred, dict):
                    continue
                for label, items in pred.items():
                    entity_type = "CN_NAME" if "名" in label else "CN_ADDRESS"
                    for item in items:
                        item_text = item.get("text", "").strip()
                        prob = float(item.get("probability", 0.85))
                        if not item_text:
                            continue

                        # If model provides local start/end
                        if "start" in item and "end" in item:
                            local_s = int(item["start"])
                            local_e = int(item["end"])
                            global_s = chunk_offset + local_s
                            global_e = chunk_offset + local_e
                            if 0 <= global_s < global_e <= len(text) and text[global_s:global_e] == item_text:
                                results.append(
                                    Entity(
                                        entity_type=entity_type,
                                        start=global_s,
                                        end=global_e,
                                        text=item_text,
                                        confidence=prob,
                                        sources=(self.name, "uie"),
                                        validated=False,
                                    )
                                )
                                continue

                        # Safe fallback alignment for identical names appearing multiple times:
                        # Find all occurrences in chunk and match sequentially with cursor
                        occurrences = [m.start() for m in re.finditer(re.escape(item_text), chunk_text)]
                        if len(occurrences) == 1:
                            global_s = chunk_offset + occurrences[0]
                            global_e = global_s + len(item_text)
                            results.append(
                                Entity(
                                    entity_type=entity_type,
                                    start=global_s,
                                    end=global_e,
                                    text=item_text,
                                    confidence=prob,
                                    sources=(self.name, "uie"),
                                    validated=False,
                                )
                            )
                        elif len(occurrences) > 1:
                            warnings.append(
                                f"实体 '{item_text}' 在当前段落中出现多次，已避免错误猜测 offset。"
                            )

        return results, warnings


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

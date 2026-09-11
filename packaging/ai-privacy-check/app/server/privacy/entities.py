"""Shared privacy-entity data structures."""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple


LABELS_ZH: Dict[str, str] = {
    "CN_ID_CARD": "身份证号",
    "CN_PHONE_NUMBER": "手机号",
    "CN_LANDLINE": "固定电话",
    "CN_BANK_CARD": "银行卡号",
    "CN_PASSPORT": "护照号",
    "CN_USCC": "统一社会信用代码",
    "CN_LICENSE_PLATE": "车牌号",
    "CN_NAME": "姓名",
    "CN_ADDRESS": "地址",
    "CN_ACCOUNT": "账号/单号",
    "CN_BIRTH_DATE": "出生日期",
    "CN_SOCIAL_ACCOUNT": "社交账号",
    "EMAIL": "电子邮箱",
    "IP_ADDRESS": "IP 地址",
    "IPV6_ADDRESS": "IPv6 地址",
    "MAC_ADDRESS": "MAC 地址",
    "PRIVATE_URL": "私有网址",
    "SECRET": "密钥/密码",
    "PRIVATE_PERSON": "姓名",
    "PRIVATE_ADDRESS": "地址",
    "PRIVATE_DATE": "私密日期",
    "ACCOUNT_NUMBER": "账号",
    "PHONE": "电话号码",
    "PASSPORT": "护照号",
    "GOVERNMENT_ID": "政府证件号",
    "IBAN": "IBAN",
    "BIC": "BIC / SWIFT",
    "USERNAME": "用户名",
    "RECORD_ID": "业务记录编号",
    "CARD_EXPIRY": "卡片有效期",
    "CARD_SECURITY_CODE": "卡片安全码",
    "IPV6_ADDRESS": "IPv6 地址",
    "MAC_ADDRESS": "MAC 地址",
}


PRIORITY: Dict[str, int] = {
    "SECRET": 120,
    "CARD_SECURITY_CODE": 119,
    "CN_ID_CARD": 115,
    "GOVERNMENT_ID": 114,
    "IBAN": 113,
    "CN_USCC": 112,
    "CN_BANK_CARD": 110,
    "CARD_EXPIRY": 109,
    "PASSPORT": 108,
    "CN_PASSPORT": 108,
    "CN_PHONE_NUMBER": 105,
    "CN_LANDLINE": 100,
    "EMAIL": 98,
    "CN_LICENSE_PLATE": 96,
    "IP_ADDRESS": 92,
    "IPV6_ADDRESS": 92,
    "MAC_ADDRESS": 91,
    "BIC": 88,
    "RECORD_ID": 87,
    "CN_ACCOUNT": 86,
    "ACCOUNT_NUMBER": 84,
    "USERNAME": 83,
    "CN_SOCIAL_ACCOUNT": 82,
    "CN_BIRTH_DATE": 76,
    "PRIVATE_DATE": 74,
    "CN_ADDRESS": 70,
    "PRIVATE_ADDRESS": 68,
    "CN_NAME": 64,
    "PRIVATE_PERSON": 62,
    "PHONE": 60,
    "PRIVATE_URL": 55,
}


@dataclass(frozen=True)
class Entity:
    """One detected span using Python's half-open character offsets."""

    entity_type: str
    start: int
    end: int
    text: str
    confidence: float
    sources: Tuple[str, ...]
    validated: bool = False

    @property
    def label_zh(self) -> str:
        return LABELS_ZH.get(self.entity_type, self.entity_type)

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.entity_type, 50)

    def with_sources(self, sources: Iterable[str], confidence: float) -> "Entity":
        return replace(
            self,
            sources=tuple(sorted(set(self.sources).union(sources))),
            confidence=max(self.confidence, confidence),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.entity_type,
            "label": self.label_zh,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "sources": list(self.sources),
            "validated": self.validated,
        }

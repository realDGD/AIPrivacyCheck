"""Shared privacy-entity data structures."""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple


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
    "SECRET": "密钥/凭证",
    "DATABASE_URI": "数据库连接串",
    "PRIVATE_KEY": "私钥",
    "API_TOKEN": "API Token",
    "PASSWORD": "密码",
    "PRIVATE_PERSON": "姓名",
    "PERSON": "姓名",
    "PRIVATE_ADDRESS": "地址",
    "ADDRESS": "地址",
    "PRIVATE_DATE": "日期",
    "DATE": "日期",
    "ACCOUNT_NUMBER": "账号",
    "PHONE": "电话号码",
    "PASSPORT": "护照号",
    "GOVERNMENT_ID": "政府证件号",
    "US_SSN": "社会安全号 (SSN)",
    "CREDIT_CARD": "信用卡号",
    "IBAN": "IBAN",
    "BIC": "BIC / SWIFT",
    "USERNAME": "用户名",
    "RECORD_ID": "业务记录编号",
    "MEDICAL_RECORD_ID": "病历号",
    "INSURANCE_ID": "社保/保险号",
    "EMPLOYEE_ID": "员工工号",
    "STUDENT_ID": "学生学号",
    "CARD_EXPIRY": "卡片有效期",
    "CARD_SECURITY_CODE": "卡片安全码",
    "ORGANIZATION": "机构/企业",
    "LOCATION": "地理位置",
    "FINANCIAL": "财务资产",
    "MEDICAL": "医疗健康",
    "BIOMETRIC": "生物特征",
    "CREDENTIAL": "认证凭据",
    "TRADE_RECORD": "交易记录",
    "LOCATION_TRAJECTORY": "轨迹位置",
    "RELATIONSHIP": "人际关系",
    "COMMUNICATION": "通信隐私",
    "JUDICIAL": "司法记录",
    "COMMERCIAL_SECRET": "商业秘密",
}


PRIORITY: Dict[str, int] = {
    "DATABASE_URI": 125,
    "PRIVATE_KEY": 124,
    "API_TOKEN": 122,
    "PASSWORD": 121,
    "SECRET": 120,
    "CARD_SECURITY_CODE": 119,
    "CREDENTIAL": 118,
    "CN_ID_CARD": 115,
    "US_SSN": 114,
    "GOVERNMENT_ID": 114,
    "IBAN": 113,
    "CN_USCC": 112,
    "CREDIT_CARD": 111,
    "CN_BANK_CARD": 110,
    "CARD_EXPIRY": 109,
    "PASSPORT": 108,
    "CN_PASSPORT": 108,
    "CN_PHONE_NUMBER": 105,
    "CN_LANDLINE": 100,
    "EMAIL": 98,
    "CN_LICENSE_PLATE": 96,
    "FINANCIAL": 95,
    "MEDICAL": 94,
    "BIOMETRIC": 93,
    "IP_ADDRESS": 92,
    "IPV6_ADDRESS": 92,
    "MAC_ADDRESS": 91,
    "MEDICAL_RECORD_ID": 90,
    "INSURANCE_ID": 89,
    "TRADE_RECORD": 89,
    "BIC": 88,
    "LOCATION_TRAJECTORY": 88,
    "RECORD_ID": 87,
    "COMMERCIAL_SECRET": 87,
    "EMPLOYEE_ID": 86,
    "JUDICIAL": 86,
    "STUDENT_ID": 85,
    "CN_ACCOUNT": 84,
    "ACCOUNT_NUMBER": 84,
    "USERNAME": 83,
    "CN_SOCIAL_ACCOUNT": 82,
    "DATE": 78,
    "CN_BIRTH_DATE": 76,
    "PRIVATE_DATE": 74,
    "CN_ADDRESS": 70,
    "ADDRESS": 69,
    "PRIVATE_ADDRESS": 68,
    "LOCATION": 67,
    "ORGANIZATION": 66,
    "CN_NAME": 65,
    "PERSON": 64,
    "PRIVATE_PERSON": 62,
    "RELATIONSHIP": 61,
    "PHONE": 60,
    "COMMUNICATION": 59,
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
    privacy_level: Optional[str] = None
    semantic_type: Optional[str] = None

    @property
    def label_zh(self) -> str:
        return LABELS_ZH.get(self.entity_type, self.entity_type)

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.entity_type, 50)

    @property
    def resolved_privacy_level(self) -> str:
        from .taxonomy import resolve_privacy_level
        return resolve_privacy_level(self.entity_type, self.semantic_type, self.privacy_level)

    def with_sources(self, sources: Iterable[str], confidence: float) -> "Entity":
        return replace(
            self,
            sources=tuple(sorted(set(self.sources).union(sources))),
            confidence=max(self.confidence, confidence),
        )

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "type": self.entity_type,
            "label": self.label_zh,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "sources": list(self.sources),
            "validated": self.validated,
            "privacy_level": self.resolved_privacy_level,
        }
        if self.semantic_type is not None:
            payload["semantic_type"] = self.semantic_type
        return payload

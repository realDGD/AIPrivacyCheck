"""Privacy Sensitivity Levels (PL1 - PL4) Taxonomy and deterministic risk policy.

Inspired by industry PII classification and semantic privacy frameworks:
- PL1: Low sensitivity / preference (not normally redacted in standard workflows)
- PL2: Identifiable PII (Direct/indirect identity indicators: name, phone, email, address)
- PL3: High sensitivity PII (Government IDs, financial cards, biometric, medical, precise geolocation)
- PL4: Critical credentials & immediately exploitable secrets (passwords, tokens, private keys, database URIs)

Deterministic entities maintain strict static risk policies that cannot be arbitrarily
downgraded by generic or generative models.
"""

from typing import Dict, Optional, Set


# Privacy Level constants
PL1 = "PL1"
PL2 = "PL2"
PL3 = "PL3"
PL4 = "PL4"

PRIVACY_LEVEL_NAMES: Dict[str, str] = {
    PL1: "低敏偏好 (PL1)",
    PL2: "可识别个人信息 (PL2)",
    PL3: "高敏合规凭证 (PL3)",
    PL4: "核心密码凭证 (PL4)",
}

PRIVACY_LEVEL_DESCRIPTIONS: Dict[str, str] = {
    PL1: "个人偏好、公开职业标签或无直接关联倾向，通常不作为默认脱敏对象。",
    PL2: "直接或间接身份标识，包含姓名、电话、邮箱、物理地址、社交账号等。",
    PL3: "受严格法律保护的高敏身份凭证与隐私资产，包含身份证号、护照、银行卡、医疗病历等。",
    PL4: "即时可利用的系统安全凭证与高危秘密，包含系统密码、私钥、API 令牌、数据库连接串等。",
}

# Static deterministic entity type mapping to PL levels
DETERMINISTIC_PL_MAPPING: Dict[str, str] = {
    # PL4: Critical credentials & secrets
    "DATABASE_URI": PL4,
    "PRIVATE_KEY": PL4,
    "API_TOKEN": PL4,
    "PASSWORD": PL4,
    "SECRET": PL4,
    "CARD_SECURITY_CODE": PL4,
    "CREDENTIAL": PL4,
    # PL3: High sensitivity PII & government IDs & financial
    "CN_ID_CARD": PL3,
    "CN_BANK_CARD": PL3,
    "CREDIT_CARD": PL3,
    "CN_PASSPORT": PL3,
    "PASSPORT": PL3,
    "GOVERNMENT_ID": PL3,
    "US_SSN": PL3,
    "IBAN": PL3,
    "CN_USCC": PL3,
    "CARD_EXPIRY": PL3,
    "MEDICAL_RECORD_ID": PL3,
    "INSURANCE_ID": PL3,
    "MEDICAL": PL3,
    "FINANCIAL": PL3,
    "BIOMETRIC": PL3,
    "TRADE_RECORD": PL3,
    "LOCATION_TRAJECTORY": PL3,
    "JUDICIAL": PL3,
    "COMMERCIAL_SECRET": PL3,
    # PL2: Identifiable PII
    "CN_NAME": PL2,
    "PERSON": PL2,
    "PRIVATE_PERSON": PL2,
    "CN_PHONE_NUMBER": PL2,
    "PHONE": PL2,
    "CN_LANDLINE": PL2,
    "EMAIL": PL2,
    "CN_ADDRESS": PL2,
    "ADDRESS": PL2,
    "PRIVATE_ADDRESS": PL2,
    "LOCATION": PL2,
    "CN_BIRTH_DATE": PL2,
    "DATE": PL2,
    "PRIVATE_DATE": PL2,
    "CN_ACCOUNT": PL2,
    "ACCOUNT_NUMBER": PL2,
    "RECORD_ID": PL2,
    "EMPLOYEE_ID": PL2,
    "STUDENT_ID": PL2,
    "CN_SOCIAL_ACCOUNT": PL2,
    "USERNAME": PL2,
    "CN_LICENSE_PLATE": PL2,
    "IP_ADDRESS": PL2,
    "IPV6_ADDRESS": PL2,
    "MAC_ADDRESS": PL2,
    "PRIVATE_URL": PL2,
    "ORGANIZATION": PL2,
    "RELATIONSHIP": PL2,
    "COMMUNICATION": PL2,
    "JOB_TITLE": PL2,
    "IDENTITY_BACKGROUND": PL2,
}

# Semantic type to default PL level mapping (used by semantic / generative privacy models)
SEMANTIC_TYPE_PL_MAPPING: Dict[str, str] = {
    "password": PL4,
    "api_token": PL4,
    "private_key": PL4,
    "database_uri": PL4,
    "otp": PL4,
    "credentials": PL4,
    "secret": PL4,
    "id_card": PL3,
    "passport": PL3,
    "bank_card": PL3,
    "credit_card": PL3,
    "medical_record": PL3,
    "health_info": PL3,
    "financial_record": PL3,
    "precise_location": PL3,
    "biometric": PL3,
    "judicial_record": PL3,
    "trade_record": PL3,
    "name": PL2,
    "phone": PL2,
    "email": PL2,
    "address": PL2,
    "social_account": PL2,
    "organization": PL2,
    "relationship": PL2,
    "communication": PL2,
    "preference": PL1,
}


def resolve_privacy_level(entity_type: str, semantic_type: Optional[str] = None, model_level: Optional[str] = None) -> str:
    """Resolve privacy level enforcing deterministic static policy superiority.

    If entity_type is a recognized deterministic type, its statutory static level takes precedence.
    Otherwise, if semantic_type or model_level is provided, it is mapped safely.
    Fallback is PL2.
    """
    if entity_type in DETERMINISTIC_PL_MAPPING:
        return DETERMINISTIC_PL_MAPPING[entity_type]

    if model_level in (PL1, PL2, PL3, PL4):
        return model_level

    if semantic_type:
        clean_sem = semantic_type.strip().lower().replace(" ", "_").replace("-", "_")
        if clean_sem in SEMANTIC_TYPE_PL_MAPPING:
            return SEMANTIC_TYPE_PL_MAPPING[clean_sem]

    return PL2

"""Multilingual, Chinese-first deterministic PII recognizers.

Rules favor validated identifiers and contextual matches. Ambiguous values such as
names, dates, account numbers, and addresses are only emitted when nearby labels
make their meaning reasonably clear.
"""

from dataclasses import dataclass
import re
from typing import Callable, Iterable, List, Match, Optional, Pattern

from .entities import Entity
from .validators import (
    card_expiry_valid,
    cn_id_card_valid,
    cn_uscc_valid,
    iban_valid,
    international_phone_valid,
    ipv4_valid,
    ipv6_valid,
    jwt_header_valid,
    bank_card_valid,
    luhn_valid,
    mac_valid,
    private_ipv4_valid,
)


Validator = Callable[[str], bool]


@dataclass(frozen=True)
class RegexRule:
    entity_type: str
    pattern: Pattern[str]
    confidence: float
    group: int = 0
    validator: Optional[Validator] = None
    validated: bool = False


def _compile(pattern: str, flags: int = 0) -> Pattern[str]:
    return re.compile(pattern, flags)


EXACT_RULES = (
    # PEM private keys: RSA/EC/DSA/OPENSSH/PGP heads (maskit parity: DSA and
    # PGP variants were missing and leak the same way).
    RegexRule(
        "SECRET",
        _compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
        1.0,
        validated=True,
    ),
    # Google API key: AIza + 35-38 key characters (official documented shape).
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35,38}(?![A-Za-z0-9_-])"),
        0.99,
        validated=True,
    ),
    # Stripe live/test keys: [sr]k_(live|test)_ + 20+ alnum (official shape).
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])[sr]k_(?:live|test)_[0-9A-Za-z]{20,}(?![A-Za-z0-9])"),
        0.99,
        validated=True,
    ),
    # Feishu app credentials: cli_ + 16+ lowercase alnum.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])cli_[a-z0-9]{16,}(?![a-z0-9])"),
        0.98,
        validated=True,
    ),
    # OpenAI keys: T3BlbkFJ (base64 of 'OpenAI') is the real discriminator
    # (Trivy production comment). Project/service-account/admin and legacy
    # shapes both carry the watermark.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,100}T3BlbkFJ[A-Za-z0-9_-]{20,100}(?![A-Za-z0-9_-])"),
        0.99,
        validated=True,
    ),
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}(?![A-Za-z0-9])"),
        0.99,
        validated=True,
    ),
    # Anthropic API keys: sk-ant-api03 + 93 chars + AA terminator (Gitleaks).
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])sk-ant-api03-[A-Za-z0-9_-]{93}AA(?![A-Za-z0-9_-])"),
        0.99,
        validated=True,
    ),
    # HuggingFace tokens: hf_ + 34-40 (Trivy modern mixed-case shape).
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{34,40}(?![A-Za-z0-9_-])"),
        0.99,
        validated=True,
    ),
    # GitLab personal access tokens.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])glpat-[0-9A-Za-z_-]{20}(?![A-Za-z0-9_-])"),
        0.98,
        validated=True,
    ),
    # Databricks: dapi + 32 hex (optional -N suffix), bounded.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])dapi[a-f0-9]{32}(?:-\d)?(?![A-Za-z0-9_-])"),
        0.98,
        validated=True,
    ),
    # Linear: lin_api_ + 40 lowercase alnum.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])lin_api_[a-z0-9]{40}(?![a-z0-9])"),
        0.98,
        validated=True,
    ),
    # Azure Entra client secrets: fixed 8Q~ watermark + 34 tail (Trivy
    # documents this literal prefix; far fewer FPs than Gitleaks' \dQ~).
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_~.\-])[A-Za-z0-9_~.-]{3}8Q~[A-Za-z0-9_~.-]{34}(?![A-Za-z0-9_~.-])"),
        0.99,
        validated=True,
    ),
    RegexRule(
        "DATABASE_URI",
        _compile(
            r"(?<![A-Za-z0-9])(?<!jdbc:)(?<!JDBC:)(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|amqps?|mssql)://[^\s<>\"'，。；;]+",
            re.IGNORECASE,
        ),
        0.998,
        validated=True,
    ),
    # JDBC connection URLs (MySQL Connector/J, PostgreSQL JDBC, Oracle JDBC thin,
    # Microsoft JDBC Driver for SQL Server, MariaDB Connector/J all document the
    # `jdbc:<subscheme>:` prefix as the constant URL head). The inner scheme
    # qualifier (e.g. `jdbc:mysql:loadbalance:`) is optional per official docs.
    RegexRule(
        "DATABASE_URI",
        _compile(
            r"(?<![A-Za-z0-9])jdbc:(?:mysql\+srv|mysql|mariadb|postgresql|oracle|sqlserver)(?::[A-Za-z0-9_-]+)?:[^\s<>\"'，。；;]+",
            re.IGNORECASE,
        ),
        0.995,
        validated=True,
    ),
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"),
        0.99,
        validator=jwt_header_valid,
        validated=True,
    ),
    # GitHub fine-grained personal access tokens: official docs structure
    # `github_pat_` + 22 chars + `_` + 59 chars. Classic/ OAuth / app tokens
    # (ghp_ / gho_ / ghu_ / ghs_ / ghr_) are covered by the gh[pousr]_ rule below.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_])github_pat_[0-9A-Za-z]{22}_[0-9A-Za-z]{59}(?![A-Za-z0-9_])"),
        0.99,
        validated=True,
    ),
    # Aliyun AccessKey ID: `LTAI` prefix is the official anchor (help.aliyun.com
    # AccessKey examples); no checksum is published, so charset/length stay tight.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9])LTAI[0-9A-Za-z]{12,20}(?![A-Za-z0-9])"),
        0.97,
        validated=True,
    ),
    # Tencent Cloud SecretId: official prefix `AKID` (China site) / `IKID`
    # (international site), 36 chars total per official masked examples.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9])(?:AKID|IKID)[0-9A-Za-z]{32}(?![A-Za-z0-9])"),
        0.97,
        validated=True,
    ),
    # Slack tokens officially documented by docs.slack.dev/authentication/tokens:
    # bot `xoxb-`, user `xoxp-`, app-level `xapp-`, workflow `xwfp-`, sections
    # separated by `-`. Legacy xoxa/xoxr/xoxs/xoxc prefixes are no longer in the
    # official docs and are deliberately NOT matched as hard rules.

    # AWS access keys: the full official prefix family (Trivy set: AKIA
    # long-lived, ASIA STS temporary, AGPA/AIDA/AROA/AIPA/ANPA/ANVA delegated
    # roles, ABIA/ACCA, A3T legacy) + 16 upper.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Z0-9])(?:A3T[A-Z0-9]|AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ABIA|ACCA)[A-Z0-9]{16}(?![A-Z0-9])"),
        0.99,
        validated=True,
    ),
    # GitHub stateless app tokens (2026-04 format): ghs_<APPID>.<JWT> - the
    # JWT dots require a dedicated charset (the main gh[pousr]_ rule excludes
    # dots to stay sentence-safe).
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9_-])ghs_[A-Za-z0-9._-]{36,}(?![A-Za-z0-9_-])"),
        0.99,
        validated=True,
    ),
    # OpenAI-style keys and GitHub classic/OAuth/app tokens (ghp_/gho_/ghu_/
    # ghs_/ghr_); fine-grained PATs have their own rule above.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{20,})(?![A-Za-z0-9])"),
        0.99,
        validated=True,
    ),
    # AWS SecretAccessKey: the half that can actually sign requests. Bare
    # 40-char base64 is refused (any digest would hit) - only the labeled
    # key=value form is accepted (maskit production lesson).
    RegexRule(
        "SECRET",
        _compile(r"(?i)aws[_-]?secret[_-]?access[_-]?key[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"),
        0.98,
        group=1,
        validated=True,
    ),
    # Bearer authorization tokens (context-validated; value 20+ token chars).
    RegexRule(
        "SECRET",
        _compile(r"(?i)(?<![A-Za-z0-9])Bearer\s+([A-Za-z0-9._~+/=-]{20,})(?![A-Za-z0-9._~+/=-]*[A-Za-z])"),
        0.95,
        group=1,
    ),
    # Slack: xoxr (refresh) / xoxs (session) added - real SDK-issued prefixes
    # (maskit production evidence) alongside the documented xoxb/xoxp.
    RegexRule(
        "SECRET",
        _compile(r"(?<![A-Za-z0-9])(?:xox[brps]|xapp|xwfp)-[0-9A-Za-z-]{20,}(?![A-Za-z0-9])"),
        0.98,
        validated=True,
    ),
    RegexRule(
        "IBAN",
        _compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?:[0-9A-Z]{11,30}|(?: [0-9A-Z]{4}){2,7}(?: [0-9A-Z]{1,4})?)(?![A-Za-z0-9])", re.IGNORECASE),
        0.995,
        validator=iban_valid,
        validated=True,
    ),
    RegexRule(
        "EMAIL",
        _compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}(?![A-Za-z0-9]|\.[A-Za-z])", re.IGNORECASE),
        0.99,
        validated=True,
    ),
    RegexRule(
        "CN_ID_CARD",
        _compile(r"(?<![0-9A-Za-z])(?:\d{17}[0-9Xx]|\d{15})(?![0-9A-Za-z])"),
        0.995,
        validator=cn_id_card_valid,
        validated=True,
    ),
    RegexRule(
        "CN_USCC",
        _compile(r"(?<![0-9A-Z])[159Y][1239]\d{6}[0-9A-HJ-NPQRTUWXY]{10}(?![0-9A-Z])", re.IGNORECASE),
        0.995,
        validator=cn_uscc_valid,
        validated=True,
    ),
    RegexRule(
        "PHONE",
        _compile(r"(?<![\w+])\+\d{1,4}(?:[ -]?(?:\(\d{1,5}\)|\d{1,5})){1,5}(?!\d)"),
        0.985,
        validator=international_phone_valid,
        validated=True,
    ),
    RegexRule(
        "IPV6_ADDRESS",
        _compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:])"),
        0.985,
        validator=ipv6_valid,
        validated=True,
    ),
    RegexRule(
        "US_SSN",
        _compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        0.98,
        validated=True,
    ),
    RegexRule(
        "CN_PHONE_NUMBER",
        # maskit lesson: separators must be consistent (one split point,
        # backreference) - mixed "138-1234 5678" runs are accidental concats,
        # not phone numbers.
        _compile(r"(?<![\d+A-Za-z])(?:(?:\+?86|0086)[ -]?)?1[3-9]\d(?:([ -])\d{4}\1\d{4}|\d{8})(?!\d)"),
        0.99,
        validator=lambda value: len(re.sub(r"\D", "", value).removeprefix("0086").removeprefix("86")) == 11,
        validated=True,
    ),
    RegexRule(
        "CN_BANK_CARD",
        # maskit 0.1.15 lesson: per-digit optional separators splice unrelated
        # numbers across a space (file size + year runs that happen to pass
        # Luhn). Two branches instead, no per-digit separators:
        #   1) continuous 13-19 digits starting with a real BIN (3-6);
        #   2) grouped digits with ONE consistent separator (backreference),
        #      first group 3-6 digits, up to 4 groups of 1-6; the validator
        #      enforces total length / BIN / Luhn.
        _compile(
            r"(?<!\d)(?:[3-6]\d{12,18}|[3-6]\d{2,5}(?:([ -])\d{1,6}){1,4})(?!\d)"
        ),
        0.97,
        validator=bank_card_valid,
        validated=True,
    ),
    RegexRule(
        "CN_PASSPORT",
        _compile(r"(?<![A-Z0-9])(?:[EG]\d{8}|[KJ]\d{7}|P\d{7}|S\d{7,8}|D\d{7,8})(?![A-Z0-9])", re.IGNORECASE),
        0.94,
        validated=True,
    ),
    RegexRule(
        "CN_LICENSE_PLATE",
        # maskit 0.1.15 lesson: the CJK lookbehind cannot stop "新README.md"
        # (新 is a province abbrev and README is a letter body). Real plates
        # always carry at least one digit in the body - one lookahead splits
        # the classes without a word list.
        _compile(r"(?<![A-Z0-9\u4e00-\u9fff])[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][·•]?(?=[A-HJ-NP-Z0-9]{0,5}\d)[A-HJ-NP-Z0-9]{5,6}(?![A-Z0-9])"),
        0.95,
        validated=True,
    ),
    RegexRule(
        "MAC_ADDRESS",
        # maskit lesson: the separator must be one consistent character
        # (backreference) - a free [: -] class splices MAC fragments across
        # spaces and eats surrounding text.
        _compile(r"(?<![0-9A-Fa-f:-])[0-9A-Fa-f]{2}([:-])(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])"),
        0.98,
        validator=mac_valid,
        validated=True,
    ),
    # Bare private/special-range IPv4 (maskit split-model): 192.168/169.254/
    # 100.64-127 are never version strings, so an unlabeled rule is safe.
    # 10.x / 172.16-31 stay label-gated (version 10.2.3.4 collision class);
    # public ranges are never matched bare.
    RegexRule(
        "IP_ADDRESS",
        _compile(
            r"(?<![0-9.])(?:192\.168\.\d{1,3}\.\d{1,3}"
            r"|169\.254\.\d{1,3}\.\d{1,3}"
            r"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})(?![0-9]|\.[0-9])"
        ),
        0.95,
        validator=private_ipv4_valid,
        validated=True,
    ),
)


def _is_valid_password_value(val: str) -> bool:
    if not val:
        return False
    val = val.strip()
    if len(val) < 6 or len(val) > 256:
        return False
    # Placeholders, templates, env vars
    if val.startswith(("${", "{{", "<", "[", "(", '"', "'", "⟦")):
        return False
    if val.endswith(("}", ">", "]", ")", '"', "'", "⟧")):
        return False
    if any(p in val for p in ["${", "{{", "os.environ", "environ[", "env[", "process.env", "⟦", "⟧"]):
        return False
    # Masked or hidden tokens
    if "*" in val or "•" in val:
        return False
    if any(h in val for h in ["已隐藏", "未设置", "REDACTED", "placeholder", "example", "from prompt", "见保险箱", "dummy"]):
        return False
    # Literals
    if val.lower() in {"null", "none", "nil", "undefined", "true", "false", "password", "passwd", "secret"}:
        return False
    # CJK sentence words or prose
    if any(ch in val for ch in "的是了在和与或很不策略要求规范"):
        return False
    # Must have at least some alphanumeric characters
    if not re.search(r"[A-Za-z0-9]", val):
        return False
    return True


PASSWORD_KEY_PATTERN = (
    r"(?<![A-Za-z0-9_])"
    r"(?:APP_PASSWORD|DB_PASSWORD|ADMIN_PASSWORD|ROOT_PASSWORD|"
    r"Temporary\s+Password|temp\s+password|initial\s+password|admin\s+password|root\s+password|"
    r"临时密码|初始密码|开机密码|支付密码|管理密码|账户密码|"
    r"temporäres\s+Passwort|mot\s+de\s+passe\s+temporaire|contraseña\s+temporal|"
    r"初期パスワード|仮パスワード|임시\s*비밀번호|"
    r"كلمة\s+المرور\s+التجريبية|รหัสผ่านชั่วคราว)"
    r"(?![A-Za-z0-9_])"
)

PASSWORD_QUOTED_KEY_PATTERN = (
    r"(?<![A-Za-z0-9_])"
    r"(?:APP_PASSWORD|DB_PASSWORD|ADMIN_PASSWORD|ROOT_PASSWORD|"
    r"Temporary\s+Password|temp\s+password|initial\s+password|admin\s+password|root\s+password|"
    r"临时密码|初始密码|开机密码|支付密码|管理密码|账户密码|"
    r"password|passwd|密码|口令|"
    r"temporäres\s+Passwort|mot\s+de\s+passe\s+temporaire|contraseña\s+temporal|"
    r"初期パスワード|仮パスワード|임시\s*비밀번호|"
    r"كلمة\s+المرور\s+التجريبية|รหัสผ่านชั่วคราว)"
    r"(?![A-Za-z0-9_])"
)

PASSWORD_DELIM_PATTERN = (
    r"(?:[ \t]*[=＝][ \t]*|[ \t]*[:：][ \t]*(?:\r?\n[ \t]*)?|[ \t]+(?:is|was|ist|lautet|est|es|为|是|هو|هي|คือ)[ \t]*)"
)


CONTEXT_RULES = (
    # Structured Quoted Password
    RegexRule(
        "PASSWORD",
        _compile(
            PASSWORD_QUOTED_KEY_PATTERN + PASSWORD_DELIM_PATTERN + r"[\"']([^\r\n\"']{6,256})[\"']",
            re.IGNORECASE,
        ),
        0.98,
        group=1,
        validator=_is_valid_password_value,
        validated=True,
    ),
    # Structured Unquoted Password
    RegexRule(
        "PASSWORD",
        _compile(
            PASSWORD_KEY_PATTERN + PASSWORD_DELIM_PATTERN + r"([^\s\"'`]{6,256})",
            re.IGNORECASE,
        ),
        0.97,
        group=1,
        validator=_is_valid_password_value,
        validated=True,
    ),
    RegexRule(
        "SECRET",
        _compile(
            r"(?<![A-Za-z0-9_.])"
            r"(?:SSH\s*私钥标识|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|GitHub\s+token|"
            r"OpenAI-style\s+test\s+token|API\s*Token|Temporary\s+password|"
            r"OTP\s+backup\s+code|recovery\s+code|登录密码|密码|口令|passwd|password|secret|token|"
            r"api[_ -]?key|access[_ -]?key|access[_ -]?token|private[_ -]?key|"
            r"令牌|密钥|秘钥|密匙|凭据|凭证|私钥|授权码|访问密钥|接口密钥|"
            r"パスワード|비밀번호|mot\s+de\s+passe|"
            r"Passwort|Contraseña|Пароль|كلمة\s+المرور|รหัสผ่าน)"
            r"(?![A-Za-z0-9_.])"
            r"[\"'“”「」]?[ \t]*[:=：＝][ \t]*[\"'“”「」]?"
            r"(?!/)"
            r"(?:(?=[A-Za-z0-9_!@#$%^&*_~+=-]*[0-9!@#$%^&*])|(?=[A-Za-z0-9_!@#$%^&*_~+=-]{16,}))"
            r"([A-Za-z0-9][A-Za-z0-9_!@#$%^&*_~+=-]{5,255})",
            re.IGNORECASE,
        ),
        0.96,
        group=1,
    ),
    RegexRule(
        "SECRET",
        _compile(
            r"(?:"
            r"(?:(?<![“\"'\w])(?:(?:My|The|my|the)\s+)?(?:temporary\s+)?(?:password|passwd|secret|passcode)\s+(?:is|was)\s+(?!not\b|a\b|the\b|an\b|only\b))"
            r"|(?:(?:登录密码|临时密码|用户密码|开机密码|支付密码|密码|口令)\s*(?:为|是)\s*)"
            r"|(?:(?:令牌|密钥|秘钥|凭据|凭证|授权码|访问密钥|接口密钥|私钥)\s*(?:为|是)\s*)"
            r"|(?:(?<![A-Za-z0-9_])(?:Passwort|Kennwort)\s+(?:ist|lautet)\s+)"
            r"|(?:(?<![A-Za-z0-9_])(?:mot\s+de\s+passe)\s+(?:est)\s+)"
            r"|(?:(?<![A-Za-z0-9_])(?:contraseña)\s+(?:es)\s+)"
            r"|(?:(?<![\u3040-\u30ff\u3400-\u9fff])(?:パスワード)\s*(?:は)\s*)"
            r"|(?:(?<![\uac00-\ud7af])(?:비밀번호)\s*(?:는|은)\s*)"
            r"|(?:كلمة\s+المرور\s+(?:هو|هي)\s+)"
            r"|(?:รหัสผ่าน\s*(?:คือ)\s*)"
            r")"
            r"(?!/)"
            r"(?:(?=[A-Za-z0-9_!@#$%^&*_~+=-]*[0-9!@#$%^&*])|(?=[A-Za-z0-9_!@#$%^&*_~+=-]{16,}))"
            r"([A-Za-z0-9][A-Za-z0-9_!@#$%^&*_~+=-]{5,255})",
            re.IGNORECASE,
        ),
        0.96,
        group=1,
    ),
    RegexRule(
        "SECRET",
        _compile(r"(?i)(?:discord[a-z0-9_ .,\-\"']{0,25})(?:=|>|:=|\|\||:|=>).{0,5}[\"']?([a-f0-9]{64})(?![a-f0-9])"),
        0.98,
        group=1,
        validated=True,
    ),
    RegexRule(
        "CARD_SECURITY_CODE",
        _compile(r"(?:CVV2?|CVC2?|card\s+security\s+code|安全代码|安全码)\s*(?:[=:：]|为|是)?\s*(\d{3,4})(?!\d)", re.IGNORECASE),
        0.99,
        group=1,
        validated=True,
    ),
    RegexRule(
        "CARD_EXPIRY",
        _compile(r"(?:有效期|expiration\s+date|expiry(?:\s+date)?)\s*(?:[=:：]|为|是|is)?\s*(\d{1,2}\s*[/.-]\s*\d{2,4})", re.IGNORECASE),
        0.97,
        group=1,
        validator=card_expiry_valid,
        validated=True,
    ),
    RegexRule(
        "PHONE",
        _compile(
            r"(?:手机号|备用电话|电话(?:号码)?|联系电话|電話番号|휴대전화\s*번호|전화번호|"
            r"phone(?:\s+number)?|Telefonnummer|numéro\s+de\s+téléphone|teléfono|"
            r"номер\s+телефона|رقم\s+الهاتف|หมายเลขโทรศัพท์)"
            r"\s*(?:[=:：]|为|是|is|est|lautet|es|هو|คือ|は|는|은)?\s*"
            r"(\+?\d(?:[\d ()-]{5,25}\d))",
            re.IGNORECASE,
        ),
        0.96,
        group=1,
        validator=international_phone_valid,
        validated=True,
    ),
    RegexRule(
        "CN_LANDLINE",
        _compile(r"(?:座机|固定电话|联系电话|传真)\s*[：:=]?\s*((?:\(?0\d{2,3}\)?[- ]?)?\d{7,8}(?:-\d{1,6})?)"),
        0.93,
        group=1,
    ),
    RegexRule(
        "IP_ADDRESS",
        _compile(r"(?:IPv4|Internal\s+IP|IP(?:地址)?|服务器地址|主机地址)\s*[：:=]?\s*((?:\d{1,3}\.){3}\d{1,3}(?!\.?\d))", re.IGNORECASE),
        0.94,
        group=1,
        validator=ipv4_valid,
        validated=True,
    ),
    RegexRule(
        "IPV6_ADDRESS",
        _compile(r"(?:IPv6(?:\s+address)?)\s*(?:[：:=]|为|是|is|est|ist|es|lautet)?\s*([0-9A-Fa-f:]{2,45})", re.IGNORECASE),
        0.97,
        group=1,
        validator=ipv6_valid,
        validated=True,
    ),
    RegexRule(
        "MAC_ADDRESS",
        _compile(r"(?:MAC(?:\s+address)?|物理地址)\s*[：:=]?\s*([0-9A-Fa-f]{2}([:-])(?:[0-9A-Fa-f]{2}\2){4}[0-9A-Fa-f]{2})", re.IGNORECASE),
        0.98,
        group=1,
        validator=mac_valid,
        validated=True,
    ),
    RegexRule(
        "PRIVATE_URL",
        _compile(r"(?:个人主页|私有网址|回调地址|内网地址|服务地址)\s*[：:=]?\s*(https?://[^\s，。；;]+)", re.IGNORECASE),
        0.9,
        group=1,
    ),
    RegexRule(
        "PRIVATE_URL",
        _compile(r"(?<![A-Za-z0-9_-])(?:Database\s+Host|Host|数据库主机)(?![A-Za-z0-9_-])\s*[：:=]\s*([A-Z0-9.-]+\.[A-Z]{2,63})", re.IGNORECASE),
        0.9,
        group=1,
    ),
    RegexRule(
        "CN_ID_CARD",
        _compile(r"(?:身份证(?:测试)?(?:号码|号)?)\s*(?:[：:=]|为|是)?\s*(\d{17}[0-9Xx]|\d{15})(?![0-9A-Za-z])"),
        0.88,
        group=1,
        validator=cn_id_card_valid,
        validated=True,
    ),
    RegexRule(
        "GOVERNMENT_ID",
        _compile(
            r"(?:SSN|social\s+security\s+number|マイナンバー|주민등록번호)"
            r"\s*(?:[：:=]|为|是|is|は|는|은)?\s*(\d[\d -]{7,18}\d)",
            re.IGNORECASE,
        ),
        0.96,
        group=1,
    ),
    RegexRule(
        "PASSPORT",
        _compile(
            r"(?:(?:test[- ]?)?passport(?:\s+number)?|(?:Test[- ]?)?Passnummer|"
            r"numéro\s+de\s+passeport(?:\s+de\s+test)?|número\s+de\s+pasaporte(?:\s+de\s+prueba)?|"
            r"номер\s+паспорта|جواز\s+السفر(?:\s+التجريبي)?|"
            r"หมายเลขหนังสือเดินทาง(?:ทดสอบ)?|パスポート番号|여권번호)"
            r"\s*(?:[：:=]|为|是|is|est|lautet|es|هو|คือ|は|는|은)?\s*"
            r"([A-Z0-9](?:[A-Z0-9 -]{4,18}[A-Z0-9]))",
            re.IGNORECASE,
        ),
        0.95,
        group=1,
    ),
    RegexRule(
        "BIC",
        _compile(r"(?:BIC|SWIFT(?:\s+code)?)\s*[：:=]?\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)", re.IGNORECASE),
        0.98,
        group=1,
        validated=True,
    ),
    RegexRule(
        "CN_BIRTH_DATE",
        _compile(
            r"(?:出生日期|出生年月|生日)\s*(?:[：:=]|为|是)?\s*"
            r"((?:19|20)\d{2}(?:"
            r"年(?:1[0-2]|0?[1-9])月(?:3[01]|[12]\d|0?[1-9])[日号]?"
            r"|年(?:1[0-2]|0?[1-9])月?"
            r"|-(?:1[0-2]|0?[1-9])-(?:3[01]|[12]\d|0?[1-9])"
            r"|/(?:1[0-2]|0?[1-9])/(?:3[01]|[12]\d|0?[1-9])"
            r"|\.(?:1[0-2]|0?[1-9])\.(?:3[01]|[12]\d|0?[1-9])"
            r"|[-/.](?:1[0-2]|0?[1-9])"
            r"))"
        ),
        0.95,
        group=1,
        validated=True,
    ),
    RegexRule(
        "CN_SOCIAL_ACCOUNT",
        _compile(r"(?:微信号|微信|WeChat|QQ(?:号)?|钉钉号)\s*[：:=]?\s*([A-Za-z][-_A-Za-z0-9]{5,19}|[1-9]\d{4,11})", re.IGNORECASE),
        0.92,
        group=1,
    ),
    RegexRule(
        "CN_ACCOUNT",
        _compile(r"(?:账号|账户|工号|学号|社保号|医保号|订单号|快递单号|客户编号|会员号|设备序列号|VIN)(?:[ \t]*[：:=][ \t]*|[ \t]*(?:为|是)[ \t]*)([A-Z0-9][A-Z0-9_.-]{4,63})", re.IGNORECASE),
        0.89,
        group=1,
    ),
    RegexRule(
        "USERNAME",
        _compile(
            r"(?<![A-Za-z0-9_])"
            r"(?:username|user|user\s+name|用户名|公司账号|登录ID|ログインID|계정\s+이름|identifiant|"
            r"Benutzername|Usuario|Логин|اسم\s+المستخدم|ชื่อผู้ใช้)"
            r"[ \t]*[=:：][ \t]*['\"]?"
            r"([\u4e00-\u9fff]{2,12}|[A-Za-z0-9][A-Za-z0-9_.-]{1,61}[A-Za-z0-9_]|[A-Za-z0-9]{2,63})",
            re.IGNORECASE,
        ),
        0.91,
        group=1,
    ),
    RegexRule(
        "USERNAME",
        _compile(
            r"(?:"
            r"(?:(?<![A-Za-z0-9_])(?:(?:My|The|my|the)\s+)?(?:username|user\s+name)\s+(?:is|was)\s+(?!not\b|a\b|the\b|an\b|only\b))"
            r"|(?:(?:用户名|公司账号|登录账号|登录ID)\s*(?:为|是)\s*)"
            r"|(?:(?<![A-Za-z0-9_])(?:Benutzername)\s+(?:ist|lautet)\s+)"
            r"|(?:(?<![A-Za-z0-9_])(?:Usuario)\s+(?:es)\s+)"
            r"|(?:(?<![A-Za-z0-9_])(?:identifiant)\s+(?:est)\s+)"
            r"|(?:(?<![\u3040-\u30ff\u3400-\u9fff])(?:ログインID)\s*(?:は)\s*)"
            r"|(?:(?<![\uac00-\ud7af])(?:계정\s*이름)\s*(?:은|는)\s*)"
            r"|(?:اسم\s+المستخدم\s+(?:هو)\s+)"
            r"|(?:ชื่อผู้ใช้\s*(?:คือ)\s*)"
            r")"
            r"['\"]?([\u4e00-\u9fff]{2,12}|[A-Za-z0-9][A-Za-z0-9_.-]{1,61}[A-Za-z0-9_]|[A-Za-z0-9]{2,63})",
            re.IGNORECASE,
        ),
        0.91,
        group=1,
    ),
    RegexRule(
        "MEDICAL_RECORD_ID",
        _compile(
            r"(?:medical\s+record(?:\s+test)?\s+(?:ID|number)|MRN|病历号|住院号|门诊号)"
            r"(?:[ \t]*[：:=][ \t]*|[ \t]+(?:is)[ \t]+|[ \t]*(?:为|是)[ \t]*|[ \t]+)"
            r"([A-Z0-9][A-Z0-9_.-]{3,63})",
            re.IGNORECASE,
        ),
        0.95,
        group=1,
    ),
    RegexRule(
        "INSURANCE_ID",
        _compile(
            r"(?:insurance\s+policy(?:\s+test)?\s+(?:ID|number)|医保号|社保卡号|保险号)"
            r"(?:[ \t]*[：:=][ \t]*|[ \t]+(?:is)[ \t]+|[ \t]*(?:为|是)[ \t]*|[ \t]+)"
            r"([A-Z0-9][A-Z0-9_.-]{3,63})",
            re.IGNORECASE,
        ),
        0.95,
        group=1,
    ),
    RegexRule(
        "EMPLOYEE_ID",
        _compile(
            r"(?:employee\s+(?:number|ID)|员工编号|员工工号|工号)"
            r"(?:[ \t]*[：:=][ \t]*|[ \t]+(?:is)[ \t]+|[ \t]*(?:为|是)[ \t]*|[ \t]+)"
            r"([A-Z0-9][A-Z0-9_.-]{3,63})",
            re.IGNORECASE,
        ),
        0.95,
        group=1,
    ),
    RegexRule(
        "STUDENT_ID",
        _compile(
            r"(?:student\s+(?:number|ID)|学号|学生证号)"
            r"(?:[ \t]*[：:=][ \t]*|[ \t]+(?:is)[ \t]+|[ \t]*(?:为|是)[ \t]*|[ \t]+)"
            r"([A-Z0-9][A-Z0-9_.-]{3,63})",
            re.IGNORECASE,
        ),
        0.95,
        group=1,
    ),
    RegexRule(
        "RECORD_ID",
        _compile(
            r"(?:employee\s+(?:number|ID)|student\s+ID|medical\s+record(?:\s+test)?\s+ID|"
            r"insurance\s+policy(?:\s+test)?\s+ID|Account\s+ID|员工编号)"
            r"(?:[ \t]*[：:=][ \t]*|[ \t]+(?:is)[ \t]+|[ \t]*(?:为|是)[ \t]*|[ \t]+)"
            r"([A-Z0-9][A-Z0-9_.-]{3,63})",
            re.IGNORECASE,
        ),
        0.93,
        group=1,
    ),
)


COMMON_SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗宣丁邓郁单杭洪包诸左石崔吉龚程嵇邢裴陆荣翁荀羊於惠甄曲封芮储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶黎乔苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
COMPOUND_SURNAMES = "欧阳|太史|端木|上官|司马|东方|独孤|南宫|万俟|闻人|夏侯|诸葛|尉迟|公羊|赫连|澹台|皇甫|宗政|濮阳|公冶|太叔|申屠|公孙|慕容|仲孙|钟离|长孙|宇文|司徒|鲜于|司空|闾丘|子车|亓官|司寇|巫马|公西|颛孙|壤驷|公良|漆雕|乐正|宰父|谷梁|拓跋|夹谷|轩辕|令狐|段干|百里|呼延|东郭|南门|羊舌|微生"
NAME_VALUE = rf"(?:(?:{COMPOUND_SURNAMES})[\u4e00-\u9fff]{{1,2}}|[{COMMON_SURNAMES}][\u4e00-\u9fff]{{1,2}}|[\u4e00-\u9fff]{{2,4}}·[A-Za-z\u4e00-\u9fff·]{{1,12}})"
NAME_PATTERNS = (
    _compile(rf"(?:姓名|联系人|收件人|患者|客户|员工|申请人|负责人|法定代表人|法人|户主|开户名|经办人)\s*[：:=]\s*({NAME_VALUE})"),
    _compile(rf"(?:负责人|联系人|法定代表人|法人|户主|开户名|经办人|姓名)\s*(?:为|是)\s*({NAME_VALUE})"),
    _compile(rf"(?:我叫|我是|本人)\s*({NAME_VALUE})"),
    _compile(rf"(?<![\u4e00-\u9fff])({NAME_VALUE})(?:先生|女士|医生|老师|经理|主任)(?![\u4e00-\u9fff])"),
)

LATIN_LETTER = r"[A-Za-zÀ-ÖØ-öø-ÿ]"
LATIN_UPPER = r"[A-ZÀ-ÖØ-Þ]"
LATIN_NAME_WORD = rf"(?:{LATIN_UPPER}\.|{LATIN_UPPER}(?:[A-Za-zÀ-ÖØ-öø-ÿ\'’-]*{LATIN_LETTER})?)"
LATIN_NAME_VALUE = rf"{LATIN_NAME_WORD}(?:[ \t]+{LATIN_NAME_WORD}){{1,4}}"

CYRILLIC_LETTER = r"[А-Яа-яЁё]"
CYRILLIC_UPPER = r"[А-ЯЁ]"
CYRILLIC_NAME_WORD = rf"(?:{CYRILLIC_UPPER}\.|{CYRILLIC_UPPER}(?:[А-Яа-яЁё'’-]*{CYRILLIC_LETTER})?)"
CYRILLIC_NAME_VALUE = rf"{CYRILLIC_NAME_WORD}(?:[ \t]+{CYRILLIC_NAME_WORD}){{1,4}}"

MULTILINGUAL_NAME_PATTERNS = (
    _compile(rf"(?:My\s+name\s+is|Customer\s+Name\s*:|Name\s*:)\s*({LATIN_NAME_VALUE})", re.IGNORECASE),
    _compile(rf"Je\s+m['’]appelle\s+({LATIN_NAME_VALUE})", re.IGNORECASE),
    _compile(rf"Mein\s+Name\s+ist\s+({LATIN_NAME_VALUE})", re.IGNORECASE),
    _compile(rf"Me\s+llamo\s+({LATIN_NAME_VALUE})", re.IGNORECASE),
    _compile(rf"Меня\s+зовут\s+({CYRILLIC_NAME_VALUE})", re.IGNORECASE),
    _compile(r"(?:私の名前は|(?:お客様の)?(?:姓名|氏名|お名前)\s*[：:]|Customer\s+Name\s*:)\s*([\u3040-\u30ff\u3400-\u9fff]{2,12})(?=[、，。\n\s]|です|$)", re.IGNORECASE),
    _compile(r"(?:제\s+이름은|负责人\s*)\s*([가-힣]{2,8})(?=입니다|\s+can|\s|[.\n]|$)", re.IGNORECASE),
    _compile(r"اسمي\s+([\u0600-\u06ff]+(?:\s+[\u0600-\u06ff]+){0,4})(?=[.\n،,;؛]|$)"),
    _compile(r"ฉันชื่อ\s+([\u0e00-\u0e7f]+(?:\s+[\u0e00-\u0e7f]+){1,3})(?=\s+ที่อยู่|[.\n]|$)"),
    _compile(rf"Please\s+contact\s+({NAME_VALUE})(?=\s+at)", re.IGNORECASE),
)

ADDRESS_PATTERN = _compile(
    r"(?:收货地址|家庭地址|通讯地址|开户地址|住址|地址|(?:目前)?住在)\s*[：:=为]?\s*"
    r"([^\n，,。；;]{5,100}(?:省|自治区|市|自治州|盟|区|县|旗|镇|乡|街道|路|街|巷|弄|村|社区|号|栋|室)[^\n，,。；;]{0,40})"
)

MULTILINGUAL_ADDRESS_PATTERNS = (
    _compile(r"(?:I\s+live\s+at|I\s+live\s+in)\s+([^\n.]{5,160})", re.IGNORECASE),
    _compile(r"J['’]habite\s+au\s+([^\n.]{5,160})", re.IGNORECASE),
    _compile(r"Ich\s+wohne\s+in\s+der\s+([^\n.]{5,160})", re.IGNORECASE),
    _compile(r"Vivo\s+en\s+([^\n.]{5,160})", re.IGNORECASE),
    _compile(r"Я\s+живу\s+по\s+адресу\s*:\s*([^\n.]{5,160})", re.IGNORECASE),
    _compile(r"أسكن\s+في\s+([^\n.]{5,160})"),
    _compile(r"(?:住所は|住所\s*[：:])\s*(.{5,160}?)(?=です|[。\n]|$)"),
    _compile(r"주소는\s*(.{5,160}?)(?=입니다|[.\n]|$)"),
    _compile(r"ที่อยู่คือ\s*(.{5,160}?)(?=\s+หมายเลขโทรศัพท์|[.\n]|$)"),
)

MULTILINGUAL_DATE_PATTERNS = (
    _compile(
        r"(?:出生日期|出生年月|生日|生年月日|생년월일)\s*(?:[：:=]|为|是|は|는|은)?\s*"
        r"((?:19|20)\d{2}\s*(?:年|년|[-/.])\s*\d{1,2}\s*(?:月|월|[-/.])\s*\d{1,2}\s*(?:日|일)?)"
    ),
    _compile(r"(?:My\s+date\s+of\s+birth\s+is|date\s+of\s+birth\s+is)\s*([^\W\d_]{3,20}\s+\d{1,2},\s*(?:19|20)\d{2})", re.IGNORECASE),
    _compile(
        r"(?:Ma\s+date\s+de\s+naissance\s+est\s+le|Mein\s+Geburtsdatum\s+ist\s+der|"
        r"Nací\s+el|Дата\s+рождения\s*:|تاريخ\s+الميلاد\s+هو|วันเกิดคือ)\s*"
        r"(\d{1,2}\s+[^\W\d_]{3,24}\s+(?:19|20)\d{2}(?:\s+года)?)",
        re.IGNORECASE,
    ),
)


def _entity_from_match(text: str, rule: RegexRule, match: Match[str]) -> Optional[Entity]:
    start, end = match.span(rule.group)
    value = text[start:end]
    left_trim = len(value) - len(value.lstrip())
    right_trim = len(value) - len(value.rstrip())
    start += left_trim
    end -= right_trim
    value = text[start:end]
    if rule.entity_type == "PASSWORD":
        punc_trimmed = value.rstrip("。，,;；")
        if punc_trimmed:
            end -= (len(value) - len(punc_trimmed))
            value = punc_trimmed
    if not value or (rule.validator is not None and not rule.validator(value)):
        return None
    # Placeholder / documentation allowlist (Gitleaks precedent: AWS
    # .+EXAMPLE$): values ENDING in "example" are documentation artifacts,
    # not credentials. Suffix-only, so real keys that merely embed the
    # letters (AWS docs SK sample) stay covered.
    if rule.entity_type == "SECRET" and re.search(r"example$", value, re.IGNORECASE):
        return None
    # USERNAME CJK values are single handles; sentence particles (的/了/是...)
    # indicate prose ("用户名是登录身份的一部分" is a sentence, not a handle).
    if rule.entity_type == "USERNAME" and any("\u4e00" <= c <= "\u9fff" for c in value):
        if any(ch in value for ch in "的是了在和与或很不"):
            return None
    if rule.entity_type == "CN_PHONE_NUMBER" and not value.lstrip().startswith(("+86", "0086")):
        prefix = text[max(0, start - 15) : start]
        if re.search(r"(?:\+\d{1,4}|00\d{1,4})[ ()-]*$", prefix):
            return None
    if rule.entity_type == "EMAIL":
        prefix = text[max(0, start - 30) : start]
        if "://" in prefix and not any(c in prefix[prefix.rfind("://") :] for c in (" ", "\n", "\t", "/")):
            return None
    return Entity(
        entity_type=rule.entity_type,
        start=start,
        end=end,
        text=value,
        confidence=rule.confidence,
        sources=("rules",),
        validated=rule.validated,
    )


class MultilingualRuleDetector:
    """Detect multilingual identifiers and context-bound PII without network calls."""

    name = "multilingual_rules"

    def detect(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for rule in EXACT_RULES + CONTEXT_RULES:
            for match in rule.pattern.finditer(text):
                entity = _entity_from_match(text, rule, match)
                if entity is not None:
                    entities.append(entity)

        for pattern in NAME_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                entities.append(Entity("CN_NAME", start, end, text[start:end], 0.84, (self.name,)))

        for pattern in MULTILINGUAL_NAME_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                entities.append(Entity("PRIVATE_PERSON", start, end, text[start:end], 0.9, (self.name,)))

        for match in ADDRESS_PATTERN.finditer(text):
            start, end = match.span(1)
            value = text[start:end].rstrip(" 的")
            end = start + len(value)
            entities.append(Entity("CN_ADDRESS", start, end, value, 0.86, (self.name,)))

        for pattern in MULTILINGUAL_ADDRESS_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                value = text[start:end].strip().rstrip("。,.،")
                end = start + len(value)
                entities.append(Entity("PRIVATE_ADDRESS", start, end, value, 0.88, (self.name,)))

        for pattern in MULTILINGUAL_DATE_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                entities.append(Entity("PRIVATE_DATE", start, end, text[start:end], 0.91, (self.name,)))

        return entities


def supported_rule_types() -> Iterable[str]:
    values = {rule.entity_type for rule in EXACT_RULES + CONTEXT_RULES}
    values.update(("CN_NAME", "CN_ADDRESS", "PRIVATE_PERSON", "PRIVATE_ADDRESS", "PRIVATE_DATE"))
    return sorted(values)


# Compatibility alias for integrations that imported the original class name.
ChineseRuleDetector = MultilingualRuleDetector

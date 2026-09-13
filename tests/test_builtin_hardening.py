"""Built-in hardening red tests (v0.6.5, maskit-derived).

Each class follows the tests-first contract: the tests below were written
against the CURRENT rules.py and observed failing (red) before the fixes.
Sources of the adversarial cases:

- maskit engine/transparent.py RULES comments (production bugs 0.1.15 etc.)
- maskit tests/test_shield.py regression cases (card BIN/luhn, secret code
  identifiers, IP internal vs version numbers, plate 新README class)
- v0.6.5 goal section 八 (adversarial negatives) and 七 (placeholder
  idempotence: mask(mask(text)) == mask(text))
"""

import re
import sys
from pathlib import Path
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.rules import MultilingualRuleDetector  # noqa: E402
from privacy.validators import cn_id_card_valid  # noqa: E402


def _detect(text: str):
    d = MultilingualRuleDetector()
    return [(e.entity_type, e.text) for e in d.detect(text)]


def _types(text: str, needle: str):
    return [t for t, x in _detect(text) if needle in x or x in needle]


def _luhn_fix(base15: str) -> str:
    """Returns the Luhn-valid 16-digit number for a 15-digit prefix."""
    digits = [int(c) for c in base15 + "0"]
    parity = len(digits) % 2
    total = 0
    for i, ch in enumerate(digits):
        n = ch
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return base15 + str((10 - total % 10) % 10)


class BankCardSeparatorTests(unittest.TestCase):
    """maskit 0.1.15: per-digit optional separators splice unrelated numbers
    across a space; a 16+ digit cross-space run that happens to pass Luhn must
    NOT be treated as a bank card."""

    def test_cross_space_luhn_run_is_not_a_card(self):
        card = _luhn_fix("4983554048202501")  # 16 digits, Luhn-valid
        split = card[:10] + " " + card[10:]   # unrelated size + date-like run
        self.assertNotIn(("CN_BANK_CARD", card), _detect(f"记录 {split} 条目"),
                         "跨空格拼接的数字串不得判定为银行卡")

    def test_single_space_inside_real_card_still_matches(self):
        # real cards split as 4-4-4-4 with a consistent separator
        self.assertIn(("CN_BANK_CARD", "4111 1111 1111 1111"), _detect("卡 4111 1111 1111 1111 结"))

    def test_bin_00_rejected_even_if_luhn_passes(self):
        bad = _luhn_fix("000000000000000")  # BIN 00
        self.assertNotIn(("CN_BANK_CARD", bad), _detect(f"序列{bad}完成"))

    def test_luhn_failure_rejected(self):
        self.assertNotIn(("CN_BANK_CARD", "4111111111111112"), _detect("卡4111111111111112尾"))

    def test_19_digit_unionpay_with_groups(self):
        card = _luhn_fix("6222020200001234567"[:18])
        grouped = " ".join([card[:4], card[4:8], card[8:12], card[12:16], card[16:]])
        self.assertIn(("CN_BANK_CARD", grouped), _detect(f"卡号 {grouped} 核对"))


class PlateDigitRuleTests(unittest.TestCase):
    """maskit 0.1.15: 新README-class false positives - the plate body must
    contain at least one digit (personal plates without digits are not issued
    in CN); the CJK lookbehind does not help when the province char follows a
    space/punctuation."""

    def test_xin_readme_is_not_a_plate(self):
        self.assertNotIn("CN_LICENSE_PLATE", [t for t, _ in _detect("文档 新README.md 的说明")])

    def test_real_plates_still_match(self):
        self.assertIn(("CN_LICENSE_PLATE", "沪A12345"), _detect("牌 沪A12345。"))
        self.assertIn(("CN_LICENSE_PLATE", "京AD12345"), _detect("牌 京AD12345。"))


class SecretContextHardeningTests(unittest.TestCase):
    """maskit: real credentials contain digits/specials and no dots; keyword
    must not be part of a larger identifier; code identifiers are not secrets."""

    def test_code_member_access_is_not_a_secret(self):
        hits = _detect("obj.secret = getSecret()")
        self.assertNotIn("SECRET", [t for t, _ in hits],
                         "obj.secret = getSecret() 不得命中（keyword 前含点号）")

    def test_code_identifier_value_is_not_a_secret(self):
        hits = _detect("const secret = ModelUtils.toStringSafe(foo)")
        self.assertNotIn("SECRET", [t for t, _ in hits],
                         "纯字母方法名不得命中（值必须含数字/特殊符号）")

    def test_pure_alphabetic_value_is_not_a_secret(self):
        self.assertNotIn("SECRET", [t for t, _ in _detect("password = abcdefgh")])

    def test_identifier_suffix_keywords_are_not_secrets(self):
        for text in ("passwordManager 保存", "token_count 统计", "api_key_name 字段",
                     "access_key_length 长度", "getPassword() 调用", "SecretService 类"):
            self.assertNotIn("SECRET", [t for t, _ in _detect(text)], text)

    def test_real_credentials_still_match(self):
        for text, val in (("password=Hn8x!qW2zLm9pR", "Hn8x!qW2zLm9pR"),
                          ("密码：Abc12345xyz", "Abc12345xyz"),
                          ("token = Abc12345XYZ", "Abc12345XYZ")):
            hits = [x for t, x in _detect(text) if t == "SECRET"]
            self.assertTrue(any(val in h for h in hits), text)

    def test_fullwidth_equals_separator(self):
        hits = [x for t, x in _detect("密码＝Abc12345xyz") if t == "SECRET"]
        self.assertTrue(any("Abc12345xyz" in h for h in hits), "全角＝必须被接受为分隔符")

    def test_new_chinese_credential_labels(self):
        for label in ("接口密钥", "访问密钥", "授权码", "凭据", "凭证", "令牌", "密钥"):
            hits = [x for t, x in _detect(f"{label}：Abc12345xyz") if t == "SECRET"]
            self.assertTrue(any("Abc12345xyz" in h for h in hits), label)


class PlaceholderIdempotenceTests(unittest.TestCase):
    """maskit: placeholder values must never be re-detected; the masking
    pipeline should be idempotent (mask(mask(x)) == mask(x))."""

    PLACEHOLDERS = [
        "⟦SECRET_01_ab12cd34⟧",           # production format (frontend)
        "{{SECRET_ab12cd34}}",             # task-spec format (defense in depth)
    ]

    def test_placeholder_value_is_not_rediscovered(self):
        for ph in self.PLACEHOLDERS:
            for label in ("password: ", "密码：", "token = "):
                hits = _detect(f"{label} {ph}")
                self.assertNotIn("SECRET", [t for t, _ in hits], f"{label}{ph}")
                self.assertNotIn("PASSWORD", [t for t, _ in hits], f"{label}{ph}")

    def test_mask_is_idempotent(self):
        d = MultilingualRuleDetector()

        def mask(text: str) -> str:
            out = text
            for e in sorted(d.detect(text), key=lambda e: e.start, reverse=True):
                out = out[:e.start] + f"⟦{e.entity_type}_01_test⟧" + out[e.end:]
            return out

        samples = [
            "password=Hn8x!qW2zLm9pR 联系 13800138000 test@example.com",
            "身份 11010519491231002X 卡 4111 1111 1111 1111",
            "postgresql://admin:s3cr3t@db.example.com:5432/app",
        ]
        for text in samples:
            once = mask(text)
            twice = mask(once)
            self.assertEqual(once, twice, f"mask 非幂等: {text!r}")


class VendorRuleAdditionsTests(unittest.TestCase):
    """Official-format vendor credentials missing from our rules (maskit
    covers them all; prefix-anchored, zero-FP class)."""

    def test_google_api_key(self):
        key = "AIzaSyA1234567890abcdefghijklmnopqrstuV"
        self.assertIn(("SECRET", key), _detect(f"key={key}"))

    def test_stripe_keys(self):
        for key in ("sk_live_abcdefghijklmnopqrst", "rk_test_abcdefghijklmnopqrst"):
            self.assertIn(("SECRET", key), _detect(f"stripe: {key}"))

    def test_aws_sts_asia(self):
        key = "ASIA" + "A1B2C3D4E5F6G7H8"
        self.assertIn(("SECRET", key), _detect(f"sts={key}"))

    def test_aws_secret_access_key_requires_label(self):
        val = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        self.assertIn(("SECRET", val), _detect(f"aws_secret_access_key = {val}"))
        self.assertNotIn("SECRET", [t for t, _ in _detect(f"random {val} end")],
                         "裸 40 位 base64 不得命中（无标签上下文）")

    def test_bearer_token(self):
        self.assertIn(("SECRET", "abcdefghijklmnopqrstuvwxyz123456"),
                      _detect("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"))

    def test_feishu_cli_app_secret(self):
        self.assertIn(("SECRET", "cli_a1b2c3d4e5f6g7h8"), _detect("app=cli_a1b2c3d4e5f6g7h8"))

    def test_slack_refresh_and_session_prefixes(self):
        for tok in ("xoxr-123456789-1234567890123-abcdefghijklmnopqrstuvwx",
                    "xoxs-123456789-1234567890123-abcdefghijklmnopqrstuvwx"):
            self.assertIn(("SECRET", tok), _detect(f"slack: {tok}"))

    def test_pem_dsa_and_pgp_variants(self):
        for head in ("DSA", "PGP"):
            pem = f"-----BEGIN {head} PRIVATE KEY-----\nMIIB\n-----END {head} PRIVATE KEY-----"
            hits = [x for t, x in _detect(pem) if t == "SECRET"]
            self.assertTrue(hits, head)

    def test_vendor_negatives(self):
        for text in ("AIza 短示例", "sk_live_ 说明", "Bearer 尾随文本", "cli_ 账号说明",
                     "prefix ASIA 也可能只是单词"):
            self.assertNotIn(("SECRET", text), _detect(text))


class PrivateIpBareRuleTests(unittest.TestCase):
    """maskit: 192.168/169.254/100.64-127 are never version strings - a bare
    (unlabeled) rule is safe for these ranges only. 10.x/172.16-31 stay
    label-gated (version 10.2.3.4 collision class)."""

    def test_private_ranges_match_without_label(self):
        for ip in ("192.168.1.50", "169.254.10.20", "100.117.224.56"):
            self.assertIn(("IP_ADDRESS", ip), _detect(f"地址 {ip} 通"))

    def test_public_and_version_like_ranges_stay_label_gated(self):
        for text in ("公共 DNS 是 8.8.8.8。", "版本 10.2.3.4 发布", "网段 172.16.3.1 示例"):
            self.assertNotIn(("IP_ADDRESS", "8.8.8.8"), _detect(text))
            self.assertNotIn(("IP_ADDRESS", "10.2.3.4"), _detect(text))
            self.assertNotIn(("IP_ADDRESS", "172.16.3.1"), _detect(text))

    def test_invalid_octets_rejected(self):
        self.assertNotIn(("IP_ADDRESS", "192.168.999.1"), _detect("地址 192.168.999.1 通"))


class SeparatorConsistencyTests(unittest.TestCase):
    def test_mac_mixed_separators_rejected(self):
        self.assertNotIn(("MAC_ADDRESS", "00:11:22-33:44:55"), _detect("MAC 00:11:22-33:44:55 组合"))

    def test_phone_mixed_separators_rejected(self):
        self.assertNotIn(("CN_PHONE_NUMBER", "138-1234 5678"), _detect("联系138-1234 5678 mixing"))

    def test_phone_consistent_separators_kept(self):
        self.assertIn(("CN_PHONE_NUMBER", "138-1234-5678"), _detect("联系138-1234-5678 mixing"))
        self.assertIn(("CN_PHONE_NUMBER", "138 0013 8000"), _detect("联系 138 0013 8000 ok"))


class IdCardProvinceTests(unittest.TestCase):
    def test_province_prefix_enforced(self):
        self.assertFalse(cn_id_card_valid("065217391304348".ljust(18, "1")[:18][:17] + "1"))
        self.assertFalse(cn_id_card_valid("990101199003074" + "514"[-1]))

    def test_valid_provinces_pass(self):
        self.assertTrue(cn_id_card_valid("11010519491231002X"))
        self.assertTrue(cn_id_card_valid("33010519880214551X"))


class PerformanceScalingTests(unittest.TestCase):
    def test_linear_scaling_on_large_inputs(self):
        import time

        d = MultilingualRuleDetector()
        unit = ("用户张三 password=Hn8x!qW2zLm9pR 联系 13800138000 "
                "postgres://admin:s3cr3t@db.example.com:5432/app "
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc Def12345 " * 40)
        timings = {}
        for size in (1024, 8192, 32768, 131072):
            text = (unit * (size // len(unit) + 1))[:size]
            t0 = time.perf_counter()
            d.detect(text)
            timings[size] = time.perf_counter() - t0
        # No pathological backtracking: doubling size must not blow up.
        self.assertLess(timings[131072], 6.0, f"128KB 耗时 {timings[131072]:.2f}s")
        ratio = timings[131072] / max(timings[32768], 1e-6)
        self.assertLess(ratio, 8.0, f"4x 输入耗时倍率 {ratio:.1f}（应近线性）")
        print("\nlatency scaling:", {k: f"{v*1000:.0f}ms" for k, v in timings.items()})


if __name__ == "__main__":
    unittest.main()


class VendorCatalogGapsTests(unittest.TestCase):
    """Gitleaks/Trivy cross-validated vendor rules missing from our set
    (watermark/prefix-anchored, zero-FP class; strictest implementation per
    type chosen after comparing both catalogs)."""

    def test_openai_watermark_keys(self):
        # Trivy: T3BlbkFJ (base64 of 'OpenAI') is the real discriminator.
        proj = "sk-proj-" + "A" * 40 + "T3BlbkFJ" + "B" * 40
        legacy = "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20
        for key in (proj, legacy):
            self.assertIn(("SECRET", key), _detect(f"openai: {key}"), key)

    def test_anthropic_api_key(self):
        key = "sk-ant-api03-" + "A" * 93 + "AA"
        self.assertIn(("SECRET", key), _detect(f"anthropic: {key}"))

    def test_huggingface_token(self):
        key = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"  # 34 after hf_
        self.assertIn(("SECRET", key), _detect(f"hf: {key}"))

    def test_gitlab_pat(self):
        key = "glpat-" + "Ab09CdEfGhIjKlMnOpQr"
        self.assertIn(("SECRET", key), _detect(f"gitlab: {key}"))

    def test_databricks_token(self):
        key = "dapi" + "a1b2c3d4" * 4  # 32 hex
        self.assertIn(("SECRET", key), _detect(f"databricks: {key}"))

    def test_linear_api_key(self):
        key = "lin_api_" + "a1b2c3d4e5" * 4  # 40 lower alnum
        self.assertIn(("SECRET", key), _detect(f"linear: {key}"))

    def test_azure_entra_client_secret(self):
        key = "ab1" + "8Q~" + "aZ09_~-." * 4 + "aZ"  # 3-char prefix + 34-char tail
        self.assertIn(("SECRET", key), _detect(f"azure: {key}"))

    def test_aws_prefix_family_widened(self):
        for prefix in ("AGPA", "AIDA", "AROA", "AIPA", "ANPA", "ANVA", "ABIA", "ACCA"):
            key = prefix + "A1B2C3D4E5F6G7H8"
            self.assertIn(("SECRET", key), _detect(f"aws: {key}"), prefix)

    def test_ghs_stateless_app_token_with_dots(self):
        key = "ghs_" + "A1b2" * 9 + "." + "a1B2" * 6 + "." + "c3D4" * 6
        self.assertIn(("SECRET", key), _detect(f"ghs: {key}"))

    def test_discord_quoted_hex_secret(self):
        secret = "a" * 64
        self.assertIn(("SECRET", secret),
                      _detect(f'"discord_token" : "{secret}"'))

    def test_format_perfect_fakes_runtime_assembled(self):
        """Format-perfect fakes whose discriminators are identical to the
        push-protection scanners (OpenAI T3BlbkFJ, Tencent AKID+32) can only
        exist at runtime - the fixture must not carry the literals."""
        import random

        rng = random.Random(20260913)
        up = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        al = "abcdefghijklmnopqrstuvwxyz0123456789"
        openai_key = "sk-proj-" + "".join(rng.choice(al + "-_") for _ in range(40)) + "T3BlbkFJ" + "".join(rng.choice(al + "-_") for _ in range(40))
        tencent_id = "AKID" + "".join(rng.choice(al) for _ in range(32))
        slack_bot = "xoxb-" + "".join(rng.choice("0123456789") for _ in range(10)) + "-" + "".join(rng.choice("0123456789") for _ in range(13)) + "-" + "".join(rng.choice(al) for _ in range(24))
        databricks = "dapi" + "".join(rng.choice("0123456789abcdef") for _ in range(32))
        linear = "lin_api_" + "".join(rng.choice(al) for _ in range(40))
        for key in (openai_key, tencent_id, slack_bot, databricks, linear):
            self.assertIn(("SECRET", key), _detect(f"sample: {key}"), key[:24])

    def test_catalog_negatives(self):
        for text in ("openai: sk-short", "databricks: dapi12345",
                     "linear: lin_api_short", "azure: xx8Q~short",
                     "hf: hf_short"):
            self.assertNotIn("SECRET", [t for t, _, _ in _detect(text)], text)

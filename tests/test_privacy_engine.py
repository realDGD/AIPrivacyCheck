import sys
from pathlib import Path
import tempfile
import unittest


SERVER_DIR = Path(__file__).resolve().parents[1] / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.entities import Entity
from privacy.merge import merge_entities
from privacy.rules import ChineseRuleDetector
from privacy.service import PrivacyService
from privacy.validators import USCC_ALPHABET, USCC_WEIGHTS, cn_id_card_valid, cn_uscc_valid, ipv4_valid, luhn_valid


def valid_uscc(prefix: str) -> str:
    total = sum(USCC_ALPHABET.index(char) * weight for char, weight in zip(prefix, USCC_WEIGHTS))
    return prefix + USCC_ALPHABET[(31 - total % 31) % 31]


class ValidatorTests(unittest.TestCase):
    def test_chinese_id_checksum_and_birth_date(self):
        self.assertTrue(cn_id_card_valid("11010519491231002X"))
        self.assertFalse(cn_id_card_valid("110105194912310021"))
        self.assertFalse(cn_id_card_valid("110105199902300021"))

    def test_luhn(self):
        self.assertTrue(luhn_valid("4532 0151 1283 0366"))
        self.assertFalse(luhn_valid("4532 0151 1283 0367"))
        self.assertFalse(luhn_valid("1111111111111111"))

    def test_unified_social_credit_code(self):
        code = valid_uscc("91350211M000100Y4")
        self.assertTrue(cn_uscc_valid(code))
        self.assertFalse(cn_uscc_valid(code[:-1] + ("0" if code[-1] != "0" else "1")))

    def test_ipv4(self):
        self.assertTrue(ipv4_valid("192.168.1.15"))
        self.assertFalse(ipv4_valid("999.168.1.15"))


class RuleDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = ChineseRuleDetector()

    def types_for(self, text: str):
        return {entity.entity_type for entity in self.detector.detect(text)}

    def test_detects_common_chinese_pii(self):
        text = (
            "联系人：张三，手机号 13800138000，邮箱 zhangsan@example.com，"
            "身份证 11010519491231002X，银行卡 4532 0151 1283 0366，"
            "收货地址：上海市浦东新区世纪大道100号。"
        )
        kinds = self.types_for(text)
        self.assertTrue({"CN_NAME", "CN_PHONE_NUMBER", "EMAIL", "CN_ID_CARD", "CN_BANK_CARD", "CN_ADDRESS"}.issubset(kinds))

    def test_context_required_for_ambiguous_values(self):
        self.assertNotIn("CN_ACCOUNT", self.types_for("发布版本为 ABCDEF123456，普通编号无需处理。"))
        self.assertIn("CN_ACCOUNT", self.types_for("客户编号：ABCDEF123456"))
        # 192.168 is never a version string: bare private-range rule (v0.6.5)
        self.assertIn("IP_ADDRESS", self.types_for("内网 192.168.1.15 可达"))
        # 10.x collides with version numbers: stays label-gated
        self.assertNotIn("IP_ADDRESS", self.types_for("版本号 10.2.3.15"))
        self.assertIn("IP_ADDRESS", self.types_for("服务器地址：10.2.3.15"))

    def test_secrets(self):
        kinds = self.types_for("password=correct-horse-battery-staple\nAuthorization: eyJabcdefgh.abcdefghijk.abcdefghijk")
        self.assertIn("SECRET", kinds)

    def test_invalid_checksums_are_rejected(self):
        text = "身份证 110105194912310021，银行卡 4532015112830367"
        kinds = self.types_for(text)
        self.assertNotIn("CN_ID_CARD", kinds)
        self.assertNotIn("CN_BANK_CARD", kinds)


class MergeTests(unittest.TestCase):
    def test_validated_rule_wins_overlap(self):
        text = "13800138000"
        model = Entity("PHONE", 0, len(text), text, 0.78, ("model",))
        rule = Entity("CN_PHONE_NUMBER", 0, len(text), text, 0.99, ("rules",), validated=True)
        result = merge_entities([model, rule])
        self.assertEqual([item.entity_type for item in result], ["CN_PHONE_NUMBER"])

    def test_identical_entities_merge_sources(self):
        first = Entity("EMAIL", 0, 7, "a@b.com", 0.8, ("rules",))
        second = Entity("EMAIL", 0, 7, "a@b.com", 0.9, ("model",))
        result = merge_entities([first, second])
        self.assertEqual(result[0].sources, ("model", "rules"))
        self.assertEqual(result[0].confidence, 0.9)


class ServiceTests(unittest.TestCase):
    def test_rules_work_without_model_install(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PrivacyService(Path(directory))
            result = service.detect("联系人张三，手机号13800138000", use_model=True)
        self.assertIn(service.rules.name, result["engines"])
        self.assertIn(service.chinese_ie.name, result["engines"])
        self.assertNotIn("gliner_pii", result["engines"])
        self.assertNotIn("memprivacy", result["engines"])
        self.assertTrue(result["warnings"])

    def test_registry_slots_and_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PrivacyService(Path(directory))
            caps = service.capabilities()
            self.assertIn("registry", caps)
            self.assertIn("slots", caps["registry"])
            slots = caps["registry"]["slots"]
            self.assertIn("built_in", slots)
            self.assertIn("chinese_ie", slots)
            self.assertIn("general_pii", slots)
            self.assertIn("semantic_privacy", slots)

    def test_privacy_policy_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PrivacyService(Path(directory))
            text = "私钥 " + "-----BEGIN " + "PRIVATE KEY----- ABC -----END " + "PRIVATE KEY----- 姓名张三"
            res_pl4 = service.detect(text, policy_level="PL4")
            for ent in res_pl4["entities"]:
                self.assertEqual(ent["privacy_level"], "PL4")


if __name__ == "__main__":
    unittest.main()

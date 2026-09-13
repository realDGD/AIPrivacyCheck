import sys
from pathlib import Path
import unittest

SERVER_DIR = Path(__file__).resolve().parents[1] / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.rules import MultilingualRuleDetector
from privacy.merge import merge_entities


class RulesFixTests(unittest.TestCase):
    def setUp(self):
        self.detector = MultilingualRuleDetector()

    def test_cn_birth_date_full_span(self):
        text = "用户档案：姓名李四，出生日期：1992年11月18日，籍贯北京。"
        entities = merge_entities(self.detector.detect(text))
        dob_entities = [e for e in entities if e.entity_type == "CN_BIRTH_DATE"]
        self.assertEqual(len(dob_entities), 1, f"Expected 1 CN_BIRTH_DATE entity, found: {dob_entities}")
        self.assertEqual(dob_entities[0].text, "1992年11月18日")

        # Variations
        text2 = "出生年月：1985-05-20，生日为 2001.08.15，出生日期 1999年2月"
        ents2 = {e.text: e.entity_type for e in merge_entities(self.detector.detect(text2))}
        self.assertIn("1985-05-20", ents2)
        self.assertIn("2001.08.15", ents2)
        self.assertIn("1999年2月", ents2)

    def test_structured_password_detection(self):
        cases = [
            ("APP_PASSWORD=v3ry_s3cr3t_p@ssw0rd!", "v3ry_s3cr3t_p@ssw0rd!"),
            ("Temporary Password:\nTempP@ss2026!", "TempP@ss2026!"),
            ("临时密码：\nTmpPass#9981", "TmpPass#9981"),
            ("كلمة المرور التجريبية: Arab!cPass2026", "Arab!cPass2026"),
            ('password="my_secure_pass123"', "my_secure_pass123"),
            ("password='another_secret_pass'", "another_secret_pass"),
            ("初始密码: InitPass9988!", "InitPass9988!"),
        ]
        for sample, expected_val in cases:
            entities = merge_entities(self.detector.detect(sample))
            pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
            self.assertTrue(pwd_entities, f"Failed to detect PASSWORD in: {sample}")
            self.assertEqual(pwd_entities[0].text, expected_val, f"Mismatch in extracted value for: {sample}")

    def test_password_strict_negatives_rejected(self):
        negatives = [
            "password = ${DB_PASSWORD}",
            "password: ******（见保险箱）",
            "config.password = os.environ['DB_PASS']",
            "密码：[已隐藏]",
            "passwordManager 同步完成",
            "setPassword(value) 后请刷新会话",
            "getPassword() 返回 None",
            "user.settings.password_policy.enabled",
            "password: (from prompt)",
            "密码策略要求至少 16 个字符。",
            "密码找回流程见帮助中心。",
        ]
        for neg in negatives:
            entities = merge_entities(self.detector.detect(neg))
            pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
            self.assertEqual(pwd_entities, [], f"Negative false positive on: {neg}")


    def test_password_with_asterisk_accepted_and_masks_rejected(self):
        # Real passwords with * must be accepted
        valid_with_asterisk = [
            ("APP_PASSWORD=MyPass*2026", "MyPass*2026"),
            ("APP_PASSWORD=A*b9X!234", "A*b9X!234"),
            ('password="P@ss*w0rd_999"', "P@ss*w0rd_999"),
        ]
        for sample, expected in valid_with_asterisk:
            entities = merge_entities(self.detector.detect(sample))
            pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
            self.assertTrue(pwd_entities, f"Failed to detect password with asterisk: {sample}")
            self.assertEqual(pwd_entities[0].text, expected)

        # Pure mask tokens must be rejected
        masks = [
            "APP_PASSWORD=******",
            "APP_PASSWORD=••••••",
            "APP_PASSWORD=************",
            "password: ******",
        ]
        for m in masks:
            entities = merge_entities(self.detector.detect(m))
            pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
            self.assertEqual(pwd_entities, [], f"Mask-only token false positive: {m}")

    def test_invalid_dob_partial_fallback_rejected(self):
        # Invalid full dates must be rejected entirely and must NOT degrade into partial year-month spans
        invalid_dates = [
            "出生日期：1992年11月99日",
            "出生日期：1992-11-99",
            "生年月日：1992年11月99日",
            "생년월일: 1992년 11월 99일",
        ]
        for inv in invalid_dates:
            entities = merge_entities(self.detector.detect(inv))
            date_entities = [e for e in entities if e.entity_type in ("CN_BIRTH_DATE", "PRIVATE_DATE")]
            self.assertEqual(date_entities, [], f"Invalid date must not partially match: {inv}")


if __name__ == "__main__":
    unittest.main()

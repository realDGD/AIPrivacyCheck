"""Targeted False-Positive and True-Positive Regression Tests for rules.py."""

import unittest
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.rules import MultilingualRuleDetector
from privacy.service import PrivacyService


class FalsePositiveRegressionTests(unittest.TestCase):
    """Ensure context-based rules do not extract false positives from ordinary sentences."""

    def setUp(self):
        self.detector = MultilingualRuleDetector()
        self.service = PrivacyService(Path("/tmp/test_privacy_fp"))

    def test_negative_secret_sentences(self):
        cases = [
            "The password policy requires at least sixteen characters.",
            "This sentence contains the word password but does not contain a password value.",
            "The API token feature is disabled.",
            "There is no API token in the previous sentence.",
            "An API key should never be committed to source control.",
            "AWS_ACCESS_KEY_ID is the name of an environment variable.",
            "AWS_SECRET_ACCESS_KEY is also only a variable name here.",
            "GitHub tokens normally begin with a recognizable prefix.",
            "The private key documentation explains how PEM files begin and end.",
            "Token budget 设置为 8192。",
            "模型的 token 数量不能超过上下文窗口。",
            "API Token 功能目前处于关闭状态，本段没有提供任何真实 Token。",
            "密码策略要求至少 16 个字符。",
            "“password”是数据库字段名称，本段没有提供实际密码。",
        ]
        for sentence in cases:
            with self.subTest(sentence=sentence):
                entities = self.detector.detect(sentence)
                secret_entities = [e for e in entities if e.entity_type in ("SECRET", "PASSWORD", "API_TOKEN", "PRIVATE_KEY")]
                self.assertEqual(
                    secret_entities,
                    [],
                    f"False positive SECRET detected in: {sentence!r} -> {secret_entities}",
                )

    def test_negative_username_sentences(self):
        sentence = "Username and Password are labels displayed on the login screen."
        entities = self.detector.detect(sentence)
        username_entities = [e for e in entities if e.entity_type == "USERNAME"]
        self.assertEqual(
            username_entities,
            [],
            f"False positive USERNAME detected in: {sentence!r} -> {username_entities}",
        )

    def test_negative_record_id_database_sentences(self):
        cases = [
            "Database credentials are managed by another service.",
            "Database connection string:",
            'The phrase "Database credentials" itself is only a section heading.',
            "The word credentials itself is not a credential.",
        ]
        for sentence in cases:
            with self.subTest(sentence=sentence):
                entities = self.detector.detect(sentence)
                record_id_entities = [e for e in entities if e.entity_type in ("RECORD_ID", "CN_ACCOUNT")]
                self.assertEqual(
                    record_id_entities,
                    [],
                    f"False positive RECORD_ID detected in: {sentence!r} -> {record_id_entities}",
                )

    def test_negative_chinese_name_sentences(self):
        cases = [
            "两处姓名应该分别识别。",
            "姓名字段目前为空。",
            "姓名识别功能已经开启。",
            "姓名检测需要人工复核。",
            "“姓名”只是字段标签。",
            "联系人字段为空。",
            "负责人字段尚未填写。",
        ]
        for sentence in cases:
            with self.subTest(sentence=sentence):
                result = self.service.detect(sentence, use_model=False)
                name_entities = [e for e in result["entities"] if e["type"] in ("CN_NAME", "PRIVATE_PERSON", "PERSON")]
                self.assertEqual(
                    name_entities,
                    [],
                    f"False positive name detected in: {sentence!r} -> {name_entities}",
                )


class TruePositiveRegressionTests(unittest.TestCase):
    """Ensure explicit fields and valid natural-language context continue to be detected accurately."""

    def setUp(self):
        self.detector = MultilingualRuleDetector()

    def test_positive_secrets(self):
        cases = [
            ("Password: Ocean-Test#2026", "Ocean-Test#2026"),
            ("Password = Ocean-Test#2026", "Ocean-Test#2026"),
            ("My password is Ocean-Test#2026", "Ocean-Test#2026"),
            ("登录密码：BlueSky-Test#2026", "BlueSky-Test#2026"),
            ("登录密码为 BlueSky-Test#2026", "BlueSky-Test#2026"),
            ("API Token: sk-proj-TESTONLY-123456", "sk-proj-TESTONLY-123456"),
            ("AWS_SECRET_ACCESS_KEY=TESTONLY_FAKE_SECRET_KEY_2026", "TESTONLY_FAKE_SECRET_KEY_2026"),
        ]
        for text, expected_value in cases:
            with self.subTest(text=text):
                entities = [e for e in self.detector.detect(text) if e.entity_type in ("SECRET", "PASSWORD", "API_TOKEN")]
                self.assertTrue(len(entities) >= 1, f"Expected SECRET not found in: {text!r}")
                matched_text = entities[0].text
                self.assertEqual(matched_text, expected_value)
                self.assertEqual(text[entities[0].start:entities[0].end], expected_value)

    def test_positive_usernames(self):
        cases = [
            ("Username: privacy_test", "privacy_test"),
            ("Username = privacy_test", "privacy_test"),
            ("My username is privacy_test", "privacy_test"),
        ]
        for text, expected_value in cases:
            with self.subTest(text=text):
                entities = [e for e in self.detector.detect(text) if e.entity_type == "USERNAME"]
                self.assertTrue(len(entities) >= 1, f"Expected USERNAME not found in: {text!r}")
                self.assertEqual(entities[0].text, expected_value)
                self.assertEqual(text[entities[0].start:entities[0].end], expected_value)

    def test_positive_chinese_names(self):
        cases = [
            ("姓名：陈思远", "陈思远"),
            ("联系人：陈思远", "陈思远"),
            ("负责人：周子涵", "周子涵"),
            ("负责人是周子涵", "周子涵"),
            ("负责人为周子涵", "周子涵"),
            ("我叫林雨辰", "林雨辰"),
        ]
        for text, expected_name in cases:
            with self.subTest(text=text):
                entities = [e for e in self.detector.detect(text) if e.entity_type in ("CN_NAME", "PRIVATE_PERSON")]
                self.assertTrue(len(entities) >= 1, f"Expected name not found in: {text!r}")
                self.assertEqual(entities[0].text, expected_name)
                self.assertEqual(text[entities[0].start:entities[0].end], expected_name)

    def test_latin_name_span_no_multiline_leak(self):
        text = "Name: Olivia Mercer\nPhone: +1 (202) 555-0186\nEmail: olivia.mercer@example.com"
        entities = self.detector.detect(text)
        name_entities = [e for e in entities if e.entity_type in ("PRIVATE_PERSON", "PERSON")]
        self.assertEqual(len(name_entities), 1)
        name = name_entities[0]
        self.assertEqual(name.text, "Olivia Mercer")
        self.assertEqual(text[name.start:name.end], "Olivia Mercer")
        self.assertNotIn("Phone", name.text)
        self.assertNotIn("Phone:", name.text)
        self.assertNotIn("\n", name.text)

    def test_latin_name_span_trailing_period_excluded(self):
        text = "My name is Olivia Mercer. I work in Seattle."
        entities = self.detector.detect(text)
        name_entities = [e for e in entities if e.entity_type in ("PRIVATE_PERSON", "PERSON")]
        self.assertEqual(len(name_entities), 1)
        name = name_entities[0]
        self.assertEqual(name.text, "Olivia Mercer")
        self.assertEqual(text[name.start:name.end], "Olivia Mercer")
        self.assertFalse(name.text.endswith("."))


if __name__ == "__main__":
    unittest.main()

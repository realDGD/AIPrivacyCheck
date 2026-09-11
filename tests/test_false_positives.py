"""Targeted False-Positive and True-Positive Regression Tests for rules.py."""

import unittest
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.chinese_ie import BuiltinChineseIE
from privacy.rules import MultilingualRuleDetector
from privacy.service import PrivacyService


class FalsePositiveRegressionTests(unittest.TestCase):
    """Ensure context-based rules do not extract false positives from ordinary sentences."""

    def setUp(self):
        self.detector = MultilingualRuleDetector()
        self.service = PrivacyService(Path("/tmp/test_privacy_fp"))
        self.chinese_ie = BuiltinChineseIE()

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
            "系统和周边，均运行正常。",
            "产品与王者，都只是项目名称。",
            "给周边设备供电。",
            "给王者客户端分配资源。",
            "用户王者，表示的是游戏用户标签。",
            "用户周边功能尚未完成。",
            "客户王者荣耀，是一个项目代号。",
            "客户资料已经归档。",
            "联系我们的客服团队。",
            "通知管理员，系统需要升级。",
            "与李子树相关的研究已经完成。",
            "和周边设备建立连接。",
            "同王者版本相比，本版本性能更高。",
            "跟周边系统保持兼容。",
            "经理高大并不是人员信息，而是测试短语。",
            "老师周边资源将在下周更新。",
            "作者王道并不是一个作者姓名，而是文档中的普通短语。",
            "姓名字段目前为空。",
            "联系人列表已经更新。",
            "负责人字段尚未填写。",
            # Additional context checks
            "两处姓名应该分别识别。",
            "姓名识别功能已经开启。",
            "姓名检测需要人工复核。",
            "“姓名”只是字段标签。",
            "联系人字段为空。",
        ]
        for sentence in cases:
            with self.subTest(sentence=sentence):
                # 1. Chinese IE extractor level
                ie_names = self.chinese_ie.extract_names(sentence)
                self.assertEqual(
                    ie_names,
                    [],
                    f"False positive in BuiltinChineseIE: {sentence!r} -> {ie_names}",
                )
                # 2. Service level integration
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
        self.service = PrivacyService(Path("/tmp/test_privacy_tp"))
        self.chinese_ie = BuiltinChineseIE()

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
            ("联系人：李明", "李明"),
            ("负责人：周子涵", "周子涵"),
            ("作者：林雨辰", "林雨辰"),
            ("客户：王晓丽", "王晓丽"),
            ("负责人是周子涵", "周子涵"),
            ("负责人为周子涵", "周子涵"),
            ("联系人是陈思远", "陈思远"),
            ("联系人为陈思远", "陈思远"),
            ("我叫林雨辰", "林雨辰"),
            ("我是林雨辰", "林雨辰"),
            ("由张伟负责。", "张伟"),
            ("通知李明。", "李明"),
            ("请通知李明。", "李明"),
            ("联系王晓丽。", "王晓丽"),
            ("请联系王晓丽。", "王晓丽"),
            ("找陈思远。", "陈思远"),
            ("转交给陈思远。", "陈思远"),
            ("拜访周子涵。", "周子涵"),
            ("采访林雨辰。", "林雨辰"),
            ("陪同张伟。", "张伟"),
            ("张伟先生", "张伟"),
            ("李明老师", "李明"),
            ("王晓丽医生", "王晓丽"),
            ("陈思远教授", "陈思远"),
        ]
        for text, expected_name in cases:
            with self.subTest(text=text):
                # 1. Chinese IE extractor verification
                ie_names = self.chinese_ie.extract_names(text)
                self.assertTrue(len(ie_names) >= 1, f"BuiltinChineseIE missed name in: {text!r}")
                self.assertEqual(ie_names[0][2], expected_name)
                self.assertEqual(text[ie_names[0][0]:ie_names[0][1]], expected_name)

                # 2. Service level integration
                result = self.service.detect(text, use_model=False)
                entities = [e for e in result["entities"] if e["type"] in ("CN_NAME", "PRIVATE_PERSON", "PERSON")]
                self.assertTrue(len(entities) >= 1, f"Expected name not found in: {text!r}")
                matched = entities[0]
                self.assertEqual(matched["text"], expected_name)
                self.assertEqual(text[matched["start"]:matched["end"]], expected_name)

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

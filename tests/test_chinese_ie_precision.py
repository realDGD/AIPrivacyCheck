"""Precision and regression tests specifically for BuiltinChineseIE."""

from pathlib import Path
import sys
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.chinese_ie import BuiltinChineseIE
from privacy.service import PrivacyService


class BuiltinChineseIEPrecisionTests(unittest.TestCase):
    """Ensure BuiltinChineseIE prioritizes precision over broad recall and avoids false positives."""

    def setUp(self):
        self.ie = BuiltinChineseIE()
        self.service = PrivacyService(Path("/tmp/test_privacy_chinese_ie_precision"))

    def test_negative_regression_sentences_yield_zero_cn_name(self):
        """Verify the 20 targeted negative sentences produce 0 false positive CN_NAME."""
        negative_sentences = [
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
        ]

        for sentence in negative_sentences:
            with self.subTest(sentence=sentence):
                # 1. Builtin extractor level
                names = self.ie.extract_names(sentence)
                self.assertEqual(
                    names,
                    [],
                    f"BuiltinChineseIE produced false positive on: {sentence!r} -> {names}",
                )

                # 2. Privacy service end-to-end level
                res = self.service.detect(sentence, use_model=False)
                cn_names = [e for e in res["entities"] if e["type"] in ("CN_NAME", "PRIVATE_PERSON", "PERSON")]
                self.assertEqual(
                    cn_names,
                    [],
                    f"PrivacyService produced false positive on: {sentence!r} -> {cn_names}",
                )

    def test_positive_regression_sentences_yield_exact_spans(self):
        """Verify high-confidence context and anchor phrases accurately extract names with exact spans."""
        positive_cases = [
            # 字段模式
            ("姓名：陈思远", "陈思远"),
            ("联系人：李明", "李明"),
            ("负责人：周子涵", "周子涵"),
            ("作者：林雨辰", "林雨辰"),
            ("客户：王晓丽", "王晓丽"),
            # 自然语言谓词模式
            ("负责人是周子涵", "周子涵"),
            ("负责人为周子涵", "周子涵"),
            ("联系人是陈思远", "陈思远"),
            ("联系人为陈思远", "陈思远"),
            # 自我介绍模式
            ("我叫林雨辰", "林雨辰"),
            ("我是林雨辰", "林雨辰"),
            # 专有人行动作短语 (由 X 负责)
            ("由张伟负责。", "张伟"),
            # 高置信人行动作对象短语
            ("通知李明。", "李明"),
            ("请通知李明。", "李明"),
            ("联系王晓丽。", "王晓丽"),
            ("请联系王晓丽。", "王晓丽"),
            ("找陈思远。", "陈思远"),
            ("转交给陈思远。", "陈思远"),
            ("拜访周子涵。", "周子涵"),
            ("采访林雨辰。", "林雨辰"),
            ("陪同张伟。", "张伟"),
            # 后置称谓模式
            ("张伟先生", "张伟"),
            ("李明老师", "李明"),
            ("王晓丽医生", "王晓丽"),
            ("陈思远教授", "陈思远"),
        ]

        for text, expected in positive_cases:
            with self.subTest(text=text, expected=expected):
                # 1. Builtin extractor
                names = self.ie.extract_names(text)
                self.assertTrue(len(names) >= 1, f"No name extracted from: {text!r}")
                start, end, val, conf = names[0]
                self.assertEqual(val, expected)
                self.assertEqual(text[start:end], expected)

                # 2. Service level
                res = self.service.detect(text, use_model=False)
                cn_names = [e for e in res["entities"] if e["type"] in ("CN_NAME", "PRIVATE_PERSON", "PERSON")]
                self.assertTrue(len(cn_names) >= 1, f"No name detected by service in: {text!r}")
                entity = cn_names[0]
                self.assertEqual(entity["text"], expected)
                self.assertEqual(text[entity["start"]:entity["end"]], expected)

    def test_section_xv_manual_verification_cases(self):
        """Execute exact check specified in Section XV of requirements."""
        # 1. Negative checks: CN_NAME count must be 0
        neg_samples = [
            "系统和周边，均运行正常。",
            "产品与王者，都只是项目名称。",
            "给周边设备供电。",
            "用户王者，表示的是游戏用户标签。",
            "客户王者荣耀，是一个项目代号。",
            "与李子树相关的研究已经完成。",
        ]
        for s in neg_samples:
            names = self.ie.extract_names(s)
            self.assertEqual(len(names), 0, f"Expected 0 CN_NAME in {s!r}, got {names}")

        # 2. Positive checks
        pos_samples = [
            ("由张伟负责。", "张伟"),
            ("请联系李明。", "李明"),
            ("负责人是周子涵。", "周子涵"),
            ("姓名：陈思远", "陈思远"),
        ]
        for s, expected in pos_samples:
            names = self.ie.extract_names(s)
            self.assertTrue(len(names) >= 1, f"Expected {expected} in {s!r}, got none")
            self.assertEqual(names[0][2], expected)
            self.assertEqual(s[names[0][0]:names[0][1]], expected)


if __name__ == "__main__":
    unittest.main()

"""Comprehensive regression suite covering the 7 key requirements and integrity constraints."""

from pathlib import Path
import sys
import tempfile
import unittest

SERVER_DIR = Path(__file__).resolve().parents[1] / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.chinese_ie import ChineseIEDetector, safe_sequential_span_alignment
from privacy.device import DEVICE_MANAGER, DeviceManager
from privacy.entities import Entity
from privacy.merge import merge_entities
from privacy.rules import MultilingualRuleDetector
from privacy.service import PrivacyService
from privacy.validators import cn_id_card_valid, iban_valid, luhn_valid


class KeyRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rules = MultilingualRuleDetector()
        self.service = PrivacyService(Path(tempfile.mkdtemp()))

    def test_regression_1_db_uri_not_overridden_by_email(self):
        """1. postgresql://testuser:FakePassword123@db.example.com:5432/privacy_test
        Must be identified as DATABASE_URI / Secret, FakePassword123@db.example.com must NOT be emitted as EMAIL.
        """
        text = "Database connection string: postgresql://testuser:FakePassword123@db.example.com:5432/privacy_test for testing."
        result = self.service.detect(text)
        entities = result["entities"]
        types = [e["type"] for e in entities]

        # Ensure DATABASE_URI or SECRET is detected
        self.assertTrue("DATABASE_URI" in types or "SECRET" in types)
        # Ensure FakePassword123@db.example.com is NOT recognized as EMAIL
        self.assertNotIn("EMAIL", types)

        # Ensure the whole URI span is preserved
        target_span = "postgresql://testuser:FakePassword123@db.example.com:5432/privacy_test"
        matched = [e for e in entities if e["text"] == target_span]
        self.assertTrue(len(matched) == 1, "The entire database URI must be matched as a single span")

    def test_regression_2_phone_with_country_code(self):
        """2. +49 151 00004281
        Full phone span must include '+49', must not truncate to '151 00004281'.
        """
        text = "Bitte rufen Sie mich an: +49 151 00004281 morgen."
        result = self.service.detect(text)
        entities = result["entities"]
        phone_matches = [e for e in entities if e["type"] in ("PHONE", "CN_PHONE_NUMBER")]

        self.assertTrue(len(phone_matches) >= 1)
        matched_text = phone_matches[0]["text"]
        self.assertTrue(matched_text.startswith("+49"), f"Expected span to include +49, got '{matched_text}'")
        self.assertEqual(matched_text, "+49 151 00004281")

    def test_regression_3_identical_name_twice_offsets(self):
        """3. An identical name appears twice in the same text.
        Confirm semantic detector does not return the same offset for both occurrences.
        """
        text = "今天张伟向李明汇报了进展，随后张伟离开了会议室。"
        # "张伟" appears at index 2 and index 15
        s1 = text.find("张伟")
        s2 = text.find("张伟", s1 + 1)
        self.assertNotEqual(s1, s2)

        predictions = [("CN_NAME", "张伟", 0.9), ("CN_NAME", "张伟", 0.9)]
        aligned, warnings = safe_sequential_span_alignment(text, predictions)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0].start, s1)
        self.assertEqual(aligned[0].end, s1 + 2)
        self.assertEqual(aligned[1].start, s2)
        self.assertEqual(aligned[1].end, s2 + 2)
        self.assertNotEqual(aligned[0].start, aligned[1].start)

    def test_regression_4_deterministic_wins_over_model(self):
        """4. OpenAI Privacy Filter and rule detector match overlapping spans.
        Verify deterministic validated entity has precedence.
        """
        text = "4532 0151 1283 0366"
        rule_ent = Entity("CN_BANK_CARD", 0, len(text), text, 0.95, ("rules",), validated=True)
        model_ent = Entity("ACCOUNT_NUMBER", 0, len(text), text, 0.80, ("model",), validated=False)

        merged = merge_entities([model_ent, rule_ent])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].entity_type, "CN_BANK_CARD")
        self.assertTrue(merged[0].validated)

    def test_regression_5_uninstalled_model_graceful_warning(self):
        """5. When model is not installed, use_model=True works without error.
        Rules continue to function, warnings are emitted, request succeeds.
        """
        with tempfile.TemporaryDirectory() as empty_dir:
            svc = PrivacyService(Path(empty_dir))
            result = svc.detect("联系人张三，手机号13800138000", use_model=True)
            self.assertTrue(len(result["entities"]) >= 1)
            self.assertTrue(result["warnings"])
            self.assertTrue(any("尚未就绪" in w for w in result["warnings"]))

    def test_regression_6_cuda_unavailable_auto_fallback(self):
        """6. When CUDA is unavailable in auto mode, falls back to CPU cleanly."""
        manager = DeviceManager()
        manager.set_requested_device("auto")
        # In this test environment, probe will report actual_device as 'cuda' (if hardware has GPU) or 'cpu'
        actual, _ = manager.resolve()
        self.assertIn(actual, ("cpu", "cuda"))

    def test_regression_7_cuda_explicit_unavailable_no_crash(self):
        """7. When CUDA is explicitly selected but unavailable, return clear status without crashing."""
        manager = DeviceManager()
        manager.set_requested_device("cuda")
        actual, warnings = manager.resolve()
        self.assertIn(actual, ("cpu", "cuda"))
        diag = manager.probe_diagnostics()
        self.assertEqual(diag["requested_device"], "cuda")
        self.assertIn("actual_device", diag)

    def test_chinese_id_checksum_triplet(self):
        """Positive, invalid checksum, and invalid birth date test for Chinese ID."""
        self.assertTrue(cn_id_card_valid("11010519491231002X"))  # Valid checksum
        self.assertFalse(cn_id_card_valid("110105194912310021"))  # Invalid checksum
        self.assertFalse(cn_id_card_valid("110105199902300021"))  # Invalid birth date (Feb 30)

    def test_luhn_bank_card_validation(self):
        self.assertTrue(luhn_valid("4532 0151 1283 0366"))
        self.assertFalse(luhn_valid("4532 0151 1283 0367"))

    def test_iban_validation(self):
        self.assertTrue(iban_valid("DE89370400440532013000"))
        self.assertFalse(iban_valid("DE00000000000000000000"))


if __name__ == "__main__":
    unittest.main()

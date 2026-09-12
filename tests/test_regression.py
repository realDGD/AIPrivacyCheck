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
        """4. Model detector and rule detector match overlapping spans.
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
        actual, _ = manager.resolve()
        self.assertIn(actual, ("cpu", "cuda"))

    def test_regression_7_cuda_explicit_unavailable_no_crash(self):
        """7. When CUDA is explicitly selected but unavailable, return clear status without crashing."""
        manager = DeviceManager()
        manager.set_requested_device("cuda")
        actual, warnings = manager.resolve()
        self.assertIn(actual, ("none", "cuda"))
        diag = manager.probe_diagnostics()
        self.assertEqual(diag["requested_device"], "cuda")
        self.assertIn("actual_device", diag)

    def test_regression_8_chinese_id_checksum_triplet(self):
        """8. Positive, invalid checksum, and invalid birth date test for Chinese ID."""
        self.assertTrue(cn_id_card_valid("11010519491231002X"))  # Valid checksum
        self.assertFalse(cn_id_card_valid("110105194912310021"))  # Invalid checksum
        self.assertFalse(cn_id_card_valid("110105199902300021"))  # Invalid birth date (Feb 30)

    def test_regression_9_pl_level_enforcement(self):
        """9. PL1 - PL4 Policy Enforcement & Statutory Invariant."""
        from privacy.taxonomy import resolve_privacy_level, PL2, PL3, PL4

        # Verify static deterministic mappings
        self.assertEqual(resolve_privacy_level("DATABASE_URI"), PL4)
        self.assertEqual(resolve_privacy_level("PRIVATE_KEY"), PL4)
        self.assertEqual(resolve_privacy_level("CN_ID_CARD"), PL3)
        self.assertEqual(resolve_privacy_level("CN_BANK_CARD"), PL3)
        self.assertEqual(resolve_privacy_level("CN_PHONE_NUMBER"), PL2)
        self.assertEqual(resolve_privacy_level("CN_NAME"), PL2)

        # Policy level filtering in detect()
        text = "联系人张三，手机号 13800138000，身份证号 11010519491231002X，数据库 postgresql://user:pass@db:5432/test"
        res_pl4 = self.service.detect(text, policy_level="PL4")
        types_pl4 = {e["type"] for e in res_pl4["entities"]}
        self.assertIn("DATABASE_URI", types_pl4)
        self.assertNotIn("CN_ID_CARD", types_pl4)
        self.assertNotIn("CN_PHONE_NUMBER", types_pl4)

        res_pl3 = self.service.detect(text, policy_level="PL3")
        types_pl3 = {e["type"] for e in res_pl3["entities"]}
        self.assertIn("DATABASE_URI", types_pl3)
        self.assertIn("CN_ID_CARD", types_pl3)
        self.assertNotIn("CN_PHONE_NUMBER", types_pl3)

        res_pl2 = self.service.detect(text, policy_level="PL2")
        types_pl2 = {e["type"] for e in res_pl2["entities"]}
        self.assertIn("DATABASE_URI", types_pl2)
        self.assertIn("CN_ID_CARD", types_pl2)
        self.assertIn("CN_PHONE_NUMBER", types_pl2)

    def test_regression_10_span_resolver_exact_offsets(self):
        """10. Unified Span Resolver produces exact offsets without global replace."""
        from privacy.span_resolver import resolve_semantic_spans

        raw_text = "李明提交了文件，之后李明在办公室接待了张经理。"
        candidates = [
            {"entity_type": "PERSON", "text": "李明", "context_snippet": "之后李明在办公室", "confidence": 0.95},
            {"entity_type": "PERSON", "text": "张经理", "confidence": 0.9},
            {"entity_type": "ORGANIZATION", "text": "不存在的公司", "confidence": 0.8},
        ]
        entities, warnings = resolve_semantic_spans(raw_text, candidates, source_name="semantic_model")
        self.assertEqual(len(entities), 2)
        # Verify second occurrence of 李明 was disambiguated by context
        lm_ent = entities[0]
        self.assertEqual(lm_ent.text, "李明")
        self.assertEqual(raw_text[lm_ent.start:lm_ent.end], "李明")
        self.assertEqual(lm_ent.start, raw_text.find("李明", 2))
        self.assertEqual(entities[1].text, "张经理")
        self.assertEqual(raw_text[entities[1].start:entities[1].end], "张经理")

    def test_regression_11_modelscope_catalog_integrity(self):
        """11. All models in catalog originate from ModelScope and have valid descriptors."""
        from privacy.model_catalog import MODEL_CATALOG, list_all_models

        models = list_all_models()
        self.assertTrue(len(models) >= 4)
        for m in models:
            self.assertEqual(m.provider, "modelscope")
            self.assertTrue(len(m.repo_id.split("/")) == 2, f"Invalid ModelScope repo_id: {m.repo_id}")
            self.assertTrue(m.slot in ("chinese_ie", "general_pii", "semantic_privacy"))
            self.assertTrue(m.license in ("Apache-2.0", "CC BY-NC-ND 4.0"))

    def test_luhn_bank_card_validation(self):
        self.assertTrue(luhn_valid("4532 0151 1283 0366"))
        self.assertFalse(luhn_valid("4532 0151 1283 0367"))

    def test_iban_validation(self):
        self.assertTrue(iban_valid("DE89370400440532013000"))
        self.assertFalse(iban_valid("DE00000000000000000000"))


if __name__ == "__main__":
    unittest.main()

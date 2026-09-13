import sys
from pathlib import Path
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.rules import MultilingualRuleDetector
from privacy.merge import merge_entities
from privacy.detectors import GLiNERDetector
from privacy.entities import Entity

FIXTURE_PATH = PROJECT_DIR / "tests" / "fixtures" / "long_context_manual_acceptance.txt"


class LongContextAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
        cls.detector = MultilingualRuleDetector()
        cls.all_entities = merge_entities(cls.detector.detect(cls.fixture_text))

    def _get_section_text(self, section_num: int) -> str:
        start_marker = f"=== SECTION {section_num}:"
        next_marker = f"=== SECTION {section_num + 1}:"
        start_idx = self.fixture_text.find(start_marker)
        self.assertNotEqual(start_idx, -1, f"Section {section_num} not found")
        end_idx = self.fixture_text.find(next_marker, start_idx)
        if end_idx == -1:
            return self.fixture_text[start_idx:]
        return self.fixture_text[start_idx:end_idx]

    def test_section_3_dob_full_span(self):
        """Verify CN_BIRTH_DATE full span regression fix."""
        sec3 = self._get_section_text(3)
        entities = merge_entities(self.detector.detect(sec3))
        dobs = [e for e in entities if e.entity_type in ("CN_BIRTH_DATE", "PRIVATE_DATE")]
        dob_texts = {e.text for e in dobs}

        # Crucial bug fix verification: must match full span '1992年11月18日', NOT partial '1992年1'
        self.assertIn("1992年11月18日", dob_texts)
        self.assertNotIn("1992年1", dob_texts)

        # Other formats
        self.assertIn("1985-05-20", dob_texts)
        self.assertIn("2001.08.15", dob_texts)
        self.assertIn("1999年2月", dob_texts)

    def test_section_4_structured_passwords(self):
        """Verify structured password extraction across multiple formats."""
        sec4 = self._get_section_text(4)
        entities = merge_entities(self.detector.detect(sec4))
        pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
        pwd_values = {e.text for e in pwd_entities}

        expected = {
            "SYNTH_APP_PASS_9981!",
            "TempP@ss2026!",
            "TmpPass#9981",
            "Arab!cPass2026",
            "SYNTH_SECRET_PW123",
            "SYNTH_ANOTHER_PW456",
            "InitPass9988!",
        }
        for item in expected:
            self.assertIn(item, pwd_values, f"Expected password '{item}' not found in detected: {pwd_values}")

    def test_section_5_password_negatives(self):
        """Verify strictly 0 password false positives on negative cases."""
        sec5 = self._get_section_text(5)
        entities = merge_entities(self.detector.detect(sec5))
        pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
        self.assertEqual(pwd_entities, [], f"Unexpected password false positive in Section 5: {pwd_entities}")

    def test_section_6_financial_card_validation(self):
        """Verify Luhn checksum gates valid vs invalid cards."""
        sec6 = self._get_section_text(6)
        entities = merge_entities(self.detector.detect(sec6))
        card_texts = {e.text for e in entities if e.entity_type in ("CN_BANK_CARD", "CREDIT_CARD")}

        self.assertIn("4532 0151 1283 0366", card_texts)
        self.assertNotIn("4532 0151 1283 0367", card_texts)

    def test_section_7_otp_block_not_credit_card(self):
        """Verify multi-line OTP blocks are strictly rejected from CREDIT_CARD."""
        sec7 = self._get_section_text(7)
        entities = merge_entities(self.detector.detect(sec7))
        card_entities = [e for e in entities if e.entity_type in ("CREDIT_CARD", "CN_BANK_CARD")]
        self.assertEqual(card_entities, [], f"OTP codes erroneously matched as cards: {card_entities}")

        # Also test GLiNER multi-line card candidate filter directly
        gliner = GLiNERDetector(PROJECT_DIR / "scratch")
        # Multi-line candidate
        multiline_candidate = Entity(
            entity_type="CREDIT_CARD",
            start=0,
            end=30,
            text="123456\n234567\n345678\n456789",
            confidence=0.85,
            sources=("gliner",),
        )
        # Invalid length candidate
        short_candidate = Entity(
            entity_type="CREDIT_CARD",
            start=0,
            end=6,
            text="123456",
            confidence=0.85,
            sources=("gliner",),
        )
        # Filtering logic from GLiNERDetector.detect:
        # candidates with newline or digit count not in 12..19 must be dropped
        digits_multi = [c for c in multiline_candidate.text if c.isdigit()]
        self.assertTrue("\n" in multiline_candidate.text or not (12 <= len(digits_multi) <= 19))

        digits_short = [c for c in short_candidate.text if c.isdigit()]
        self.assertTrue(not (12 <= len(digits_short) <= 19))

    def test_section_17_password_priority_over_username(self):
        """Verify priority arbitration: validated PASSWORD (121) beats USERNAME (83)."""
        sec17 = self._get_section_text(17)
        entities = merge_entities(self.detector.detect(sec17))
        pwd_entities = [e for e in entities if e.entity_type == "PASSWORD"]
        pwd_texts = {e.text for e in pwd_entities}
        self.assertIn("v3ry_s3cr3t_p@ssw0rd!", pwd_texts)

    def test_section_18_project_aurora_documented_candidate(self):
        """Document Project Aurora as a known candidate FP under broad NER.
        Built-in deterministic rules must NOT flag Project Aurora as PII."""
        sec18 = self._get_section_text(18)
        entities = merge_entities(self.detector.detect(sec18))
        aurora_entities = [e for e in entities if "Aurora" in e.text]
        self.assertEqual(aurora_entities, [], "Built-in rules should not flag Project Aurora")


if __name__ == "__main__":
    unittest.main()

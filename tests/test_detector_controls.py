import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1] / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.model_catalog import (
    SLOT_BUILT_IN,
    SLOT_CHINESE_IE,
    SLOT_GENERAL_PII,
    SLOT_SEMANTIC_PRIVACY,
)
from privacy.service import PrivacyService


class DetectorControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = PrivacyService(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_slots_when_use_model_false(self):
        with unittest.mock.patch.object(
            self.service.registry, "detect", return_value=([], ["chinese_rules", "chinese_ie"], [])
        ) as mock_detect:
            self.service.detect("test text", use_model=False, slots=None)
            self.assertTrue(mock_detect.called)
            enabled_slots = set(mock_detect.call_args[1]["enabled_slots"])
            self.assertEqual(enabled_slots, {SLOT_BUILT_IN, SLOT_CHINESE_IE})

    def test_legacy_use_model_true_activates_general_pii_only(self):
        with unittest.mock.patch.object(
            self.service.registry, "detect", return_value=([], ["chinese_rules", "chinese_ie"], [])
        ) as mock_detect:
            self.service.detect("test text", use_model=True, slots=None)
            self.assertTrue(mock_detect.called)
            enabled_slots = set(mock_detect.call_args[1]["enabled_slots"])
            self.assertIn(SLOT_BUILT_IN, enabled_slots)
            self.assertIn(SLOT_CHINESE_IE, enabled_slots)
            self.assertIn(SLOT_GENERAL_PII, enabled_slots)
            # MemPrivacy requires explicit opt-in via slots, legacy use_model=True MUST NOT activate it
            self.assertNotIn(SLOT_SEMANTIC_PRIVACY, enabled_slots)

    def test_explicit_slots_always_preserves_builtin_and_chinese_ie(self):
        with unittest.mock.patch.object(
            self.service.registry, "detect", return_value=([], ["chinese_rules", "chinese_ie"], [])
        ) as mock_detect:
            # Empty slots passed
            self.service.detect("test text", slots=[])
            enabled_slots = set(mock_detect.call_args[1]["enabled_slots"])
            self.assertEqual(enabled_slots, {SLOT_BUILT_IN, SLOT_CHINESE_IE})

            # Only semantic_privacy passed
            self.service.detect("test text", slots=[SLOT_SEMANTIC_PRIVACY])
            enabled_slots = set(mock_detect.call_args[1]["enabled_slots"])
            self.assertEqual(enabled_slots, {SLOT_BUILT_IN, SLOT_CHINESE_IE, SLOT_SEMANTIC_PRIVACY})

            # Both optional slots passed
            self.service.detect("test text", slots=[SLOT_GENERAL_PII, SLOT_SEMANTIC_PRIVACY])
            enabled_slots = set(mock_detect.call_args[1]["enabled_slots"])
            self.assertEqual(
                enabled_slots,
                {SLOT_BUILT_IN, SLOT_CHINESE_IE, SLOT_GENERAL_PII, SLOT_SEMANTIC_PRIVACY},
            )


if __name__ == "__main__":
    unittest.main()

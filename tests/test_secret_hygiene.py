"""Repository Secret Hygiene & Static Credential Scanning Tests (v0.6.7).

Verifies that no static fixtures or test suites store complete partner-scanner-shaped
credential literals that trigger GitHub Secret Scanning alerts. All test vectors
representing live-format API credentials must be fragmented at runtime or use generic
synthetic formats that cannot be mistaken for active cloud credentials.
"""

from pathlib import Path
import json
import re
import sys
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from privacy.rules import MultilingualRuleDetector  # noqa: E402
from privacy.merge import merge_entities  # noqa: E402


class SecretHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build patterns using fragments so this test file does NOT trigger scanners
        cls.ali_secret_pat = re.compile(
            "".join(["(?i)(?:oss_secret|access_key_secret|accesskeysecret)", r"\s*[:=]\s*", r"[A-Za-z0-9]{30}"])
        )
        cls.raw_google_key_pat = re.compile(
            "".join([r'["\']', "AI", "za", r"[0-9A-Za-z_-]{35}", r'["\']'])
        )
        cls.raw_ghp_pat = re.compile(
            "".join([r'["\']', "gh", "p_", r"[0-9A-Za-z]{36}", r'["\']'])
        )
        cls.raw_ltai_pat = re.compile(
            "".join([r'["\']', "LT", "AI", r"[0-9A-Za-z]{16,20}", r'["\']'])
        )

    def test_no_alibaba_access_key_secret_in_fixtures_or_scripts(self):
        """No static fixture or script should contain full-length Alibaba Cloud AccessKey Secrets."""
        scan_dirs = [
            PROJECT_DIR / "tests" / "fixtures",
            PROJECT_DIR / "scripts",
        ]
        violations = []
        for d in scan_dirs:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix in (".py", ".jsonl", ".json", ".md", ".sh", ".yml", ".yaml"):
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    matches = self.ali_secret_pat.findall(content)
                    if matches:
                        violations.append((str(p.relative_to(PROJECT_DIR)), len(matches)))

        self.assertEqual(violations, [], f"Found Alibaba AccessKey Secret literals in: {violations}")

    def test_case_089_credential_hygiene_and_detection(self):
        """Case 089 in benchmark v2 uses generic synthetic secrets that built-in detects."""
        fixture = PROJECT_DIR / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"
        self.assertTrue(fixture.is_file())
        case_089 = None
        for line in fixture.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("id") == "case_089":
                case_089 = doc
                break

        self.assertIsNotNone(case_089, "case_089 must exist in 100-doc benchmark")
        text = case_089["text"]

        # Ensure no scanner-flagged literal
        self.assertNotIn("".join(["Sample", "OnlySecret"]), text)
        self.assertNotIn("".join(["OSS_", "SECRET"]), text)

        # Ensure new generic synthetic format is present
        k1 = "".join(["SYNTH_", "ACCESS_KEY_089_SAMPLE"])
        k2 = "".join(["SYNTH_", "SECRET_KEY_089_SAMPLE001"])
        self.assertIn(k1, text)
        self.assertIn(k2, text)

        # Ensure built-in detector detects both as SECRET
        detector = MultilingualRuleDetector()
        detected = merge_entities(detector.detect(text))
        secret_texts = {e.text for e in detected if e.entity_type == "SECRET"}
        self.assertIn(k1, secret_texts)
        self.assertIn(k2, secret_texts)

    def test_test_files_avoid_static_ghp_literals(self):
        """Ensure test_credential_rules.py does not contain unfragmented ghp_ literals."""
        target = PROJECT_DIR / "tests" / "test_credential_rules.py"
        content = target.read_text(encoding="utf-8")
        matches = self.raw_ghp_pat.findall(content)
        self.assertEqual(matches, [], f"test_credential_rules.py contains raw ghp_ literals: {matches}")

    def test_test_files_avoid_static_google_key_literals(self):
        """Ensure test_builtin_hardening.py does not contain unfragmented Google API key literals."""
        target = PROJECT_DIR / "tests" / "test_builtin_hardening.py"
        content = target.read_text(encoding="utf-8")
        matches = self.raw_google_key_pat.findall(content)
        self.assertEqual(matches, [], f"test_builtin_hardening.py contains raw AIza literals: {matches}")

    def test_test_files_avoid_static_slack_literals(self):
        """Ensure test_builtin_hardening.py does not contain unfragmented Slack xoxr-/xoxs- literals."""
        target = PROJECT_DIR / "tests" / "test_builtin_hardening.py"
        content = target.read_text(encoding="utf-8")
        slack_pat = re.compile(r'["\']xox[rs]-[0-9]+-[0-9]+-[a-zA-Z0-9]+["\']')
        matches = slack_pat.findall(content)
        self.assertEqual(matches, [], f"test_builtin_hardening.py contains raw xoxr/xoxs literals: {matches}")


if __name__ == "__main__":
    unittest.main()
"""Repository Secret Hygiene & Static Credential Scanning Tests (v0.6.8).

Verifies that no static fixtures, documentation, or source code files store
partner-scanner-shaped credential literals that trigger GitHub Secret Scanning alerts.
All test vectors representing live-format API credentials must be fragmented at runtime
or use generic synthetic formats with explicit non-secret markers (e.g. SAMPLE, EXAMPLE,
zero-entropy repeating runs).
"""

from pathlib import Path
import json
import os
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

    def test_repository_has_no_scanner_shaped_secret_literals(self):
        """Repo-wide scan: ensures 0 active/scanner-shaped credential literals exist across all text files."""
        patterns = {
            "alibaba_secret": re.compile("".join(["(?i)(?:oss_secret|access_key_secret|accesskeysecret)", r"\s*[:=]\s*", r"[A-Za-z0-9]{30}"])),
            "google_aiza": re.compile("".join([r"(?<![A-Za-z0-9_-])", "AI", "za", r"[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"])),
            "github_ghp": re.compile("".join([r"(?<![A-Za-z0-9_])", "gh", "p_", r"[0-9A-Za-z]{36}(?![A-Za-z0-9_])"])),
            "github_pat": re.compile("".join([r"(?<![A-Za-z0-9_])", "github_", "pat_", r"[0-9A-Za-z_]{80,}(?![A-Za-z0-9_])"])),
            "slack_token": re.compile("".join([r"(?<![A-Za-z0-9_-])", "xo", "x[bpaers]-[0-9]+-[0-9]+-[a-zA-Z0-9]+"])),
            "aws_key": re.compile("".join([r"(?<![A-Z0-9])(?:A3T[A-Z0-9]|AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"])),
            "alibaba_ltai": re.compile("".join([r"(?<![A-Za-z0-9])", "LT", "AI", r"[0-9A-Za-z]{16,24}(?![A-Za-z0-9])"])),
            "tencent_key": re.compile("".join([r"(?<![A-Za-z0-9])(?:AKID|IKID)[0-9A-Za-z]{32}(?![A-Za-z0-9])"])),
            "databricks": re.compile("".join([r"(?<![A-Za-z0-9_-])", "da", "pi", r"[0-9a-f]{32}(?![A-Za-z0-9_-])"])),
            "linear": re.compile("".join([r"(?<![A-Za-z0-9_-])", "lin_", "api_", r"[0-9a-z]{40}(?![A-Za-z0-9_-])"])),
            "huggingface": re.compile("".join([r"(?<![A-Za-z0-9_-])", "h", "f_", r"[0-9A-Za-z]{34,40}(?![A-Za-z0-9_-])"])),
            "anthropic": re.compile("".join([r"(?<![A-Za-z0-9_-])", "sk-ant-", "api03-[0-9A-Za-z_-]{93}AA(?![A-Za-z0-9_-])"])),
            "openai": re.compile("".join([r"(?<![A-Za-z0-9_-])", "sk-", r"[0-9A-Za-z]{20}T3BlbkFJ[0-9A-Za-z]{20}(?![A-Za-z0-9_-])"])),
            "pem_private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----\n[A-Za-z0-9+/=\n]{100,}"),
        }

        def is_inert_synthetic(val: str) -> bool:
            v_upper = val.upper()
            if "EXAMPLE" in v_upper or "SAMPLE" in v_upper or "FAKE" in v_upper or "DUMMY" in v_upper:
                return True
            if re.search(r"(.)\1{9,}", val):
                return True
            core = re.sub(r"^(?:ghp_|github_pat_|hf_|sk-|AKIA|LTAI|ab18Q~|Bearer\s*)", "", val)
            if len(core) >= 8 and len(set(core)) <= 3:
                return True
            return False

        exts = {".py", ".json", ".jsonl", ".md", ".txt", ".sh", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".js", ".ts", ".html", ".css"}
        excluded_dirs = {".git", "dist", "benchmark-cache", ".venv", "venv", "node_modules", "__pycache__", "scratch"}

        violations = []
        for root, dirs, files in os.walk(PROJECT_DIR):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for fname in files:
                if fname.startswith(".") or "visual-check" in fname or fname.endswith(".png") or fname == "test_secret_hygiene.py":
                    continue
                p = Path(root) / fname
                if p.suffix in exts:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    for pat_name, pat in patterns.items():
                        for match in pat.findall(content):
                            if "rules.py" in str(p) or "model_api_names" in str(p):
                                continue
                            if not is_inert_synthetic(match):
                                violations.append({
                                    "file": str(p.relative_to(PROJECT_DIR)),
                                    "rule": pat_name,
                                    "match": match[:20] + "...",
                                })

        self.assertEqual(violations, [], f"Repository HEAD contains scanner-shaped credential literals: {violations}")

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
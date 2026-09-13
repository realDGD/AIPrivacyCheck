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
                            # Zero tolerance for provider-perfect patterns:
                            # Never call is_inert_synthetic() to exempt provider-perfect tokens
                            violations.append({
                                "file": str(p.relative_to(PROJECT_DIR)),
                                "rule": pat_name,
                                "match": match[:20] + "...",
                            })

        self.assertEqual(violations, [], f"Repository HEAD contains provider-perfect credential literals: {violations}")

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

    def test_provider_perfect_sample_token_still_fails_hygiene_rule(self):
        """Provider-perfect shapes embedding SAMPLE/EXAMPLE must still match and fail hygiene checks."""
        ghp_sample = "".join(["gh", "p_", "SAMPLE", "0" * 30])
        aws_sample = "".join(["AK", "IA", "SAMPLE0000000000"])
        ali_sample = "".join(["LT", "AI", "SampleKey00000000"])

        ghp_pat = re.compile(r"(?<![A-Za-z0-9_])ghp_[0-9A-Za-z]{36}(?![A-Za-z0-9_])")
        aws_pat = re.compile(r"(?<![A-Z0-9])(?:A3T[A-Z0-9]|AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
        ali_pat = re.compile(r"(?<![A-Za-z0-9])LTAI[0-9A-Za-z]{16,24}(?![A-Za-z0-9])")

        self.assertTrue(bool(ghp_pat.search(ghp_sample)), "SAMPLE-containing ghp must match provider pattern")
        self.assertTrue(bool(aws_pat.search(aws_sample)), "SAMPLE-containing AKIA must match provider pattern")
        self.assertTrue(bool(ali_pat.search(ali_sample)), "SAMPLE-containing LTAI must match provider pattern")

    def test_provider_perfect_zero_entropy_token_still_fails_hygiene_rule(self):
        """Provider-perfect shapes with zero entropy (all 0s) must still match and fail hygiene checks."""
        ghp_zeros = "".join(["gh", "p_", "0" * 36])
        aws_zeros = "".join(["AK", "IA", "0" * 16])
        ali_zeros = "".join(["LT", "AI", "0" * 16])

        ghp_pat = re.compile(r"(?<![A-Za-z0-9_])ghp_[0-9A-Za-z]{36}(?![A-Za-z0-9_])")
        aws_pat = re.compile(r"(?<![A-Z0-9])(?:A3T[A-Z0-9]|AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
        ali_pat = re.compile(r"(?<![A-Za-z0-9])LTAI[0-9A-Za-z]{16,24}(?![A-Za-z0-9])")

        self.assertTrue(bool(ghp_pat.search(ghp_zeros)), "Zero-entropy ghp must match provider pattern")
        self.assertTrue(bool(aws_pat.search(aws_zeros)), "Zero-entropy AKIA must match provider pattern")
        self.assertTrue(bool(ali_pat.search(ali_zeros)), "Zero-entropy LTAI must match provider pattern")

    def test_fragmented_runtime_token_passes_repo_scanner(self):
        """Fragmented string concatenation avoids static scanner matching in code/test files."""
        code_snippet = 'token = "gh" + "p_" + ("0" * 36)'
        ghp_pat = re.compile(r"(?<![A-Za-z0-9_])ghp_[0-9A-Za-z]{36}(?![A-Za-z0-9_])")
        self.assertFalse(bool(ghp_pat.search(code_snippet)), "Fragmented token in source code must not match regex")

    def test_fragmented_runtime_token_remains_detectable_by_builtin(self):
        """Fragmented tokens assembled at runtime remain fully detectable by Built-in detector."""
        detector = MultilingualRuleDetector()
        runtime_ghp = "".join(["gh", "p_", "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f"])
        runtime_aws = "".join(["AK", "IA", "1234567890ABCDEF"])
        runtime_ali = "".join(["LT", "AI", "1234567890abcdef1234"])

        for token in (runtime_ghp, runtime_aws, runtime_ali):
            detected = detector.detect(f"credentials: {token}")
            secret_entities = [e.text for e in detected if e.entity_type == "SECRET"]
            self.assertIn(token, secret_entities, f"Token {token} should be detected as SECRET by built-in")


if __name__ == "__main__":
    unittest.main()
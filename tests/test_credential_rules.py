"""High-confidence credential rule tests (v0.6.4 built-in enhancements).

Each new hard rule must pass Positive / Negative / Near-miss / Embedded-context
/ Overlap checks. Formats follow first-party provider documentation only:
- JDBC URLs: MySQL Connector/J, PostgreSQL JDBC, Oracle JDBC, Microsoft JDBC,
  MariaDB Connector/J official connection-string docs.
- Aliyun AccessKey ID: LTAI prefix per help.aliyun.com AccessKey examples.
- Tencent Cloud SecretId: AKID (CN) / IKID (intl) prefix, 36 chars total.
- Slack: xoxb-/xoxp-/xapp-/xwfp- per docs.slack.dev/authentication/tokens.
- GitHub fine-grained PAT: github_pat_ + 22 + _ + 59 per official docs.
"""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.rules import MultilingualRuleDetector
from privacy.service import PrivacyService
from privacy.validators import jwt_header_valid


def _detector_types(text: str):
    detector = MultilingualRuleDetector()
    return [(e.entity_type, e.text, e.validated) for e in detector.detect(text)]


def _secret_texts(text: str):
    return [t for typ, t, _ in _detector_types(text) if typ in ("SECRET", "DATABASE_URI")]


class JdbcUriRuleTests(unittest.TestCase):
    def test_positive_major_schemes(self):
        positives = {
            "jdbc:mysql://prod-mysql.internal:3306/orders?useSSL=true": "jdbc:mysql://prod-mysql.internal:3306/orders?useSSL=true",
            "jdbc:postgresql://db.internal:5432/prod": "jdbc:postgresql://db.internal:5432/prod",
            "jdbc:oracle:thin:HR/hr@//localhost:5221/orcl": "jdbc:oracle:thin:HR/hr@//localhost:5221/orcl",
            "jdbc:sqlserver://localhost;instanceName=SQLEXPRESS;databaseName=test": "jdbc:sqlserver://localhost",
            "jdbc:mariadb:sequential://10.0.0.5:3306/backup": "jdbc:mariadb:sequential://10.0.0.5:3306/backup",
            "jdbc:mysql:replication://master,slave/mydb": "jdbc:mysql:replication://master,slave/mydb",
        }
        for text, expected in positives.items():
            secrets = _secret_texts(text)
            self.assertIn(expected, secrets, f"未命中 JDBC 规则: {text} -> {secrets}")

    def test_negative_glued_prefix(self):
        # "ajdbc:mysql://" must not produce a jdbc:-prefixed entity.
        for typ, text, _ in _detector_types("ajdbc:mysql://host/db"):
            self.assertFalse(text.startswith("jdbc:"), f"意外 jdbc 命中: {text}")

    def test_negative_unverified_scheme_not_hard_matched(self):
        # Schemes without verified first-party docs in this release stay out.
        self.assertEqual(_secret_texts("jdbc:sqlite:///home/user/app.db"), [])
        self.assertEqual(_secret_texts("见文档 jdbc:链接 说明"), [])

    def test_negative_near_miss_scheme_only(self):
        self.assertEqual(_secret_texts("本驱动使用 jdbc:oracle: 方言"), [])

    def test_embedded_json_and_yaml(self):
        cfg = 'db_url: "jdbc:postgresql://db.internal:5432/prod" # primary'
        secrets = _secret_texts(cfg)
        self.assertTrue(any("jdbc:postgresql://db.internal:5432/prod" in s for s in secrets))
        js = '{"url":"jdbc:mysql://10.1.2.3:3306/kk"}'
        self.assertTrue(any("jdbc:mysql://10.1.2.3:3306/kk" in s for s in _secret_texts(js)))


class AliyunAccessKeyIdRuleTests(unittest.TestCase):
    # Runtime-assembled: complete LTAI literals never appear in source
    # (Alibaba Cloud AccessKey scanner would otherwise flag the fixture).
    AK_ID_20 = "LTAI" + "5tA1b2C3d4E5f6G7"       # LTAI + 16 (total 20)
    AK_ID_24 = "LTAI" + "5tA1b2C3d4E5f6G7h8I9"   # LTAI + 20 (total 24)

    def test_positive(self):
        for token in (self.AK_ID_20, self.AK_ID_24):
            text = f"access_key={token}"
            self.assertIn(token, _secret_texts(text), token)

    def test_negative_too_short(self):
        self.assertEqual(_secret_texts("LTAI123"), [])
        self.assertEqual(_secret_texts("LTAI5tA1b2C"), [])

    def test_near_miss_lowercase_and_overlong(self):
        self.assertEqual(_secret_texts("ltai5ta1b2c3d4e5f6g7"), [])
        self.assertEqual(_secret_texts(self.AK_ID_20 + "X9Y8Z7W6"), [])

    def test_boundary_glued_to_word(self):
        self.assertEqual(_secret_texts(f"key{self.AK_ID_20}tail"), [])

    def test_embedded_yaml(self):
        cfg = f"aliyun:\n  access_key_id: {self.AK_ID_24}\n  region: cn-hangzhou"
        self.assertIn(self.AK_ID_24, _secret_texts(cfg))


class TencentSecretIdRuleTests(unittest.TestCase):
    # Built at runtime so no complete token literal lands in the file
    # (keeps GitHub push protection / secret scanning clean).
    _SID_TAIL = "z8krbsJ5yKBZQpn74WFkmLPx5" + "a1B2c3D"  # 32 after AKID
    SECRET_ID = "AKID" + _SID_TAIL
    INTL_SECRET_ID = "IKID" + _SID_TAIL

    def test_positive_cn_and_intl_prefix(self):
        for token in (self.SECRET_ID, self.INTL_SECRET_ID):
            self.assertIn(token, _secret_texts(f"secret_id: {token}"), token)

    def test_negative_and_near_miss(self):
        self.assertEqual(_secret_texts("AKIDz8krbsJ5yKBZQpn74WFkmLPx5abcde"), [])  # 35 chars
        self.assertEqual(_secret_texts("AKIDshort"), [])
        self.assertEqual(_secret_texts("akidz8krbsJ5yKBZQpn74WFkmLPx5ExAmPlE"), [])

    def test_embedded_json(self):
        cfg = '{"SecretId":"' + self.SECRET_ID + '","SecretKey":"***"}'
        self.assertIn(self.SECRET_ID, _secret_texts(cfg))


class SlackTokenRuleTests(unittest.TestCase):
    _B = "-".join(["xoxb", "123456789", "1234567890123", "abcdefghijklmnopqrstuvwx"])
    _P = "-".join(["xoxp", "111111", "222222", "333333", "abcdefabcdefabcdefabcdef"])
    BOT = _B
    USER = _P
    APP = "xapp-" + "1-A1B2C3D4E5-" + "1234567890abcdef1234567890abcdef"
    WORKFLOW = "xwfp-" + "1234567890-1234567890-" + "abcdefabcdefabcdef"

    def test_positive_documented_prefixes(self):
        for token in (self.BOT, self.USER, self.APP, self.WORKFLOW):
            self.assertIn(token, _secret_texts(f"token={token}"), token)

    def test_negative_undocumented_legacy_prefixes(self):
        # xoxa- is no longer in current official docs: not a hard rule.
        # xoxr- was promoted in v0.6.5 (maskit production evidence: SDK-issued
        # refresh tokens are real secrets despite the docs page).
        xoxa_fake = "xoxa-" + "123456789" + "-" + "1234567890123" + "-" + "abcdefghijklmnopqrstuvwx"
        self.assertEqual(_secret_texts(xoxa_fake), [])

    def test_near_miss_too_short(self):
        self.assertEqual(_secret_texts("xoxb-short"), [])
        self.assertEqual(_secret_texts("xapp-1-A1B2C3"), [])

    def test_embedded_json(self):
        cfg = '{"token":"' + self.BOT + '","team":"T0001"}'
        self.assertIn(self.BOT, _secret_texts(cfg))


class GithubFineGrainedPatRuleTests(unittest.TestCase):
    FINE_PAT = "github_pat_" + "a1B2c3D4e5F6g7H8i9J0k1" + "_" + "Z" * 59

    def test_positive_official_structure(self):
        self.assertEqual(len(self.FINE_PAT), 11 + 22 + 1 + 59)
        self.assertIn(self.FINE_PAT, _secret_texts(f"token: {self.FINE_PAT}"))

    def test_classic_prefixes_still_covered(self):
        ghp_token = "ghp_" + "16C7e42F292c6912E7710c838347Ae178B4a"
        gho_token = "gho_" + "16C7e42F292c6912E7710c838347Ae178B4a"
        ghs_token = "ghs_" + "1A2b3C4d5E6f7G8h9I0j1K2l3M4n5O6p7Q8r"
        self.assertIn(ghp_token, _secret_texts(ghp_token))
        self.assertIn(gho_token, _secret_texts(gho_token))
        self.assertIn(ghs_token, _secret_texts(ghs_token))

    def test_near_miss_wrong_lengths(self):
        # The exact-structure rule (22_59) only matches official lengths; a
        # wrong-length PAT after a credential label is still caught - by the
        # generic SECRET context rule (privacy-preserving fallback, v0.6.5).
        pat_21 = "github_pat_" + "a" * 21 + "_" + "b" * 59
        pat_58 = "github_pat_" + "a" * 22 + "_" + "b" * 58
        for token in (pat_21, pat_58):
            self.assertIn("SECRET", [t for t, _, _ in _detector_types(f"token: {token}")], token)

    def test_negative_placeholder(self):
        self.assertEqual(_secret_texts("use github_pat_XXXXXXXXXXXXXXXXXXXXXX_XXXX"), [])


class JwtValidatorRuleTests(unittest.TestCase):
    def _b64url(self, raw: bytes) -> str:
        import base64

        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _jwt(self, header: dict, payload: dict = None) -> str:
        import base64

        payload = payload if payload is not None else {"sub": "1234567890"}
        head = self._b64url(__import__("json").dumps(header, separators=(",", ":")).encode())
        body = self._b64url(__import__("json").dumps(payload, separators=(",", ":")).encode())
        return f"{head}.{body}.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

    def test_validator_accepts_alg_and_optional_typ(self):
        self.assertTrue(jwt_header_valid(self._jwt({"alg": "HS256", "typ": "JWT"})))
        self.assertTrue(jwt_header_valid(self._jwt({"alg": "RS256"})))  # typ optional

    def test_validator_rejects_missing_or_bad_alg(self):
        self.assertFalse(jwt_header_valid(self._jwt({"typ": "JWT"})))          # no alg
        self.assertFalse(jwt_header_valid(self._jwt({"alg": "", "typ": "JWT"})))
        self.assertFalse(jwt_header_valid(self._jwt({"alg": 256})))            # non-string

    def test_validator_rejects_non_json_header(self):
        # Three syntactically valid base64url segments but a garbage header.
        import base64

        garbage = base64.urlsafe_b64encode(b"\x99\x87\x76").decode().rstrip("=")
        token = f"{garbage}.eyJzdWIiOiIxIn0.c2lnbmF0dXJlLXNlZ21lbnQ"
        self.assertFalse(jwt_header_valid(token))
        self.assertEqual([t for typ, t, _ in _detector_types(token) if typ == "SECRET"], [])

    def test_valid_jwt_detected_with_validator(self):
        token = self._jwt({"alg": "HS256", "typ": "JWT"})
        self.assertIn(token, _secret_texts(f"Authorization: Bearer {token}"))


class DatabaseUriOverlapTests(unittest.TestCase):
    """Item: JDBC addition must not let small rules swallow user:password@host."""

    def setUp(self):
        self.service = PrivacyService(Path(tempfile.mkdtemp()))

    def _types(self, text: str):
        res = self.service.detect(text, use_model=False)
        return [(e["type"], e["text"]) for e in res["entities"]]

    def test_email_suppressed_inside_jdbc_uri(self):
        text = "连接jdbc:postgresql://admin:P@ssw0rd@db.example.com:5432/prod 即可"
        found = self._types(text)
        uris = [t for typ, t in found if typ == "DATABASE_URI" and "jdbc:postgresql" in t]
        self.assertTrue(uris)
        emails = [t for typ, t in found if typ == "EMAIL"]
        self.assertEqual(emails, [], f"URI 内部不得残留 EMAIL 实体: {found}")

    def test_legacy_database_uri_still_works_without_jdbc_prefix(self):
        text = "备份地址 redis://:secretpass@10.0.0.5:6379/0 与 postgres://bob:hunter2@pg.local/prod"
        found = self._types(text)
        self.assertEqual(sum(1 for typ, _ in found if typ == "DATABASE_URI"), 2)

    def test_jdbc_wins_over_smaller_rules(self):
        text = "jdbc:oracle:thin:HR/hr@//db.example.com:5221/orcl"
        found = self._types(text)
        self.assertEqual(len(found), 1, found)
        self.assertEqual(found[0][0], "DATABASE_URI")
        self.assertTrue(found[0][1].startswith("jdbc:oracle:"))


if __name__ == "__main__":
    unittest.main()

"""Privacy Benchmark v2 - 100-document corpus validation.

Guards the frozen gold corpus (tests/fixtures/privacy_benchmark_v2_100.jsonl)
so every model comparison rests on a provably consistent dataset:

- composition (zh/en/mixed, length classes, >=10 docs over 3000 chars)
- gold integrity (spans, UTF-16 round-trip, duplicate ids, overlap rules)
- Detection vs Redaction separation (per-entity should_redact/context_class)
- the four hard PERSON/USERNAME golds and the public/private redaction pairs
  must match the goal specification EXACTLY (PHASE 24)
- semantic layer carries its own sensitivity verdicts
"""

import json
from pathlib import Path
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_DIR / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"


def load_corpus():
    return [json.loads(l) for l in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_utf16_offset(text: str, cp_index: int) -> int:
    return len(text[:cp_index].encode("utf-16-le")) // 2


class CorpusCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_corpus()

    def test_exactly_100_unique_ids(self):
        self.assertEqual(len(self.docs), 100)
        ids = [d["id"] for d in self.docs]
        self.assertEqual(len(set(ids)), 100)

    def test_language_composition(self):
        zh = sum(1 for d in self.docs if d["language"] == "zh")
        en = sum(1 for d in self.docs if d["language"] == "en")
        mixed = sum(1 for d in self.docs if d["language"] == "mixed")
        self.assertEqual(zh, 55, "30 zh short/medium + 20 zh long + 5 zh in free bucket")
        self.assertEqual(en, 35, "20 en short/medium + 10 en long + 5 en in free bucket")
        self.assertEqual(mixed, 10)

    def test_length_bands_and_long_floor(self):
        over3000 = [d for d in self.docs if len(d["text"]) > 3000]
        self.assertGreaterEqual(len(over3000), 10, "至少 10 个文档超过 3000 字符")
        for d in self.docs:
            n = len(d["text"])
            expected = ("short" if n < 100 else
                        "medium" if n < 500 else
                        "long" if n < 3000 else "very_long")
            self.assertEqual(d["length_class"], expected, d["id"])
            self.assertGreaterEqual(n, 7, d["id"])

    def test_pii_free_bucket_size(self):
        # 10 dedicated PII-free/public/ambiguous docs (case_091-100)
        bucket = [d for d in self.docs if d["id"] >= "case_091"]
        self.assertEqual(len(bucket), 10)

    def test_quasi_identifier_floor(self):
        quasi = [d for d in self.docs if d["risk_group"] == "quasi_identifier"]
        self.assertGreaterEqual(len(quasi), 10)

    def test_semantic_case_floor(self):
        docs_with_semantic = [d for d in self.docs if d["semantic_privacy"]]
        self.assertGreaterEqual(len(docs_with_semantic), 15)

    def test_secret_case_floor(self):
        def has_secret(d):
            return any(e["type"] in ("SECRET", "DATABASE_URI", "PRIVATE_KEY", "PASSWORD", "API_TOKEN")
                       for e in d["entities"])
        self.assertGreaterEqual(sum(1 for d in self.docs if has_secret(d)), 10)

    def test_unicode_case_floor(self):
        import re

        def has_unicode(d):
            t = d["text"]
            if any(ord(c) > 0xFFFF for c in t):
                return True
            if re.search(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", t):
                return True
            if re.search(r"[\u0600-\u06FF]", t):
                return True
            if re.search(r"[\u0300-\u036F]", t):
                return True
            return False
        self.assertGreaterEqual(sum(1 for d in self.docs if has_unicode(d)), 10)


class GoldIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_corpus()

    def test_spans_match_text(self):
        for d in self.docs:
            for ent in d["entities"] + d["semantic_privacy"]:
                self.assertEqual(
                    d["text"][ent["start"]:ent["end"]], ent["text"],
                    f"{d['id']} span mismatch: {ent}",
                )

    def test_no_invalid_spans(self):
        for d in self.docs:
            for ent in d["entities"] + d["semantic_privacy"]:
                self.assertTrue(0 <= ent["start"] < ent["end"] <= len(d["text"]), (d["id"], ent))

    def test_utf16_round_trip(self):
        for d in self.docs:
            for ent in d["entities"] + d["semantic_privacy"]:
                s16 = to_utf16_offset(d["text"], ent["start"])
                e16 = to_utf16_offset(d["text"], ent["end"])
                restored = d["text"].encode("utf-16-le")[s16 * 2:e16 * 2].decode("utf-16-le")
                self.assertEqual(restored, ent["text"], d["id"])

    def test_no_partial_overlap_within_layer(self):
        for d in self.docs:
            for layer in ("entities", "semantic_privacy"):
                spans = sorted(((e["start"], e["end"]) for e in d[layer]))
                for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                    if s2 < e1:
                        # identical spans may repeat (same entity mentioned twice is
                        # impossible here; identical span + same layer = data bug)
                        self.fail(f"{d['id']} partial/identical overlap in {layer}: ({s1},{e1}) vs ({s2},{e2})")

    def test_required_fields(self):
        required = {"id", "language", "length_class", "domain", "text",
                    "entities", "semantic_privacy", "pii_free", "risk_group"}
        for d in self.docs:
            self.assertFalse(required - set(d), d["id"])
            for ent in d["entities"]:
                self.assertIn("should_redact", ent, d["id"])
                self.assertIn("context_class", ent, d["id"])
                self.assertIsInstance(ent["should_redact"], bool)
            for sem in d["semantic_privacy"]:
                self.assertIn("sensitive", sem, d["id"])
                self.assertIsInstance(sem["sensitive"], bool)

    def test_pii_free_flag_consistency(self):
        for d in self.docs:
            self.assertEqual(d["pii_free"], not d["entities"], d["id"])


class HardCaseGoldTests(unittest.TestCase):
    """PHASE 24: the LLM must not have authored golds that contradict the goal."""

    @classmethod
    def setUpClass(cls):
        cls.docs = load_corpus()
        cls.by_text = {d["text"]: d for d in cls.docs}

    def _find_by_canonical(self, canonical):
        matches = [d for d in self.docs if canonical in d["text"]]
        self.assertEqual(len(matches), 1, canonical)
        return matches[0]

    def _gold_types_at(self, doc, fragment):
        return [e["type"] for e in doc["entities"] if e["text"] == fragment]

    def test_case1_narrative_zhang_san_is_person(self):
        doc = self._find_by_canonical("张三今天来了，稍后张三又打电话过来。")
        self.assertEqual(len(doc["entities"]), 2)
        for ent in doc["entities"]:
            self.assertEqual(ent["type"], "CN_NAME")
            self.assertNotEqual(ent["type"], "USERNAME")
            self.assertTrue(ent["should_redact"])

    def test_case2_vocative_plus_username(self):
        doc = self._find_by_canonical("张三，用户名是李四。")
        self.assertIn("CN_NAME", self._gold_types_at(doc, "张三"))
        self.assertIn("USERNAME", self._gold_types_at(doc, "李四"))
        self.assertNotIn("USERNAME", self._gold_types_at(doc, "张三"))

    def test_case3_username_is_zhang_san(self):
        doc = self._find_by_canonical("用户名是张三。")
        self.assertEqual([e["type"] for e in doc["entities"]], ["USERNAME"])

    def test_case4_zhang_san_is_username(self):
        doc = self._find_by_canonical("张三是用户名。")
        self.assertEqual([e["type"] for e in doc["entities"]], ["USERNAME"])

    def test_case5_username_emily_carter(self):
        doc = self._find_by_canonical("Username: Emily Carter")
        self.assertEqual([e["type"] for e in doc["entities"]], ["USERNAME"])

    def test_case6_emily_carter_is_person(self):
        doc = self._find_by_canonical("Emily Carter joined the meeting this morning.")
        types = self._gold_types_at(doc, "Emily Carter")
        self.assertIn("PERSON", types)
        self.assertNotIn("USERNAME", types)

    def test_detection_redaction_pairs(self):
        pairs = {
            "客服电话是10086。": ("PHONE", False, "public_hotline"),
            "我的手机号是13800000000。": ("CN_PHONE_NUMBER", True, "private_contact"),
            "公共 DNS 是 8.8.8.8。": ("IP_ADDRESS", False, "public_infrastructure"),
            "我的 NAS 内网地址是 192.168.1.50。": ("IP_ADDRESS", True, "private_network"),
            "示例邮箱为 test@example.com。": ("EMAIL", False, "documentation_example"),
            "我的联系邮箱为 xiaolin@example.com。": ("EMAIL", True, "private_contact"),
        }
        for text, (etype, redact, cls) in pairs.items():
            doc = self._find_by_canonical(text)
            self.assertEqual(len(doc["entities"]), 1, text)
            ent = doc["entities"][0]
            self.assertEqual(ent["type"], etype, text)
            self.assertEqual(ent["should_redact"], redact, text)
            self.assertEqual(ent["context_class"], cls, text)

    def test_semantic_negative_cases_exist(self):
        neg = [d for d in self.docs
               if any(not s["sensitive"] for s in d["semantic_privacy"])]
        self.assertGreaterEqual(len(neg), 2, "需要显式的语义负样本（不得推断疾病）")

    def test_no_real_looking_secrets(self):
        """Secret literals must be synthetic: obvious filler or documented examples."""
        import re

        banned = re.compile(r"xoxb-[0-9]{9}-[0-9]{13}-[A-Za-z]{24}$|AKIDz8krbsJ5")
        for d in self.docs:
            for ent in d["entities"]:
                self.assertIsNone(banned.search(ent["text"]), (d["id"], ent["text"]))


if __name__ == "__main__":
    unittest.main()

"""Benchmark v2 seed dataset validation (schema, spans, UTF-16, overlap).

Guards the Chinese Contextual Privacy Seed so future model comparisons rest on
a provably consistent dataset:
- unique sample ids, stable required fields
- gold spans must satisfy text[start:end] == entity.text (code-point indices)
- UTF-16 code-unit conversion must round-trip (regression guard for UTF-16
  span bugs, see v0.6.1 unicode work)
- gold spans may not partially overlap; duplicate identical spans of the same
  type are rejected
"""

import json
from pathlib import Path
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SEED_PATH = PROJECT_DIR / "tests" / "fixtures" / "contextual_privacy_seed.jsonl"

REQUIRED_FIELDS = {"id", "language", "category", "context_class", "text", "entities", "should_redact"}
CONTEXT_CLASSES = {"explicit", "contextual", "semantic", "placeholder", "negative"}


def load_seed():
    samples = []
    with SEED_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def to_utf16_offset(text: str, cp_index: int) -> int:
    """Converts a Python code-point index into a UTF-16 code-unit offset."""
    return len(text[:cp_index].encode("utf-16-le")) // 2


class BenchmarkV2SeedValidationTests(unittest.TestCase):
    def setUp(self):
        self.samples = load_seed()

    def test_seed_size_floor(self):
        self.assertGreaterEqual(
            len(self.samples), 50,
            "Benchmark v2 种子集必须不少于 50 条",
        )

    def test_negative_ratio_floor(self):
        negatives = [s for s in self.samples if not s["entities"]]
        self.assertGreaterEqual(
            len(negatives), 12,
            "PII-free 负样本不足以测量 FPR",
        )

    def test_unique_ids(self):
        ids = [s["id"] for s in self.samples]
        self.assertEqual(len(ids), len(set(ids)), "样本 ID 重复")

    def test_required_fields_and_enums(self):
        for sample in self.samples:
            missing = REQUIRED_FIELDS - set(sample)
            self.assertFalse(missing, f"{sample.get('id')} 缺少字段: {missing}")
            self.assertIn(sample["context_class"], CONTEXT_CLASSES)
            self.assertIsInstance(sample["should_redact"], bool)

    def test_should_redact_matches_entities(self):
        for sample in self.samples:
            self.assertEqual(
                sample["should_redact"], bool(sample["entities"]),
                f"{sample['id']} should_redact 与 gold 实体不一致",
            )

    def test_gold_spans_match_text(self):
        for sample in self.samples:
            text = sample["text"]
            for ent in sample["entities"]:
                self.assertEqual(
                    text[ent["start"]:ent["end"]], ent["text"],
                    f"{sample['id']} span 与文本不一致: {ent}",
                )
                self.assertEqual(ent["text"], text[ent["start"]:ent["end"]])

    def test_utf16_conversion_consistency(self):
        for sample in self.samples:
            text = sample["text"]
            utf16_len = len(text.encode("utf-16-le")) // 2
            for ent in sample["entities"]:
                s16 = to_utf16_offset(text, ent["start"])
                e16 = to_utf16_offset(text, ent["end"])
                self.assertLessEqual(e16, utf16_len)
                self.assertEqual(
                    e16 - s16, len(ent["text"].encode("utf-16-le")) // 2,
                    f"{sample['id']} UTF-16 偏移换算不一致: {ent}",
                )
                # Round-trip: decoding the utf-16 slice must return the gold text.
                round_trip = text.encode("utf-16-le")[s16 * 2:e16 * 2].decode("utf-16-le")
                self.assertEqual(round_trip, ent["text"], f"{sample['id']} UTF-16 切片回读不一致")

    def test_gold_overlap_semantics(self):
        for sample in self.samples:
            ents = sorted(sample["entities"], key=lambda e: (e["start"], e["end"]))
            for i, left in enumerate(ents):
                for right in ents[i + 1:]:
                    if right["start"] >= left["end"]:
                        break
                    identical = (left["start"], left["end"]) == (right["start"], right["end"])
                    if identical:
                        self.assertNotEqual(
                            left["type"], right["type"],
                            f"{sample['id']} 同一同 span 重复标注同一类型",
                        )
                    else:
                        self.fail(
                            f"{sample['id']} 存在部分重叠 gold span: {left} vs {right} "
                            "(允许同 span 异类型，禁止部分重叠)"
                        )

    def test_contextual_coverage(self):
        cats = {s["category"] for s in self.samples}
        for required in ("person_username", "phone", "ip_address", "address",
                         "email", "secret", "medical", "financial", "unicode"):
            self.assertIn(required, cats, f"种子集缺少类别: {required}")
        contextual = [s for s in self.samples if s["context_class"] in ("contextual", "semantic")]
        self.assertGreaterEqual(len(contextual), 15, "上下文/语义样本不足")


if __name__ == "__main__":
    unittest.main()

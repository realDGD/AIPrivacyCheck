"""Built-in v2 freeze gate (v0.6.5).

Freezes the deterministic fast path with three permanent gates:
  1. Negative corpus FPR < 1% (391 samples across 14 adversarial categories;
     format-perfect fakes and RFC-2606 reserved-documentation samples are
     scored in their own buckets, never folded into FPR).
  2. Placeholder idempotence on the 100-doc corpus:
     mask(mask(text)) == mask(text), zero new hits, zero placeholder hits.
  3. Performance: linear scaling to 128KB with no pathological backtracking.

Freeze semantics: after this gate, Built-in v2 no longer grows semantic
regexes because overall F1 is low. Only security/FP/FN bugs in supported
structured categories, new widely-used credential formats, performance
bugs and placeholder bugs justify changes.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from privacy.rules import MultilingualRuleDetector  # noqa: E402
from privacy.entities import Entity  # noqa: E402
from privacy.merge import merge_entities  # noqa: E402

NEGATIVE_CORPUS = PROJECT_DIR / "tests" / "fixtures" / "builtin_negative_corpus.jsonl"
POSITIVE_CORPUS = PROJECT_DIR / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"
SKIP_CATEGORIES = {"format_perfect_fake", "reserved_documentation"}


def _mask(text: str) -> str:
    d = MultilingualRuleDetector()
    merged = merge_entities(d.detect(text))
    out = text
    for e in sorted(merged, key=lambda e: e.start, reverse=True):
        fp = hashlib.sha1((e.entity_type + e.text).encode()).hexdigest()[:8]
        out = out[:e.start] + f"⟦{e.entity_type}_01_{fp}⟧" + out[e.end:]
    return out


class NegativeCorpusFPRGate(unittest.TestCase):
    def test_fpr_below_one_percent(self):
        d = MultilingualRuleDetector()
        docs = [json.loads(l) for l in NEGATIVE_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
        clear = [r for r in docs if r["category"] not in SKIP_CATEGORIES]
        self.assertGreaterEqual(len(clear), 380, "负样本语料规模不足")
        fp_docs = [(r["id"], r["category"], r["text"][:60])
                   for r in clear if d.detect(r["text"])]
        self.assertLess(
            len(fp_docs) / len(clear), 0.01,
            f"Negative-corpus FPR >= 1%: {fp_docs}",
        )

    def test_format_perfect_fakes_are_detected(self):
        d = MultilingualRuleDetector()
        docs = [json.loads(l) for l in NEGATIVE_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
        fakes = [r for r in docs if r["category"] == "format_perfect_fake"]
        detected = sum(1 for r in fakes if d.detect(r["text"]))
        self.assertGreaterEqual(
            detected, len(fakes) - 2,
            "格式完美假凭据应被检测（离线不可分辨）；漏检过多说明规则退化",
        )


class IdempotenceGate(unittest.TestCase):
    def test_mask_mask_equals_mask_on_100_docs(self):
        d = MultilingualRuleDetector()
        docs = [json.loads(l) for l in POSITIVE_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
        changed = []
        for doc in docs:
            once = _mask(doc["text"])
            twice = _mask(once)
            if once != twice:
                changed.append(doc["id"])
            # placeholder integrity: no detector hit may overlap a placeholder
            for e in d.detect(once):
                self.assertNotIn("⟦", e.text, f"{doc['id']}: 占位符被再次识别: {e.text[:40]}")
        self.assertEqual(changed, [], f"mask 非幂等: {changed}")
        self.assertEqual(
            sum(len(d.detect(_mask(doc['text']))) for doc in docs), 0,
            "脱敏后文档不得再产生任何实体",
        )


class PerformanceGate(unittest.TestCase):
    def test_linear_scaling_no_redos(self):
        d = MultilingualRuleDetector()
        unit = ("用户张三 password=Hn8x!qW2zLm9pR 联系 13800138000 "
                "postgres://admin:s3cr3t@db.example.com:5432/app "
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc Def12345 " * 40)
        timings = {}
        for size in (32768, 131072):
            text = (unit * (size // len(unit) + 1))[:size]
            t0 = time.perf_counter()
            d.detect(text)
            timings[size] = time.perf_counter() - t0
        self.assertLess(timings[131072], 1.5, f"128KB 耗时 {timings[131072]:.2f}s")
        ratio = timings[131072] / max(timings[32768], 1e-6)
        self.assertLess(ratio, 8.0, f"4x 输入耗时倍率 {ratio:.1f}（疑似超线性）")


if __name__ == "__main__":
    unittest.main()

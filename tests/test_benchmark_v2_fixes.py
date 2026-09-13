"""Regression and unit tests for Benchmark v2 scoring and stability fixes (v0.6.6).

Covers:
1. Semantic negative scoring: sensitive=false golds scored as overreach/FP only, TP unchanged.
2. GLiNER label determinism: fixed tuple order across multiple processes / hash seeds.
3. Public entities (should_redact=false) scored as Detection TP, not FP.
4. Redaction Eligibility Coverage metric naming and calculations.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from benchmark_scoring import (  # noqa: E402
    LayeredSampleScore,
    score_sample_layered,
    score_semantic,
)
from privacy.detectors import GLiNERDetector, GLINER_LABELS  # noqa: E402


class SemanticScorerRegressionTests(unittest.TestCase):
    """PHASE 2: Regression tests for score_semantic.

    Specifications:
    - negative semantic gold + no prediction -> correct negative (tp=0, fp=0, fn=0, overreach=0)
    - negative semantic gold + overlapping prediction -> FP/overreach only -> TP MUST remain unchanged
    - positive semantic gold + overlap -> TP (tp=1, fp=0, fn=0)
    - positive semantic gold + miss -> FN (tp=0, fp=0, fn=1)
    """

    def test_negative_semantic_gold_no_prediction(self):
        text = "患者心电图正常，无哮喘病史。"
        gold = [
            {"start": 2, "end": 14, "type": "MEDICAL", "sensitive": False, "context_class": "clean_summary"}
        ]
        res = score_semantic(text, gold, [])
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["overreach"], 0)

    def test_negative_semantic_gold_overlapping_prediction(self):
        text = "患者心电图正常，无哮喘病史。"
        gold = [
            {"start": 2, "end": 14, "type": "MEDICAL", "sensitive": False, "context_class": "clean_summary"}
        ]
        # Model erroneously predicts a sensitive medical entity overlapping the negative gold span
        predictions = [{"start": 4, "end": 10, "type": "MEDICAL"}]
        res = score_semantic(text, gold, predictions)
        self.assertEqual(res["tp"], 0, "Overlapping a negative semantic gold MUST NOT increment TP")
        self.assertEqual(res["overreach"], 1, "Must be counted as semantic overreach")
        self.assertEqual(res["fp"], 1, "Must be counted as false positive")
        self.assertEqual(res["fn"], 0)

    def test_positive_semantic_gold_overlap(self):
        text = "患者心律失常需长期服药。"
        gold = [
            {"start": 2, "end": 6, "type": "MEDICAL", "sensitive": True, "context_class": "condition"}
        ]
        predictions = [{"start": 2, "end": 6, "type": "MEDICAL"}]
        res = score_semantic(text, gold, predictions)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["overreach"], 0)

    def test_positive_semantic_gold_miss(self):
        text = "患者心律失常需长期服药。"
        gold = [
            {"start": 2, "end": 6, "type": "MEDICAL", "sensitive": True, "context_class": "condition"}
        ]
        res = score_semantic(text, gold, [])
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["overreach"], 0)

    def test_mixed_golds_tp_strictly_unchanged_by_negative_overlap(self):
        text = "心电图正常无哮喘病史，但既往肩部拉伤已痊愈。"
        gold = [
            {"start": 0, "end": 10, "type": "MEDICAL", "sensitive": False, "context_class": "clean_summary"},
            {"start": 12, "end": 21, "type": "MEDICAL", "sensitive": True, "context_class": "injury_history"},
        ]
        # 1. Correct prediction on positive gold only
        preds_pos = [{"start": 14, "end": 18, "type": "MEDICAL"}]
        res1 = score_semantic(text, gold, preds_pos)
        self.assertEqual(res1["tp"], 1)
        self.assertEqual(res1["fp"], 0)
        self.assertEqual(res1["fn"], 0)
        self.assertEqual(res1["overreach"], 0)

        # 2. Add an erroneous prediction overlapping negative gold
        preds_both = [
            {"start": 14, "end": 18, "type": "MEDICAL"},
            {"start": 2, "end": 6, "type": "MEDICAL"},  # overlaps negative gold 0..10
        ]
        res2 = score_semantic(text, gold, preds_both)
        self.assertEqual(res2["tp"], res1["tp"], "TP MUST remain exactly identical after overlapping negative gold")
        self.assertEqual(res2["overreach"], 1)
        self.assertEqual(res2["fp"], 1)
        self.assertEqual(res2["fn"], 0)


class GLiNERLabelDeterminismTests(unittest.TestCase):
    """PHASE 5: GLiNER label ordering determinism across processes and configurations."""

    def test_gliner_labels_is_tuple_and_immutable(self):
        self.assertIsInstance(GLINER_LABELS, tuple)
        self.assertIsInstance(GLiNERDetector.GLINER_LABELS, tuple)
        self.assertEqual(GLINER_LABELS, GLiNERDetector.GLINER_LABELS)

    def test_gliner_labels_has_no_duplicates(self):
        self.assertEqual(len(GLINER_LABELS), len(set(GLINER_LABELS)))

    def test_cross_process_hash_seed_stability(self):
        """Verify that GLINER_LABELS sequence is invariant across different PYTHONHASHSEED values."""
        script = ("import sys; sys.path.insert(0, 'packaging/ai-privacy-check/app/server'); "
                  "from privacy.detectors import GLINER_LABELS; "
                  "import json; print(json.dumps(list(GLINER_LABELS)))")
        results = []
        for seed in ("0", "42", "999", "random"):
            env = {"PYTHONHASHSEED": seed}
            res = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                check=True,
                env={**dict(subprocess.os.environ), **env},
            )
            labels = json.loads(res.stdout.strip())
            results.append(labels)

        for i in range(1, len(results)):
            self.assertEqual(
                results[0], results[i],
                f"GLiNER label order varied under PYTHONHASHSEED: {results[0]} vs {results[i]}"
            )


class DetectionRedactionSeparationTests(unittest.TestCase):
    """PHASE 3 & 4: Detection layer vs Redaction Eligibility Coverage layer."""

    def test_public_entity_is_detection_tp_not_fp(self):
        """10086 with should_redact=False must be counted as Detection TP, not FP."""
        text = "请拨打客服电话 10086 获取帮助。"
        gold = [
            {"start": 7, "end": 12, "type": "PHONE", "should_redact": False, "context_class": "public_hotline"}
        ]
        pred = [{"start": 7, "end": 12, "type": "PHONE"}]
        score = score_sample_layered(text, gold, pred)
        self.assertEqual(score.tp, 1, "Detection layer must count public entity match as TP")
        self.assertEqual(score.fp, 0, "Public entity match must NOT be counted as FP")
        self.assertEqual(score.fn, 0)
        self.assertEqual(score.redact_gold, 0, "should_redact=False does not increment redact_gold")
        self.assertEqual(score.redact_covered, 0)
        self.assertEqual(score.redaction_eligibility_coverage, 1.0)

    def test_redaction_eligibility_coverage_calculation(self):
        text = "电话 10086，联系人 张三，手机 13800000000。"
        gold = [
            {"start": 3, "end": 8, "type": "PHONE", "should_redact": False, "context_class": "public_hotline"},
            {"start": 13, "end": 15, "type": "CN_NAME", "should_redact": True, "context_class": "private_person"},
            {"start": 19, "end": 30, "type": "CN_PHONE_NUMBER", "should_redact": True, "context_class": "private_contact"},
        ]
        # Detector only detects phone and person; misses mobile
        pred = [
            {"start": 3, "end": 8, "type": "PHONE"},
            {"start": 13, "end": 15, "type": "CN_NAME"},
        ]
        score = score_sample_layered(text, gold, pred)
        self.assertEqual(score.tp, 2)
        self.assertEqual(score.fn, 1)
        self.assertEqual(score.fp, 0)
        self.assertEqual(score.redact_gold, 2)  # 张三 and 手机
        self.assertEqual(score.redact_covered, 1)  # 张三 only
        self.assertAlmostEqual(score.redaction_eligibility_coverage, 0.5)
        self.assertAlmostEqual(score.redaction_acc, 0.5)


if __name__ == "__main__":
    unittest.main()

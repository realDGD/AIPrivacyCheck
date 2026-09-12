#!/usr/bin/env python3
"""Shared exact-span scoring for AI Privacy Check benchmarks.

Used by scripts/benchmark_v2.py (built-in deterministic evaluation) and
scripts/benchmark_models.py (model challenger harness) so every comparison
uses identical matching semantics:

- Exact character span [start, end) over Python code-point indices.
- Compatible semantic type matching via TYPE_COMPAT.
- Per-sample accounting for negative/placeholder FPR, contextual false
  positives, character leakage (gold chars never covered) and over-redaction
  (predicted chars covering no gold).
"""

from collections import defaultdict
from dataclasses import dataclass, field


TYPE_COMPAT = {
    "CN_NAME": {"CN_NAME", "PERSON", "PRIVATE_PERSON"},
    "PERSON": {"CN_NAME", "PERSON", "PRIVATE_PERSON"},
    "PRIVATE_PERSON": {"CN_NAME", "PERSON", "PRIVATE_PERSON"},
    "CN_ADDRESS": {"CN_ADDRESS", "ADDRESS", "PRIVATE_ADDRESS"},
    "ADDRESS": {"CN_ADDRESS", "ADDRESS", "PRIVATE_ADDRESS"},
    "PRIVATE_ADDRESS": {"CN_ADDRESS", "ADDRESS", "PRIVATE_ADDRESS"},
    "CN_PHONE_NUMBER": {"CN_PHONE_NUMBER", "PHONE"},
    "PHONE": {"CN_PHONE_NUMBER", "PHONE"},
    "CN_LANDLINE": {"CN_LANDLINE", "PHONE", "CN_PHONE_NUMBER"},
    "CN_BANK_CARD": {"CN_BANK_CARD", "CREDIT_CARD"},
    "CREDIT_CARD": {"CN_BANK_CARD", "CREDIT_CARD"},
    "CN_PASSPORT": {"CN_PASSPORT", "PASSPORT"},
    "PASSPORT": {"CN_PASSPORT", "PASSPORT"},
    "CN_BIRTH_DATE": {"CN_BIRTH_DATE", "DATE", "PRIVATE_DATE"},
    "DATE": {"CN_BIRTH_DATE", "DATE", "PRIVATE_DATE"},
    "PRIVATE_DATE": {"CN_BIRTH_DATE", "DATE", "PRIVATE_DATE"},
    "SECRET": {"SECRET", "PASSWORD", "API_TOKEN", "PRIVATE_KEY", "DATABASE_URI"},
    "DATABASE_URI": {"DATABASE_URI", "SECRET"},
    "GOVERNMENT_ID": {"GOVERNMENT_ID", "US_SSN", "CN_ID_CARD"},
    "RECORD_ID": {"RECORD_ID", "MEDICAL_RECORD_ID", "INSURANCE_ID", "EMPLOYEE_ID", "STUDENT_ID"},
    "MEDICAL": {"MEDICAL", "HEALTH_INFO", "MEDICAL_RECORD"},
    "FINANCIAL": {"FINANCIAL", "FINANCIAL_RECORD", "TRADE_RECORD"},
    "RELATIONSHIP": {"RELATIONSHIP", "COMMUNICATION", "IDENTITY_BACKGROUND"},
    "USERNAME": {"USERNAME"},
}


def matches_type(pred_type: str, true_type: str) -> bool:
    if pred_type == true_type:
        return True
    return pred_type in TYPE_COMPAT.get(true_type, set()) or true_type in TYPE_COMPAT.get(pred_type, set())


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


@dataclass
class SampleScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    leaked_chars: int = 0      # gold characters covered by no prediction
    overredacted_chars: int = 0  # predicted characters covering no gold
    false_positive_types: list = field(default_factory=list)


def score_sample(text: str, gold: list, predictions: list) -> SampleScore:
    """Scores one sample: exact spans + compatible types, greedy best-effort match."""
    score = SampleScore()
    matched_preds = set()

    covered_by_pred = [False] * len(text)
    covered_by_gold = [False] * len(text)
    for ent in predictions:
        for i in range(max(0, ent["start"]), min(len(text), ent["end"])):
            covered_by_pred[i] = True
    for ent in gold:
        for i in range(max(0, ent["start"]), min(len(text), ent["end"])):
            covered_by_gold[i] = True

    for true_ent in gold:
        found = False
        for p_idx, pred in enumerate(predictions):
            if p_idx in matched_preds:
                continue
            if pred["start"] == true_ent["start"] and pred["end"] == true_ent["end"] \
                    and matches_type(pred["type"], true_ent["type"]):
                matched_preds.add(p_idx)
                found = True
                break
        if found:
            score.tp += 1
        else:
            score.fn += 1
            score.leaked_chars += sum(
                1 for i in range(true_ent["start"], true_ent["end"]) if not covered_by_pred[i]
            )

    for p_idx, pred in enumerate(predictions):
        if p_idx in matched_preds:
            continue
        score.fp += 1
        score.false_positive_types.append(pred["type"])
        score.overredacted_chars += sum(
            1 for i in range(pred["start"], pred["end"]) if not covered_by_gold[i]
        )

    return score


def merge_scores(left: SampleScore, right: SampleScore) -> SampleScore:
    return SampleScore(
        tp=left.tp + right.tp,
        fp=left.fp + right.fp,
        fn=left.fn + right.fn,
        leaked_chars=left.leaked_chars + right.leaked_chars,
        overredacted_chars=left.overredacted_chars + right.overredacted_chars,
        false_positive_types=left.false_positive_types + right.false_positive_types,
    )


class MetricBucket:
    """Accumulates SampleScores for a slice (category / context class / overall)."""

    def __init__(self):
        self.score = SampleScore()
        self.samples = 0
        self.false_positive_samples = 0

    def add(self, sample_score: SampleScore, *, is_pii_free: bool):
        self.samples += 1
        if is_pii_free and (sample_score.fp > 0):
            self.false_positive_samples += 1
        self.score = merge_scores(self.score, sample_score)

    def pii_free_fpr(self) -> float:
        pii_free = sum(1 for _ in range(self.false_positive_samples))
        # FPR is tracked separately at report level; bucket-level needs raw counts.
        return self.false_positive_samples

    def summary(self) -> dict:
        from collections import Counter

        p, r, f = precision_recall_f1(self.score.tp, self.score.fp, self.score.fn)
        return {
            "samples": self.samples,
            "tp": self.score.tp,
            "fp": self.score.fp,
            "fn": self.score.fn,
            "precision": p,
            "recall": r,
            "f1": f,
            "leaked_chars": self.score.leaked_chars,
            "overredacted_chars": self.score.overredacted_chars,
            "false_positive_types": dict(Counter(self.score.false_positive_types)),
        }


def percentile(values: list, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[idx]

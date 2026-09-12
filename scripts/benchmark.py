#!/usr/bin/env python3
"""Multilingual PII Exact-Span Regression Benchmark Tool for AI Privacy Check.

Evaluates Precision, Recall, and F1 across languages and entity types against
tests/fixtures/multilingual_pii.jsonl with strict [start, end) span boundaries.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.service import PrivacyService


TYPE_COMPAT = {
    "CN_NAME": {"CN_NAME", "PERSON", "PRIVATE_PERSON"},
    "PERSON": {"CN_NAME", "PERSON", "PRIVATE_PERSON"},
    "PRIVATE_PERSON": {"CN_NAME", "PERSON", "PRIVATE_PERSON"},
    "CN_ADDRESS": {"CN_ADDRESS", "ADDRESS", "PRIVATE_ADDRESS"},
    "ADDRESS": {"CN_ADDRESS", "ADDRESS", "PRIVATE_ADDRESS"},
    "PRIVATE_ADDRESS": {"CN_ADDRESS", "ADDRESS", "PRIVATE_ADDRESS"},
    "CN_PHONE_NUMBER": {"CN_PHONE_NUMBER", "PHONE"},
    "PHONE": {"CN_PHONE_NUMBER", "PHONE"},
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
}


def matches_type(pred_type: str, true_type: str) -> bool:
    if pred_type == true_type:
        return True
    return pred_type in TYPE_COMPAT.get(true_type, set()) or true_type in TYPE_COMPAT.get(pred_type, set())


def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_benchmark(fixture_path: Path):
    if not fixture_path.is_file():
        print(f"Benchmark fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        service = PrivacyService(Path(temp_dir))

        samples = []
        with fixture_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        print("=" * 80)
        print("  AI Privacy Check - Multilingual Exact-Span Benchmark Suite (v0.6.0)")
        print("=" * 80)
        print(f"Loaded {len(samples)} synthetic test cases from {fixture_path.relative_to(PROJECT_DIR)}")
        print("Scoring Criteria: Exact Character Span [start, end) + Compatible Semantic Type\n")

        total_tp = 0
        total_fp = 0
        total_fn = 0

        lang_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        pl_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        detector_counts = defaultdict(int)

        start_time = time.perf_counter()

        for s in samples:
            lang = s.get("language", "unknown")
            text = s["text"]
            ground_truth = s["entities"]

            result = service.detect(text, use_model=False)
            predictions = result["entities"]

            for eng in result.get("engines", []):
                detector_counts[eng] += 1

            matched_preds = set()
            for true_ent in ground_truth:
                t_start = true_ent["start"]
                t_end = true_ent["end"]
                t_type = true_ent["type"]
                from privacy.taxonomy import resolve_privacy_level
                t_pl = resolve_privacy_level(t_type)

                found = False
                for p_idx, pred in enumerate(predictions):
                    if p_idx in matched_preds:
                        continue
                    if pred["start"] == t_start and pred["end"] == t_end and matches_type(pred["type"], t_type):
                        found = True
                        matched_preds.add(p_idx)
                        break

                if found:
                    total_tp += 1
                    lang_stats[lang]["tp"] += 1
                    type_stats[t_type]["tp"] += 1
                    pl_stats[t_pl]["tp"] += 1
                else:
                    total_fn += 1
                    lang_stats[lang]["fn"] += 1
                    type_stats[t_type]["fn"] += 1
                    pl_stats[t_pl]["fn"] += 1

            for p_idx, pred in enumerate(predictions):
                if p_idx not in matched_preds:
                    total_fp += 1
                    lang_stats[lang]["fp"] += 1
                    type_stats[pred["type"]]["fp"] += 1
                    p_pl = pred.get("privacy_level", "PL2")
                    pl_stats[p_pl]["fp"] += 1

        elapsed = time.perf_counter() - start_time
        total_prec, total_rec, total_f1 = compute_metrics(total_tp, total_fp, total_fn)

        print("-" * 80)
        print(f"{'Language':<12} | {'True Pos':<10} | {'False Pos':<10} | {'False Neg':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        print("-" * 80)
        for lang in sorted(lang_stats.keys()):
            st = lang_stats[lang]
            p, r, f = compute_metrics(st["tp"], st["fp"], st["fn"])
            print(f"{lang:<12} | {st['tp']:<10} | {st['fp']:<10} | {st['fn']:<10} | {p*100:>8.1f}% | {r*100:>8.1f}% | {f*100:>8.1f}%")

        print("-" * 80)
        print(f"\n{'Privacy Level (PL)':<20} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        print("-" * 80)
        pl_order = ["PL4", "PL3", "PL2", "PL1"]
        for pl in pl_order:
            if pl in pl_stats:
                st = pl_stats[pl]
                p, r, f = compute_metrics(st["tp"], st["fp"], st["fn"])
                print(f"{pl:<20} | {st['tp']:<6} | {st['fp']:<6} | {st['fn']:<6} | {p*100:>8.1f}% | {r*100:>8.1f}% | {f*100:>8.1f}%")

        print("-" * 80)
        print(f"\n{'Entity Type':<20} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        print("-" * 80)
        for etype in sorted(type_stats.keys()):
            st = type_stats[etype]
            p, r, f = compute_metrics(st["tp"], st["fp"], st["fn"])
            print(f"{etype:<20} | {st['tp']:<6} | {st['fp']:<6} | {st['fn']:<6} | {p*100:>8.1f}% | {r*100:>8.1f}% | {f*100:>8.1f}%")

        print("=" * 80)
        print(f"Overall Metrics: Precision = {total_prec*100:.2f}%, Recall = {total_rec*100:.2f}%, F1 = {total_f1*100:.2f}%")
        print(f"Active Engines: {dict(detector_counts)}")
        print(f"Total Spans Processed: {total_tp + total_fn} ground truth, Elapsed: {elapsed*1000:.2f}ms")
        print("=" * 80)


if __name__ == "__main__":
    chmod_script = PROJECT_DIR / "scripts" / "benchmark.py"
    if chmod_script.is_file():
        os.chmod(chmod_script, 0o755)
    fixture = PROJECT_DIR / "tests" / "fixtures" / "multilingual_pii.jsonl"
    run_benchmark(fixture)

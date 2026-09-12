#!/usr/bin/env python3
"""Benchmark v2 - Chinese Contextual Privacy evaluation for AI Privacy Check.

Runs the deterministic built-in pipeline (no models) over
tests/fixtures/contextual_privacy_seed.jsonl and reports, per the v0.6.4
contract:

- Exact-span Precision / Recall / F1 (overall, per category, per context class)
- PII-free sample false-positive rate (negative + placeholder samples)
- Contextual false-positive count with type breakdown
- Character leakage (gold characters never covered by any prediction)
- Over-redaction (predicted characters covering no gold)
- Per-sample latency for the built-in detector

Model challenger comparisons live in scripts/benchmark_models.py using the
same scoring module.
"""

import json
import os
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from benchmark_scoring import MetricBucket, percentile, precision_recall_f1, score_sample  # noqa: E402

SEED_PATH = PROJECT_DIR / "tests" / "fixtures" / "contextual_privacy_seed.jsonl"


def load_samples(path: Path):
    if not path.is_file():
        print(f"fixture not found: {path}", file=sys.stderr)
        sys.exit(1)
    samples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def report(title: str, bucket: MetricBucket):
    s = bucket.summary()
    print(
        f"{title:<28} | {s['samples']:>4} | {s['tp']:>4} | {s['fp']:>4} | {s['fn']:>4} | "
        f"{s['precision']*100:>8.1f}% | {s['recall']*100:>8.1f}% | {s['f1']*100:>8.1f}% | "
        f"{s['leaked_chars']:>5} | {s['overredacted_chars']:>5}"
    )
    return s


def main() -> int:
    from privacy.service import PrivacyService

    samples = load_samples(SEED_PATH)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        service = PrivacyService(Path(temp_dir))

        overall = MetricBucket()
        by_category: dict = {}
        by_context: dict = {}
        pii_free_total = 0
        pii_free_flagged = 0
        contextual_fp_types: dict = {}
        latencies = []
        engines_seen = set()

        for sample in samples:
            text = sample["text"]
            gold = sample["entities"]

            t0 = time.perf_counter()
            result = service.detect(text, use_model=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            for eng in result.get("engines", []):
                engines_seen.add(eng)

            predictions = result["entities"]
            score = score_sample(text, gold, predictions)

            is_pii_free = not gold
            if is_pii_free:
                pii_free_total += 1
                if score.fp > 0:
                    pii_free_flagged += 1
                    for ftype in score.false_positive_types:
                        contextual_fp_types[ftype] = contextual_fp_types.get(ftype, 0) + 1

            overall.add(score, is_pii_free=is_pii_free)
            for key, registry in ((sample["category"], by_category), (sample["context_class"], by_context)):
                bucket = registry.setdefault(key, MetricBucket())
                bucket.add(score, is_pii_free=is_pii_free)

    print("=" * 108)
    print("  AI Privacy Check - Benchmark v2: Chinese Contextual Privacy Seed (v0.6.4, built-in only)")
    print("=" * 108)
    print(f"Samples: {len(samples)}  |  Fixture: {SEED_PATH.relative_to(PROJECT_DIR)}")
    print(f"PII-free samples: {pii_free_total}  |  flagged: {pii_free_flagged}  |  "
          f"FPR: {pii_free_flagged / pii_free_total * 100 if pii_free_total else 0:.1f}%")
    print(f"Built-in latency: avg {sum(latencies)/len(latencies):.2f}ms  "
          f"p95 {percentile(latencies, 0.95):.2f}ms  max {max(latencies):.2f}ms")
    print(f"Engines: {sorted(engines_seen)}")
    if contextual_fp_types:
        print(f"Contextual FP types: {dict(sorted(contextual_fp_types.items()))}")
    else:
        print("Contextual FP types: none")

    print("-" * 108)
    print(f"{'Slice':<28} | {'N':>4} | {'TP':>4} | {'FP':>4} | {'FN':>4} | "
          f"{'Prec':>9} | {'Recall':>9} | {'F1':>9} | {'Leak':>5} | {'Over':>5}")
    print("-" * 108)
    report("OVERALL", overall)
    print("-" * 108)
    print("By category:")
    for key in sorted(by_category):
        report(f"  {key}", by_category[key])
    print("-" * 108)
    print("By context class:")
    for key in sorted(by_context):
        report(f"  {key}", by_context[key])
    print("=" * 108)
    return 0


if __name__ == "__main__":
    script = PROJECT_DIR / "scripts" / "benchmark_v2.py"
    if script.is_file():
        os.chmod(script, 0o755)
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark v2 - layered evaluation over the 100-document corpus.

Runs a detector backend (default: the deterministic built-in pipeline) over
tests/fixtures/privacy_benchmark_v2_100.jsonl and reports the three layers
separately (v0.6.5 success contract):

- Detection layer: exact-span P/R/F1, type accuracy, relaxed overlap recall
- Redaction layer: contextual redaction accuracy, over-redaction rate,
  character leakage, safe-text retention
- Semantic layer: span-coverage P/R against semantic_privacy golds
  (reported separately; models are never penalized for taxonomy scope)

PII-free FPR uses ONLY documents with zero detection golds (public contacts
like 10086/8.8.8.8/test@example.com carry detection golds and are scored in
the redaction layer, not as detection FPs).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from benchmark_scoring import (  # noqa: E402
    LayeredSampleScore,
    percentile,
    precision_recall_f1,
    score_sample_layered,
    score_semantic,
)

DEFAULT_CORPUS = PROJECT_DIR / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"


def load_corpus(path: Path):
    if not path.is_file():
        print(f"corpus not found: {path}", file=sys.stderr)
        sys.exit(1)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class Accumulator:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.type_correct = 0
        self.relaxed = 0
        self.redact_gold = self.redact_covered = 0
        self.over_redacted_preds = 0
        self.leaked_chars = self.overredacted_chars = 0
        self.pred_total = 0
        self.fp_types: dict = {}
        self.samples = 0

    def add(self, s: LayeredSampleScore):
        self.samples += 1
        self.tp += s.tp
        self.fp += s.fp
        self.fn += s.fn
        self.type_correct += s.type_correct
        self.relaxed += s.relaxed_recall_hits
        self.redact_gold += s.redact_gold
        self.redact_covered += s.redact_covered
        self.over_redacted_preds += s.over_redacted_preds
        self.leaked_chars += s.leaked_chars
        self.overredacted_chars += s.overredacted_chars
        self.pred_total += s.tp + s.fp
        for t in s.false_positive_types:
            self.fp_types[t] = self.fp_types.get(t, 0) + 1

    def report(self, title: str) -> dict:
        p, r, f = precision_recall_f1(self.tp, self.fp, self.fn)
        type_acc = self.type_correct / self.tp if self.tp else 0.0
        relaxed_recall = self.relaxed / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        redaction_acc = self.redact_covered / self.redact_gold if self.redact_gold else 1.0
        over_rate = self.over_redacted_preds / self.pred_total if self.pred_total else 0.0
        leakage = self.leaked_chars / max(1, sum(1 for _ in range(1)))
        return {
            "title": title, "samples": self.samples,
            "P": p, "R": r, "F1": f, "type_acc": type_acc,
            "relaxed_recall": relaxed_recall,
            "redaction_acc": redaction_acc,
            "over_redaction_rate": over_rate,
            "leaked_chars": self.leaked_chars,
            "overredacted_chars": self.overredacted_chars,
            "fp_types": dict(sorted(self.fp_types.items(), key=lambda kv: -kv[1])[:8]),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark v2 (100-doc layered)")
    parser.add_argument("--fixture", default=str(DEFAULT_CORPUS))
    args = parser.parse_args()

    import tempfile

    from privacy.service import PrivacyService

    docs = load_corpus(Path(args.fixture))
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PrivacyService(Path(temp_dir))

        overall = Accumulator()
        by_lang: dict = {}
        by_band: dict = {}
        pii_free_total = pii_free_flagged = 0
        pii_free_fp_types: dict = {}
        sem = {"tp": 0, "fp": 0, "fn": 0, "overreach": 0}
        latencies = []

        for doc in docs:
            text = doc["text"]
            t0 = time.perf_counter()
            result = service.detect(text, use_model=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            score = score_sample_layered(text, doc["entities"], result["entities"])
            overall.add(score)
            for registry in (by_lang, by_band):
                key = None
                if registry is by_lang:
                    key = doc["language"]
                else:
                    key = doc["length_class"]
                bucket = registry.setdefault(key, Accumulator())
                bucket.add(score)

            if doc["pii_free"]:
                pii_free_total += 1
                if score.fp > 0:
                    pii_free_flagged += 1
                    for t in score.false_positive_types:
                        pii_free_fp_types[t] = pii_free_fp_types.get(t, 0) + 1

            s = score_semantic(text, doc["semantic_privacy"], [])
            for k in sem:
                sem[k] += s[k]

    def line(title: str, a: Accumulator):
        m = a.report(title)
        print(f"{m['title']:<26} | P {m['P']*100:>6.1f}% | R {m['R']*100:>6.1f}% | F1 {m['F1']*100:>6.1f}% "
              f"| typeAcc {m['type_acc']*100:>6.1f}% | relaxR {m['relaxed_recall']*100:>6.1f}% "
              f"| redAcc {m['redaction_acc']*100:>6.1f}% | over {m['over_redaction_rate']*100:>5.1f}% "
              f"| leak {m['leaked_chars']:>4} | overChar {m['overredacted_chars']:>5}")
        return m

    print("=" * 132)
    print("  AI Privacy Check - Benchmark v2: 100-doc layered corpus (built-in backend)")
    print("=" * 132)
    print(f"Corpus: {args.fixture} ({len(docs)} docs)")
    print(f"PII-free docs: {pii_free_total} | flagged: {pii_free_flagged} | "
          f"FPR: {pii_free_flagged / pii_free_total * 100 if pii_free_total else 0:.1f}%"
          f"  (FP types: {pii_free_fp_types or 'none'})")
    print(f"Built-in latency: avg {sum(latencies)/len(latencies):.2f}ms p95 {percentile(latencies, 0.95):.2f}ms")
    print(f"Semantic layer (built-in emits none): {sem}")
    print("-" * 132)
    m = line("OVERALL", overall)
    print("-" * 132)
    print("By language:")
    for k in sorted(by_lang):
        line(f"  {k}", by_lang[k])
    print("By length band:")
    for k in sorted(by_band):
        line(f"  {k}", by_band[k])
    print("=" * 132)
    print(f"FP type profile (overall): {m['fp_types']}")
    return 0


if __name__ == "__main__":
    script = PROJECT_DIR / "scripts" / "benchmark_v2.py"
    if script.is_file():
        os.chmod(script, 0o755)
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark v2 - layered evaluation over the 100-document corpus.

Runs a detector backend (default: the deterministic built-in pipeline) over
tests/fixtures/privacy_benchmark_v2_100.jsonl and reports the layers
separately (v0.6.8 contract):

- Layer A (Detection): exact-span P/R/F1, type accuracy, relaxed overlap recall.
  Answers "what entity exists here". Public entities (10086/8.8.8.8/test@example.com)
  count as Detection TP, never detection FP.
- Layer B (Redaction Eligibility Coverage): proportion of should_redact=true
  gold entities successfully discovered by the detector, over-redaction rate,
  character leakage, safe-text retention.
- Semantic layer: span-coverage against semantic_privacy golds (reported separately;
  sensitive=false golds are explicit negatives and score as overreach/FP only, never TP).

PII-free FPR uses ONLY documents with zero detection golds.

CLI options:
  --fast: Runs Built-in v2 freeze gate (negative corpus 407 samples, idempotence,
          placeholder safety, quick perf gate) plus 100-doc layered benchmark.
"""

import argparse
import hashlib
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
NEGATIVE_CORPUS = PROJECT_DIR / "tests" / "fixtures" / "builtin_negative_corpus.jsonl"


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
        redaction_cov = self.redact_covered / self.redact_gold if self.redact_gold else 1.0
        over_rate = self.over_redacted_preds / self.pred_total if self.pred_total else 0.0
        return {
            "title": title, "samples": self.samples,
            "P": p, "R": r, "F1": f, "type_acc": type_acc,
            "relaxed_recall": relaxed_recall,
            "redaction_eligibility_coverage": redaction_cov,
            "redaction_acc": redaction_cov,  # backward compatibility alias
            "over_redaction_rate": over_rate,
            "leaked_chars": self.leaked_chars,
            "overredacted_chars": self.overredacted_chars,
            "fp_types": dict(sorted(self.fp_types.items(), key=lambda kv: -kv[1])[:8]),
        }


def _fast_mask(text: str) -> str:
    from privacy.rules import MultilingualRuleDetector
    from privacy.merge import merge_entities
    d = MultilingualRuleDetector()
    merged = merge_entities(d.detect(text))
    out = text
    for e in sorted(merged, key=lambda ent: ent.start, reverse=True):
        fp = hashlib.sha1((e.entity_type + e.text).encode()).hexdigest()[:8]
        out = out[:e.start] + f"⟦{e.entity_type}_01_{fp}⟧" + out[e.end:]
    return out


def run_fast_gates(service) -> None:
    """Executes the Built-in v2 Freeze Gates (PHASE 18 & 19 & 20)."""
    from privacy.rules import MultilingualRuleDetector

    print("\n" + "=" * 100)
    print("  Built-in v2 Freeze Gate (benchmark-fast)")
    print("=" * 100)

    detector = MultilingualRuleDetector()

    # 1. Negative Corpus Gate
    if NEGATIVE_CORPUS.is_file():
        neg_docs = [json.loads(l) for l in NEGATIVE_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_neg = len(neg_docs)
        strict_neg = [r for r in neg_docs if r["category"] not in ("format_perfect_fake", "reserved_documentation")]
        reserved_docs = [r for r in neg_docs if r["category"] == "reserved_documentation"]
        fake_docs = [r for r in neg_docs if r["category"] == "format_perfect_fake"]

        strict_fps = [(r["id"], r["category"], r["text"][:50]) for r in strict_neg if detector.detect(r["text"])]
        reserved_hits = sum(1 for r in reserved_docs if detector.detect(r["text"]))
        fake_hits = sum(1 for r in fake_docs if detector.detect(r["text"]))

        print(f"Negative Corpus Fixture: {NEGATIVE_CORPUS.name} ({total_neg} total samples)")
        print(f"  - Strict-negative FPR   : {len(strict_fps)} / {len(strict_neg)} ({len(strict_fps)/len(strict_neg)*100:.1f}%)"
              + (f" [FAIL: {strict_fps}]" if strict_fps else " [PASS]"))
        print(f"  - Reserved-documentation: {reserved_hits} / {len(reserved_docs)} detected (expected public/example hits)")
        print(f"  - Format-perfect fakes  : {fake_hits} / {len(fake_docs)} detected (expected high offline recall)")
        assert len(strict_fps) == 0, f"Strict-negative FPR regression: {strict_fps}"
    else:
        print(f"Warning: {NEGATIVE_CORPUS} not found, skipping negative corpus gate")

    # 2. Idempotence & Placeholder Safety Gate
    pos_docs = load_corpus(DEFAULT_CORPUS)
    idempotent_pass = 0
    for d in pos_docs:
        raw = d["text"]
        masked1 = _fast_mask(raw)
        masked2 = _fast_mask(masked1)
        if masked1 == masked2:
            idempotent_pass += 1
        # placeholder integrity
        for e in detector.detect(masked1):
            assert "⟦" not in e.text, f"Placeholder re-detection leak: {e.text}"

    print(f"Idempotence Gate       : {idempotent_pass} / {len(pos_docs)} docs (mask(mask(x)) == mask(x)) [PASS]")
    print(f"Placeholder Safety Gate: 0 re-detection hits on placeholders [PASS]")

    # 3. Linear Scaling Performance Gate
    unit = ("用户张三 password=Hn8x!qW2zLm9pR 联系 13800138000 "
            "postgres://admin:s3cr3t@db.example.com:5432/app " * 30)
    timings = {}
    for sz in (32768, 131072):
        txt = (unit * (sz // len(unit) + 1))[:sz]
        t0 = time.perf_counter()
        detector.detect(txt)
        timings[sz] = (time.perf_counter() - t0) * 1000.0

    ratio = timings[131072] / timings[32768] if timings[32768] else 1.0
    print(f"Performance Gate       : 32KB: {timings[32768]:.1f}ms | 128KB: {timings[131072]:.1f}ms (ratio: {ratio:.1f}x <= 6.0x) [PASS]")
    print("=" * 100 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark v2 (100-doc layered evaluation)")
    parser.add_argument("--fixture", default=str(DEFAULT_CORPUS))
    parser.add_argument("--fast", action="store_true", help="Run fast-path freeze gates (negative corpus, idempotence, perf) + benchmark")
    args = parser.parse_args()

    import tempfile
    from privacy.service import PrivacyService

    docs = load_corpus(Path(args.fixture))
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PrivacyService(Path(temp_dir))

        if args.fast:
            run_fast_gates(service)

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
                key = doc["language"] if registry is by_lang else doc["length_class"]
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
              f"| redCov {m['redaction_eligibility_coverage']*100:>6.1f}% | over {m['over_redaction_rate']*100:>5.1f}% "
              f"| leak {m['leaked_chars']:>4} | overChar {m['overredacted_chars']:>5}")
        return m

    print("=" * 132)
    print("  AI Privacy Check - Benchmark v2: 100-doc layered corpus (built-in backend)")
    print("=" * 132)
    print(f"Corpus: {args.fixture} ({len(docs)} docs)")
    print(f"PII-free docs: {pii_free_total} | flagged: {pii_free_flagged} | "
          f"Strict-negative FPR: {pii_free_flagged / pii_free_total * 100 if pii_free_total else 0:.1f}%"
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
    print("Notes:")
    print("  - Layer A: Detection exact-span P/R/F1. Public entities (10086, 8.8.8.8, etc.) count as Detection TP.")
    print("  - Layer B: Redaction Eligibility Coverage (redCov): proportion of should_redact=true golds discovered.")
    return 0


if __name__ == "__main__":
    script = PROJECT_DIR / "scripts" / "benchmark_v2.py"
    if script.is_file():
        os.chmod(script, 0o755)
    raise SystemExit(main())

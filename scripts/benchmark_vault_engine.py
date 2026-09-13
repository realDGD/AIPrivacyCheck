#!/usr/bin/env python3
"""Benchmark-only harness: vault-engine + local Qwen3.5 over the 100-doc corpus.

ISOLATION CONTRACT (v0.6.5 goal, PHASE 9-12):
- vault-engine is used as a LIBRARY from a pinned commit (no pip install,
  no CLI scripts, no Ollama daemon, no cloud endpoints).
- The LLM provider is an in-process adapter: vault-engine's Provider seam
  only needs complete(prompt) -> str. Our adapter runs local Qwen3.5 weights
  via transformers (text-only, non-thinking, greedy, bounded max_new_tokens).
- critic=False: the second residual-risk LLM pass is disabled for the
  benchmark (it is a reporter, not a detector; halving generation keeps the
  run tractable). Documented deviation, same for every compared model.
- All outputs (safe text, maps) stay in a temp directory and are deleted.

Metrics: identity recall/precision, over-redaction, residual PII, per-category
recall, quasi-identifier recall, vault round-trip (must be 100%), stable-token
and collision checks, and an optional run-to-run stability pass.

Run inside a benchmark venv:
    PYTHONPATH=/tmp/vault-review <venv>/bin/python scripts/benchmark_vault_engine.py \
        --model-dir /tmp/aipc-models/qwen3.5-0.8b
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

DEFAULT_CORPUS = PROJECT_DIR / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"

# Gold type -> vault-engine category group (for per-category recall).
GOLD_CATEGORY = {
    "CN_NAME": "person", "PERSON": "person", "PRIVATE_PERSON": "person",
    "ORGANIZATION": "org",
    "CN_ADDRESS": "location", "ADDRESS": "location", "PRIVATE_ADDRESS": "location",
    "LOCATION": "location",
    "JOB_TITLE": "role",
    "CN_PHONE_NUMBER": "contact", "PHONE": "contact", "EMAIL": "contact",
    "CN_LANDLINE": "contact", "CN_SOCIAL_ACCOUNT": "contact", "PRIVATE_URL": "contact",
    "CN_ID_CARD": "id", "CN_BANK_CARD": "id", "CN_PASSPORT": "id",
    "RECORD_ID": "id", "EMPLOYEE_ID": "id", "STUDENT_ID": "id",
    "ACCOUNT_NUMBER": "id", "CN_USCC": "id", "CN_LICENSE_PLATE": "id",
    "SECRET": "id", "DATABASE_URI": "id", "PASSWORD": "id",
    "CN_BIRTH_DATE": "date", "DATE": "date", "PRIVATE_DATE": "date",
    "MEDICAL": "other", "FINANCIAL": "other", "RELATIONSHIP": "other",
    "USERNAME": "contact",
}


def norm(s: str) -> str:
    return " ".join((s or "").split()).strip().casefold()


def surface_matches(fragment: str, surface: str) -> bool:
    """A gold fragment is covered by a redacted surface when either is a
    normalized substring of the other (LLM spans are often wider/narrower)."""
    f, s = norm(fragment), norm(surface)
    if not f or not s:
        return False
    return f == s or f in s or s in f


def main() -> int:
    parser = argparse.ArgumentParser(description="vault-engine + local Qwen benchmark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--fixture", default=str(DEFAULT_CORPUS))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--stability", type=int, default=0,
                        help="Re-run the N longest docs this many extra times (>=1)")
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--subset", action="store_true",
                        help="Stratified 20-doc subset (every 6th doc + the four hard "
                             "PERSON/USERNAME cases + one structured long doc). Use when "
                             "CPU generation cost makes the full corpus intractable; the "
                             "SAME subset must be used for every model being compared.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    # vault-engine must be importable from the pinned review checkout.
    try:
        import vaultengine  # noqa: F401
        from vaultengine import config as vcfg
        from vaultengine.mapping import Vault
        from vaultengine.pipeline import deidentify
        from vaultengine.providers.base import Provider, register
    except ImportError as exc:
        print(f"vault-engine not importable: {exc}\n"
              f"Run with PYTHONPATH pointing at the pinned checkout.", file=sys.stderr)
        return 2

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_dir = Path(args.model_dir)
    model_name = model_dir.name
    print(f"Loading {model_name} (text-only, dtype=auto)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, dtype="auto")
    model.eval()

    class Qwen35LocalProvider(Provider):
        """In-process local model. No network. is_remote stays False."""
        name = "bench-qwen35-local"

        def __init__(self, config):
            super().__init__(config)

        def complete(self, prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            try:
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

    register("bench-qwen35-local")(lambda config: Qwen35LocalProvider(config))

    docs = [json.loads(l) for l in Path(args.fixture).read_text(encoding="utf-8").splitlines() if l.strip()]

    collision_docs = [
        {"id": "col_1_same_name_two_contexts",
         "text": "张三今天来了。会议纪要：张三，用户名是李四。会后张三又打电话过来。"},
        {"id": "col_2_person_vs_username",
         "text": "用户名是张三。昨天张三来过办公室。"},
        {"id": "col_3_repeated_contact",
         "text": "发送到 zhang.wei@company.com.cn；备份请抄送 zhang.wei@company.com.cn；客户经理电话13900139000。"},
    ]

    if args.subset:
        subset = docs[::6] + [d for d in docs if d["id"] in
                              ("case_001", "case_002", "case_003", "case_004", "case_090")]
        seen = set()
        docs = [d for d in subset if d["id"] not in seen and not seen.add(d["id"])]
        print(f"STRATIFIED SUBSET: {len(docs)} docs", flush=True)
    if args.collision_only:
        docs = collision_docs
    else:
        docs = docs + collision_docs

    def run_one(doc):
        cfg = vcfg.Config(provider="bench-qwen35-local", model=model_name,
                          policy=vcfg.POLICY_BALANCED, critic=False,
                          chunk_chars=6000, chunk_overlap=400).validate()
        t0 = time.perf_counter()
        result = deidentify(doc["text"], cfg)
        wall = time.perf_counter() - t0
        safe, vault = result.text, result.vault
        restored = vault.rehydrate_text(safe)
        surfaces = []
        for token, entry in vault._tokens.items():
            surfaces.append({"token": token, "surface": entry["surface"],
                             "category": entry.get("category", ""),
                             "count": entry.get("count", 0)})
            for alias in entry.get("aliases", []):
                surfaces.append({"token": token, "surface": alias,
                                 "category": entry.get("category", ""),
                                 "count": 0})
        return {"safe": safe, "restored": restored, "roundtrip": restored == doc["text"],
                "surfaces": surfaces, "wall_s": wall}

    def score_doc(doc, run):
        gold_ents = doc.get("entities", [])
        gold_red = [e for e in gold_ents if e.get("should_redact")]
        gold_no = [e for e in gold_ents if not e.get("should_redact")]
        surfaces = run["surfaces"]
        s_texts = [s["surface"] for s in surfaces]

        covered_gold = set()
        matched_surfaces = set()
        for gi, g in enumerate(gold_red):
            for si, s in enumerate(s_texts):
                if surface_matches(g["text"], s):
                    covered_gold.add(gi)
                    matched_surfaces.add(si)
                    break
        over_redacted = []
        for si, s in enumerate(surfaces):
            if si in matched_surfaces:
                continue
            if any(surface_matches(g["text"], s["surface"]) for g in gold_no):
                over_redacted.append((s["surface"], "should_not_redact"))
            elif not any(surface_matches(g["text"], s["surface"]) for g in gold_red):
                over_redacted.append((s["surface"], "no_gold"))

        recall = len(covered_gold) / len(gold_red) if gold_red else 1.0
        precision = (len(s_texts) - len(over_redacted)) / len(s_texts) if s_texts else 1.0

        residual_chars = sum(len(g["text"]) for i, g in enumerate(gold_red) if i not in covered_gold)

        # per-category recall
        cat = {}
        for gi, g in enumerate(gold_red):
            c = GOLD_CATEGORY.get(g["type"], "other")
            st = cat.setdefault(c, [0, 0])
            st[1] += 1
            if gi in covered_gold:
                st[0] += 1

        quasi_ok = None
        if doc.get("risk_group") == "quasi_identifier":
            quasi_gold = [g for g in doc["entities"] if g.get("context_class") == "quasi_identifier"] or gold_red
            quasi_ok = sum(1 for gi, g in enumerate(gold_red)
                           if gi in covered_gold) / len(quasi_gold) if quasi_gold else None

        return {
            "id": doc["id"],
            "identity_recall": recall,
            "identity_precision": precision,
            "over_redaction_count": len(over_redacted),
            "over_redaction_examples": over_redacted[:6],
            "residual_chars": residual_chars,
            "roundtrip": run["roundtrip"],
            "surfaces": len(s_texts),
            "category_recall": {k: f"{v[0]}/{v[1]}" for k, v in cat.items()},
            "quasi_recall": quasi_ok,
            "wall_s": run["wall_s"],
        }

    results = []
    with tempfile.TemporaryDirectory(prefix="vault_bench_") as tmp:
        os.chdir(tmp)  # any sidecars land here; directory is deleted after
        for doc in docs:
            run = run_one(doc)
            results.append(score_doc(doc, run))
            r = results[-1]
            flag = "" if r["roundtrip"] else "  [ROUND-TRIP FAILURE]"
            print(f"{doc['id']:<34} R {r['identity_recall']*100:>5.1f}% "
                  f"P {r['identity_precision']*100:>5.1f}% "
                  f"surfaces {r['surfaces']:>3} over {r['over_redaction_count']:>2} "
                  f"residual {r['residual_chars']:>4}{flag}", flush=True)

        # stable-token + collision checks (PHASE 12.2/12.3)
        stable_checks = []
        col_results = {}
        for doc in docs:
            if not (doc["id"].startswith("col_") or doc["id"] in ("case_001", "case_002")):
                continue
            run = run_one(doc)
            safe = run["safe"]
            person_tokens = {s["token"] for s in run["surfaces"] if s["token"].startswith("P-n")}
            detail = {"id": doc["id"], "safe": safe[:300], "person_tokens": sorted(person_tokens)}
            if doc["id"] == "col_1_same_name_two_contexts":
                counts = [safe.count(t) for t in person_tokens]
                detail["verdict"] = ("STABLE" if any(c == 3 for c in counts) else "UNSTABLE")
            elif doc["id"] == "col_3_repeated_contact":
                email_toks = {s["token"] for s in run["surfaces"] if "EMAIL" in s["token"]}
                counts = [safe.count(t) for t in email_toks]
                detail["verdict"] = ("STABLE" if any(c == 2 for c in counts) else "UNSTABLE")
            elif doc["id"] == "col_2_person_vs_username":
                detail["verdict"] = "INFO"
            stable_checks.append(detail)
            if doc["id"].startswith("col_"):
                col_results[doc["id"]] = detail

        # stability runs (PHASE 14): N longest docs x extra passes
        stability = None
        if args.stability > 0:
            hardest = sorted(docs, key=lambda d: -len(d["text"]))[:10]
            stability = []
            for doc in hardest:
                base = run_one(doc)
                base_set = sorted({norm(s["surface"]) for s in base["surfaces"]})
                disagreements = 0
                for _ in range(args.stability):
                    again = run_one(doc)
                    again_set = sorted({norm(s["surface"]) for s in again["surfaces"]})
                    if again_set != base_set:
                        disagreements += 1
                stability.append({"id": doc["id"], "extra_runs": args.stability,
                                  "disagreements": disagreements})
                print(f"stability {doc['id']}: {disagreements}/{args.stability} disagreements", flush=True)

    scored = [r for r in results if not r["id"].startswith("col_")]
    agg_r = sum(r["identity_recall"] for r in scored) / len(scored)
    agg_p = sum(r["identity_precision"] for r in scored) / len(scored)
    rt_fail = sum(1 for r in results if not r["roundtrip"])
    residual_total = sum(r["residual_chars"] for r in scored)

    print("=" * 90)
    print(f"vault-engine + {model_name} over {len(results)} docs")
    print(f"Identity Recall(avg) {agg_r*100:.1f}% | Identity Precision(avg) {agg_p*100:.1f}%")
    print(f"Round-trip failures: {rt_fail} | Residual PII chars: {residual_total}")
    print(f"Collision/stable checks: {[c.get('verdict') for c in stable_checks]}")
    if stability:
        dis = sum(s["disagreements"] for s in stability)
        print(f"Stability: {dis} disagreements across {len(stability)} docs x {args.stability} re-runs")

    out = {"model": model_name, "per_doc": results, "collision": col_results,
           "stability": stability,
           "summary": {"identity_recall": agg_r, "identity_precision": agg_p,
                       "roundtrip_failures": rt_fail, "residual_chars": residual_total}}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

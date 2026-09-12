#!/usr/bin/env python3
"""Model Benchmark Harness for AI Privacy Check.

Fair challenger comparison for Benchmark v2: runs INSTALLED/AVAILABLE models
over the same exact-span fixture (default: tests/fixtures/contextual_privacy_seed.jsonl)
with the shared scoring module (scripts/benchmark_scoring.py).

Contract (v0.6.4):
- Never downloads anything: models must already exist under --models-dir.
  Missing models are reported as "SKIP (NOT INSTALLED)".
- Never touches the production service: this is an offline, in-process harness
  intended to be executed by an isolated runtime interpreter, e.g.:
      <runtime-venv>/bin/python scripts/benchmark_models.py --models-dir <dir>
- Uses production worker logic wherever possible (siamese-uie pipeline +
  extraction, MemPrivacy prompt/JSON parsing, GLiNER production label set).
- Records wall time and process peak RSS; CUDA peak allocated VRAM only when
  running on CUDA. Everything not actually measured is reported as such.

Resource measurement notes:
- RSS is process high-water (resource.getrusage ru_maxrss): monotonically
  increasing across models in one run. macOS reports bytes, Linux KiB.
- CUDA numbers are "peak allocated by this process via torch caching allocator",
  not total system VRAM.
"""

import argparse
from pathlib import Path
import gc
import json
import resource
import sys
import time
import traceback

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(SERVER_DIR / "privacy" / "workers"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from benchmark_scoring import MetricBucket, percentile, score_sample  # noqa: E402

DEFAULT_FIXTURE = PROJECT_DIR / "tests" / "fixtures" / "contextual_privacy_seed.jsonl"

DEFAULT_SEED_SCHEMA = {
    "人物": None,
    "地理位置": None,
    "组织机构": None,
    "学校": None,
    "职位": None,
}

SIAMESE_TYPE_MAP = {
    "人物": "CN_NAME",
    "地理位置": "CN_ADDRESS",
    "组织机构": "ORGANIZATION",
    "学校": "ORGANIZATION",
    "职位": "JOB_TITLE",
}

# Mirrors privacy.detectors.GLiNERDetector.GLINER_LABEL_MAP essentials.
GLINER_LABEL_MAP = {
    "person": "CN_NAME",
    "people": "CN_NAME",
    "name": "CN_NAME",
    "organization": "ORGANIZATION",
    "company": "ORGANIZATION",
    "phone number": "CN_PHONE_NUMBER",
    "phone": "CN_PHONE_NUMBER",
    "mobile phone": "CN_PHONE_NUMBER",
    "email": "EMAIL",
    "email address": "EMAIL",
    "passport number": "CN_PASSPORT",
    "passport": "CN_PASSPORT",
    "driver license": "GOVERNMENT_ID",
    "driving license": "GOVERNMENT_ID",
    "social security number": "US_SSN",
    "ssn": "US_SSN",
    "credit card number": "CREDIT_CARD",
    "credit card": "CREDIT_CARD",
    "bank account": "ACCOUNT_NUMBER",
    "bank account number": "ACCOUNT_NUMBER",
    "address": "CN_ADDRESS",
    "street address": "CN_ADDRESS",
    "location": "LOCATION",
    "city": "LOCATION",
    "country": "LOCATION",
    "username": "USERNAME",
    "user name": "USERNAME",
    "date of birth": "CN_BIRTH_DATE",
    "birth date": "CN_BIRTH_DATE",
    "ip address": "IP_ADDRESS",
    "ipv4": "IP_ADDRESS",
    "ipv6": "IPV6_ADDRESS",
    "mac address": "MAC_ADDRESS",
}
GLINER_LABELS = list(dict.fromkeys([
    "person", "organization", "phone number", "email", "passport number",
    "driver license", "social security number", "credit card number",
    "bank account", "address", "username", "date of birth",
]))

AIGUARD_TYPE_MAP = {
    "name": "CN_NAME",
    "mobile": "CN_PHONE_NUMBER",
    "address": "CN_ADDRESS",
    "id_card": "CN_ID_CARD",
    "bank_card": "CN_BANK_CARD",
    "credit_card": "CREDIT_CARD",
    "email": "EMAIL",
    "passport": "CN_PASSPORT",
    "hkmtp_pass": "CN_PASSPORT",
    "birth_date": "CN_BIRTH_DATE",
    "drivers_license": "GOVERNMENT_ID",
    "plate_number": "CN_LICENSE_PLATE",
    "social_security": "INSURANCE_ID",
    "insurance_policy": "INSURANCE_ID",
    "bank_password": "SECRET",
    "jd_order": "RECORD_ID",
    "taobao_order": "RECORD_ID",
    "pdd_order": "RECORD_ID",
    "sf_tracking": "RECORD_ID",
    "ems_tracking": "RECORD_ID",
    "yto_tracking": "RECORD_ID",
}

OPENMED_TYPE_MAP = {
    "ACCOUNTNAME": "USERNAME",
    "USERNAME": "USERNAME",
    "AGE": "CN_BIRTH_DATE",
    "BANKACCOUNT": "ACCOUNT_NUMBER",
    "BIC": "BIC",
    "BUILDINGNUMBER": "CN_ADDRESS",
    "CITY": "LOCATION",
    "COUNTY": "LOCATION",
    "CREDITCARD": "CREDIT_CARD",
    "DATEOFBIRTH": "CN_BIRTH_DATE",
    "DRIVERLICENSE": "GOVERNMENT_ID",
    "EMAIL": "EMAIL",
    "GIVENNAME": "CN_NAME",
    "IDCARD": "CN_ID_CARD",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IPV6_ADDRESS",
    "MAC": "MAC_ADDRESS",
    "MEDICALCONDITION": "MEDICAL",
    "NAME": "CN_NAME",
    "ORGANIZATION": "ORGANIZATION",
    "PASSPORT": "PASSPORT",
    "PERSON": "CN_NAME",
    "PHONENUMBER": "CN_PHONE_NUMBER",
    "SOCIALSECURITYNUMBER": "US_SSN",
    "SSN": "US_SSN",
    "STREETADDRESS": "CN_ADDRESS",
    "TAXNUMBER": "GOVERNMENT_ID",
    "TELEPHONENUMBER": "CN_PHONE_NUMBER",
    "URL": "PRIVATE_URL",
    "USERNAME": "USERNAME",
    "ZIPCODE": "CN_ADDRESS",
}

RANER_TYPE_MAP = {
    "PER": "CN_NAME",
    "NAME": "CN_NAME",
    "人名": "CN_NAME",
    "LOC": "CN_ADDRESS",
    "GPE": "LOCATION",
    "地名": "CN_ADDRESS",
    "ORG": "ORGANIZATION",
    "机构名": "ORGANIZATION",
}

SEMANTIC_TYPE_MAP = {
    "name": "CN_NAME",
    "person": "CN_NAME",
    "phone": "CN_PHONE_NUMBER",
    "mobile": "CN_PHONE_NUMBER",
    "email": "EMAIL",
    "address": "CN_ADDRESS",
    "precise_location": "CN_ADDRESS",
    "location": "LOCATION",
    "id_card": "CN_ID_CARD",
    "government_id": "GOVERNMENT_ID",
    "passport": "CN_PASSPORT",
    "bank_card": "CN_BANK_CARD",
    "credit_card": "CREDIT_CARD",
    "financial_record": "FINANCIAL",
    "financial": "FINANCIAL",
    "medical_record": "MEDICAL",
    "health_info": "MEDICAL",
    "medical": "MEDICAL",
    "relationship": "RELATIONSHIP",
    "social_account": "USERNAME",
    "username": "USERNAME",
    "password": "SECRET",
    "api_token": "SECRET",
    "otp": "SECRET",
    "credentials": "SECRET",
    "secret": "SECRET",
    "private_key": "SECRET",
    "database_uri": "DATABASE_URI",
    "biometric": "MEDICAL",
}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def detect_adapter(model_dir: Path) -> str | None:
    """Classifies an installed model directory into a harness adapter."""
    if not (model_dir / "config.json").is_file() and not (model_dir / "configuration.json").is_file():
        return None

    ms_config = _read_json(model_dir / "configuration.json") or {}
    hf_config = _read_json(model_dir / "config.json") or {}

    task = str(ms_config.get("task") or "").lower()
    if task == "siamese-uie":
        return "siamese"

    if (model_dir / "gliner_config.json").is_file():
        return "gliner"

    pipeline_type = str((ms_config.get("pipeline") or {}).get("type") or "").lower()
    if pipeline_type in ("named-entity-recognition", "ner"):
        return "ms_ner"

    archs = hf_config.get("architectures") or []
    joined = ",".join(str(a) for a in archs)
    if "ForTokenClassification" in joined:
        return "token_cls"
    if "ForCausalLM" in joined:
        return "generative"
    return None


def resolve_model_type(label: str, mapping: dict) -> str:
    clean = str(label).strip()
    for prefix in ("B-", "I-", "E-", "S-"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    clean_lower = clean.lower()
    if clean in mapping:
        return mapping[clean]
    if clean.upper() in mapping:
        return mapping[clean.upper()]
    return mapping.get(clean_lower, clean_lower)


# ---------------------------------------------------------------- adapters

def run_siamese(model_dir: Path, samples: list, device: str, notes: list):
    from modelscope.pipelines import pipeline  # type: ignore
    from modelscope.utils.constant import Tasks  # type: ignore
    sys.path.insert(0, str(SERVER_DIR / "privacy" / "workers"))
    from siamese_uie_worker import extract_entities_from_output  # noqa: E402

    pipe = pipeline(Tasks.siamese_uie, model=str(model_dir), device="cuda:0" if device == "cuda" else "cpu")
    import time as _time

    latencies = []
    out = []
    for sample in samples:
        text = sample["text"]
        t0 = _time.perf_counter()
        raw = pipe(input=text, schema=DEFAULT_SEED_SCHEMA)
        latencies.append((_time.perf_counter() - t0) * 1000.0)
        entities = []
        for ent in extract_entities_from_output(raw, text):
            mapped = SIAMESE_TYPE_MAP.get(ent["label"], ent["label"])
            entities.append({"type": mapped, "start": ent["start"], "end": ent["end"], "text": ent["text"]})
        out.append(entities)
    return out, latencies


def run_gliner(model_dir: Path, samples: list, device: str, notes: list):
    """Mirrors production GLiNERDetector: same label set, threshold 0.40 and the
    USERNAME plausibility post-filter, imported from the production module as
    the single source of truth."""
    import time as _time

    from gliner import GLiNER  # type: ignore
    from privacy.detectors import GLiNERDetector  # noqa: E402

    model = GLiNER.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to("cuda" if device == "cuda" else "cpu")

    labels = list(set(GLiNERDetector.GLINER_LABEL_MAP.keys()))
    detector = GLiNERDetector(Path("/tmp"))
    latencies = []
    out = []
    for sample in samples:
        text = sample["text"]
        t0 = _time.perf_counter()
        raw_entities = model.predict_entities(text, labels, threshold=0.40)
        latencies.append((_time.perf_counter() - t0) * 1000.0)
        entities = []
        for ent in raw_entities:
            s, e = int(ent.get("start", 0)), int(ent.get("end", 0))
            if not (0 <= s < e <= len(text) and text[s:e] == ent.get("text", "")):
                continue
            label_raw = str(ent.get("label", "")).strip().lower()
            if label_raw == "username" and not detector._is_plausible_username(text, s, e, text[s:e]):
                continue
            mapped = GLiNERDetector.GLINER_LABEL_MAP.get(label_raw, (label_raw.upper(), label_raw))
            entities.append({"type": mapped[0], "start": s, "end": e, "text": text[s:e]})
        out.append(entities)
    return out, latencies


def run_ms_ner(model_dir: Path, samples: list, device: str, notes: list):
    """ModelScope legacy NER pipeline (e.g. iic RANER): bare transformers
    loading yields randomly initialized heads for these checkpoints."""
    import time as _time

    from modelscope.pipelines import pipeline  # type: ignore
    from modelscope.utils.constant import Tasks  # type: ignore
    from siamese_uie_worker import extract_entities_from_output  # noqa: E402

    pipe = pipeline(Tasks.named_entity_recognition, model=str(model_dir),
                    device="cuda:0" if device == "cuda" else "cpu")
    latencies = []
    out = []
    for sample in samples:
        text = sample["text"]
        t0 = _time.perf_counter()
        raw = pipe(input=text)
        latencies.append((_time.perf_counter() - t0) * 1000.0)
        entities = []
        for ent in extract_entities_from_output(raw, text):
            mapped = RANER_TYPE_MAP.get(ent["label"], ent["label"])
            entities.append({"type": mapped, "start": ent["start"], "end": ent["end"], "text": ent["text"]})
        out.append(entities)
    return out, latencies


def run_token_cls(model_dir: Path, samples: list, device: str, notes: list):
    import torch  # type: ignore
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore

    hf_config = _read_json(model_dir / "config.json") or {}
    archs = [str(a) for a in (hf_config.get("architectures") or [])]
    # Security contract: never execute third-party remote code (trust_remote_code).
    if any("OpenAIPrivacyFilter" in a for a in archs):
        notes.append("custom architecture requires trust_remote_code - refused by policy")
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device if device in ("cuda", "cpu") else "cpu")
    model.eval()

    id2label = {int(k): v for k, v in (hf_config.get("id2label") or {}).items()}
    has_e_tags = any(str(v).startswith(("B-", "I-", "E-", "S-")) for v in id2label.values())

    import time as _time

    out = []
    latencies = []
    with torch.no_grad():
        for sample in samples:
            text = sample["text"]
            t0 = _time.perf_counter()
            # Character-level models need no word alignment; batch of one.
            inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                               truncation=True, max_length=4096)
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits[0]

            labels = logits.argmax(dim=-1).tolist()
            spans = []
            open_span = None  # [start, end, label]
            for (tok_start, tok_end), label_id in zip(offset_mapping, labels):
                if tok_start == tok_end:  # special tokens
                    continue
                raw_label = id2label.get(label_id, "O")
                if has_e_tags and raw_label != "O":
                    prefix, body = raw_label.split("-", 1)
                else:
                    prefix, body = ("S", raw_label) if raw_label != "O" else ("O", "")
                if prefix == "S":
                    if open_span:
                        spans.append(open_span)
                        open_span = None
                    spans.append([tok_start, tok_end, body])
                elif prefix == "B":
                    if open_span:
                        spans.append(open_span)
                    open_span = [tok_start, tok_end, body]
                elif prefix == "I" and open_span and open_span[2] == body:
                    open_span[1] = tok_end
                elif prefix == "E":
                    if open_span and open_span[2] == body:
                        open_span[1] = tok_end
                        spans.append(open_span)
                        open_span = None
                    else:
                        spans.append([tok_start, tok_end, body])
            if open_span:
                spans.append(open_span)

            entities = []
            for s, e, body in spans:
                frag = text[s:e].strip()
                if not frag:
                    continue
                entities.append({
                    "type": resolve_model_type(body, TOKEN_CLS_MAPS.get(model_dir.name, {})),
                    "start": s, "end": e, "text": frag,
                })
            out.append(entities)
            latencies.append((_time.perf_counter() - t0) * 1000.0)
    return out, latencies


def _pick_token_map(model_dir: Path) -> dict:
    hf_config = _read_json(model_dir / "config.json") or {}
    model_type = str(hf_config.get("model_type") or "")
    if "qwen3" in model_type or "aiguard" in model_dir.name.lower():
        return AIGUARD_TYPE_MAP
    if "privacy" in model_dir.name.lower():
        return OPENMED_TYPE_MAP
    return RANER_TYPE_MAP


def run_generative(model_dir: Path, samples: list, device: str, notes: list):
    """Semantic/generative challenger path.

    memprivacy-* runs through the production worker request handler (same
    prompt template, same JSON parsing, real_name=unknown). Other causal-LM
    challengers (Qwen) run the shared AIPrivacyCheck extraction prompt with
    Qwen3 thinking disabled, parsed by the production JSON parser.
    """
    import time as _time

    import torch  # type: ignore

    latencies = []
    out = []

    if "memprivacy" in model_dir.name.lower():
        import memprivacy_worker  # noqa: E402

        resp_load = memprivacy_worker.handle_request(
            {"action": "load", "model_path": str(model_dir), "device": device})
        if not resp_load.get("ok"):
            notes.append(f"load failed: {resp_load.get('error')}"[:160])
            return None, latencies

        from privacy.detectors import MEMPRIVACY_TYPE_MAP  # noqa: E402

        for sample in samples:
            text = sample["text"]
            t0 = _time.perf_counter()
            resp = memprivacy_worker.handle_request({
                "action": "detect",
                "model_path": str(model_dir),
                "device": device,
                "text": text,
                "real_name": "unknown",
                "max_new_tokens": 256,
            })
            latencies.append((_time.perf_counter() - t0) * 1000.0)
            entities = []
            if resp.get("ok"):
                cursor = 0
                for item in resp.get("entities", []):
                    frag = str(item.get("original_text", "")).strip()
                    if not frag:
                        continue
                    idx = text.find(frag, cursor)
                    if idx == -1:
                        idx = text.find(frag)
                        if idx == -1:
                            continue
                    ptype = str(item.get("privacy_type", "PII")).strip().lower()
                    mapped = MEMPRIVACY_TYPE_MAP.get(ptype, (ptype.upper(), ptype))[0]
                    entities.append({"type": mapped, "start": idx, "end": idx + len(frag), "text": frag})
                    cursor = idx + len(frag)
            else:
                notes.append(f"detect failed: {resp.get('error')}"[:160])
            out.append(entities)
        return out, latencies

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from memprivacy_worker import parse_extracted_json, strip_think_tags  # noqa: E402
    from modelscope_downloader import AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT  # noqa: E402
    from privacy.detectors import MEMPRIVACY_TYPE_MAP  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype="auto")
    model.to(device if device in ("cuda", "cpu") else "cpu")
    model.eval()

    for sample in samples:
        text = sample["text"]
        messages = [
            {"role": "system", "content": AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = _time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latencies.append((_time.perf_counter() - t0) * 1000.0)
        raw = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        raw = strip_think_tags(raw)
        items, ok, err = parse_extracted_json(raw, text)
        if not ok and err:
            notes.append(f"json parse issue: {err}"[:120])
        entities = []
        cursor = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            frag = str(item.get("original_text") or item.get("text") or "").strip()
            if not frag:
                continue
            idx = text.find(frag, cursor)
            if idx == -1:
                idx = text.find(frag)
                if idx == -1:
                    continue
            sem_type = str(item.get("privacy_type") or item.get("type") or "PII")
            mapped = MEMPRIVACY_TYPE_MAP.get(sem_type.strip().lower(), (sem_type.upper(), sem_type))[0]
            entities.append({"type": mapped, "start": idx, "end": idx + len(frag), "text": frag})
            cursor = idx + len(frag)
        out.append(entities)

    del model
    gc.collect()
    return out, latencies


TOKEN_CLS_MAPS = None  # resolved lazily per model dir


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: bytes; Linux: KiB
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def main() -> int:
    global TOKEN_CLS_MAPS
    parser = argparse.ArgumentParser(description="AIPrivacyCheck model benchmark harness")
    parser.add_argument("--models-dir", default=str(PROJECT_DIR / "data" / "models"),
                        help="Directory containing installed model folders")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--model", action="append", default=None,
                        help="Only benchmark these model folder names (repeatable)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N samples (0 = all)")
    parser.add_argument("--json-out", default=None, help="Write raw results JSON to this path")
    args = parser.parse_args()

    try:
        import torch  # type: ignore
        HAS_TORCH = True
    except Exception:
        print("torch is not importable: run this harness inside an isolated runtime interpreter, e.g.")
        print("  <runtime-venv>/bin/python scripts/benchmark_models.py --models-dir <dir>")
        return 2

    cuda_ok = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"

    models_dir = Path(args.models_dir).resolve()
    samples = [json.loads(line) for line in Path(args.fixture).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]

    candidates = sorted(p for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")) if models_dir.is_dir() else []
    if args.model:
        wanted = set(args.model)
        candidates = [c for c in candidates if c.name in wanted]

    print("=" * 100)
    print("  AI Privacy Check - Model Benchmark Harness (v0.6.4)")
    print("=" * 100)
    print(f"Fixture: {args.fixture} ({len(samples)} samples) | device: {device}"
          + ("" if cuda_ok else "  [CUDA: Not Executed - no CUDA device in this environment]"))

    raw_results = {}
    rows = []

    for model_dir in candidates:
        adapter = detect_adapter(model_dir)
        name = model_dir.name
        if adapter is None:
            print(f"\n>>> {name}: SKIP (NOT INSTALLED or unrecognized layout)")
            continue
        if args.model and name not in set(args.model):
            continue

        TOKEN_CLS_MAPS = {name: _pick_token_map(model_dir)}
        notes: list = []
        rss_before = peak_rss_mb()
        t0 = time.perf_counter()
        try:
            runner = {
                "siamese": run_siamese,
                "gliner": run_gliner,
                "ms_ner": run_ms_ner,
                "token_cls": run_token_cls,
                "generative": run_generative,
            }[adapter]
            predictions, sample_latencies = runner(model_dir, samples, device, notes)
        except Exception as exc:
            print(f"\n>>> {name}: FAILED ({type(exc).__name__}: {exc})")
            traceback.print_exc(limit=3)
            rows.append({"name": name, "adapter": adapter, "status": f"FAILED: {type(exc).__name__}: {exc}"[:160]})
            raw_results[name] = {"status": "failed", "error": str(exc)[:400]}
            gc.collect()
            continue
        load_s = time.perf_counter() - t0

        if predictions is None:
            print(f"\n>>> {name}: NOT EXECUTED ({'; '.join(notes) or 'adapter refused'})")
            rows.append({"name": name, "adapter": adapter, "status": "NOT EXECUTED: " + "; ".join(notes)})
            raw_results[name] = {"status": "not_executed", "notes": notes}
            continue

        bucket = MetricBucket()
        pii_free_total = pii_free_flagged = 0
        fp_types: dict = {}
        latencies = sample_latencies or [0.0] * len(samples)
        per_sample = []
        for sample, preds in zip(samples, predictions):
            score = score_sample(sample["text"], sample["entities"], preds)
            is_pii_free = not sample["entities"]
            if is_pii_free:
                pii_free_total += 1
                if score.fp > 0:
                    pii_free_flagged += 1
                    for ftype in score.false_positive_types:
                        fp_types[ftype] = fp_types.get(ftype, 0) + 1
            bucket.add(score, is_pii_free=is_pii_free)
            per_sample.append(preds)
        run_s = time.perf_counter() - t0 - load_s

        s = bucket.summary()
        rows.append({
            "name": name, "adapter": adapter, "status": "ok",
            "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "pii_free_fpr": pii_free_flagged / pii_free_total if pii_free_total else 0.0,
            "leaked_chars": s["leaked_chars"], "overredacted_chars": s["overredacted_chars"],
            "latency_avg_ms": sum(latencies) / len(latencies),
            "latency_p95_ms": percentile(latencies, 0.95),
            "load_s": load_s, "run_s": run_s,
            "peak_rss_mb_after": peak_rss_mb(),
            "rss_before_mb": rss_before,
            "cuda_peak_alloc_mb": (torch.cuda.max_memory_allocated() / 1048576) if cuda_ok else None,
            "notes": notes,
            "fp_types": fp_types,
        })
        raw_results[name] = {"status": "ok", "predictions": per_sample, "summary": rows[-1]}
        gc.collect()

    print("\n" + "=" * 100)
    print(f"{'Model':<28} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>6} {'Leak':>5} {'Over':>5} "
          f"{'Lat(ms)':>8} {'p95':>7} {'Load(s)':>8}")
    print("-" * 100)
    for row in rows:
        if row.get("status") != "ok":
            print(f"{row['name']:<28} {row['status']}")
            continue
        print(f"{row['name']:<28} {row['precision']*100:>6.1f}% {row['recall']*100:>6.1f}% "
              f"{row['f1']*100:>6.1f}% {row['pii_free_fpr']*100:>5.0f}% "
              f"{row['leaked_chars']:>5} {row['overredacted_chars']:>5} "
              f"{row['latency_avg_ms']:>8.1f} {row['latency_p95_ms']:>7.1f} {row['load_s']:>8.1f}")
    print("=" * 100)
    print("Notes:")
    print("  - RSS is process high-water (ru_maxrss), cumulative across models within this run.")
    if not cuda_ok:
        print("  - CUDA VRAM: Not Executed (no CUDA device); Tesla P4 numbers require fnOS hardware.")
    for row in rows:
        if row.get("notes"):
            print(f"  - {row['name']}: {row['notes']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(raw_results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Raw results written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

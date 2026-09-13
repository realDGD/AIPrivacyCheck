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
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(SERVER_DIR / "privacy" / "workers"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from benchmark_scoring import (  # noqa: E402
    CRITICAL_ENTITY_TYPES,
    MetricBucket,
    matches_type,
    percentile,
    score_sample,
)
from benchmark_cache import BenchmarkPredictionCache  # noqa: E402
from selective_downloader import ensure_selective_model  # noqa: E402

DEFAULT_FIXTURE = PROJECT_DIR / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"
if not DEFAULT_FIXTURE.is_file():
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

# Single source of truth: import directly from production GLiNERDetector
from privacy.detectors import GLiNERDetector, GLINER_LABELS  # noqa: E402
GLINER_LABEL_MAP = {k: v[0] for k, v in GLiNERDetector.GLINER_LABEL_MAP.items()}

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
    if "ForCausalLM" in joined or "ForConditionalGeneration" in joined:
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


def run_gliner(model_dir: Path, samples: list, device: str, notes: list, min_threshold: float = 0.30):
    """Mirrors production GLiNERDetector: deterministic GLINER_LABELS, min_threshold,
    and USERNAME plausibility post-filter. Retains prediction scores for offline threshold sweep."""
    import time as _time

    from gliner import GLiNER  # type: ignore
    from privacy.detectors import GLiNERDetector, GLINER_LABELS  # noqa: E402

    model = GLiNER.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to("cuda" if device == "cuda" else "cpu")

    labels = list(GLINER_LABELS)
    detector = GLiNERDetector(Path("/tmp"))
    default_threshold = GLiNERDetector.GLINER_DEFAULT_THRESHOLD
    latencies = []
    out = []
    for sample in samples:
        text = sample["text"]
        t0 = _time.perf_counter()
        raw_entities = model.predict_entities(text, labels, threshold=min_threshold)
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
            score = float(ent.get("score", 0.85))
            entities.append({
                "type": mapped[0],
                "start": s,
                "end": e,
                "text": text[s:e],
                "score": round(score, 4),
                "label_raw": label_raw,
            })
        out.append(entities)
    return out, latencies


def evaluate_gliner_threshold_sweep(samples: list, all_predictions: list, thresholds: list) -> Tuple[list, dict]:
    hard_cases_queries = [
        ("case_001", "张三今天来了，稍后张三又打电话过来。"),
        ("case_002", "张三，用户名是李四。"),
        ("case_003", "用户名是张三。"),
        ("case_004", "张三是用户名。"),
    ]
    hard_indices = {}
    for cid, query in hard_cases_queries:
        for idx, s in enumerate(samples):
            if s.get("id") == cid or query in s["text"]:
                hard_indices[cid] = (idx, query)
                break

    rows = []
    hard_case_reports = {cid: [] for cid, _ in hard_cases_queries}

    for t in thresholds:
        bucket = MetricBucket()
        pii_free_total = pii_free_flagged = 0
        person_to_username = 0
        username_to_person = 0
        redactable_fn = 0
        critical_fn = 0
        type_correct = 0

        for sample, preds in zip(samples, all_predictions):
            t_preds = [p for p in preds if p.get("score", 1.0) >= t]
            score = score_sample(sample["text"], sample["entities"], t_preds)
            is_pii_free = not sample["entities"]
            if is_pii_free:
                pii_free_total += 1
                if score.fp > 0:
                    pii_free_flagged += 1
            bucket.add(score, is_pii_free=is_pii_free)

            # Check confusion, redactable FN, and critical FN
            for true_ent in sample["entities"]:
                true_type = true_ent["type"]
                should_redact = true_ent.get("should_redact", True)
                matched = any(
                    p["start"] == true_ent["start"] and p["end"] == true_ent["end"]
                    and matches_type(p["type"], true_ent["type"])
                    for p in t_preds
                )
                if not matched and should_redact:
                    redactable_fn += 1
                    if true_type in CRITICAL_ENTITY_TYPES:
                        critical_fn += 1

                # Overlap-based confusion
                overlaps = [
                    p for p in t_preds
                    if p["start"] < true_ent["end"] and true_ent["start"] < p["end"]
                ]
                for p in overlaps:
                    if true_type in ("CN_NAME", "PERSON") and p["type"] == "USERNAME":
                        person_to_username += 1
                    elif true_type == "USERNAME" and p["type"] in ("CN_NAME", "PERSON"):
                        username_to_person += 1

            # Check exact type correctness on matched TPs
            matched_p = set()
            for true_ent in sample["entities"]:
                for p_idx, p in enumerate(t_preds):
                    if p_idx in matched_p:
                        continue
                    if p["start"] == true_ent["start"] and p["end"] == true_ent["end"] and matches_type(p["type"], true_ent["type"]):
                        matched_p.add(p_idx)
                        if p["type"] == true_ent["type"]:
                            type_correct += 1
                        break

        s = bucket.summary()
        type_acc = type_correct / s["tp"] if s["tp"] else 0.0
        fpr = pii_free_flagged / pii_free_total if pii_free_total else 0.0

        row = {
            "threshold": t,
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "type_acc": type_acc,
            "pii_free_fpr": fpr,
            "pii_free_flagged": pii_free_flagged,
            "pii_free_total": pii_free_total,
            "person_to_username": person_to_username,
            "username_to_person": username_to_person,
            "redactable_fn": redactable_fn,
            "critical_fn": critical_fn,
        }
        rows.append(row)

        for cid, (idx, qtext) in hard_indices.items():
            t_preds = [p for p in all_predictions[idx] if p.get("score", 1.0) >= t]
            hard_case_reports[cid].append({
                "threshold": t,
                "preds": [(p["type"], p["text"], p.get("score")) for p in t_preds],
            })

    return rows, hard_case_reports


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
    failures = 0
    for sample in samples:
        text = sample["text"]
        t0 = _time.perf_counter()
        try:
            raw = pipe(input=text)
        except Exception as exc:
            # Legacy NER pipelines cap at 512 positions: long documents crash
            # inside the model. Record as a model limitation (empty prediction),
            # never as a silent skip.
            failures += 1
            latencies.append((_time.perf_counter() - t0) * 1000.0)
            out.append([])
            continue
        latencies.append((_time.perf_counter() - t0) * 1000.0)
        entities = []
        for ent in extract_entities_from_output(raw, text):
            mapped = RANER_TYPE_MAP.get(ent["label"], ent["label"])
            entities.append({"type": mapped, "start": ent["start"], "end": ent["end"], "text": ent["text"]})
        out.append(entities)
    if failures:
        notes.append(f"{failures}/{len(samples)} samples crashed the 512-position model")
    return out, latencies


def run_native_token_cls(model_dir: Path, samples: list, device: str, notes: list):
    """Token classification through a runtime that natively registers the
    architecture (e.g. openai_privacy_filter under transformers >=5.6).
    The repo ships no custom code, so no trust_remote_code is involved."""
    import time as _time

    import torch  # type: ignore
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device if device in ("cuda", "cpu") else "cpu")
    model.eval()

    hf_config = _read_json(model_dir / "config.json") or {}
    id2label = {int(k): v for k, v in (hf_config.get("id2label") or {}).items()}
    has_e_tags = any(str(v).startswith(("B-", "I-", "E-", "S-")) for v in id2label.values())

    out = []
    latencies = []
    with torch.no_grad():
        for sample in samples:
            text = sample["text"]
            t0 = _time.perf_counter()
            inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                               truncation=True, max_length=8192)
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits[0]
            labels = logits.argmax(dim=-1).tolist()
            latencies.append((_time.perf_counter() - t0) * 1000.0)

            spans = []
            open_span = None
            for (tok_start, tok_end), label_id in zip(offset_mapping, labels):
                if tok_start == tok_end:
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
                    "type": resolve_model_type(body, {}),
                    "start": s, "end": e, "text": frag,
                })
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
            model_max = getattr(tokenizer, "model_max_length", None) or 512
            if model_max > 100000:  # sentinel for unset
                model_max = 512
            inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                               truncation=True, max_length=model_max)
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
    parser.add_argument("--force-adapter", default=None,
                        help="Override adapter detection (e.g. native_token_cls for a "
                             "benchmark-only runtime that natively registers the arch)")
    parser.add_argument("--download-missing", action="store_true", default=False,
                        help="Allow selectively downloading required model files if missing")
    parser.add_argument("--score-cache", default=None,
                        help="Load and score from benchmark prediction cache directory without re-running inference")
    parser.add_argument("--min-threshold", type=float, default=0.30,
                        help="Minimum inference threshold for GLiNER single-pass inference (default 0.30)")
    parser.add_argument("--sweep-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65",
                        help="Comma-separated threshold sweep values for offline re-scoring")
    parser.add_argument("--allow-legacy-unverified-cache", action="store_true",
                        help="Allow loading legacy prediction cache without cache-manifest.json (UNVERIFIED)")
    args = parser.parse_args()

    cache_mgr = BenchmarkPredictionCache()
    samples = [json.loads(line) for line in Path(args.fixture).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]

    # Offline score-cache mode: score directly without model weights or GPU
    if args.score_cache:
        print("=" * 100)
        print("  AI Privacy Check - Benchmark Cache Offline Scoring (v0.6.8)")
        print("=" * 100)
        cfg, cached_preds, _ = cache_mgr.load(
            Path(args.score_cache),
            allow_legacy_unverified=args.allow_legacy_unverified_cache,
        )
        print(f"Loaded cache from: {args.score_cache}")
        print(f"Model: {cfg.get('model_id')} | Corpus: {cfg.get('corpus_file')} ({len(cached_preds)} predictions)")

        thresholds = [float(x.strip()) for x in args.sweep_thresholds.split(",") if x.strip()]
        all_preds = [p["entities"] for p in cached_preds]
        sweep_rows, hard_cases = evaluate_gliner_threshold_sweep(samples, all_preds, thresholds)

        _print_gliner_sweep_tables(sweep_rows, hard_cases, thresholds)
        return 0

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
    target_models = list(args.model) if args.model else [p.name for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")] if models_dir.is_dir() else []

    print("=" * 100)
    print("  AI Privacy Check - Model Benchmark Harness (v0.6.8)")
    print("=" * 100)
    print(f"Fixture: {args.fixture} ({len(samples)} samples) | device: {device}"
          + ("" if cuda_ok else "  [CUDA: Not Executed - no CUDA device in this environment]"))

    raw_results = {}
    rows = []

    for name in target_models:
        model_dir = models_dir / name
        if not model_dir.is_dir():
            ok, msg, manifest = ensure_selective_model(name, models_dir, download_missing=args.download_missing)
            if not ok:
                print(f"\n>>> {name}: SKIP ({msg})")
                continue
            model_dir = models_dir / name

        adapter = detect_adapter(model_dir)
        if adapter is None:
            print(f"\n>>> {name}: SKIP (NOT INSTALLED or unrecognized layout)")
            continue

        TOKEN_CLS_MAPS = {name: _pick_token_map(model_dir)}
        notes: list = []
        rss_before = peak_rss_mb()
        t0 = time.perf_counter()

        predictions = None
        sample_latencies = None
        load_s = 0.0

        # Check raw prediction cache for GLiNER
        if adapter == "gliner":
            inference_config = {
                "device": device,
                "min_threshold": args.min_threshold,
                "label_order": list(GLINER_LABELS),
            }
            key = cache_mgr.build_cache_key(
                corpus_path=Path(args.fixture),
                model_id=name,
                revision="master",
                model_dir=model_dir,
                inference_config=inference_config,
            )
            cache_hit, cached_dir = cache_mgr.has_valid_cache(key)
            if cache_hit:
                print(f"\n>>> {name}: [CACHE HIT] Loaded raw predictions from {cached_dir}")
                _, loaded_raw, _ = cache_mgr.load(
                    cached_dir,
                    allow_legacy_unverified=args.allow_legacy_unverified_cache,
                )
                predictions = [p["entities"] for p in loaded_raw]
                sample_latencies = [p.get("latency_ms", 0.0) for p in loaded_raw]
            else:
                try:
                    predictions, sample_latencies = run_gliner(model_dir, samples, device, notes, min_threshold=args.min_threshold)
                    load_s = time.perf_counter() - t0
                    # Persist to cache
                    preds_to_cache = [
                        {"id": s.get("id", f"sample_{i}"), "text": s["text"], "entities": p, "latency_ms": lat}
                        for i, (s, p, lat) in enumerate(zip(samples, predictions, sample_latencies))
                    ]
                    saved_dir = cache_mgr.save(key, preds_to_cache)
                    print(f"\n>>> {name}: [CACHE SAVED] Persisted {len(preds_to_cache)} raw predictions to {saved_dir}")
                except Exception as exc:
                    print(f"\n>>> {name}: FAILED ({type(exc).__name__}: {exc})")
                    traceback.print_exc(limit=3)
                    rows.append({"name": name, "adapter": adapter, "status": f"FAILED: {type(exc).__name__}: {exc}"[:160]})
                    raw_results[name] = {"status": "failed", "error": str(exc)[:400]}
                    gc.collect()
                    continue
        else:
            try:
                runner = {
                    "siamese": run_siamese,
                    "ms_ner": run_ms_ner,
                    "token_cls": run_token_cls,
                    "native_token_cls": run_native_token_cls,
                    "generative": run_generative,
                }[args.force_adapter or adapter]
                predictions, sample_latencies = runner(model_dir, samples, device, notes)
                load_s = time.perf_counter() - t0
            except Exception as exc:
                print(f"\n>>> {name}: FAILED ({type(exc).__name__}: {exc})")
                traceback.print_exc(limit=3)
                rows.append({"name": name, "adapter": adapter, "status": f"FAILED: {type(exc).__name__}: {exc}"[:160]})
                raw_results[name] = {"status": "failed", "error": str(exc)[:400]}
                gc.collect()
                continue

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
            eval_preds = preds
            if adapter == "gliner":
                from privacy.detectors import GLiNERDetector
                eval_preds = [p for p in preds if p.get("score", 1.0) >= GLiNERDetector.GLINER_DEFAULT_THRESHOLD]
            score = score_sample(sample["text"], sample["entities"], eval_preds)
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

        # If GLiNER, run full offline threshold sweep
        if adapter == "gliner":
            thresholds = [float(x.strip()) for x in args.sweep_thresholds.split(",") if x.strip()]
            sweep_rows, hard_cases = evaluate_gliner_threshold_sweep(samples, predictions, thresholds)
            raw_results[name]["threshold_sweep"] = sweep_rows
            raw_results[name]["hard_cases"] = hard_cases
            _print_gliner_sweep_tables(sweep_rows, hard_cases, thresholds)

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

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(raw_results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Raw results written: {args.json_out}")
    return 0


def _print_gliner_sweep_tables(sweep_rows: list, hard_cases: dict, thresholds: list) -> None:
    print("\n" + "=" * 128)
    print("  GLiNER Offline Threshold Sweep (Deterministic Labels: 33 types, Single-pass Inference)")
    print("=" * 128)
    print(f"{'Thresh':<8} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'TypeAcc':>8} | {'PII-free FPR':>12} | {'P->USER':>8} | {'USER->P':>8} | {'Redactable FN':>13} | {'Critical FN':>11}")
    print("-" * 128)
    for r in sweep_rows:
        print(f"{r['threshold']:<8.2f} | {r['precision']*100:>9.1f}% | {r['recall']*100:>7.1f}% | {r['f1']*100:>7.1f}% | {r['type_acc']*100:>7.1f}% | "
              f"{r['pii_free_flagged']}/{r['pii_free_total']} ({r['pii_free_fpr']*100:>4.1f}%) | {r['person_to_username']:>8} | {r['username_to_person']:>8} | {r['redactable_fn']:>13} | {r['critical_fn']:>11}")
    print("-" * 128)

    print("\n" + "=" * 116)
    print("  Hard Cases Detailed Predictions Across Thresholds")
    print("=" * 116)
    case_names = {
        "case_001": "Case 1: 张三今天来了，稍后张三又打电话过来。 (Gold: 2x CN_NAME)",
        "case_002": "Case 2: 张三，用户名是李四。 (Gold: 张三 CN_NAME, 李四 USERNAME)",
        "case_003": "Case 3: 用户名是张三。 (Gold: 张三 USERNAME)",
        "case_004": "Case 4: 张三是用户名。 (Gold: 张三 USERNAME)",
    }
    for cid, title in case_names.items():
        print(f"\n[{cid}] {title}")
        for entry in hard_cases.get(cid, []):
            t = entry["threshold"]
            preds = entry["preds"]
            pred_strs = [f"{etype}('{ptext}', {score:.3f})" for etype, ptext, score in preds]
            print(f"  t={t:.2f} -> {', '.join(pred_strs) if pred_strs else 'None'}")
    print("=" * 116 + "\n")



if __name__ == "__main__":
    raise SystemExit(main())

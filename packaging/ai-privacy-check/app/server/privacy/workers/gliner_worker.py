#!/usr/bin/env python3
"""Isolated GLiNER Inference Worker for AI Privacy Check.

Runs exclusively within an isolated PyTorch runtime virtual environment (torch-cpu or torch-cuda).
Communicates strictly via JSONL over stdin/stdout.
Enforces offline inference mode and never outputs raw sensitive user text to logs.
"""

import json
import os
import sys
import traceback

# Enforce offline inference mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"

_ACTIVE_MODEL = None
_ACTIVE_MODEL_PATH = None
_ACTIVE_DEVICE = None


def get_gliner_model(model_path: str, device: str = "cpu"):
    global _ACTIVE_MODEL, _ACTIVE_MODEL_PATH, _ACTIVE_DEVICE
    target_device = "cuda" if device == "cuda" else "cpu"
    if _ACTIVE_MODEL is not None and _ACTIVE_MODEL_PATH == model_path and _ACTIVE_DEVICE == target_device:
        return _ACTIVE_MODEL

    from gliner import GLiNER  # type: ignore

    model = GLiNER.from_pretrained(model_path, local_files_only=True)
    if target_device == "cuda":
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                model = model.to("cuda")
                target_device = "cuda"
            else:
                target_device = "cpu"
        except Exception:
            target_device = "cpu"
    else:
        model = model.to("cpu")

    _ACTIVE_MODEL = model
    _ACTIVE_MODEL_PATH = model_path
    _ACTIVE_DEVICE = target_device
    return _ACTIVE_MODEL


def handle_request(req: dict) -> dict:
    action = req.get("action", "detect")

    if action == "ping":
        return {"ok": True, "status": "pong", "device": _ACTIVE_DEVICE}

    model_path = req.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        return {"ok": False, "error_type": "ValueError", "error": f"Invalid model path: {model_path}"}

    device = req.get("device", "cpu")

    if action == "load":
        get_gliner_model(model_path, device=device)
        return {"ok": True, "device": _ACTIVE_DEVICE}

    if action == "detect":
        text = req.get("text", "")
        if not text:
            return {"ok": True, "entities": []}

        labels = req.get("labels", [])
        if not labels:
            labels = [
                "person",
                "organization",
                "phone number",
                "email",
                "passport number",
                "driver license",
                "social security number",
                "credit card number",
                "bank account",
                "address",
                "username",
                "date of birth",
            ]
        threshold = float(req.get("threshold", 0.3))

        model = get_gliner_model(model_path, device=device)
        raw_entities = model.predict_entities(text, labels, threshold=threshold)

        sanitized_entities = []
        for ent in raw_entities:
            s = int(ent.get("start", 0))
            e = int(ent.get("end", 0))
            ent_text = ent.get("text", "")
            lbl = str(ent.get("label", "")).strip()
            score = float(ent.get("score", 0.9))

            # Enforce exact character offset invariant
            if s >= 0 and e <= len(text) and s < e and text[s:e] == ent_text:
                sanitized_entities.append({
                    "start": s,
                    "end": e,
                    "text": ent_text,
                    "label": lbl,
                    "score": score,
                })
            else:
                # Fallback: find if offset drifted slightly
                pass

        return {"ok": True, "entities": sanitized_entities}

    return {"ok": False, "error_type": "ValueError", "error": f"Unknown action: {action}"}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
        except Exception as exc:
            resp = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

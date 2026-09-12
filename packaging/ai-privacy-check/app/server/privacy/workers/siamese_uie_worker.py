#!/usr/bin/env python3
"""Isolated SiameseUIE Inference Worker for AI Privacy Check.

Uses ModelScope's PyTorch pipeline ('siamese-uie') within isolated PyTorch runtime.
Extracts structured Chinese entities (人名, 地理位置, 组织机构, 学校, 职位) with exact character spans.
"""

import json
import os
import sys

# Enforce offline inference mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"

_ACTIVE_PIPELINE = None
_ACTIVE_MODEL_PATH = None
_ACTIVE_DEVICE = None

DEFAULT_SCHEMA = {
    "人物": None,
    "地理位置": None,
    "组织机构": None,
    "学校": None,
    "职位": None,
}


def get_pipeline(model_path: str, device: str = "cpu"):
    global _ACTIVE_PIPELINE, _ACTIVE_MODEL_PATH, _ACTIVE_DEVICE
    target_device = "cuda:0" if device == "cuda" else "cpu"
    if _ACTIVE_PIPELINE is not None and _ACTIVE_MODEL_PATH == model_path and _ACTIVE_DEVICE == target_device:
        return _ACTIVE_PIPELINE

    from modelscope.pipelines import pipeline  # type: ignore
    from modelscope.utils.constant import Tasks  # type: ignore

    # Deliberately NO trust_remote_code here: AIPrivacyCheck refuses to import
    # or execute code shipped inside model directories. The installer strips
    # `allow_remote`/`plugins` from configuration.json at install time, so the
    # catalog siamese-uie checkpoint loads through ModelScope built-in
    # pipeline/model/preprocessor classes only.
    pipe = pipeline(
        Tasks.siamese_uie,
        model=model_path,
        device=target_device,
    )
    _ACTIVE_PIPELINE = pipe
    _ACTIVE_MODEL_PATH = model_path
    _ACTIVE_DEVICE = target_device
    return _ACTIVE_PIPELINE


def extract_entities_from_output(raw_output, text: str) -> list:
    """Normalizes ModelScope SiameseUIE output into standard span objects."""
    entities = []
    # Output can be:
    # 1. {'output': [[{'type': '地理位置', 'span': '...', 'offset': [7, 19]}]]}
    #    (newer ModelScope: one inner list per schema key, half-open offsets)
    # 2. [{'type': '人物', 'span': '张三', 'start': 0, 'end': 2, 'probability': 0.98}]
    # 3. {'output': [{'type': '人物', 'span': '张三', ...}]}
    # 4. {'人物': [{'span': '张三', 'start': 0, 'end': 2}]}
    items = []
    if isinstance(raw_output, dict):
        if "output" in raw_output and isinstance(raw_output["output"], list):
            flattened = []
            for group in raw_output["output"]:
                if isinstance(group, list):
                    flattened.extend(group)
                else:
                    flattened.append(group)
            items = flattened
        else:
            for label, val_list in raw_output.items():
                if isinstance(val_list, list):
                    for v in val_list:
                        if isinstance(v, dict):
                            item = dict(v)
                            item.setdefault("type", label)
                            items.append(item)
                        elif isinstance(v, str):
                            items.append({"type": label, "span": v})
    elif isinstance(raw_output, list):
        items = raw_output

    cursor_by_span = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("type") or item.get("label") or "实体"
        span = item.get("span") or item.get("text") or item.get("entity") or ""
        if not span:
            continue

        score = float(item.get("probability") or item.get("score") or 0.92)

        start = item.get("start")
        end = item.get("end")
        if start is None and isinstance(item.get("offset"), (list, tuple)) and len(item["offset"]) == 2:
            start, end = item["offset"]

        if start is not None and end is not None:
            s = int(start)
            e = int(end)
            if 0 <= s < e <= len(text) and text[s:e] == span:
                entities.append({
                    "start": s,
                    "end": e,
                    "text": span,
                    "label": label,
                    "score": score,
                })
                continue

        # Sequential cursor lookup to avoid repetitive first-occurrence trap
        start_search = cursor_by_span.get(span, 0)
        found_idx = text.find(span, start_search)
        if found_idx != -1:
            s = found_idx
            e = found_idx + len(span)
            cursor_by_span[span] = e
            entities.append({
                "start": s,
                "end": e,
                "text": span,
                "label": label,
                "score": score,
            })

    return entities


def handle_request(req: dict) -> dict:
    action = req.get("action", "detect")

    if action == "ping":
        return {"ok": True, "status": "pong", "device": _ACTIVE_DEVICE}

    model_path = req.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        return {"ok": False, "error_type": "ValueError", "error": f"Invalid model path: {model_path}"}

    device = req.get("device", "cpu")

    if action == "load":
        get_pipeline(model_path, device=device)
        return {"ok": True, "device": _ACTIVE_DEVICE}

    if action == "detect":
        text = req.get("text", "")
        if not text:
            return {"ok": True, "entities": []}

        schema = req.get("schema") or DEFAULT_SCHEMA
        pipe = get_pipeline(model_path, device=device)
        raw_res = pipe(input=text, schema=schema)
        entities = extract_entities_from_output(raw_res, text)
        return {"ok": True, "entities": entities}

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

#!/usr/bin/env python3
"""Isolated MemPrivacy Inference Worker for AI Privacy Check.

Runs in isolated PyTorch runtime. Uses AutoTokenizer and AutoModelForCausalLM.
Loads prompt from model_path/privacy_prompt.txt, formats messages with user real_name and dialogue text,
strips reasoning <think> tags, parses JSON entities, and returns structured results.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Enforce offline inference mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"

_ACTIVE_MODEL = None
_ACTIVE_TOKENIZER = None
_ACTIVE_MODEL_PATH = None
_ACTIVE_DEVICE = None

# AIPrivacyCheck Semantic Privacy Extraction Prompt (Apache-2.0)
# Independently authored for AIPrivacyCheck local extraction.
AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT = """You are a precise data privacy inspection assistant.
Your task is to analyze the provided input text, identify privacy-sensitive spans, and classify their sensitivity level.

Classification Categories:
- PL2: Personal identifiable data (direct identifiers, contact details, account handles, demographic facts).
- PL3: High-sensitivity personal data (official identity numbers, financial accounts, health records, biometric indicators, precise private locations).
- PL4: Critical security secrets (passwords, tokens, API credentials, private encryption keys, authentication material).

Output Rules:
1. Extract only the minimal sensitive span; never return full sentences.
2. Return strictly a JSON array without additional commentary or Markdown formatting outside JSON.
3. Each item must have:
   - "original_text": exact substring from input
   - "privacy_type": semantic category tag
   - "privacy_level": "PL2", "PL3", or "PL4"
4. If no privacy data is found, return [].
"""

DEFAULT_SYSTEM_PROMPT = AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT


def load_prompt_template(model_path: str) -> str:
    prompt_file = os.path.join(model_path, "privacy_prompt.txt")
    if os.path.isfile(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return DEFAULT_SYSTEM_PROMPT


def get_model_and_tokenizer(model_path: str, device: str = "cpu"):
    global _ACTIVE_MODEL, _ACTIVE_TOKENIZER, _ACTIVE_MODEL_PATH, _ACTIVE_DEVICE
    target_device = "cuda" if device == "cuda" else "cpu"

    if _ACTIVE_MODEL is not None and _ACTIVE_TOKENIZER is not None and _ACTIVE_MODEL_PATH == model_path and _ACTIVE_DEVICE == target_device:
        return _ACTIVE_MODEL, _ACTIVE_TOKENIZER

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if target_device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
        )
        _ACTIVE_DEVICE = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model = model.to("cpu")
        _ACTIVE_DEVICE = "cpu"

    _ACTIVE_MODEL = model
    _ACTIVE_TOKENIZER = tokenizer
    _ACTIVE_MODEL_PATH = model_path
    return _ACTIVE_MODEL, _ACTIVE_TOKENIZER


def strip_think_tags(text: str) -> str:
    """Removes Qwen <think>...</think> reasoning blocks from generated text."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also handle unclosed <think> if truncated
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


def parse_extracted_json(raw_output: str, original_text: str) -> Tuple[list, bool, Optional[str]]:
    """Extracts and sanitizes JSON entity array from model output.

    Returns:
        (entities, has_parse_error, error_detail)
    """
    cleaned = strip_think_tags(raw_output)
    if not cleaned:
        return [], False, None

    # Check for markdown code fence
    cand = cleaned
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence_match:
        cand = fence_match.group(1).strip()
    else:
        array_match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if array_match:
            cand = array_match.group(0).strip()
        else:
            # Check if empty array was output: '[]'
            empty_match = re.search(r"\[\s*\]", cleaned)
            if empty_match:
                return [], False, None

    # Clean trailing commas
    cand_cleaned = re.sub(r",\s*([\]\}])", r"\1", cand)
    data = None
    try:
        data = json.loads(cand_cleaned)
    except Exception:
        # Fallback: extract individual JSON objects
        extracted_items = []
        for obj_m in re.finditer(r"\{[^{}]*\}", cand_cleaned):
            try:
                item = json.loads(re.sub(r",\s*\}", "}", obj_m.group(0)))
                extracted_items.append(item)
            except Exception:
                continue
        if extracted_items:
            data = extracted_items

    if data is None:
        # Generated non-empty text that failed to parse into JSON
        return [], True, f"Failed to parse generated text as JSON array: {cleaned[:150]}"

    if not isinstance(data, list):
        return [], True, f"Model output JSON is not a list (got {type(data).__name__})"

    entities = []
    for item in data:
        if not isinstance(item, dict):
            continue
        orig = item.get("original_text") or item.get("text") or item.get("entity") or ""
        if not isinstance(orig, str) or not orig.strip():
            continue
        orig_clean = orig.strip()
        if orig_clean not in original_text:
            continue

        ptype = str(item.get("privacy_type") or item.get("type") or "PII").strip()
        plevel = str(item.get("privacy_level") or item.get("level") or "PL2").strip().upper()
        entities.append({
            "original_text": orig_clean,
            "privacy_type": ptype,
            "privacy_level": plevel,
            "context": item.get("context"),
        })

    return entities, False, None


def handle_request(req: dict) -> dict:
    action = req.get("action", "detect")

    if action == "ping":
        return {"ok": True, "status": "pong", "device": _ACTIVE_DEVICE}

    model_path = req.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        return {"ok": False, "error_type": "ValueError", "error": f"Invalid model path: {model_path}"}

    device = req.get("device", "cpu")

    if action == "load":
        get_model_and_tokenizer(model_path, device=device)
        return {"ok": True, "device": _ACTIVE_DEVICE}

    if action == "detect":
        text = req.get("text", "")
        if not text:
            return {"ok": True, "entities": []}

        real_name = req.get("real_name", "unknown")
        max_new_tokens = int(req.get("max_new_tokens", 256))

        model, tokenizer = get_model_and_tokenizer(model_path, device=device)
        system_prompt = load_prompt_template(model_path)

        user_content = f"User Name: {real_name}\nDialogue Text: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer([prompt_str], return_tensors="pt")
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        import torch  # type: ignore
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        is_truncated = len(gen_tokens) >= max_new_tokens
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        entities, parse_err, err_msg = parse_extracted_json(gen_text, text)
        if parse_err:
            if is_truncated and entities:
                return {
                    "ok": True,
                    "entities": entities,
                    "truncated": True,
                }
            return {
                "ok": False,
                "error_type": "ModelOutputParseError",
                "error": err_msg or "Model output could not be parsed as JSON",
            }

        return {
            "ok": True,
            "entities": entities,
            "truncated": is_truncated,
        }

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

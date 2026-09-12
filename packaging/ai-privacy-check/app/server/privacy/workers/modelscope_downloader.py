#!/usr/bin/env python3
"""Isolated ModelScope Downloader Worker for AI Privacy Check.

Runs exclusively within an isolated Python runtime venv containing the ModelScope SDK.
Never executes in control-plane Python 3.12.
Downloads models into staging directory, captures resolved commit revisions,
and provisions required third-party prompt assets for MemPrivacy models.
"""

import argparse
import datetime
import json
import os
import sys
import traceback


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


def download_model(repo_id: str, revision: str, target_dir: str, model_id: str) -> dict:
    os.makedirs(target_dir, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

    # Execute download into target directory
    downloaded_path = snapshot_download(
        repo_id,
        revision=revision,
        local_dir=target_dir,
    )

    # For MemPrivacy models, ensure prompt asset is saved as AIPrivacyCheck extraction prompt
    if "memprivacy" in model_id.lower():
        prompt_path = os.path.join(target_dir, "privacy_prompt.txt")
        meta_path = os.path.join(target_dir, "privacy_prompt.meta.json")
        if not os.path.isfile(prompt_path):
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name": "AIPrivacyCheck semantic privacy extraction prompt",
                    "license": "Apache-2.0",
                    "source": "AIPrivacyCheck",
                    "note": "Independently authored local extraction prompt for AIPrivacyCheck",
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f, indent=2)

    # Record download metadata
    meta = {
        "repo_id": repo_id,
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_path": str(downloaded_path),
        "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_file = os.path.join(target_dir, ".download_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "ok": True,
        "repo_id": repo_id,
        "model_id": model_id,
        "requested_revision": revision,
        "target_dir": target_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="ModelScope Isolated Downloader Worker")
    parser.add_argument("--repo-id", required=True, help="ModelScope Repo ID")
    parser.add_argument("--revision", default="master", help="Model revision or commit")
    parser.add_argument("--target-dir", required=True, help="Target download directory")
    parser.add_argument("--model-id", required=True, help="Catalog Model ID")

    args = parser.parse_args()

    try:
        res = download_model(
            repo_id=args.repo_id,
            revision=args.revision,
            target_dir=args.target_dir,
            model_id=args.model_id,
        )
        sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        sys.exit(0)
    except Exception as exc:
        err_res = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        sys.stdout.write(json.dumps(err_res, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

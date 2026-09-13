#!/usr/bin/env python3
"""Selective Model Downloader for AI Privacy Check Benchmarks (v0.6.9).

Strict selective download policy:
- NEVER downloads the entire repository snapshot by default.
- Statically enumerates remote repository files via ModelScope REST API.
- Categorizes files into Required, Optional, and Rejected/Unnecessary.
- Downloads ONLY the minimal set of files required for local PyTorch inference.
- Verifies local file sizes / integrity and reuses existing files.
- Generates a persistent download-manifest.json recording downloaded file details.
- Prohibits downloading expensive or out-of-scope challenger models (MemPrivacy, Qwen, AIguard, etc.).
- Strictly requires explicit --download-missing; otherwise reports MODEL NOT INSTALLED and skips.
"""

from dataclasses import dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request
import urllib.error

from model_integrity import (
    ModelIntegrityError,
    sha256_file,
    validate_manifest_schema,
    verify_existing_model_integrity,
)

# Allowed models for selective download in benchmark
DOWNLOADABLE_MODEL_CATALOG = {
    "gliner-pii-edge": {
        "repo_id": "knowledgator/gliner-pii-edge-v1.0",
        "revision": "master",
        "required_files": {
            "gliner_config.json",
            "configuration.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        },
        "forbidden_files_prefix": ("onnx", ".git", "README"),
    },
}

# Backward compatibility alias
BENCHMARK_MODEL_CATALOG = DOWNLOADABLE_MODEL_CATALOG

# Blocked challenger models: strictly prohibited from downloading in this release
BLOCKED_MODELS = {
    "memprivacy-1.7b",
    "memprivacy-4b",
    "openai-privacy-filter",
    "aiguard",
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "raner",
    "vault-engine",
}


@dataclass
class RemoteFileInfo:
    path: str
    size: int
    type: str  # 'blob' or 'tree'


def query_modelscope_repo_files(repo_id: str, revision: str = "master", root: str = "") -> List[RemoteFileInfo]:
    """Queries ModelScope repository file list via public REST API."""
    url = f"https://modelscope.cn/api/v1/models/{repo_id}/repo/files?Revision={revision}"
    if root:
        url += f"&Root={urllib.parse.quote(root)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AIPrivacyCheck-Benchmark/0.6.9"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if data.get("Code") != 200:
            raise RuntimeError(f"ModelScope API error ({data.get('Code')}): {data.get('Message')}")
        files = data.get("Data", {}).get("Files", [])
        return [
            RemoteFileInfo(
                path=f["Path"] if not root else f"{root}/{f['Path']}",
                size=int(f.get("Size", 0)),
                type=f.get("Type", "blob"),
            )
            for f in files
        ]


def categorize_files(
    repo_files: List[RemoteFileInfo],
    required_names: Set[str],
    forbidden_prefixes: Tuple[str, ...],
) -> Tuple[List[RemoteFileInfo], List[RemoteFileInfo], List[RemoteFileInfo]]:
    """Partitions repository files into required, optional, and rejected."""
    required = []
    optional = []
    rejected = []

    for f in repo_files:
        p = f.path.strip("/")
        base = os.path.basename(p)
        if any(p.startswith(pref) for pref in forbidden_prefixes) or base.startswith(".git") or base.endswith(".md"):
            rejected.append(f)
        elif base in required_names or p in required_names:
            required.append(f)
        else:
            optional.append(f)

    return required, optional, rejected




def download_single_file(
    repo_id: str,
    revision: str,
    remote_path: str,
    target_path: Path,
    expected_size: int,
    expected_sha256: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[bool, int, str]:
    """Downloads a single file from ModelScope with integrity check.

    When force_download=True, existing target_path content is never reused.
    When force_download=False, existing file is reused ONLY if expected_sha256
    is provided and matches actual file SHA-256.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if existing file is already verified
    if not force_download and target_path.is_file() and target_path.stat().st_size == expected_size:
        actual_sha = sha256_file(target_path)
        if expected_sha256 is not None and actual_sha == expected_sha256:
            print(f"  [REUSE] {remote_path} ({expected_size:,} bytes, verified)")
            return True, expected_size, actual_sha

    temp_path = target_path.with_name(target_path.name + ".tmp_download")
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)

    url = f"https://modelscope.cn/api/v1/models/{repo_id}/repo?Revision={revision}&FilePath={urllib.parse.quote(remote_path)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AIPrivacyCheck-Benchmark/0.6.9"})

    h = hashlib.sha256()
    downloaded = 0
    success = False
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(temp_path, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    downloaded += len(chunk)
                    if expected_size > 0:
                        pct = downloaded / expected_size * 100
                        sys.stdout.write(f"\r  [DOWNLOADING] {remote_path} ... {downloaded:,}/{expected_size:,} bytes ({pct:.1f}%)")
                        sys.stdout.flush()

        sys.stdout.write("\n")
        if expected_size > 0 and downloaded != expected_size:
            raise IOError(f"Downloaded size mismatch for {remote_path}: got {downloaded}, expected {expected_size}")
        digest = h.hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise IOError(f"Downloaded SHA-256 mismatch for {remote_path}: got {digest}, expected {expected_sha256}")

        os.replace(temp_path, target_path)
        success = True
        return True, downloaded, digest
    finally:
        if not success:
            temp_path.unlink(missing_ok=True)


def ensure_selective_model(
    model_name: str,
    models_dir: Path,
    download_missing: bool = False,
) -> Tuple[bool, str, Optional[Dict]]:
    """Ensures a model is present under models_dir using selective download.

    Returns: (success: bool, status_message: str, manifest: dict|None)
    """
    if model_name in BLOCKED_MODELS:
        return False, (
            f"FORBIDDEN: Re-downloading '{model_name}' is strictly prohibited "
            "(Historical challenger model policy: historical results preserved, no expensive re-download)"
        ), None

    spec = DOWNLOADABLE_MODEL_CATALOG.get(model_name)
    if not spec:
        # Check if model folder exists locally
        target_dir = models_dir / model_name
        if target_dir.is_dir():
            return True, f"Custom model directory {target_dir} exists", None
        return False, f"Unknown model '{model_name}' and folder does not exist", None

    repo_id = spec["repo_id"]
    revision = spec.get("revision", "master")
    target_dir = models_dir / model_name
    required_names = spec["required_files"]
    forbidden_prefixes = spec.get("forbidden_files_prefix", ())
    manifest_file = target_dir / "download-manifest.json"

    files_to_download_names = set(required_names)

    # Check if target already has verified files
    if target_dir.is_dir():
        if manifest_file.is_file():
            is_valid, msg, manifest, corrupted = verify_existing_model_integrity(target_dir, required_names)
            if is_valid:
                return True, f"Model '{model_name}' already installed and complete ({len(required_names)} files verified against manifest)", manifest
            if not download_missing:
                return False, f"MODEL CORRUPTED: '{model_name}' failed integrity check ({msg}). Pass --download-missing to repair.", None
            print(f"Model integrity check failed: {msg}. Repairing {len(corrupted)} files...")
            files_to_download_names = set(corrupted)
        else:
            if not download_missing:
                return False, f"UNVERIFIED MODEL DIRECTORY: '{model_name}' exists under {models_dir} without download-manifest.json. Pass --download-missing to verify.", None
            files_to_download_names = set(required_names)
    else:
        if not download_missing:
            return False, f"MODEL NOT INSTALLED: '{model_name}' is missing under {models_dir} (pass --download-missing to selectively download)", None
        files_to_download_names = set(required_names)

    print("\n" + "=" * 80)
    print(f"  Selective Model Downloader: {model_name} ({repo_id})")
    print("=" * 80)
    print("Querying remote repository file list via ModelScope API...")
    remote_files = query_modelscope_repo_files(repo_id, revision)

    # Check for onnx directory or nested files to report skipped bytes
    onnx_files = []
    try:
        onnx_files = query_modelscope_repo_files(repo_id, revision, root="onnx")
    except Exception:
        pass
    all_remote = []
    seen_paths = set()
    for f in remote_files + onnx_files:
        if f.path not in seen_paths:
            seen_paths.add(f.path)
            all_remote.append(f)

    required, optional, rejected = categorize_files(all_remote, required_names, forbidden_prefixes)

    total_repo_bytes = sum(f.size for f in all_remote)
    selected_bytes = sum(f.size for f in required)
    skipped_bytes = total_repo_bytes - selected_bytes

    print(f"Repository total files : {len(all_remote)}")
    print(f"Selected required files: {len(required)} ({selected_bytes:,} bytes / {selected_bytes/1048576:.1f} MB)")
    print(f"Skipped / rejected files: {len(rejected) + len(optional)} ({skipped_bytes:,} bytes / {skipped_bytes/1048576:.1f} MB)")
    print("-" * 80)
    print("Required files for inference:")
    for f in required:
        print(f"  + {f.path:<30} {f.size:>12,} bytes")
    print("\nRejected / unnecessary files (will NOT be downloaded):")
    for f in (rejected + optional)[:8]:
        print(f"  - {f.path:<30} {f.size:>12,} bytes (REJECTED)")
    if len(rejected + optional) > 8:
        print(f"    ... and {len(rejected + optional) - 8} more skipped files")
    print("-" * 80)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Preserve existing valid records from manifest if repairing
    existing_records: Dict[str, Dict[str, Any]] = {}
    if manifest_file.is_file():
        try:
            m = json.loads(manifest_file.read_text(encoding="utf-8"))
            f_list = m.get("files", [])
            if isinstance(f_list, list):
                for item in f_list:
                    if isinstance(item, dict) and "path" in item:
                        existing_records[item["path"]] = item
        except Exception:
            pass

    downloaded_files_record = []
    total_downloaded = 0
    upstream_content_changed = False

    for f in required:
        dest = target_dir / f.path
        if f.path not in files_to_download_names and dest.is_file() and f.path in existing_records:
            # File is verified and not corrupted
            rec = existing_records[f.path]
            downloaded_files_record.append(rec)
            continue

        # File is missing or corrupted: unconditionally force download to bypass corrupted local reuse
        ok, sz, digest = download_single_file(
            repo_id,
            revision,
            f.path,
            dest,
            f.size,
            expected_sha256=None,
            force_download=True,
        )
        rec = {
            "path": f.path,
            "size": sz,
            "sha256": digest,
        }
        downloaded_files_record.append(rec)
        total_downloaded += sz

        # Check if upstream content changed compared to existing manifest
        if f.path in existing_records:
            prev_sha = existing_records[f.path].get("sha256")
            if prev_sha and prev_sha.lower() != digest.lower():
                upstream_content_changed = True
                print(
                    f"  [WARNING] The mutable ModelScope revision returned content for '{f.path}' "
                    f"different from the previous locally recorded fingerprint ({prev_sha} -> {digest})"
                )

    manifest_data = {
        "repo_id": repo_id,
        "revision": revision,
        "requested_revision": revision,
        "resolved_revision": None,  # Upstream immutable revision not available through current endpoint
        "revision_provenance_note": "Upstream immutable revision not available through the current endpoint.",
        "upstream_content_changed": upstream_content_changed,
        "integrity_policy": "local_content_integrity_fingerprint",
        "files": downloaded_files_record,
        "total_download_bytes": total_downloaded,
        "purpose": "benchmark",
        "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selective_download_policy": "minimal_pytorch_inference_only",
        "repo_total_files": len(all_remote),
        "selected_files_count": len(required),
        "skipped_bytes": skipped_bytes,
    }

    manifest_file.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Download complete: {manifest_file} written ({total_downloaded:,} bytes downloaded)")
    print("=" * 80 + "\n")
    return True, f"Successfully selectively downloaded {len(required)} files ({total_downloaded:,} bytes)", manifest_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Selective Benchmark Model Downloader")
    parser.add_argument("--model", required=True, help="Model name (e.g. gliner-pii-edge)")
    parser.add_argument("--models-dir", default="/tmp/aipc-models", help="Target models directory")
    parser.add_argument("--download-missing", action="store_true", help="Allow downloading if missing")
    args = parser.parse_args()

    ok, msg, manifest = ensure_selective_model(args.model, Path(args.models_dir), args.download_missing)
    print(msg)
    sys.exit(0 if ok else 1)

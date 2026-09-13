#!/usr/bin/env python3
"""Centralized Model Integrity & Content Fingerprinting Module (v0.6.8).

Single Source of Truth for:
- Streaming SHA-256 computation (sha256_file).
- Strict download-manifest.json schema and path traversal validation.
- Disk content verification against manifest (size and per-file SHA256).
- Model content fingerprint generation (Case A: verified manifest, Case B: streaming fallback, Case C: raise ModelIntegrityError).
"""

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
CANONICAL_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelIntegrityError(Exception):
    """Raised when model files fail integrity verification or manifest is corrupted."""
    pass


def sha256_file(path: Path) -> str:
    """Computes streaming SHA-256 for a local file in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest_schema(
    manifest: Any,
    target_dir: Optional[Path] = None,
) -> Tuple[bool, str, Dict[str, Dict[str, Any]]]:
    """Validates the structure and entries of download-manifest.json.

    Enforces:
    1. manifest is a dictionary.
    2. 'files' is a list or dictionary of file entries.
    3. Each entry has:
       - non-empty 'path' without path traversal (no '..', no leading '/', resolves inside target_dir).
       - non-negative integer 'size'.
       - 64-hex character 'sha256'.

    Returns:
        (is_valid: bool, error_message: str, manifest_files_map: Dict[path, entry_dict])
    """
    if not isinstance(manifest, dict):
        return False, "manifest must be a JSON object", {}

    files_entry = manifest.get("files")
    if files_entry is None:
        return False, "manifest missing 'files' field", {}

    raw_items: List[Dict[str, Any]] = []
    if isinstance(files_entry, list):
        for item in files_entry:
            if isinstance(item, dict):
                raw_items.append(item)
            else:
                return False, f"manifest 'files' list contains non-dict item: {item!r}", {}
    elif isinstance(files_entry, dict):
        for path_key, item in files_entry.items():
            if isinstance(item, dict):
                item_copy = dict(item)
                item_copy.setdefault("path", path_key)
                raw_items.append(item_copy)
            else:
                return False, f"manifest 'files' dict contains non-dict value for key '{path_key}': {item!r}", {}
    else:
        return False, f"manifest 'files' must be a list or object, got {type(files_entry).__name__}", {}

    manifest_files_map: Dict[str, Dict[str, Any]] = {}

    target_resolved = target_dir.resolve() if target_dir else None

    for entry in raw_items:
        rel_path = entry.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            return False, f"manifest entry missing or empty 'path': {entry!r}", {}

        # Path traversal and format hardening
        cleaned_path = rel_path.replace("\\", "/").strip()
        if WINDOWS_DRIVE_RE.match(rel_path) or WINDOWS_DRIVE_RE.match(cleaned_path):
            return False, f"manifest path traversal forbidden (Windows drive path): '{rel_path}'", {}

        if rel_path.startswith(("\\\\", "//")) or cleaned_path.startswith("//"):
            return False, f"manifest path traversal forbidden (UNC path): '{rel_path}'", {}

        if cleaned_path.startswith("/") or Path(cleaned_path).is_absolute():
            return False, f"manifest path traversal forbidden (absolute path): '{rel_path}'", {}

        norm = os.path.normpath(cleaned_path)
        if norm == ".." or norm.startswith(".." + os.sep) or norm.startswith("../") or "/../" in cleaned_path or norm.startswith(".."):
            return False, f"manifest path traversal forbidden ('..'): '{rel_path}'", {}

        if target_resolved is not None:
            resolved_file = (target_resolved / norm).resolve()
            try:
                resolved_file.relative_to(target_resolved)
            except ValueError:
                return False, f"manifest path escapes target directory: '{rel_path}'", {}

        # Size validation: non-negative integer (not bool)
        sz = entry.get("size")
        if type(sz) is not int or sz < 0:
            return False, f"manifest entry for '{rel_path}' invalid 'size': expected non-negative int, got {sz!r}", {}

        # SHA-256 validation: canonical lowercase 64 hex characters
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not CANONICAL_SHA256_RE.match(sha):
            return False, f"manifest entry for '{rel_path}' invalid 'sha256': expected canonical lowercase 64-char hex, got {sha!r}", {}

        manifest_files_map[norm] = entry

    return True, "manifest schema valid", manifest_files_map


def verify_existing_model_integrity(
    target_dir: Path,
    required_names: Optional[Set[str]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]], List[str]]:
    """Verifies local target_dir against download-manifest.json.

    Validates:
    - download-manifest.json exists and satisfies strict schema.
    - Every required file (or all declared files) exists on disk.
    - File size on disk matches manifest size.
    - File actual SHA-256 on disk matches manifest sha256.

    Returns:
        (is_valid: bool, status_message: str, manifest_data: dict|None, corrupted_files: list)
    """
    manifest_file = target_dir / "download-manifest.json"
    if not manifest_file.is_file():
        missing_list = sorted(required_names) if required_names else []
        return False, "missing download-manifest.json", None, missing_list

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        missing_list = sorted(required_names) if required_names else []
        return False, f"corrupted download-manifest.json ({exc})", None, missing_list

    schema_ok, schema_msg, manifest_files_map = validate_manifest_schema(manifest, target_dir)
    if not schema_ok:
        missing_list = sorted(required_names) if required_names else []
        return False, f"invalid manifest schema: {schema_msg}", manifest, missing_list

    if required_names is not None:
        files_to_check = set(required_names)
        missing_declarations = [f for f in files_to_check if f not in manifest_files_map]
        if missing_declarations:
            return False, f"manifest missing required file declarations: {missing_declarations}", manifest, missing_declarations
    else:
        files_to_check = set(manifest_files_map.keys())

    corrupted = []
    for fname in sorted(files_to_check):
        fpath = target_dir / fname
        if not fpath.is_file():
            corrupted.append(fname)
            continue
        entry = manifest_files_map[fname]
        expected_size = entry["size"]
        expected_sha = entry["sha256"].lower()

        if fpath.stat().st_size != expected_size:
            corrupted.append(fname)
            continue

        actual_sha = sha256_file(fpath).lower()
        if actual_sha != expected_sha:
            corrupted.append(fname)
            continue

    if corrupted:
        return False, f"{len(corrupted)} file(s) failed size/hash integrity: {corrupted}", manifest, corrupted

    return True, f"all {len(files_to_check)} required files verified against manifest", manifest, []


def compute_model_files_hash(
    model_dir: Path,
    required_names: Optional[Set[str]] = None,
) -> str:
    """Computes deterministic compound SHA-256 over model files content.

    Case A (Preferred): download-manifest.json exists AND passes full disk integrity
        (exists, size match, actual SHA-256 matches manifest SHA-256 for all declared files).
        Fast compound fingerprint: SHA256(sorted(path + ":" + size + ":" + sha256)).
    Case B (Fallback): download-manifest.json is missing entirely.
        Streams actual model files on disk, computes streaming SHA-256 for each.
        Never relies on mtime or size alone.
    Case C: download-manifest.json exists but verification fails (corruption, mismatch,
        tampering, missing sha256, path traversal).
        RAISES ModelIntegrityError. Never silently falls back to compute hash over corrupted weights.
    """
    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / "download-manifest.json"

    if manifest_path.is_file():
        is_valid, msg, manifest, corrupted = verify_existing_model_integrity(model_dir, required_names)
        if not is_valid:
            raise ModelIntegrityError(f"Model integrity verification failed for '{model_dir}': {msg}")

        # Case A: Manifest validated and files on disk strictly match manifest
        _, _, manifest_files_map = validate_manifest_schema(manifest, model_dir)
        items = []
        for path_key in sorted(manifest_files_map.keys()):
            entry = manifest_files_map[path_key]
            items.append(f"{entry['path']}:{entry['size']}:{entry['sha256'].lower()}")

        h = hashlib.sha256()
        for line in items:
            h.update(line.encode("utf-8"))
        return h.hexdigest()

    # Case B: Manifest missing entirely -> fallback to streaming actual content
    items = []
    for root, _, files in os.walk(model_dir):
        for fname in sorted(files):
            if (
                fname.startswith(".")
                or fname.endswith(".tmp")
                or fname.endswith(".tmp_download")
                or fname == "download-manifest.json"
            ):
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(model_dir).as_posix()
            stat = fpath.stat()
            file_sha = sha256_file(fpath)
            items.append(f"{rel}:{stat.st_size}:{file_sha}")

    h = hashlib.sha256()
    for line in sorted(items):
        h.update(line.encode("utf-8"))
    return h.hexdigest()

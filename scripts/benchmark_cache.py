#!/usr/bin/env python3
"""Persistent Raw Prediction Cache for AI Privacy Check Benchmarks (v0.6.7).

Enables offline re-scoring without re-running expensive model inference:
- Cache key binds: corpus SHA256, model_id, model revision, model file hash,
  runtime version, transformers version, modelscope version, inference config,
  label list/order, prompt version, minimum inference threshold.
- Directory structure:
    benchmark-cache/
      <corpus_hash>/
        <model_id>/
          <revision>/
            inference_config.json
            runtime.json
            predictions.jsonl
            metrics.json
- GLiNER inference runs ONCE at minimum threshold (e.g. 0.30); subsequent
  threshold sweeps (0.35..0.65) execute strictly offline in pure scoring layer.
- Security: Caches ONLY synthetic benchmark corpus outputs; production user
  data is strictly forbidden from persistence.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = PROJECT_DIR / "benchmark-cache"

# Known synthetic benchmark corpus signatures allowed for caching
ALLOWED_CORPUS_FILENAMES = {
    "privacy_benchmark_v2_100.jsonl",
    "contextual_privacy_seed.jsonl",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


RUNTIME_CRITICAL_KEYS = (
    "python_version",
    "platform",
    "machine",
    "torch_version",
    "transformers_version",
    "modelscope_version",
    "gliner_version",
)


def compute_model_files_hash(model_dir: Path) -> str:
    """Computes deterministic compound SHA-256 over model files content.

    1. Preferred: if download-manifest.json exists with per-file SHA256,
       model_fingerprint = SHA256(sorted(path + size + sha256)).
    2. Fallback: computes true streaming SHA-256 over all non-temporary model files.
       Never relies on mtime.
    """
    manifest_path = model_dir / "download-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files_entry = manifest.get("files")
            items = []
            if isinstance(files_entry, list):
                for item in files_entry:
                    if isinstance(item, dict) and "path" in item and "size" in item and "sha256" in item:
                        items.append(f"{item['path']}:{item['size']}:{item['sha256']}")
            elif isinstance(files_entry, dict):
                for p, item in files_entry.items():
                    if isinstance(item, dict) and "size" in item and "sha256" in item:
                        items.append(f"{p}:{item['size']}:{item['sha256']}")
            if items:
                h = hashlib.sha256()
                for line in sorted(items):
                    h.update(line.encode("utf-8"))
                return h.hexdigest()
        except Exception:
            pass

    # Fallback: compute actual content SHA-256 for all model files
    h = hashlib.sha256()
    for root, _, files in os.walk(model_dir):
        for fname in sorted(files):
            if fname.startswith(".") or fname.endswith(".tmp") or fname.endswith(".tmp_download") or fname == "download-manifest.json":
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(model_dir).as_posix()
            stat = fpath.stat()
            file_sha = sha256_file(fpath)
            h.update(f"{rel}:{stat.st_size}:{file_sha}".encode("utf-8"))
    return h.hexdigest()


def compute_cache_signature(key_data: Dict[str, Any]) -> str:
    """Computes a deterministic 16-hex-character signature representing the execution context:
    model content fingerprint, runtime fingerprint, and inference config (threshold, labels, device).
    """
    sig_components = {
        "model_files_hash": key_data.get("model_files_hash", ""),
        "inference_config": key_data.get("inference_config", {}),
        "runtime": {
            k: key_data.get("runtime", {}).get(k)
            for k in sorted(RUNTIME_CRITICAL_KEYS)
        },
    }
    dumped = json.dumps(sig_components, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def get_runtime_environment_metadata() -> Dict[str, Any]:
    """Captures runtime environment versions for cache key binding."""
    meta = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in ("torch", "transformers", "modelscope", "gliner"):
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            meta[f"{pkg}_version"] = None
    return meta


class BenchmarkPredictionCache:
    """Manages reading and writing raw model predictions under benchmark-cache/."""

    def __init__(self, cache_root: Path = DEFAULT_CACHE_ROOT):
        self.cache_root = Path(cache_root).resolve()

    def build_cache_key(
        self,
        corpus_path: Path,
        model_id: str,
        revision: str,
        model_dir: Optional[Path],
        inference_config: Dict[str, Any],
        runtime_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        corpus_path = Path(corpus_path).resolve()
        corpus_sha = sha256_file(corpus_path)
        model_files_hash = compute_model_files_hash(model_dir) if (model_dir and model_dir.is_dir()) else "no_model_dir"
        runtime = runtime_meta or get_runtime_environment_metadata()

        key_data = {
            "corpus_sha256": corpus_sha,
            "corpus_file": corpus_path.name,
            "model_id": model_id,
            "revision": revision,
            "model_files_hash": model_files_hash,
            "inference_config": inference_config,
            "runtime": runtime,
        }
        key_data["cache_signature"] = compute_cache_signature(key_data)
        return key_data

    def get_cache_dir(
        self,
        corpus_sha256: str,
        model_id: str,
        revision: str,
        cache_signature: Optional[str] = None,
    ) -> Path:
        short_hash = corpus_sha256[:16]
        safe_model = model_id.replace("/", "_").replace("\\", "_")
        safe_rev = revision.replace("/", "_").replace("\\", "_")
        base = self.cache_root / short_hash / safe_model / safe_rev
        if cache_signature:
            return base / cache_signature[:16]
        return base

    def has_valid_cache(self, key_data: Dict[str, Any]) -> Tuple[bool, Optional[Path]]:
        corpus_sha = key_data["corpus_sha256"]
        model_id = key_data["model_id"]
        revision = key_data["revision"]
        sig = key_data.get("cache_signature") or compute_cache_signature(key_data)

        # Check signature-specific directory first, then fallback to base directory
        candidate_dirs = [
            self.get_cache_dir(corpus_sha, model_id, revision, sig),
            self.get_cache_dir(corpus_sha, model_id, revision),
        ]

        for cdir in candidate_dirs:
            cfg_file = cdir / "inference_config.json"
            preds_file = cdir / "predictions.jsonl"

            if not (cfg_file.is_file() and preds_file.is_file()):
                continue

            try:
                cached_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                # Compare key components:
                if cached_cfg.get("corpus_sha256") != corpus_sha:
                    continue
                if cached_cfg.get("model_id") != model_id:
                    continue
                if cached_cfg.get("revision") != revision:
                    continue
                if cached_cfg.get("model_files_hash") != key_data["model_files_hash"]:
                    continue
                # Check inference config match (e.g. label order, min threshold)
                if cached_cfg.get("inference_config") != key_data["inference_config"]:
                    continue
                # Check runtime fingerprint match
                cached_rt = cached_cfg.get("runtime", {})
                curr_rt = key_data.get("runtime", {})
                runtime_mismatch = False
                for rkey in RUNTIME_CRITICAL_KEYS:
                    if cached_rt.get(rkey) != curr_rt.get(rkey):
                        runtime_mismatch = True
                        break
                if runtime_mismatch:
                    continue

                return True, cdir
            except Exception:
                continue

        return False, None

    def save(
        self,
        key_data: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        corpus_file = key_data.get("corpus_file", "")
        if corpus_file and corpus_file not in ALLOWED_CORPUS_FILENAMES:
            raise SecurityError(
                f"Prediction cache security refusal: '{corpus_file}' is not a permitted synthetic benchmark fixture."
            )

        cdir = self.get_cache_dir(
            key_data["corpus_sha256"],
            key_data["model_id"],
            key_data["revision"],
            key_data.get("cache_signature"),
        )
        cdir.mkdir(parents=True, exist_ok=True)

        cfg_file = cdir / "inference_config.json"
        cfg_file.write_text(json.dumps(key_data, ensure_ascii=False, indent=2), encoding="utf-8")

        rt_file = cdir / "runtime.json"
        rt_file.write_text(json.dumps(key_data.get("runtime", {}), ensure_ascii=False, indent=2), encoding="utf-8")

        preds_file = cdir / "predictions.jsonl"
        with open(preds_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        if metrics is not None:
            metrics_file = cdir / "metrics.json"
            metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        return cdir

    def load(self, cache_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        cache_dir = Path(cache_dir).resolve()
        cfg_file = cache_dir / "inference_config.json"
        preds_file = cache_dir / "predictions.jsonl"
        metrics_file = cache_dir / "metrics.json"

        # If passed parent directory (e.g. .../master), search for signature subdirectories
        if not (cfg_file.is_file() and preds_file.is_file()):
            if cache_dir.is_dir():
                for sub in sorted(cache_dir.iterdir()):
                    if sub.is_dir() and (sub / "inference_config.json").is_file() and (sub / "predictions.jsonl").is_file():
                        cfg_file = sub / "inference_config.json"
                        preds_file = sub / "predictions.jsonl"
                        metrics_file = sub / "metrics.json"
                        cache_dir = sub
                        break

        if not (cfg_file.is_file() and preds_file.is_file()):
            raise FileNotFoundError(f"Missing cache files in {cache_dir}")

        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in preds_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else None
        return cfg, predictions, metrics


class SecurityError(Exception):
    pass

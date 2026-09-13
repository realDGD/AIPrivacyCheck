#!/usr/bin/env python3
"""Persistent Raw Prediction Cache for AI Privacy Check Benchmarks (v0.6.8).

Enables offline re-scoring without re-running expensive model inference:
- Cache key binds: corpus SHA256, model_id, model revision, model file hash,
  runtime version, transformers version, modelscope version, inference config,
  label list/order, prompt version, minimum inference threshold.
- Directory structure:
    benchmark-cache/
      <corpus_hash>/
        <model_id>/
          <revision>/
            <signature>/
              inference_config.json
              runtime.json
              predictions.jsonl
              cache-manifest.json
              metrics.json
- GLiNER inference runs ONCE at minimum threshold (e.g. 0.30); subsequent
  threshold sweeps (0.35..0.65) execute strictly offline in pure scoring layer.
- Security: Caches ONLY synthetic benchmark corpus outputs; production user
  data is strictly forbidden from persistence.
"""

from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

from model_integrity import (
    ModelIntegrityError,
    compute_model_files_hash,
    sha256_file,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = PROJECT_DIR / "benchmark-cache"

# Known synthetic benchmark corpus signatures allowed for caching
ALLOWED_CORPUS_FILENAMES = {
    "privacy_benchmark_v2_100.jsonl",
    "contextual_privacy_seed.jsonl",
}


RUNTIME_CRITICAL_KEYS = (
    "python_version",
    "platform",
    "machine",
    "torch_version",
    "transformers_version",
    "modelscope_version",
    "gliner_version",
)


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
        allow_corrupted: bool = False,
    ) -> Dict[str, Any]:
        corpus_path = Path(corpus_path).resolve()
        corpus_sha = sha256_file(corpus_path)
        if model_dir and model_dir.is_dir():
            try:
                model_files_hash = compute_model_files_hash(model_dir)
            except ModelIntegrityError:
                if allow_corrupted:
                    model_files_hash = "INVALID_CORRUPTED_MODEL"
                else:
                    raise
        else:
            model_files_hash = "no_model_dir"

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

        # Write predictions file first
        preds_file = cdir / "predictions.jsonl"
        with open(preds_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        preds_sha = sha256_file(preds_file)

        # Write cache-manifest.json recording prediction checksum and count
        cache_manifest = {
            "predictions_sha256": preds_sha,
            "prediction_count": len(predictions),
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (cdir / "cache-manifest.json").write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")

        # Save inference config with predictions metadata
        saved_key_data = dict(key_data)
        saved_key_data["predictions_sha256"] = preds_sha
        saved_key_data["prediction_count"] = len(predictions)

        cfg_file = cdir / "inference_config.json"
        cfg_file.write_text(json.dumps(saved_key_data, ensure_ascii=False, indent=2), encoding="utf-8")

        rt_file = cdir / "runtime.json"
        rt_file.write_text(json.dumps(key_data.get("runtime", {}), ensure_ascii=False, indent=2), encoding="utf-8")

        if metrics is not None:
            metrics_file = cdir / "metrics.json"
            metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        return cdir

    def load(self, cache_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        cache_dir = Path(cache_dir).resolve()
        cfg_file = cache_dir / "inference_config.json"
        preds_file = cache_dir / "predictions.jsonl"
        target_dir = cache_dir

        # If passed parent directory (e.g. .../master), search for signature subdirectories
        if not (cfg_file.is_file() and preds_file.is_file()):
            if cache_dir.is_dir():
                sig_dirs = [
                    sub for sub in sorted(cache_dir.iterdir())
                    if sub.is_dir() and (sub / "inference_config.json").is_file() and (sub / "predictions.jsonl").is_file()
                ]
                if len(sig_dirs) == 1:
                    target_dir = sig_dirs[0]
                    cfg_file = target_dir / "inference_config.json"
                    preds_file = target_dir / "predictions.jsonl"
                elif len(sig_dirs) > 1:
                    available = [s.name for s in sig_dirs]
                    raise AmbiguousCacheError(
                        f"Ambiguous cache directory '{cache_dir}': multiple signature directories found {available}. "
                        "Please specify the exact signature subdirectory."
                    )
                else:
                    raise FileNotFoundError(f"Missing cache files in {cache_dir}")
            else:
                raise FileNotFoundError(f"Missing cache files in {cache_dir}")

        if not (cfg_file.is_file() and preds_file.is_file()):
            raise FileNotFoundError(f"Missing cache files in {target_dir}")

        # Check predictions file integrity against cache-manifest.json if present
        cmanifest_file = target_dir / "cache-manifest.json"
        if cmanifest_file.is_file():
            try:
                cmanifest = json.loads(cmanifest_file.read_text(encoding="utf-8"))
                expected_sha = cmanifest.get("predictions_sha256")
                expected_count = cmanifest.get("prediction_count")
                actual_sha = sha256_file(preds_file)
                if expected_sha and actual_sha != expected_sha:
                    raise CacheCorruptedError(
                        f"Prediction cache integrity verification failed in '{target_dir}': "
                        f"predictions.jsonl checksum mismatch (got {actual_sha}, expected {expected_sha})"
                    )
            except (json.JSONDecodeError, CacheCorruptedError) as exc:
                if isinstance(exc, CacheCorruptedError):
                    raise
                raise CacheCorruptedError(f"Corrupted cache-manifest.json in '{target_dir}': {exc}")

        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in preds_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if cmanifest_file.is_file() and expected_count is not None:
            if len(predictions) != expected_count:
                raise CacheCorruptedError(
                    f"Prediction cache integrity verification failed in '{target_dir}': "
                    f"prediction count mismatch (got {len(predictions)}, expected {expected_count})"
                )

        metrics_file = target_dir / "metrics.json"
        metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else None
        return cfg, predictions, metrics


class SecurityError(Exception):
    pass


class AmbiguousCacheError(ValueError):
    """Raised when a parent cache directory contains multiple ambiguous cache signatures."""
    pass


class CacheCorruptedError(Exception):
    """Raised when predictions file content does not match recorded checksum or count."""
    pass

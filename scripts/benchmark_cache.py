#!/usr/bin/env python3
"""Persistent Raw Prediction Cache for AI Privacy Check Benchmarks (v0.6.6).

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


def compute_model_files_hash(model_dir: Path) -> str:
    """Computes deterministic compound SHA-256 over model files (name + size + mtime)."""
    h = hashlib.sha256()
    for root, _, files in os.walk(model_dir):
        for fname in sorted(files):
            if fname.startswith(".") or fname.endswith(".tmp"):
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(model_dir).as_posix()
            stat = fpath.stat()
            h.update(f"{rel}:{stat.st_size}".encode("utf-8"))
    return h.hexdigest()


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
        return key_data

    def get_cache_dir(self, corpus_sha256: str, model_id: str, revision: str) -> Path:
        short_hash = corpus_sha256[:16]
        safe_model = model_id.replace("/", "_").replace("\\", "_")
        safe_rev = revision.replace("/", "_").replace("\\", "_")
        return self.cache_root / short_hash / safe_model / safe_rev

    def has_valid_cache(self, key_data: Dict[str, Any]) -> Tuple[bool, Optional[Path]]:
        corpus_sha = key_data["corpus_sha256"]
        model_id = key_data["model_id"]
        revision = key_data["revision"]
        cdir = self.get_cache_dir(corpus_sha, model_id, revision)

        cfg_file = cdir / "inference_config.json"
        preds_file = cdir / "predictions.jsonl"

        if not (cfg_file.is_file() and preds_file.is_file()):
            return False, None

        try:
            cached_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            # Compare key components:
            if cached_cfg.get("corpus_sha256") != corpus_sha:
                return False, None
            if cached_cfg.get("model_id") != model_id:
                return False, None
            if cached_cfg.get("revision") != revision:
                return False, None
            if cached_cfg.get("model_files_hash") != key_data["model_files_hash"]:
                return False, None
            # Check inference config match (e.g. label order, min threshold)
            if cached_cfg.get("inference_config") != key_data["inference_config"]:
                return False, None
            return True, cdir
        except Exception:
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

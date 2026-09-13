"""Runtime environment and storage path helpers for AI Privacy Check."""

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Dict, Optional


def get_runtime_storage_map(data_dir: Path) -> Dict[str, Path]:
    """Returns mapping of environment variable names to their dedicated writable storage paths."""
    data_path = Path(data_dir).resolve()
    return {
        "HOME": data_path.parent / "home",
        "XDG_CACHE_HOME": data_path / "cache",
        "UV_CACHE_DIR": data_path / "cache" / "uv",
        "UV_PYTHON_INSTALL_DIR": data_path / "python" / "installations",
        "MODELSCOPE_HOME": data_path / "modelscope-home",
        "MODELSCOPE_CACHE": data_path / "modelscope",
        "HF_HOME": data_path / "huggingface",
        "PIP_CACHE_DIR": data_path / "pip-cache",
    }


def prepare_runtime_dirs(data_dir: Path) -> Dict[str, Path]:
    """Ensures all writable storage paths for ML SDKs and caches exist.

    Redirects away from system /home/ai-privacy-check or /root.
    """
    storage_map = get_runtime_storage_map(data_dir)
    for path in storage_map.values():
        path.mkdir(parents=True, exist_ok=True)
    return storage_map


def build_runtime_env(
    data_dir: Path,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Builds environment dictionary with all ML SDK homes redirected to writable storage."""
    dirs = prepare_runtime_dirs(data_dir)
    env = dict(base_env) if base_env is not None else os.environ.copy()

    for var_name, path in dirs.items():
        env[var_name] = str(path)

    return env

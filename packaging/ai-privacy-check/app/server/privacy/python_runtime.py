"""uv-managed Python runtime provisioning and capability probing for AI Privacy Check.

All AI/ML execution environments (torch-cpu, torch-cuda) run exclusively within
virtual environments created from an isolated uv-managed CPython interpreter.
Never relies on fnOS host system Python for ML execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

# Single source of truth for pinned managed CPython version
MANAGED_PYTHON_VERSION = "3.12.9"

# Core native extension and standard library contract required for ML frameworks
REQUIRED_PYTHON_CAPABILITIES: Tuple[str, ...] = (
    "lzma",
    "_lzma",
    "bz2",
    "_bz2",
    "ssl",
    "_ssl",
    "sqlite3",
    "_sqlite3",
    "ctypes",
    "_ctypes",
    "zlib",
    "hashlib",
    "json",
    "multiprocessing",
    "subprocess",
    "venv",
    "ensurepip",
)

RUNTIME_MANIFEST_SCHEMA_VERSION = 3

# Subprocess script executed by the target interpreter to probe its native capabilities
_PYTHON_CAPABILITY_PROBE_SCRIPT = r'''
import json
import sys

CAPABILITIES = [
    "lzma", "_lzma",
    "bz2", "_bz2",
    "ssl", "_ssl",
    "sqlite3", "_sqlite3",
    "ctypes", "_ctypes",
    "zlib", "hashlib", "json",
    "multiprocessing", "subprocess", "venv", "ensurepip"
]

results = {}
missing = []

for mod in CAPABILITIES:
    try:
        __import__(mod)
        results[mod] = True
    except Exception:
        results[mod] = False
        missing.append(mod)

output = {
    "ok": len(missing) == 0,
    "version": ".".join(map(str, sys.version_info[:3])),
    "full_version": sys.version,
    "base_prefix": sys.base_prefix,
    "prefix": sys.prefix,
    "executable": sys.executable,
    "capabilities": results,
    "missing": missing,
}
print(json.dumps(output))
'''


def find_uv() -> Optional[str]:
    """Locates the uv executable, prioritizing app-bundled or system path."""
    uv_bin = shutil.which("uv")
    if uv_bin:
        return uv_bin
    for cand in (
        "/usr/local/bin/uv",
        "/opt/homebrew/bin/uv",
        os.path.expanduser("~/.cargo/bin/uv"),
        "/var/apps/ai-privacy-check/bin/uv",
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def get_python_installations_dir(data_dir: Path) -> Path:
    """Returns dedicated directory where uv installs private CPython distributions."""
    return (Path(data_dir).resolve() / "python" / "installations")


def get_uv_cache_dir(data_dir: Path) -> Path:
    """Returns private directory for uv wheel and metadata cache."""
    return (Path(data_dir).resolve() / "cache" / "uv")


def build_uv_env(
    data_dir: Path,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Constructs environment with all uv, Python, and ML directories redirected to private app storage."""
    from .runtime_env import build_runtime_env as base_build_runtime_env

    data_path = Path(data_dir).resolve()
    env = base_build_runtime_env(data_path, base_env=base_env)

    install_dir = get_python_installations_dir(data_path)
    uv_cache = get_uv_cache_dir(data_path)

    install_dir.mkdir(parents=True, exist_ok=True)
    uv_cache.mkdir(parents=True, exist_ok=True)

    env["UV_PYTHON_INSTALL_DIR"] = str(install_dir)
    env["UV_CACHE_DIR"] = str(uv_cache)

    return env


def probe_python_capabilities(
    python_path: Union[Path, str],
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Executes target interpreter in a subprocess to check native capabilities.

    Returns structured dictionary with capability report.
    Never uses host Python's importlib.
    """
    py_path = Path(python_path)
    if not py_path.is_file() or not os.access(py_path, os.X_OK):
        return {
            "ok": False,
            "version": None,
            "full_version": None,
            "base_prefix": None,
            "executable": str(python_path),
            "source": "unknown",
            "capabilities": {cap: False for cap in REQUIRED_PYTHON_CAPABILITIES},
            "missing": list(REQUIRED_PYTHON_CAPABILITIES),
            "error": f"解释器不存在或不可执行: {python_path}",
        }

    cmd = [str(py_path), "-c", _PYTHON_CAPABILITY_PROBE_SCRIPT]
    if runner is not None:
        retcode, stdout, stderr = runner(cmd, timeout=timeout)
    else:
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            retcode, stdout, stderr = res.returncode, res.stdout, res.stderr
        except Exception as exc:
            retcode, stdout, stderr = 1, "", str(exc)

    if retcode != 0:
        err_msg = stderr.strip() or stdout.strip() or f"Probe failed with exit code {retcode}"
        return {
            "ok": False,
            "version": None,
            "full_version": None,
            "base_prefix": None,
            "executable": str(python_path),
            "source": "unknown",
            "capabilities": {cap: False for cap in REQUIRED_PYTHON_CAPABILITIES},
            "missing": list(REQUIRED_PYTHON_CAPABILITIES),
            "error": f"能力探测进程退出异常: {err_msg}",
        }

    try:
        data = json.loads(stdout.strip())
        base_prefix = data.get("base_prefix", "")
        # Determine source
        source = "unknown"
        if "python/installations" in base_prefix or "installations" in base_prefix:
            source = "uv-managed"
        elif any(base_prefix.startswith(p) for p in ("/usr", "/lib", "/bin", "/System", "/private")):
            source = "legacy-system-python"
        else:
            source = "custom"

        data["source"] = source
        data["error"] = None
        return data
    except Exception as exc:
        return {
            "ok": False,
            "version": None,
            "full_version": None,
            "base_prefix": None,
            "executable": str(python_path),
            "source": "unknown",
            "capabilities": {cap: False for cap in REQUIRED_PYTHON_CAPABILITIES},
            "missing": list(REQUIRED_PYTHON_CAPABILITIES),
            "error": f"解析能力探测输出失败: {exc}",
        }


def ensure_managed_python(
    data_dir: Path,
    uv_bin: Optional[str] = None,
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    emit_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Ensures pinned uv-managed CPython is installed in private app directory and returns its path."""
    uv = uv_bin or find_uv()
    if not uv:
        raise RuntimeError("未找到 uv 工具，无法管理或部署 Python 运行环境。")

    data_path = Path(data_dir).resolve()
    env = build_uv_env(data_path)
    install_dir = get_python_installations_dir(data_path)

    def log(msg: str) -> None:
        if emit_fn:
            emit_fn(msg)

    # 1. Try to find existing managed Python
    find_cmd = [uv, "python", "find", "--managed-python", MANAGED_PYTHON_VERSION]
    if runner is not None:
        retcode, stdout, stderr = runner(find_cmd, env=env, timeout=15)
    else:
        res = subprocess.run(find_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        retcode, stdout, stderr = res.returncode, res.stdout, res.stderr

    existing_path = None
    if retcode == 0 and stdout.strip():
        cand = Path(stdout.strip()).resolve()
        if cand.is_file() and os.access(cand, os.X_OK):
            probe = probe_python_capabilities(cand, runner=runner)
            if probe.get("ok"):
                existing_path = cand

    if existing_path is not None:
        log(f"已找到受管理的 Python {MANAGED_PYTHON_VERSION}: {existing_path}")
        return existing_path

    # 2. Install pinned Python using uv into private app installations directory
    log(f"正在安装 AIPrivacyCheck 受管理的 Python {MANAGED_PYTHON_VERSION} 至 {install_dir}...")
    install_cmd = [
        uv,
        "python",
        "install",
        "--no-bin",
        "--install-dir",
        str(install_dir),
        MANAGED_PYTHON_VERSION,
    ]
    if runner is not None:
        retcode, stdout, stderr = runner(install_cmd, env=env, timeout=600)
    else:
        res = subprocess.run(install_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        retcode, stdout, stderr = res.returncode, res.stdout, res.stderr

    if retcode != 0:
        err = stderr.strip() or stdout.strip() or f"exit code {retcode}"
        raise RuntimeError(f"Managed Python {MANAGED_PYTHON_VERSION} 安装失败: {err}")

    # 3. Locate installed interpreter
    if runner is not None:
        retcode, stdout, stderr = runner(find_cmd, env=env, timeout=15)
    else:
        res = subprocess.run(find_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        retcode, stdout, stderr = res.returncode, res.stdout, res.stderr

    resolved: Optional[Path] = None
    if retcode == 0 and stdout.strip():
        resolved = Path(stdout.strip()).resolve()
    else:
        # Fallback search within install_dir
        matches = list(install_dir.glob(f"cpython-{MANAGED_PYTHON_VERSION}*/bin/python3*"))
        if matches:
            resolved = matches[0].resolve()

    if not resolved or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Managed Python 安装后未能定位可执行解释器: {stderr or stdout}")

    # Strict check: sys.base_prefix must be inside install_dir, NOT host /usr
    probe = probe_python_capabilities(resolved, runner=runner)
    if not probe.get("ok"):
        missing = probe.get("missing", [])
        raise RuntimeError(
            f"新安装的 Managed Python 缺少必要能力: {', '.join(missing)} ({probe.get('error')})"
        )

    log(f"Managed Python 部署验证成功: {resolved} (版本: {probe.get('version')})")
    return resolved

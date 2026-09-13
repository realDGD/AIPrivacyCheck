"""uv-managed Python runtime provisioning and capability probing for AI Privacy Check.

All AI/ML execution environments (torch-cpu, torch-cuda) run exclusively within
virtual environments created from an isolated uv-managed CPython interpreter.
Never relies on fnOS host system Python for ML execution.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from .runtime_sources import (
    UV_SYSTEM_CERTS,
    CERNET_PYTHON_INSTALL_MIRROR,
    OFFICIAL_PYTHON_INSTALL_SOURCE,
    TLS_MODE_UV_NATIVE,
    TLS_MODE_SYSTEM_CERTS,
    TLS_MODE_EXPLICIT_CA,
    FAIL_TLS_TRUST,
    find_explicit_ca_bundle,
    classify_uv_failure,
    build_attempt_env,
    is_integrity_or_corruption_error,
    is_retryable_network_error,
    format_user_friendly_network_error,
)

_VERIFIED_BUNDLED_UV_PATHS: set[str] = set()


def clear_verified_uv_cache() -> None:
    """Clears cached bundled uv SHA verification results (for testing)."""
    _VERIFIED_BUNDLED_UV_PATHS.clear()

# Pinned official Astral uv binary distribution for Linux
UV_VERSION = "0.12.13"
UV_X86_64_SHA256 = "37a89bb1ffa013a95f81f888ef445fef056eb877ac994b713158cbe185809ac9"
UV_AARCH64_SHA256 = "9ea448a9d8534ec5143dc1328a7a3e865391f851115e2ca5b266fdd6c8a1790e"

# Single source of truth for pinned managed CPython version
MANAGED_PYTHON_VERSION = "3.12.9"

# Core native extension and standard library contract required for ML frameworks (17 items)
PYTHON_CAPABILITY_CONTRACT: Tuple[str, ...] = (
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
REQUIRED_PYTHON_CAPABILITIES: Tuple[str, ...] = PYTHON_CAPABILITY_CONTRACT

PROBE_BASE_PACKAGES: Tuple[str, ...] = (
    "torch",
    "modelscope",
    "numpy",
    "packaging",
    "tqdm",
    "transformers",
    "accelerate",
    "gliner",
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

_BASE_CONTRACT_PROBE_SCRIPT = r'''
import json
import sys

BASE_PACKAGES = [
    "torch", "modelscope", "numpy", "packaging",
    "tqdm", "transformers", "accelerate", "gliner"
]

packages = {}
missing = []
for p in BASE_PACKAGES:
    try:
        mod = __import__(p)
        packages[p] = getattr(mod, "__version__", "unknown")
    except Exception:
        missing.append(p)

violations = []
if "transformers" in packages and packages["transformers"] != "unknown":
    tf_ver = packages["transformers"]
    try:
        from packaging.version import parse as v_parse
        pv = v_parse(tf_ver)
        if pv < v_parse("4.51") or pv >= v_parse("5.0"):
            violations.append(f"transformers=={tf_ver} violates >=4.51,<5")
    except Exception:
        parts = [int(x) for x in tf_ver.split(".")[:2] if x.isdigit()]
        if len(parts) >= 2:
            if (parts[0], parts[1]) < (4, 51) or parts[0] >= 5:
                violations.append(f"transformers=={tf_ver} violates >=4.51,<5")

output = {
    "base_packages_ready": len(missing) == 0 and len(violations) == 0,
    "missing_base_packages": missing,
    "base_contract_violations": violations,
    "packages": packages,
}
print(json.dumps(output))
'''


def get_linux_arch_subdir(machine: Optional[str] = None) -> str:
    """Maps Linux machine architecture identifier to bundled bin subdirectory."""
    mach = (machine or platform.machine()).lower()
    if mach in ("x86_64", "amd64"):
        return "linux-x86_64"
    elif mach in ("aarch64", "arm64"):
        return "linux-aarch64"
    raise ValueError(f"Unsupported machine architecture for bundled uv: {mach}")


def verify_bundled_uv(
    uv_path: Union[str, Path],
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    expected_version: str = UV_VERSION,
    verify_sha: bool = True,
    expected_sha256: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Verifies that the specified uv executable exists, is executable, matches expected version and SHA-256."""
    path = Path(uv_path)
    if not path.is_file():
        return False, f"bundled uv binary not found: {path}"
    if not os.access(path, os.X_OK):
        return False, f"bundled uv binary is not executable: {path}"

    if verify_sha:
        resolved_str = str(path.resolve())
        if resolved_str not in _VERIFIED_BUNDLED_UV_PATHS:
            target_sha = expected_sha256
            if not target_sha:
                path_lower = str(path).lower()
                if "x86_64" in path_lower or "amd64" in path_lower:
                    target_sha = UV_X86_64_SHA256
                elif "aarch64" in path_lower or "arm64" in path_lower:
                    target_sha = UV_AARCH64_SHA256
            if target_sha:
                try:
                    h = hashlib.sha256()
                    with path.open("rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    actual_sha = h.hexdigest()
                    if actual_sha != target_sha:
                        return False, f"bundled uv SHA-256 mismatch: expected {target_sha}, got {actual_sha}"
                    _VERIFIED_BUNDLED_UV_PATHS.add(resolved_str)
                except Exception as exc:
                    return False, f"failed to compute bundled uv SHA-256: {exc}"

    cmd = [str(path), "--version"]
    if runner is not None:
        retcode, stdout, stderr = runner(cmd, timeout=10)
    else:
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            retcode, stdout, stderr = res.returncode, res.stdout, res.stderr
        except Exception as exc:
            return False, f"bundled uv execution check failed: {exc}"

    if retcode != 0:
        return False, f"bundled uv exited with code {retcode}: {stderr.strip() or stdout.strip()}"

    out_ver = stdout.strip()
    if expected_version not in out_ver:
        return False, f"bundled uv version mismatch: expected {expected_version}, got {out_ver}"

    return True, None


def find_uv_info(
    app_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    target_machine: Optional[str] = None,
    target_platform: Optional[str] = None,
    strict_bundled_check: bool = True,
) -> Tuple[Optional[str], str]:
    """Locates uv executable with strict priority:

    Priority 1: Bundled uv matching current Linux architecture
    Priority 2: AIPrivacyCheck managed uv directory (${DATA_DIR}/tools/uv/... or ${DATA_DIR}/bin/uv)
    Priority 3: System PATH uv
    Priority 4: Legacy known locations
    """
    sys_plat = target_platform or sys.platform

    # Priority 1: Bundled uv for Linux
    if sys_plat.startswith("linux") or target_platform is not None:
        try:
            arch_dir = get_linux_arch_subdir(target_machine)
            bundled_candidates: List[Path] = []
            if app_dir:
                bundled_candidates.append(Path(app_dir).resolve() / "bin" / arch_dir / "uv")
            if data_dir:
                bundled_candidates.append(Path(data_dir).resolve() / "app" / "bin" / arch_dir / "uv")

            try:
                # Relative to this file: server/privacy/python_runtime.py -> parents[2] is app
                module_app_dir = Path(__file__).resolve().parents[2]
                bundled_candidates.append(module_app_dir / "bin" / arch_dir / "uv")
            except Exception:
                pass

            for base in ("/var/apps/ai-privacy-check", "/usr/local/apps/ai-privacy-check"):
                bundled_candidates.append(Path(base) / "bin" / arch_dir / "uv")

            for cand in bundled_candidates:
                if cand.is_file():
                    valid, err = verify_bundled_uv(cand, runner=runner, verify_sha=(runner is None))
                    if valid:
                        return str(cand), "bundled"
                    else:
                        if strict_bundled_check:
                            raise RuntimeError(f"Bundled uv binary invalid at {cand}: {err}")
        except ValueError as val_err:
            if target_machine is not None or sys_plat.startswith("linux"):
                raise RuntimeError(f"Unsupported architecture for uv runtime: {val_err}") from val_err

    # Priority 2: AIPrivacyCheck managed uv directory
    if data_dir:
        for managed_cand in (
            Path(data_dir).resolve() / "tools" / "uv" / "uv",
            Path(data_dir).resolve() / "bin" / "uv",
        ):
            if managed_cand.is_file() and os.access(managed_cand, os.X_OK):
                return str(managed_cand), "managed"

    # Priority 3: System PATH uv
    sys_uv = shutil.which("uv")
    if sys_uv:
        return sys_uv, "system"

    # Priority 4: Legacy known locations
    for cand_str in (
        "/usr/local/bin/uv",
        "/opt/homebrew/bin/uv",
        os.path.expanduser("~/.cargo/bin/uv"),
        "/var/apps/ai-privacy-check/bin/uv",
    ):
        if os.path.isfile(cand_str) and os.access(cand_str, os.X_OK):
            return cand_str, "legacy"

    return None, "none"


def find_uv(
    app_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
) -> Optional[str]:
    """Locates the uv executable, returning path or None."""
    path, _ = find_uv_info(app_dir=app_dir, data_dir=data_dir, runner=runner)
    return path


def probe_base_runtime_contract(
    python_path: Union[Path, str],
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Probes the target Python environment for Base ML Contract packages and constraints."""
    py_path = Path(python_path)
    if not py_path.is_file() or not os.access(py_path, os.X_OK):
        return {
            "base_packages_ready": False,
            "missing_base_packages": list(PROBE_BASE_PACKAGES),
            "base_contract_violations": [],
            "packages": {},
            "error": f"解释器不存在或不可执行: {python_path}",
        }

    cmd = [str(py_path), "-c", _BASE_CONTRACT_PROBE_SCRIPT]
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
            "base_packages_ready": False,
            "missing_base_packages": list(PROBE_BASE_PACKAGES),
            "base_contract_violations": [],
            "packages": {},
            "error": f"Base ML Contract 探测异常: {err_msg}",
        }

    try:
        data = json.loads(stdout.strip())
        data["error"] = None
        return data
    except Exception as exc:
        return {
            "base_packages_ready": False,
            "missing_base_packages": list(PROBE_BASE_PACKAGES),
            "base_contract_violations": [],
            "packages": {},
            "error": f"解析 Base ML Contract 输出失败: {exc}",
        }


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
    env.pop("UV_SYSTEM_CERTS", None)

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


def ensure_managed_python_info(
    data_dir: Path,
    uv_bin: Optional[str] = None,
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    emit_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, str]:
    """Ensures pinned uv-managed CPython is installed in private app directory.

    Reuses existing valid installation if available (0 bytes downloaded).
    Otherwise downloads using Cernet mirror with automatic fallback to official source.
    Returns (interpreter_path, download_source) where download_source is:
    "existing-local", "cernet-mirror", or "official".
    """
    uv_path, uv_source = (uv_bin, "custom") if uv_bin else find_uv_info(data_dir=data_dir, runner=runner)
    if not uv_path:
        raise RuntimeError("未找到 uv 工具，无法管理或部署 Python 运行环境。")
    uv = uv_path

    data_path = Path(data_dir).resolve()
    env = build_uv_env(data_path)
    install_dir = get_python_installations_dir(data_path)

    def log(msg: str) -> None:
        if emit_fn:
            emit_fn(msg)

    uv_source_desc = {
        "bundled": "内置",
        "managed": "托管",
        "system": "系统",
        "legacy": "系统备用",
        "custom": "自定义",
    }.get(uv_source, "可用")
    log(f"已选择 AIPrivacyCheck {uv_source_desc} uv: {uv}")

    # 1. Try to find existing managed Python (via uv python find or direct scan)
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

    if existing_path is None and install_dir.is_dir():
        for cand in sorted(install_dir.glob(f"cpython-{MANAGED_PYTHON_VERSION}*/bin/python3*")):
            if cand.is_file() and os.access(cand, os.X_OK):
                probe = probe_python_capabilities(cand, runner=runner)
                if probe.get("ok"):
                    existing_path = cand.resolve()
                    break

    if existing_path is not None:
        log(f"已复用现有托管 Python {MANAGED_PYTHON_VERSION} (0 字节下载): {existing_path}")
        return existing_path, "existing-local"

    # 2. Install pinned Python using uv into private app installations directory
    install_cmd = [
        uv,
        "python",
        "install",
        "--no-bin",
        "--install-dir",
        str(install_dir),
        MANAGED_PYTHON_VERSION,
    ]

    def _exec_install(target_env: Dict[str, str]) -> Tuple[int, str, str]:
        if runner is not None:
            return runner(install_cmd, env=target_env, timeout=600)
        res = subprocess.run(install_cmd, env=target_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode, res.stdout, res.stderr

    def _run_with_tls_progression(base_target_env: Dict[str, str], is_mirror: bool) -> Tuple[int, str, str, str]:
        # 1. TLS Mode A: uv-native
        env_try = build_attempt_env(base_target_env, TLS_MODE_UV_NATIVE, is_mirror=is_mirror)
        r, out, err = _exec_install(env_try)
        if r == 0:
            return 0, out, err, TLS_MODE_UV_NATIVE

        cat = classify_uv_failure(err or out)
        if cat == FAIL_TLS_TRUST:
            log("uv 默认 CA 无法验证证书，正在使用 fnOS 系统 CA 重试...")
            env_try = build_attempt_env(base_target_env, TLS_MODE_SYSTEM_CERTS, is_mirror=is_mirror)
            r, out, err = _exec_install(env_try)
            if r == 0:
                return 0, out, err, TLS_MODE_SYSTEM_CERTS

            cat2 = classify_uv_failure(err or out)
            if cat2 == FAIL_TLS_TRUST:
                ca_bundle = find_explicit_ca_bundle()
                if ca_bundle:
                    log(f"正在使用显式 CA 证书链 ({ca_bundle}) 重试...")
                    env_try = build_attempt_env(base_target_env, TLS_MODE_EXPLICIT_CA, is_mirror=is_mirror, ca_bundle_path=ca_bundle)
                    r, out, err = _exec_install(env_try)
                    if r == 0:
                        return 0, out, err, TLS_MODE_EXPLICIT_CA

        return r, out, err, TLS_MODE_UV_NATIVE

    # Supply Chain A - Attempt 1: Primary Cernet Mirror
    log(f"正在通过 Cernet 镜像源安装 AIPrivacyCheck 受管理的 Python {MANAGED_PYTHON_VERSION} 至 {install_dir}...")
    env_primary = dict(env)
    env_primary["UV_PYTHON_INSTALL_MIRROR"] = CERNET_PYTHON_INSTALL_MIRROR

    download_source = "cernet-mirror"
    retcode, stdout, stderr, _ = _run_with_tls_progression(env_primary, is_mirror=True)

    if retcode != 0:
        err1 = stderr.strip() or stdout.strip() or f"exit code {retcode}"
        # Fail closed immediately on archive corruption or SHA mismatch
        if is_integrity_or_corruption_error(err1):
            raise RuntimeError(f"Managed Python 镜像包完整性校验失败 (不可降级重试): {err1}")

        if is_retryable_network_error(err1):
            log(f"Cernet 镜像下载 Python 失败 ({err1})，正在降级回退至官方源重试...")
            env_official = dict(env)
            env_official.pop("UV_PYTHON_INSTALL_MIRROR", None)
            download_source = "official"

            retcode2, stdout2, stderr2, _ = _run_with_tls_progression(env_official, is_mirror=False)
            if retcode2 != 0:
                err2 = stderr2.strip() or stdout2.strip() or f"exit code {retcode2}"
                if is_integrity_or_corruption_error(err2):
                    raise RuntimeError(f"Managed Python 官方源完整性校验失败: {err2}")
                raise RuntimeError(format_user_friendly_network_error(err2 or err1))
        else:
            raise RuntimeError(f"Managed Python {MANAGED_PYTHON_VERSION} 安装失败: {err1}")

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

    log(f"Managed Python 部署验证成功: {resolved} (版本: {probe.get('version')}, 来源: {download_source})")
    log("Python Capability Contract: PASS")
    return resolved, download_source


def ensure_managed_python(
    data_dir: Path,
    uv_bin: Optional[str] = None,
    runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    emit_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Ensures pinned uv-managed CPython is installed in private app directory and returns its path."""
    resolved, _ = ensure_managed_python_info(data_dir=data_dir, uv_bin=uv_bin, runner=runner, emit_fn=emit_fn)
    return resolved

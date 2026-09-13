"""Model lifecycle management for AI Privacy Check.

All officially supported models originate exclusively from ModelScope (魔搭社区).
Supports:
1. Online transactional download and installation via ModelScope.
2. Local manual import from user-authorized fnOS directories (e.g. data-share models)
   with strict authorization boundary validation, symlink escape protection,
   and atomic replacement without mutating user source files.
3. Scanning user-visible fnOS data-share models folder (AI 脱敏器/models).
4. Isolated runtime virtual environments (torch-cpu, torch-cuda)
   with real subprocess framework verification.
5. Model uninstallation, status reporting, and safe hot-reload.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from privacy.device import DEVICE_MANAGER
from privacy.model_catalog import MODEL_CATALOG, check_model_integrity, get_model_descriptor
from privacy.runtime_manager import (
    ALL_PROFILES,
    PROFILE_TORCH_CPU,
    PROFILE_TORCH_CUDA,
    PYPI_MIRROR_URL,
    PYTORCH_CPU_INDEX,
    PYTORCH_CUDA_INDEX,
    RuntimeManager,
    get_runtime_manager,
)
from privacy.python_runtime import (
    MANAGED_PYTHON_VERSION,
    PROBE_BASE_PACKAGES,
    PYTHON_CAPABILITY_CONTRACT,
    REQUIRED_PYTHON_CAPABILITIES,
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    UV_VERSION,
    build_uv_env,
    ensure_managed_python,
    ensure_managed_python_info,
    find_uv,
    find_uv_info,
    probe_base_runtime_contract,
    probe_python_capabilities,
)
from privacy.runtime_sources import (
    CERNET_PYPI_INDEX,
    OFFICIAL_PYPI_INDEX,
    SJTUG_TORCH_INDEX_CPU,
    SJTUG_TORCH_INDEX_CUDA,
    OFFICIAL_TORCH_INDEX_CPU,
    OFFICIAL_TORCH_INDEX_CUDA,
    get_torch_index_url,
    get_pypi_index_url,
    find_explicit_ca_bundle,
    classify_uv_failure,
    build_attempt_env,
    InstallResult,
    TLS_MODE_UV_NATIVE,
    TLS_MODE_SYSTEM_CERTS,
    TLS_MODE_EXPLICIT_CA,
    FAIL_TLS_TRUST,
    FAIL_INTEGRITY,
    FAIL_LOCAL_IO,
    is_integrity_or_corruption_error,
    is_local_io_error,
    is_retryable_network_error,
    format_user_friendly_network_error,
)
from privacy.runtime_env import build_runtime_env, prepare_runtime_dirs
from privacy.worker_client import get_worker_client

_LOCK_STATE = threading.local()

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Pinned PyTorch version for stable, predictable ML runtime execution
PINNED_TORCH_VERSION = "2.6.0"
PINNED_TORCH_REQUIREMENT = f"torch=={PINNED_TORCH_VERSION}"

# Shared torch runtime transformers constraint:
#   >=4.51 - floor required by Qwen3ForCausalLM (MemPrivacy catalog models and
#            Qwen3 challenger weights fail to load on older releases).
#   <5     - ceiling required by ModelScope legacy pipelines/models: their NLP
#            configuration modules import `transformers.onnx`, which was
#            removed in transformers 5.x.
TRANSFORMERS_REQUIREMENT = "transformers>=4.51,<5"

# Distinguishes the dependency probe subprocess from other interpreter calls.
DEPENDENCY_PROBE_MARKER = "AIPRIVACY_DEPENDENCY_PROBE"

# Minimal spec evaluation running inside the isolated runtime interpreter.
# Uses only the stdlib: importlib for presence/version probing.
_DEPENDENCY_PROBE_SCRIPT = r'''
import importlib.metadata
import importlib.util
import json
import re
import sys

IMPORT_ALIASES = {
    "pillow": "PIL",
}

def dist_name(spec):
    return re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec).group(0)

def version_tuple(value):
    parts = []
    for chunk in re.split(r"[.+]", value.strip()):
        digits = re.match(r"\d+", chunk)
        if not digits:
            break
        parts.append(int(digits.group(0)))
    return tuple(parts) or (0,)

def satisfies(installed, constraints):
    for op, expected in constraints:
        a, b = version_tuple(installed), version_tuple(expected)
        ok = {
            ">=": a >= b, "<=": a <= b, "==": a == b,
            ">": a > b, "<": a < b, "!=": a != b,
        }[op]
        if not ok:
            return False
    return True

specs = json.loads(sys.argv[1])
missing = []
for spec in specs:
    name = dist_name(spec)
    constraints = re.findall(r"(>=|<=|==|!=|>|<)\s*([0-9A-Za-z.]+)", spec[len(name):])
    try:
        installed_version = importlib.metadata.version(name)
    except Exception:
        missing.append(spec)
        continue
    if constraints and not satisfies(installed_version, constraints):
        missing.append(spec)
        continue
    import_name = IMPORT_ALIASES.get(name.lower(), name.lower())
    try:
        if importlib.util.find_spec(import_name) is None:
            missing.append(spec)
    except Exception:
        missing.append(spec)

print(json.dumps({"ok": True, "missing": missing, "error": None}))
'''


def validate_model_id(model_id: str):
    """Validates model_id format and ensures it exists in the official ModelScope catalog.

    Defense-in-depth against directory traversal (e.g. '../runtimes').
    """
    if not isinstance(model_id, str):
        raise ValueError(f"model_id 类型无效: {type(model_id).__name__}")
    clean_id = model_id.strip()
    if not clean_id or not MODEL_ID_RE.match(clean_id):
        raise ValueError(f"模型标识格式非法: {model_id}")
    descriptor = get_model_descriptor(clean_id)
    if descriptor is None:
        raise ValueError(f"未知模型标识: {model_id}")
    return descriptor


@contextmanager
def model_operation_lock(data_dir: Path, model_id: str, non_blocking: bool = True):
    """Acquires a cross-process mutual exclusion file lock for a specific model operation.

    Uses fcntl.flock on ${DATA_DIR}/locks/{model_id}.lock.
    Re-entrant within the same thread.
    """
    validate_model_id(model_id)
    locks_dir = (Path(data_dir) / "locks").resolve()
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (locks_dir / f"{model_id}.lock").resolve()
    if lock_file.parent != locks_dir:
        raise ValueError(f"非法锁文件路径越界: {model_id}")

    if not hasattr(_LOCK_STATE, "acquired"):
        _LOCK_STATE.acquired = {}

    key = str(lock_file)
    if key in _LOCK_STATE.acquired:
        _LOCK_STATE.acquired[key]["count"] += 1
        try:
            yield lock_file
        finally:
            _LOCK_STATE.acquired[key]["count"] -= 1
            if _LOCK_STATE.acquired[key]["count"] <= 0:
                del _LOCK_STATE.acquired[key]
        return

    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o666)
    flags = fcntl.LOCK_EX
    if non_blocking:
        flags |= fcntl.LOCK_NB

    try:
        try:
            fcntl.flock(fd, flags)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            raise RuntimeError(f"模型 [{model_id}] 当前正在执行安装、导入或卸载操作，请稍后重试。") from exc

        _LOCK_STATE.acquired[key] = {"fd": fd, "count": 1}
        try:
            yield lock_file
        finally:
            if key in _LOCK_STATE.acquired:
                _LOCK_STATE.acquired[key]["count"] -= 1
                if _LOCK_STATE.acquired[key]["count"] <= 0:
                    del _LOCK_STATE.acquired[key]
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
    except Exception:
        raise


MIN_FREE_DISK_BYTES_CPU = 2 * 1024 * 1024 * 1024  # 2.0 GB
MIN_FREE_DISK_BYTES_CUDA = int(4.5 * 1024 * 1024 * 1024)  # 4.5 GB


def check_disk_space_for_rebuild(data_dir: Path, profile: str) -> Tuple[bool, Optional[str]]:
    """Checks whether sufficient free disk space is available for staging runtime rebuild."""
    try:
        usage = shutil.disk_usage(data_dir)
        free_bytes = usage.free
    except Exception:
        return True, None

    required_bytes = MIN_FREE_DISK_BYTES_CUDA if profile == PROFILE_TORCH_CUDA else MIN_FREE_DISK_BYTES_CPU
    if free_bytes < required_bytes:
        req_gb = required_bytes / (1024 ** 3)
        avail_gb = free_bytes / (1024 ** 3)
        return False, f"磁盘剩余空间不足: 重建 [{profile}] 需要至少 {req_gb:.1f} GB 可用空间，当前仅有 {avail_gb:.1f} GB"
    return True, None


@contextmanager
def runtime_operation_lock(data_dir: Path, profile: str, non_blocking: bool = True):
    """Acquires a cross-process mutual exclusion file lock for a runtime profile operation.

    Uses fcntl.flock on ${DATA_DIR}/locks/runtime_{profile}.lock.
    Re-entrant within the same thread.
    """
    if profile not in (PROFILE_TORCH_CPU, PROFILE_TORCH_CUDA):
        raise ValueError(f"不支持的运行时 Profile: {profile}")
    locks_dir = (Path(data_dir) / "locks").resolve()
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (locks_dir / f"runtime_{profile}.lock").resolve()

    if not hasattr(_LOCK_STATE, "acquired"):
        _LOCK_STATE.acquired = {}

    key = str(lock_file)
    if key in _LOCK_STATE.acquired:
        _LOCK_STATE.acquired[key]["count"] += 1
        try:
            yield lock_file
        finally:
            _LOCK_STATE.acquired[key]["count"] -= 1
            if _LOCK_STATE.acquired[key]["count"] <= 0:
                del _LOCK_STATE.acquired[key]
        return

    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o666)
    flags = fcntl.LOCK_EX
    if non_blocking:
        flags |= fcntl.LOCK_NB

    try:
        try:
            fcntl.flock(fd, flags)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            raise RuntimeError(f"运行时 [{profile}] 当前正在执行环境部署或重建操作，请稍后重试。") from exc

        _LOCK_STATE.acquired[key] = {"fd": fd, "count": 1}
        try:
            yield lock_file
        finally:
            if key in _LOCK_STATE.acquired:
                _LOCK_STATE.acquired[key]["count"] -= 1
                if _LOCK_STATE.acquired[key]["count"] <= 0:
                    del _LOCK_STATE.acquired[key]
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
    except Exception:
        raise


def emit(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def get_model_dir(data_dir: Path, model_id: str) -> Path:
    validate_model_id(model_id)
    base = (data_dir / "models").resolve()
    target = (data_dir / "models" / model_id).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"非法模型路径越界: {model_id}")
    return target


def get_staging_dir(data_dir: Path, model_id: str) -> Path:
    validate_model_id(model_id)
    base = (data_dir / "models" / ".staging").resolve()
    target = (data_dir / "models" / ".staging" / model_id).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"非法暂存路径越界: {model_id}")
    return target


def write_state(data_dir: Path, state: str, detail: str, model_id: str) -> None:
    status_dir = data_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_id,
        "state": state,
        "detail": detail,
        "updated_at": int(time.time()),
    }
    target = status_dir / f"{model_id}-install.json"
    temporary = status_dir / f"{model_id}-install.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def get_dir_size(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def get_shared_models_dirs() -> List[Path]:
    """Discover user-managed shared model source directories from fnOS environment.

    Reads TRIM_DATA_SHARE_PATHS, /var/apps/ai-privacy-check/shares, and local fallbacks.
    """
    candidates: List[Path] = []

    # 1. TRIM_DATA_SHARE_PATHS (colon-separated list set by fnOS gateway/resource manager)
    share_env = os.environ.get("TRIM_DATA_SHARE_PATHS", "").strip()
    if share_env:
        for p in share_env.split(":"):
            clean_p = p.strip()
            if clean_p:
                cand = Path(clean_p).resolve()
                if cand not in candidates:
                    candidates.append(cand)

    # 2. Standard fnOS symlink locations
    for std_share in (
        Path("/var/apps/ai-privacy-check/shares/ai-privacy-check/models"),
        Path("/var/apps/ai-privacy-check/share/models"),
        Path("/var/apps/ai-privacy-check/shares"),
    ):
        if std_share.exists():
            resolved = std_share.resolve()
            if resolved not in candidates:
                candidates.append(resolved)

    return candidates


def validate_import_source_path(source_path: Union[Path, str], data_dir: Path) -> Tuple[bool, str, Optional[Path]]:
    """Strictly validates source path against fnOS authorization boundaries.

    Ensures:
    1. Source path exists.
    2. Canonical realpath resolution prevents symlink escape.
    3. Target is within authorized roots:
       - TRIM_DATA_SHARE_PATHS (user model shares)
       - TRIM_DATA_ACCESSIBLE_PATHS (fnOS authorized user paths)
       - App's private data_dir
    4. Rejects root '/', system dirs, '/etc', '/root', '/proc', '/sys', other app dirs.
    """
    try:
        path_obj = Path(source_path) if isinstance(source_path, str) else source_path
        resolved_src = path_obj.resolve(strict=True)
    except Exception as exc:
        return False, f"源路径无效或不存在: {exc}", None

    real_src_str = str(resolved_src)

    # Reject dangerous system roots immediately (accounting for Linux and macOS canonical paths)
    dangerous_prefixes = (
        "/etc",
        "/private/etc",
        "/root",
        "/var/root",
        "/private/var/root",
        "/proc",
        "/sys",
        "/dev",
        "/boot",
        "/bin",
        "/sbin",
        "/usr",
    )
    for dangerous in dangerous_prefixes:
        if real_src_str == dangerous or real_src_str.startswith(dangerous + "/"):
            return False, f"拒绝访问系统受限路径: {source_path}", None

    allowed_roots: List[Path] = []

    # 1. User shared model directories
    allowed_roots.extend(get_shared_models_dirs())

    # 2. fnOS authorized data accessible paths
    acc_env = os.environ.get("TRIM_DATA_ACCESSIBLE_PATHS", "").strip()
    if acc_env:
        for p in acc_env.split(":"):
            if p.strip():
                try:
                    cand = Path(p.strip()).resolve()
                    if cand not in allowed_roots:
                        allowed_roots.append(cand)
                except Exception:
                    pass

    # 3. App data dir & local fallback
    allowed_roots.append(data_dir.resolve())
    local_shared = (data_dir / "shared_models").resolve()
    if local_shared not in allowed_roots:
        allowed_roots.append(local_shared)

    # Discard root filesystem or empty paths
    allowed_roots = [r for r in allowed_roots if str(r) not in ("/", "")]

    # Check if resolved_src is contained in any allowed root
    is_allowed = False
    for root in allowed_roots:
        try:
            resolved_src.relative_to(root)
            is_allowed = True
            break
        except (ValueError, Exception):
            continue

    if not is_allowed:
        return (
            False,
            f"路径未经 fnOS 授权或超出允许的模型导入目录范围: {source_path}",
            None,
        )

    return True, "路径合法", resolved_src


def scan_shared_models_directory(data_dir: Path) -> List[Dict[str, Any]]:
    """Scans user-managed shared model directories for importable models."""
    shared_dirs = get_shared_models_dirs()
    fallback = data_dir / "shared_models"
    if fallback.is_dir() and fallback.resolve() not in shared_dirs:
        shared_dirs.append(fallback.resolve())

    found: List[Dict[str, Any]] = []
    seen_ids = set()

    for s_dir in shared_dirs:
        if not s_dir.is_dir():
            continue
        try:
            for entry in s_dir.iterdir():
                if not entry.is_dir():
                    continue
                folder_name = entry.name.strip()
                # Match folder name against catalog model IDs or repo basenames
                for model_id, desc in MODEL_CATALOG.items():
                    repo_suffix = desc.repo_id.split("/")[-1].lower()
                    if (folder_name.lower() == model_id.lower() or folder_name.lower() == repo_suffix) and model_id not in seen_ids:
                        ok, reason = verify_model_integrity(entry, model_id)
                        found.append({
                            "model_id": model_id,
                            "display_name": desc.display_name,
                            "folder_name": folder_name,
                            "source_path": str(entry.resolve()),
                            "shared_dir": str(s_dir.resolve()),
                            "valid": ok,
                            "reason": reason,
                            "approx_size": desc.approx_size,
                            "slot": desc.slot,
                        })
                        seen_ids.add(model_id)
        except Exception:
            continue

    return found


def verify_model_integrity(model_dir: Path, model_id: str) -> Tuple[bool, str]:
    """Verify integrity of model weights and configs based on catalog descriptor."""
    return check_model_integrity(model_id, model_dir)


def sanitize_model_config(model_dir: Path) -> Tuple[bool, str]:
    """Strips remote-code triggers from a model's ModelScope configuration.json.

    Thin wrapper over privacy.model_security.gate_model_security, the single
    source of truth shared with the pre-load security gate (which also covers
    models installed by v0.6.3 and earlier). Idempotent.
    """
    from privacy.model_security import gate_model_security

    result = gate_model_security(model_dir)
    if result["status"] == "sanitized":
        emit(f"已从模型配置中移除远程代码声明: {model_dir / 'configuration.json'}")
        return True, result["detail"]
    return False, result["detail"]


def _find_uv() -> Optional[str]:
    return find_uv()


def _pip_install_command(uv_bin: Optional[str], interp: Path, venv_dir: Path, args: List[str]) -> List[str]:
    """Builds a pip/uv install command targeting the isolated venv interpreter."""
    if uv_bin:
        return [uv_bin, "pip", "install", "--python", str(interp), *args]
    pip_bin = str(venv_dir / "bin" / "pip")
    return [pip_bin, "install", "--disable-pip-version-check", "--no-input", "--upgrade", *args]


def _default_dependency_runner(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 15,
) -> Tuple[int, str, str]:
    try:
        run_kwargs: Dict[str, Any] = {"check": False}
        if env is not None:
            run_kwargs["env"] = env
        if cwd is not None:
            run_kwargs["cwd"] = cwd
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                **run_kwargs,
            )
        except TypeError:
            # Fallback for simple unit test mocks with signature (cmd, check=..., env=...)
            res = subprocess.run(cmd, **run_kwargs)
        retcode = getattr(res, "returncode", 0)
        stdout = getattr(res, "stdout", "") or ""
        stderr = getattr(res, "stderr", "") or ""
        return retcode, stdout, stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0] if cmd else ''}"
    except subprocess.TimeoutExpired:
        return 124, "", "dependency command timed out"
    except Exception as exc:
        return 1, "", str(exc)


def run_install_pypi_with_fallback(
    runner: Callable[..., Tuple[int, str, str]],
    uv_bin: Optional[str],
    interp: Path,
    venv_dir: Path,
    packages: List[str],
    env: Dict[str, str],
    emit_fn: Optional[Callable[[str], None]] = None,
    timeout: int = 1800,
) -> InstallResult:
    """Installs PyPI packages using Cernet mirror with official fallback and layered TLS."""
    log = emit_fn or emit
    sources = [
        ("cernet", CERNET_PYPI_INDEX, True),
        ("official", OFFICIAL_PYPI_INDEX, False),
    ]

    attempts: List[Dict[str, Any]] = []
    last_err = ""

    for source_name, index_url, is_mirror in sources:
        cmd = _pip_install_command(uv_bin, interp, venv_dir, [*packages, "-i", index_url])

        tls_modes_to_try = [TLS_MODE_UV_NATIVE]
        idx = 0
        while idx < len(tls_modes_to_try):
            tls_mode = tls_modes_to_try[idx]
            idx += 1

            ca_bundle = find_explicit_ca_bundle() if tls_mode == TLS_MODE_EXPLICIT_CA else None
            attempt_env = build_attempt_env(env, tls_mode, is_mirror=is_mirror, ca_bundle_path=ca_bundle)
            attempt_cmd = list(cmd)
            if tls_mode == TLS_MODE_EXPLICIT_CA and ca_bundle and uv_bin:
                attempt_cmd.extend(["--cert", ca_bundle])

            start_t = time.time()
            retcode, stdout, stderr = runner(attempt_cmd, env=attempt_env, timeout=timeout)
            elapsed_ms = int((time.time() - start_t) * 1000)

            if retcode == 0:
                attempts.append({
                    "source": source_name,
                    "tls_mode": tls_mode,
                    "result": "success",
                    "retryable": False,
                    "elapsed_ms": elapsed_ms,
                })
                return InstallResult(source_name, tls_mode, attempts)

            output = stderr.strip() or stdout.strip() or f"exit code {retcode}"
            last_err = output
            category = classify_uv_failure(output)
            attempts.append({
                "source": source_name,
                "tls_mode": tls_mode,
                "result": category.lower(),
                "retryable": is_retryable_network_error(output),
                "elapsed_ms": elapsed_ms,
            })

            # Fail closed immediately on integrity violation or corruption
            if is_integrity_or_corruption_error(output):
                raise RuntimeError(f"PyPI 镜像包校验失败 (不可降级重试): {output}")

            # Fail closed immediately on local disk / permission error
            if is_local_io_error(output):
                raise RuntimeError(f"本地存储或权限受阻 (不可换源重试): {output}")

            # Check if TLS retry is warranted
            if category == FAIL_TLS_TRUST:
                if tls_mode == TLS_MODE_UV_NATIVE and TLS_MODE_SYSTEM_CERTS not in tls_modes_to_try:
                    log("uv 默认 CA 无法验证证书，正在使用 fnOS 系统 CA 重试...")
                    tls_modes_to_try.append(TLS_MODE_SYSTEM_CERTS)
                    continue
                elif tls_mode == TLS_MODE_SYSTEM_CERTS and TLS_MODE_EXPLICIT_CA not in tls_modes_to_try:
                    explicit_ca = find_explicit_ca_bundle()
                    if explicit_ca:
                        log(f"正在使用显式 CA 证书链 ({explicit_ca}) 重试...")
                        tls_modes_to_try.append(TLS_MODE_EXPLICIT_CA)
                        continue

            # Non-TLS failure or TLS modes exhausted for this source
            break

        # If primary mirror failed with retryable error, proceed to official fallback
        if is_mirror and is_retryable_network_error(last_err):
            log(f"Cernet PyPI 镜像安装失败 ({last_err})，正在降级回退至官方源重试...")
            continue
        elif not is_mirror:
            break
        else:
            raise RuntimeError(f"安装依赖失败 ({' '.join(packages[:3])}): {last_err}")

    raise RuntimeError(format_user_friendly_network_error(last_err))


def run_install_torch_with_fallback(
    runner: Callable[..., Tuple[int, str, str]],
    uv_bin: Optional[str],
    interp: Path,
    venv_dir: Path,
    profile: str,
    env: Dict[str, str],
    emit_fn: Optional[Callable[[str], None]] = None,
    timeout: int = 1800,
) -> InstallResult:
    """Installs PyTorch using SJTUG wheel mirror with official fallback, layered TLS, and index separation."""
    log = emit_fn or emit
    sources = [
        ("sjtug", get_torch_index_url(profile, use_mirror=True), get_pypi_index_url(use_mirror=True), True),
        ("official", get_torch_index_url(profile, use_mirror=False), get_pypi_index_url(use_mirror=False), False),
    ]

    attempts: List[Dict[str, Any]] = []
    last_err = ""

    for source_name, torch_index, pypi_index, is_mirror in sources:
        # Build command ensuring index separation:
        # torch from dedicated wheel index, general dependencies from PyPI
        if uv_bin:
            cmd_args = [
                PINNED_TORCH_REQUIREMENT,
                "--index", torch_index,
                "--default-index", pypi_index,
                "--index-strategy", "first-index",
            ]
        else:
            cmd_args = [
                PINNED_TORCH_REQUIREMENT,
                "--index-url", torch_index,
                "--extra-index-url", pypi_index,
            ]
        cmd = _pip_install_command(uv_bin, interp, venv_dir, cmd_args)

        tls_modes_to_try = [TLS_MODE_UV_NATIVE]
        idx = 0
        while idx < len(tls_modes_to_try):
            tls_mode = tls_modes_to_try[idx]
            idx += 1

            ca_bundle = find_explicit_ca_bundle() if tls_mode == TLS_MODE_EXPLICIT_CA else None
            attempt_env = build_attempt_env(env, tls_mode, is_mirror=is_mirror, ca_bundle_path=ca_bundle)
            attempt_cmd = list(cmd)
            if tls_mode == TLS_MODE_EXPLICIT_CA and ca_bundle and uv_bin:
                attempt_cmd.extend(["--cert", ca_bundle])

            start_t = time.time()
            retcode, stdout, stderr = runner(attempt_cmd, env=attempt_env, timeout=timeout)
            elapsed_ms = int((time.time() - start_t) * 1000)

            if retcode == 0:
                attempts.append({
                    "source": source_name,
                    "tls_mode": tls_mode,
                    "result": "success",
                    "retryable": False,
                    "elapsed_ms": elapsed_ms,
                })
                return InstallResult(source_name, tls_mode, attempts)

            output = stderr.strip() or stdout.strip() or f"exit code {retcode}"
            last_err = output
            category = classify_uv_failure(output)
            attempts.append({
                "source": source_name,
                "tls_mode": tls_mode,
                "result": category.lower(),
                "retryable": is_retryable_network_error(output),
                "elapsed_ms": elapsed_ms,
            })

            # Fail closed immediately on integrity violation or corruption
            if is_integrity_or_corruption_error(output):
                raise RuntimeError(f"PyTorch 镜像包校验失败 (不可降级重试): {output}")

            # Fail closed immediately on local disk / permission error
            if is_local_io_error(output):
                raise RuntimeError(f"本地存储或权限受阻 (不可换源重试): {output}")

            # Check if TLS retry is warranted
            if category == FAIL_TLS_TRUST:
                if tls_mode == TLS_MODE_UV_NATIVE and TLS_MODE_SYSTEM_CERTS not in tls_modes_to_try:
                    log("uv 默认 CA 无法验证证书，正在使用 fnOS 系统 CA 重试...")
                    tls_modes_to_try.append(TLS_MODE_SYSTEM_CERTS)
                    continue
                elif tls_mode == TLS_MODE_SYSTEM_CERTS and TLS_MODE_EXPLICIT_CA not in tls_modes_to_try:
                    explicit_ca = find_explicit_ca_bundle()
                    if explicit_ca:
                        log(f"正在使用显式 CA 证书链 ({explicit_ca}) 重试...")
                        tls_modes_to_try.append(TLS_MODE_EXPLICIT_CA)
                        continue

            # Non-TLS failure or TLS modes exhausted for this source
            break

        # If primary mirror failed with retryable error, proceed to official fallback
        if is_mirror and is_retryable_network_error(last_err):
            log(f"SJTUG PyTorch 镜像安装失败 ({last_err})，正在降级回退至官方源重试...")
            continue
        elif not is_mirror:
            break
        else:
            raise RuntimeError(f"安装 PyTorch 失败: {last_err}")

    raise RuntimeError(format_user_friendly_network_error(last_err))


# Base runtime compatibility contract: checked EVEN when the profile probe
# reports verified, because historical runtimes (pre-0.6.4) may carry
# incompatible packages (e.g. transformers 5.16.1) that the torch-only probe
# cannot see. Repairs only the violating packages; never reinstalls torch,
# never rebuilds the venv.
BASE_RUNTIME_REQUIREMENTS: Tuple[str, ...] = (
    "modelscope",
    "numpy",
    "packaging",
    "tqdm",
    "torch",
    TRANSFORMERS_REQUIREMENT,
    "accelerate",
    "gliner",
)


def _probe_dependency_specs(
    python_bin: Path,
    specs: Tuple[str, ...],
    runner: Callable[..., Tuple[int, str, str]],
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Probes the isolated interpreter for requirement satisfaction.

    Returns (missing_specs, error). missing_specs is None on probe failure.
    """
    cmd = [
        str(python_bin),
        "-c",
        f'"""{DEPENDENCY_PROBE_MARKER}"""' + "\n" + _DEPENDENCY_PROBE_SCRIPT,
        json.dumps(list(specs)),
    ]
    retcode, stdout, stderr = runner(cmd, timeout=120)
    if retcode != 0:
        return None, stderr.strip() or stdout.strip() or f"依赖探测退出码 {retcode}"
    try:
        parsed = json.loads(stdout.strip().splitlines()[-1])
        if not parsed.get("ok"):
            return None, parsed.get("error") or "依赖探测返回失败"
        return list(parsed.get("missing") or []), None
    except Exception as exc:
        return None, f"无法解析依赖探测输出: {exc}"


def ensure_base_runtime_contract(
    data_dir: Path,
    profile: str,
    command_runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
) -> Dict[str, Any]:
    """Lightweight compatibility check for an ALREADY-VERIFIED runtime profile.

    Migrates historical runtimes (e.g. transformers 5.16.1 installed before
    the pin existed) to the base contract by installing only the violating
    packages. Torch itself is never reinstalled here: a missing torch means
    a broken venv that only the full install path may rebuild.
    """
    rt_manager = get_runtime_manager(data_dir)
    python_bin = rt_manager.get_python_bin(profile)
    if not python_bin.is_file():
        raise RuntimeError(f"运行时 [{profile}] 解释器不存在，无法校验 Base Runtime Contract: {python_bin}")

    runner = command_runner or _default_dependency_runner
    summary: Dict[str, Any] = {
        "profile": profile,
        "required": BASE_RUNTIME_REQUIREMENTS,
        "missing_initial": [],
        "installed": [],
        "satisfied": True,
    }

    missing, probe_err = _probe_dependency_specs(python_bin, BASE_RUNTIME_REQUIREMENTS, runner)
    if probe_err is not None:
        raise RuntimeError(f"运行时 [{profile}] Base Contract 探测失败: {probe_err}")
    summary["missing_initial"] = list(missing or [])
    if not missing:
        return summary

    violated = [spec for spec in missing if spec.split(">")[0].split("<")[0].split("=")[0].strip() == "torch"]
    if violated:
        raise RuntimeError(
            f"运行时 [{profile}] 缺少 PyTorch，属于损坏环境，请重建运行时而不应增量修复"
        )

    emit(f"运行时 [{profile}] Base Contract 迁移: 补齐 {', '.join(missing)}...")
    uv_bin = _find_uv()
    venv_dir = rt_manager.venv_dir(profile)
    env_pip = build_uv_env(data_dir) if uv_bin else build_runtime_env(data_dir)
    try:
        run_install_pypi_with_fallback(
            runner=runner,
            uv_bin=uv_bin,
            interp=python_bin,
            venv_dir=venv_dir,
            packages=missing,
            env=env_pip,
            emit_fn=emit,
        )
    except Exception as exc:
        raise RuntimeError(f"运行时 [{profile}] Base Contract 修复失败 ({', '.join(missing)}): {exc}")

    missing_after, probe_err = _probe_dependency_specs(python_bin, BASE_RUNTIME_REQUIREMENTS, runner)
    if probe_err is not None:
        raise RuntimeError(f"运行时 [{profile}] Base Contract 复核失败: {probe_err}")
    if missing_after:
        raise RuntimeError(f"运行时 [{profile}] Base Contract 修复后仍缺失: {', '.join(missing_after)}")

    summary["installed"] = list(missing)
    emit(f"运行时 [{profile}] Base Contract 迁移完成: {', '.join(missing)}")
    return summary


def ensure_model_runtime_dependencies(
    data_dir: Path,
    model_id: str,
    profile: str,
    command_runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
) -> Dict[str, Any]:
    """Resolves the model-specific runtime dependency contract inside an existing profile venv.

    Runs AFTER install_isolated_runtime (including its "already installed and
    verified, skip" fast path) and BEFORE model download / smoke inference.
    Idempotent: probes the target interpreter for declared requirements and
    installs ONLY what is missing; the shared runtime and PyTorch itself are
    never reinstalled.

    Raises RuntimeError when the venv cannot be probed or missing dependencies
    cannot be installed, which must prevent the model from reaching ready state.
    """
    descriptor = get_model_descriptor(model_id)
    if not descriptor:
        raise ValueError(f"未知的 ModelScope 模型标识: {model_id}")

    required = tuple(descriptor.runtime_dependencies)
    runner = command_runner or _default_dependency_runner
    summary: Dict[str, Any] = {
        "model_id": model_id,
        "profile": profile,
        "required": required,
        "missing_initial": [],
        "installed": [],
        "satisfied": True,
    }
    if not required:
        return summary

    rt_manager = get_runtime_manager(data_dir)
    python_bin = rt_manager.get_python_bin(profile)
    if not python_bin.is_file():
        raise RuntimeError(f"运行时 [{profile}] 解释器不存在，无法校验模型专属依赖: {python_bin}")

    def probe() -> Tuple[Optional[List[str]], Optional[str]]:
        return _probe_dependency_specs(python_bin, required, runner)

    emit(f"正在校验模型 [{descriptor.display_name}] 的专属运行依赖 ({len(required)} 项)...")
    missing, probe_err = probe()
    if probe_err is not None:
        raise RuntimeError(f"模型 [{model_id}] 依赖探测失败: {probe_err}")
    summary["missing_initial"] = list(missing or [])

    if not missing:
        emit(f"模型 [{model_id}] 专属运行依赖已满足，跳过安装。")
        return summary

    emit(f"模型 [{model_id}] 缺少专属运行依赖: {', '.join(missing)}，开始增量安装...")
    uv_bin = _find_uv()
    venv_dir = rt_manager.venv_dir(profile)
    env_pip = build_uv_env(data_dir) if uv_bin else build_runtime_env(data_dir)
    try:
        run_install_pypi_with_fallback(
            runner=runner,
            uv_bin=uv_bin,
            interp=python_bin,
            venv_dir=venv_dir,
            packages=missing,
            env=env_pip,
            emit_fn=emit,
        )
    except Exception as exc:
        raise RuntimeError(f"模型 [{model_id}] 专属依赖安装失败 ({', '.join(missing)}): {exc}")

    missing_after, probe_err = probe()
    if probe_err is not None:
        raise RuntimeError(f"模型 [{model_id}] 依赖复核失败: {probe_err}")
    if missing_after:
        raise RuntimeError(
            f"模型 [{model_id}] 依赖安装后仍缺失: {', '.join(missing_after)}"
        )

    summary["installed"] = list(missing)
    summary["satisfied"] = True
    clear_dependency_probe_cache()
    emit(f"模型 [{model_id}] 专属运行依赖安装完成: {', '.join(missing)}")
    return summary


_PROBE_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_PROBE_CACHE_LOCK = threading.Lock()


def clear_dependency_probe_cache() -> None:
    """Clears in-memory dependency probe cache."""
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE.clear()


def probe_model_runtime_dependencies(
    data_dir: Path,
    model_id: str,
    profile: str,
    command_runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Lightweight, read-only probe for model-specific runtime dependency satisfaction.

    Uses stdlib importlib inside the isolated target venv without loading weights
    or importing heavy machine learning frameworks.
    """
    descriptor = get_model_descriptor(model_id)
    if not descriptor:
        raise ValueError(f"未知的 ModelScope 模型标识: {model_id}")

    required = list(descriptor.runtime_dependencies)
    if not required:
        return {
            "model_id": model_id,
            "profile": profile,
            "required": [],
            "missing": [],
            "satisfied": True,
            "error": None,
        }

    cache_key = (str(Path(data_dir).resolve()), model_id, profile)
    if not force_refresh and command_runner is None:
        with _PROBE_CACHE_LOCK:
            if cache_key in _PROBE_CACHE:
                return dict(_PROBE_CACHE[cache_key])

    rt_manager = get_runtime_manager(data_dir)
    python_bin = rt_manager.get_python_bin(profile)
    if not python_bin.is_file():
        res = {
            "model_id": model_id,
            "profile": profile,
            "required": required,
            "missing": required,
            "satisfied": False,
            "error": f"运行时 [{profile}] 解释器不存在: {python_bin}",
        }
        return res

    runner = command_runner or _default_dependency_runner
    missing, probe_err = _probe_dependency_specs(python_bin, tuple(required), runner)
    if probe_err is not None:
        res = {
            "model_id": model_id,
            "profile": profile,
            "required": required,
            "missing": required,
            "satisfied": False,
            "error": probe_err,
        }
    else:
        missing_list = list(missing or [])
        res = {
            "model_id": model_id,
            "profile": profile,
            "required": required,
            "missing": missing_list,
            "satisfied": len(missing_list) == 0,
            "error": None,
        }

    if command_runner is None:
        with _PROBE_CACHE_LOCK:
            _PROBE_CACHE[cache_key] = res

    return res


def rebuild_runtime(
    data_dir: Path,
    profile: str,
    command_runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    emit_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Atomically rebuilds an isolated runtime venv using uv-managed Python.

    1. Acquires runtime_operation_lock.
    2. Verifies disk space preflight (fail-fast without mutating state).
    3. Finds all installed models that utilize this profile, and unions their runtime_dependencies.
       NEVER touches or re-downloads model weights in ${DATA_DIR}/models/*.
    4. Ensures uv-managed Python 3.12.9 is provisioned in ${DATA_DIR}/python/installations.
    5. Creates a staging venv in ${DATA_DIR}/runtime/${profile}/venv.rebuild-<timestamp>.
    6. Installs base ML packages in staging venv.
    7. Installs aggregated model dependencies in staging venv.
    8. Validates staging interpreter capabilities and runs real worker smoke tests.
    9. On success: stops running profile workers, atomically swaps staging -> venv, writes manifest v3.
    10. On failure: deletes staging venv, leaves existing venv untouched.
    """
    if profile not in (PROFILE_TORCH_CPU, PROFILE_TORCH_CUDA):
        raise ValueError(f"不支持的运行时 Profile: {profile}")

    log_emit = emit_fn or emit
    runner = command_runner or _default_dependency_runner

    with runtime_operation_lock(data_dir, profile):
        # 1. Disk space check
        space_ok, space_err = check_disk_space_for_rebuild(data_dir, profile)
        if not space_ok:
            log_emit(f"运行时 [{profile}] 重建中止: {space_err}")
            return False, space_err or "磁盘剩余空间不足"

        # 2. Discover installed models using this profile and aggregate dependencies
        installed_models: List[str] = []
        aggregated_deps: set = set()
        for mid, desc in MODEL_CATALOG.items():
            m_dir = get_model_dir(data_dir, mid)
            if m_dir.is_dir():
                integ_ok, _ = verify_model_integrity(m_dir, mid)
                if integ_ok:
                    if profile == PROFILE_TORCH_CUDA and desc.supports_cuda:
                        installed_models.append(mid)
                        aggregated_deps.update(desc.runtime_dependencies)
                    elif profile == PROFILE_TORCH_CPU and desc.supports_cpu:
                        installed_models.append(mid)
                        aggregated_deps.update(desc.runtime_dependencies)

        log_emit(f"运行时 [{profile}] 准备重建，关联已安装模型: {installed_models or '无'}")

        # 3. Ensure uv and managed Python
        uv_bin, uv_source = find_uv_info(data_dir=data_dir, runner=command_runner)
        if not uv_bin:
            err_msg = "未找到 uv 工具，无法构建托管 Python 运行环境"
            log_emit(f"错误: {err_msg}")
            return False, err_msg

        uv_source_desc = {
            "bundled": "内置",
            "managed": "托管",
            "system": "系统",
            "legacy": "系统备用",
            "custom": "自定义",
        }.get(uv_source, "可用")
        log_emit(f"已选择 AIPrivacyCheck {uv_source_desc} uv: {uv_bin} (版本: {UV_VERSION})")

        try:
            managed_python, py_download_source = ensure_managed_python_info(
                data_dir, uv_bin=uv_bin, runner=command_runner, emit_fn=log_emit
            )
        except Exception as exc:
            err_msg = f"准备托管 Python 失败: {exc}"
            log_emit(f"错误: {err_msg}")
            return False, err_msg

        # 4. Prepare staging directory
        rt_manager = get_runtime_manager(data_dir)
        profile_dir = rt_manager.profile_dir(profile)
        profile_dir.mkdir(parents=True, exist_ok=True)
        venv_dir = rt_manager.venv_dir(profile)
        staging_id = int(time.time() * 1000)
        staging_venv = profile_dir / f"venv.rebuild-{staging_id}"
        staging_interp = staging_venv / "bin" / "python"

        try:
            log_emit(f"正在创建 [{profile}] 重建环境: {staging_venv}...")
            uv_env = build_uv_env(data_dir)
            create_cmd = [uv_bin, "venv", "--python", str(managed_python), str(staging_venv)]
            ret, stdout, stderr = runner(create_cmd, env=uv_env)
            if ret != 0:
                raise RuntimeError(f"创建临时虚拟环境失败: {stderr.strip() or stdout.strip()}")

            pypi_sources: set[str] = set()
            pypi_tls_modes: set[str] = set()
            torch_source: str = "unknown"
            torch_tls_mode: str = "uv-native"

            log_emit(f"正在临时环境中安装 [{profile}] 基础依赖 (modelscope, numpy, packaging, tqdm)...")
            s1 = run_install_pypi_with_fallback(
                runner=runner,
                uv_bin=uv_bin,
                interp=staging_interp,
                venv_dir=staging_venv,
                packages=["modelscope", "numpy", "packaging", "tqdm"],
                env=uv_env,
                emit_fn=log_emit,
            )
            pypi_sources.add(s1.source if hasattr(s1, "source") else str(s1))
            if hasattr(s1, "tls_mode"):
                pypi_tls_modes.add(s1.tls_mode)

            if profile == PROFILE_TORCH_CPU:
                log_emit(f"正在安装 PyTorch CPU ({PINNED_TORCH_REQUIREMENT})...")
                s_torch = run_install_torch_with_fallback(
                    runner=runner,
                    uv_bin=uv_bin,
                    interp=staging_interp,
                    venv_dir=staging_venv,
                    profile=profile,
                    env=uv_env,
                    emit_fn=log_emit,
                )
                torch_source = s_torch.source if hasattr(s_torch, "source") else str(s_torch)
                if hasattr(s_torch, "tls_mode"):
                    torch_tls_mode = s_torch.tls_mode
                log_emit("正在恢复 Base Runtime Contract (transformers, accelerate, gliner)...")
                s2 = run_install_pypi_with_fallback(
                    runner=runner,
                    uv_bin=uv_bin,
                    interp=staging_interp,
                    venv_dir=staging_venv,
                    packages=[TRANSFORMERS_REQUIREMENT, "accelerate", "gliner"],
                    env=uv_env,
                    emit_fn=log_emit,
                )
                pypi_sources.add(s2.source if hasattr(s2, "source") else str(s2))
                if hasattr(s2, "tls_mode"):
                    pypi_tls_modes.add(s2.tls_mode)
            elif profile == PROFILE_TORCH_CUDA:
                log_emit(f"正在安装 PyTorch CUDA ({PINNED_TORCH_REQUIREMENT})...")
                s_torch = run_install_torch_with_fallback(
                    runner=runner,
                    uv_bin=uv_bin,
                    interp=staging_interp,
                    venv_dir=staging_venv,
                    profile=profile,
                    env=uv_env,
                    emit_fn=log_emit,
                )
                torch_source = s_torch.source if hasattr(s_torch, "source") else str(s_torch)
                if hasattr(s_torch, "tls_mode"):
                    torch_tls_mode = s_torch.tls_mode
                log_emit("正在恢复 Base Runtime Contract (transformers, accelerate, gliner)...")
                s2 = run_install_pypi_with_fallback(
                    runner=runner,
                    uv_bin=uv_bin,
                    interp=staging_interp,
                    venv_dir=staging_venv,
                    packages=[TRANSFORMERS_REQUIREMENT, "accelerate", "gliner"],
                    env=uv_env,
                    emit_fn=log_emit,
                )
                pypi_sources.add(s2.source if hasattr(s2, "source") else str(s2))
                if hasattr(s2, "tls_mode"):
                    pypi_tls_modes.add(s2.tls_mode)

            if aggregated_deps:
                deps_list = sorted(aggregated_deps)
                log_emit(f"正在恢复已安装模型依赖 ({len(deps_list)} 项): {', '.join(deps_list)}...")
                s3 = run_install_pypi_with_fallback(
                    runner=runner,
                    uv_bin=uv_bin,
                    interp=staging_interp,
                    venv_dir=staging_venv,
                    packages=deps_list,
                    env=uv_env,
                    emit_fn=log_emit,
                )
                pypi_sources.add(s3.source if hasattr(s3, "source") else str(s3))
                if hasattr(s3, "tls_mode"):
                    pypi_tls_modes.add(s3.tls_mode)

            pypi_source_final = "official" if "official" in pypi_sources and len(pypi_sources) == 1 else ("cernet" if any("cernet" in s for s in pypi_sources) else ("official" if "official" in pypi_sources else "cernet"))
            pypi_tls_final = "system-certs" if "system-certs" in pypi_tls_modes else ("explicit-ca" if "explicit-ca" in pypi_tls_modes else "uv-native")
            download_sources_telemetry = {
                "managed_python": py_download_source,
                "python": py_download_source,
                "pypi": pypi_source_final,
                "torch": torch_source,
            }
            tls_modes_telemetry = {
                "pypi": pypi_tls_final,
                "torch": torch_tls_mode,
            }

            # Validate staging interpreter native capabilities
            log_emit("正在验证临时环境 Python 原生能力...")
            caps_report = probe_python_capabilities(staging_interp, runner=command_runner)
            if not caps_report.get("ok", False):
                missing_caps = caps_report.get("missing", [])
                raise RuntimeError(f"临时环境 Python 原生能力缺失: {', '.join(missing_caps)}")
            log_emit("Python Capability Contract: PASS")

            # Validate staging Base ML Contract
            log_emit("正在验证临时环境 Base ML Contract...")
            base_contract_report = probe_base_runtime_contract(staging_interp, runner=command_runner)
            if not base_contract_report.get("base_packages_ready", False):
                missing_base = base_contract_report.get("missing_base_packages", [])
                violations = base_contract_report.get("base_contract_violations", [])
                raise RuntimeError(f"临时环境 Base ML Contract 未满足: {', '.join(missing_base + violations)}")
            staged_packages = dict(base_contract_report.get("packages", {}))
            actual_torch_version = staged_packages.get("torch", PINNED_TORCH_VERSION)
            log_emit(f"Base ML Contract: PASS (torch {actual_torch_version})")

            # Smoke test installed models using staging interpreter
            client = get_worker_client(data_dir)
            for mid in installed_models:
                desc = get_model_descriptor(mid)
                m_dir = get_model_dir(data_dir, mid)
                m_device = "cuda" if profile == PROFILE_TORCH_CUDA else "cpu"
                log_emit(f"正在对模型 [{desc.display_name if desc else mid}] 进行临时环境冒烟测试...")
                smoke_ok, smoke_err = client.run_smoke_test(
                    model_id=mid,
                    model_path=m_dir,
                    profile=profile,
                    device=m_device,
                    python_bin=staging_interp,
                )
                if not smoke_ok:
                    raise RuntimeError(f"模型 [{mid}] 在临时环境中冒烟测试未通过: {smoke_err}")

            # All tests passed! Proceed with atomic switch
            log_emit(f"临时环境验证通过，正在停止 [{profile}] 旧 Worker 并原子切换...")
            client.stop_workers_for_profile(profile)

            manifest_data = {
                "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
                "profile": profile,
                "created_at": int(time.time()),
                "created_by": "AIPrivacyCheck/0.6.14",
                "uv": {
                    "version": UV_VERSION,
                    "source": uv_source,
                },
                "python": {
                    "provider": "uv-managed",
                    "version": MANAGED_PYTHON_VERSION,
                    "executable": str(venv_dir / "bin" / "python"),
                    "managed_python_path": str(managed_python),
                },
                "python_runtime_source": "managed",
                "python_runtime_version": MANAGED_PYTHON_VERSION,
                "packages": staged_packages,
                "capabilities": list(PYTHON_CAPABILITY_CONTRACT),
                "capabilities_verified_at": int(time.time()),
                "framework_version": actual_torch_version,
                "torch_version": actual_torch_version,
                "cuda_available": profile == PROFILE_TORCH_CUDA,
                "download_sources": download_sources_telemetry,
                "tls_modes": tls_modes_telemetry,
            }
            (staging_venv / "runtime-manifest.json").write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Transactional atomic swap with metadata consistency
            old_backup = profile_dir / f"venv.old.{staging_id}"
            old_manifest_backup = profile_dir / f"runtime-manifest.old.{staging_id}"
            old_installed_backup = profile_dir / f"installed.old.{staging_id}"

            manifest_target = profile_dir / "runtime-manifest.json"
            installed_target = profile_dir / "installed.json"
            manifest_tmp = profile_dir / "runtime-manifest.json.tmp"
            installed_tmp = profile_dir / "installed.json.tmp"

            old_moved = False
            new_promoted = False
            old_manifest_backed = False
            old_installed_backed = False

            installed_meta = {
                "profile": profile,
                "installed_at": int(time.time()),
                "rebuilt_at": int(time.time()),
                "python_runtime_source": "uv-managed",
                "python_runtime_version": MANAGED_PYTHON_VERSION,
                "torch_version": actual_torch_version,
                "transformers_requirement": TRANSFORMERS_REQUIREMENT,
                "packages": staged_packages,
                "download_sources": download_sources_telemetry,
                "tls_modes": tls_modes_telemetry,
            }

            try:
                # 1. Back up existing metadata if present
                if manifest_target.is_file():
                    shutil.copy2(manifest_target, old_manifest_backup)
                    old_manifest_backed = True
                if installed_target.is_file():
                    shutil.copy2(installed_target, old_installed_backup)
                    old_installed_backed = True

                # 2. Swap venv
                if venv_dir.exists():
                    if old_backup.exists():
                        shutil.rmtree(old_backup, ignore_errors=True)
                    os.replace(venv_dir, old_backup)
                    old_moved = True

                os.replace(staging_venv, venv_dir)
                new_promoted = True

                # 3. Final verification probe on promoted runtime FIRST
                clear_dependency_probe_cache()
                final_probe = rt_manager.probe_profile(profile, force_refresh=True)
                if not final_probe.get("verified", False) or final_probe.get("runtime_rebuild_required", False):
                    raise RuntimeError(
                        f"最终验证探针未通过: {final_probe.get('error') or 'Promoted runtime not verified'}"
                    )

                # 4. ONLY after final probe PASSES, write .tmp files and atomically replace
                manifest_tmp.write_text(
                    json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                installed_tmp.write_text(
                    json.dumps(installed_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                os.replace(manifest_tmp, manifest_target)
                os.replace(installed_tmp, installed_target)

                # 5. Success: clean up backups
                if old_backup.exists():
                    shutil.rmtree(old_backup, ignore_errors=True)
                if old_manifest_backup.exists():
                    old_manifest_backup.unlink(missing_ok=True)
                if old_installed_backup.exists():
                    old_installed_backup.unlink(missing_ok=True)

            except Exception as swap_exc:
                log_emit(f"运行时切换/验证发生异常: {swap_exc}，启动事务回滚...")
                if manifest_tmp.exists():
                    manifest_tmp.unlink(missing_ok=True)
                if installed_tmp.exists():
                    installed_tmp.unlink(missing_ok=True)

                rb_err = None
                try:
                    if new_promoted and venv_dir.exists():
                        shutil.rmtree(venv_dir, ignore_errors=True)
                    if old_moved and old_backup.exists():
                        os.replace(old_backup, venv_dir)
                        log_emit(f"旧运行环境已成功恢复回原路径: {venv_dir}")

                    if old_manifest_backed and old_manifest_backup.exists():
                        os.replace(old_manifest_backup, manifest_target)
                    elif not old_manifest_backed and manifest_target.exists():
                        manifest_target.unlink(missing_ok=True)

                    if old_installed_backed and old_installed_backup.exists():
                        os.replace(old_installed_backup, installed_target)
                    elif not old_installed_backed and installed_target.exists():
                        installed_target.unlink(missing_ok=True)

                except Exception as rollback_exc:
                    rb_err = f"CRITICAL RECOVERY ERROR: 运行环境回滚失败 ({rollback_exc})，旧环境备份保留在 {old_backup}"
                    log_emit(f"严重错误: {rb_err}")

                if rb_err:
                    raise RuntimeError(f"{swap_exc} | {rb_err}") from swap_exc
                raise swap_exc

            for mid in installed_models:
                write_state(data_dir, "ready", f"环境重建完成，模型已就绪: {get_model_dir(data_dir, mid)}", mid)

            clear_dependency_probe_cache()
            rt_manager.probe_profile(profile, force_refresh=True)
            DEVICE_MANAGER.invalidate_runtime_state()
            DEVICE_MANAGER.invalidate_cache()
            try:
                from privacy.service import PRIVACY
                PRIVACY.reset_models()
            except Exception:
                pass

            log_emit(f"运行时 [{profile}] 重建并切换完成。")
            return True, f"运行时 [{profile}] 重建完成"

        except Exception as exc:
            log_emit(f"运行时 [{profile}] 重建失败: {exc}，保留原环境")
            if staging_venv.exists():
                try:
                    shutil.rmtree(staging_venv, ignore_errors=True)
                except Exception:
                    pass
            return False, f"运行时 [{profile}] 重建失败: {exc}"


def repair_model_runtime(
    data_dir: Path,
    model_id: str,
    command_runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
) -> Tuple[bool, str]:
    """Repairs runtime environment and model-specific dependencies for an already-installed model.

    Preserves existing weights and intact runtime:
    - Never re-downloads model weights
    - Never rebuilds functional venv or reinstalls intact PyTorch
    - Triggers atomic rebuild if Python native capabilities are missing
    - Idempotently migrates base runtime contract
    - Installs only missing model-specific dependencies
    - Runs real worker smoke test to verify readiness
    - Invalidates runtime caches and resets models
    """
    descriptor = validate_model_id(model_id)
    target_dir = get_model_dir(data_dir, model_id)
    weights_ok, integrity_err = verify_model_integrity(target_dir, model_id)
    if not weights_ok:
        return False, f"模型权重校验失败，无法仅修复运行环境: {integrity_err}"

    runner = command_runner or _default_dependency_runner

    with model_operation_lock(data_dir, model_id):
        DEVICE_MANAGER.set_data_dir(data_dir)
        diag = DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
        req = DEVICE_MANAGER.get_requested_device()
        has_nv = bool(diag["hardware"].get("nvidia_available", False))

        target_profile: str
        target_device: str

        if req == "cuda":
            if not descriptor.supports_cuda:
                return False, f"模型 [{model_id}] 不支持 CUDA 加速。"
            if not has_nv:
                return False, "当前配置为 CUDA 模式，但主机未检测到可用 NVIDIA GPU 或驱动。"
            target_profile = PROFILE_TORCH_CUDA
            target_device = "cuda"
        elif req == "cpu":
            if not descriptor.supports_cpu:
                return False, f"模型 [{model_id}] 仅支持 CUDA 运行，不支持 CPU 模式。"
            target_profile = PROFILE_TORCH_CPU
            target_device = "cpu"
        else:  # "auto"
            if has_nv and descriptor.supports_cuda:
                target_profile = PROFILE_TORCH_CUDA
                target_device = "cuda"
            else:
                if not descriptor.supports_cpu:
                    return False, f"主机未检测到可用 NVIDIA GPU，且模型 [{model_id}] 不支持 CPU。"
                target_profile = PROFILE_TORCH_CPU
                target_device = "cpu"

        rt_manager = get_runtime_manager(data_dir)
        python_bin = rt_manager.get_python_bin(target_profile)
        if not python_bin.is_file():
            return False, f"运行时 [{target_profile}] 解释器不存在，请先安装运行时环境: {python_bin}"

        # Two-tier check: if Python native capabilities are missing or rebuild is required, rebuild runtime atomically!
        profile_probe = rt_manager.probe_profile(target_profile)
        if not profile_probe.get("python_runtime_ready", True) or profile_probe.get("runtime_rebuild_required", False):
            missing_caps = profile_probe.get("missing_python_capabilities", [])
            caps_desc = f" ({', '.join(missing_caps)})" if missing_caps else ""
            emit(f"检测到运行时 [{target_profile}] Python 原生能力不满足{caps_desc}，需要重建隔离环境...")
            write_state(data_dir, "installing", f"正在为 [{target_profile}] 重建完整 Python 隔离运行环境...", model_id)
            rebuild_ok, rebuild_err = rebuild_runtime(data_dir, target_profile, command_runner=runner)
            if not rebuild_ok:
                write_state(data_dir, "error", f"环境重建失败: {rebuild_err}", model_id)
                return False, f"环境重建失败: {rebuild_err}"
            emit(f"运行时 [{target_profile}] 隔离环境重建完成，继续校验模型专属状态...")

        emit(f"正在检查 [{target_profile}] 基础运行环境...")
        write_state(data_dir, "installing", f"正在检查 [{target_profile}] 基础运行环境...", model_id)
        try:
            ensure_base_runtime_contract(data_dir, target_profile, command_runner=runner)
            emit("基础运行环境已就绪。")
        except Exception as exc:
            write_state(data_dir, "error", f"基础运行环境校验失败: {exc}", model_id)
            return False, f"基础运行环境校验失败: {exc}"

        emit(f"正在检查 [{descriptor.display_name}] 模型专属依赖...")
        write_state(data_dir, "installing", f"正在检查并补齐模型 [{model_id}] 专属依赖...", model_id)
        try:
            ensure_model_runtime_dependencies(data_dir, model_id, target_profile, command_runner=runner)
        except Exception as exc:
            write_state(data_dir, "error", f"模型专属依赖安装失败: {exc}", model_id)
            return False, f"模型专属依赖安装失败: {exc}"

        emit("正在执行模型专属依赖 Probe...")
        probe_res = probe_model_runtime_dependencies(
            data_dir, model_id, target_profile, command_runner=runner, force_refresh=True
        )
        if not probe_res.get("satisfied"):
            missing = probe_res.get("missing", [])
            err_msg = f"依赖复核未通过，仍缺少: {', '.join(missing)}"
            write_state(data_dir, "error", err_msg, model_id)
            return False, err_msg
        emit("模型专属运行依赖已就绪。")

        # Worker smoke test
        emit(f"正在执行 {descriptor.display_name} 冒烟推理...")
        write_state(data_dir, "testing", "正在执行端到端冒烟推理验证...", model_id)
        client = get_worker_client(data_dir)
        smoke_ok, smoke_err = client.run_smoke_test(
            model_id=model_id,
            model_path=target_dir,
            profile=target_profile,
            device=target_device,
        )
        if not smoke_ok:
            write_state(data_dir, "error", f"冒烟推理验证未通过: {smoke_err}", model_id)
            emit(f"错误: 冒烟推理未通过 ({smoke_err})，模型权重已保留供排查。")
            return False, f"冒烟推理验证未通过: {smoke_err}"

        emit("冒烟推理通过。")
        write_state(data_dir, "ready", f"模型已就绪: {target_dir}", model_id)
        emit("运行环境修复完成。")

        # Invalidate caches
        clear_dependency_probe_cache()
        DEVICE_MANAGER.invalidate_runtime_state()
        DEVICE_MANAGER.invalidate_cache()
        try:
            from privacy.service import PRIVACY
            PRIVACY.reset_models()
        except Exception:
            pass

        return True, "运行环境修复完成"


def install_isolated_runtime(data_dir: Path, profile: str) -> Path:
    """Creates isolated Python virtual environment for a runtime profile and verifies it."""
    if profile not in (PROFILE_TORCH_CPU, PROFILE_TORCH_CUDA):
        raise ValueError(f"不支持的运行时 Profile: {profile}")

    rt_manager = get_runtime_manager(data_dir)
    profile_dir = rt_manager.profile_dir(profile)
    venv_dir = rt_manager.venv_dir(profile)
    profile_dir.mkdir(parents=True, exist_ok=True)

    interp = venv_dir / "bin" / "python"
    installed_file = profile_dir / "installed.json"

    if interp.is_file() and installed_file.is_file():
        probe = rt_manager.probe_profile(profile)
        if probe.get("verified", False) and not probe.get("runtime_rebuild_required", False):
            emit(f"运行时 [{profile}] 已就绪并通过验证，跳过安装。")
            # Verified only proves torch imports: historical runtimes may still
            # violate the base contract (e.g. transformers 5.x). Repair in place.
            ensure_base_runtime_contract(data_dir, profile)
            return venv_dir

    emit(f"正在为 [{profile}] 创建隔离 Python 运行环境: {venv_dir}...")

    uv_bin = find_uv()
    managed_python: Optional[Path] = None
    if uv_bin:
        try:
            managed_python = ensure_managed_python(data_dir, uv_bin=uv_bin)
        except Exception as exc:
            emit(f"获取托管 Python 提示: {exc}")

    if not interp.is_file():
        env_init = build_uv_env(data_dir) if uv_bin else build_runtime_env(data_dir)
        if uv_bin:
            cmd = [uv_bin, "venv"]
            if managed_python and managed_python.is_file():
                cmd.extend(["--python", str(managed_python)])
            cmd.append(str(venv_dir))
            subprocess.run(cmd, check=True, env=env_init)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, env=env_init)

    env_pip = build_uv_env(data_dir) if uv_bin else build_runtime_env(data_dir)

    pypi_sources: set[str] = set()
    pypi_tls_modes: set[str] = set()
    emit(f"正在安装 [{profile}] 基础依赖 (modelscope, numpy, packaging, tqdm)...")
    s1 = run_install_pypi_with_fallback(
        runner=_default_dependency_runner,
        uv_bin=uv_bin,
        interp=interp,
        venv_dir=venv_dir,
        packages=["modelscope", "numpy", "packaging", "tqdm"],
        env=env_pip,
        emit_fn=emit,
    )
    pypi_sources.add(s1.source if hasattr(s1, "source") else str(s1))
    if hasattr(s1, "tls_mode"):
        pypi_tls_modes.add(s1.tls_mode)

    torch_source: str = "unknown"
    torch_tls_mode: str = "uv-native"
    if profile == PROFILE_TORCH_CPU:
        emit(f"正在安装 [{profile}] PyTorch CPU ({PINNED_TORCH_REQUIREMENT})...")
        s_torch = run_install_torch_with_fallback(
            runner=_default_dependency_runner,
            uv_bin=uv_bin,
            interp=interp,
            venv_dir=venv_dir,
            profile=profile,
            env=env_pip,
            emit_fn=emit,
        )
        torch_source = s_torch.source if hasattr(s_torch, "source") else str(s_torch)
        if hasattr(s_torch, "tls_mode"):
            torch_tls_mode = s_torch.tls_mode
        s2 = run_install_pypi_with_fallback(
            runner=_default_dependency_runner,
            uv_bin=uv_bin,
            interp=interp,
            venv_dir=venv_dir,
            packages=[TRANSFORMERS_REQUIREMENT, "accelerate", "gliner"],
            env=env_pip,
            emit_fn=emit,
        )
        pypi_sources.add(s2.source if hasattr(s2, "source") else str(s2))
        if hasattr(s2, "tls_mode"):
            pypi_tls_modes.add(s2.tls_mode)
    elif profile == PROFILE_TORCH_CUDA:
        emit(f"正在安装 [{profile}] PyTorch CUDA (cu124) ({PINNED_TORCH_REQUIREMENT})...")
        s_torch = run_install_torch_with_fallback(
            runner=_default_dependency_runner,
            uv_bin=uv_bin,
            interp=interp,
            venv_dir=venv_dir,
            profile=profile,
            env=env_pip,
            emit_fn=emit,
        )
        torch_source = s_torch.source if hasattr(s_torch, "source") else str(s_torch)
        if hasattr(s_torch, "tls_mode"):
            torch_tls_mode = s_torch.tls_mode
        s2 = run_install_pypi_with_fallback(
            runner=_default_dependency_runner,
            uv_bin=uv_bin,
            interp=interp,
            venv_dir=venv_dir,
            packages=[TRANSFORMERS_REQUIREMENT, "accelerate", "gliner"],
            env=env_pip,
            emit_fn=emit,
        )
        pypi_sources.add(s2.source if hasattr(s2, "source") else str(s2))
        if hasattr(s2, "tls_mode"):
            pypi_tls_modes.add(s2.tls_mode)

    pypi_source_final = "official" if "official" in pypi_sources and len(pypi_sources) == 1 else ("cernet" if any("cernet" in s for s in pypi_sources) else ("official" if "official" in pypi_sources else "cernet"))
    pypi_tls_final = "system-certs" if "system-certs" in pypi_tls_modes else ("explicit-ca" if "explicit-ca" in pypi_tls_modes else "uv-native")
    download_sources_telemetry = {
        "managed_python": "existing-local" if managed_python else "system",
        "python": "managed" if managed_python else "system",
        "pypi": pypi_source_final,
        "torch": torch_source,
    }
    tls_modes_telemetry = {
        "pypi": pypi_tls_final,
        "torch": torch_tls_mode,
    }

    # Run genuine probe verification via isolated interpreter
    emit(f"正在对 [{profile}] 运行环境执行真实子进程 Probe 验证...")
    probe_result = rt_manager.probe_profile(profile, force_refresh=True)
    if not probe_result.get("verified", False):
        err = probe_result.get("error") or "探针验证失败"
        emit(f"错误: 运行时 [{profile}] 验证未通过: {err}")
        raise RuntimeError(f"隔离运行时 [{profile}] 验证未通过: {err}")

    probed_torch = probe_result.get("torch_version") or probe_result.get("framework_version")
    manifest_data = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "profile": profile,
        "created_at": int(time.time()),
        "created_by": "AIPrivacyCheck/0.6.14",
        "python_runtime_source": "managed" if managed_python else "system",
        "python_runtime_version": MANAGED_PYTHON_VERSION if managed_python else sys.version.split()[0],
        "python_interpreter": str(interp),
        "managed_python_path": str(managed_python) if managed_python else None,
        "packages": probe_result.get("packages", {}),
        "capabilities": list(PYTHON_CAPABILITY_CONTRACT),
        "capabilities_verified_at": int(time.time()),
        "framework_version": probed_torch,
        "torch_version": probed_torch,
        "cuda_available": probe_result.get("cuda_available", False),
        "download_sources": download_sources_telemetry,
        "tls_modes": tls_modes_telemetry,
    }
    (profile_dir / "runtime-manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "profile": profile,
        "installed_at": int(time.time()),
        "probe": probe_result,
        "torch_version": probed_torch,
        "packages": probe_result.get("packages", {}),
        "transformers_requirement": TRANSFORMERS_REQUIREMENT,
        "download_sources": download_sources_telemetry,
        "tls_modes": tls_modes_telemetry,
    }
    installed_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"运行时 [{profile}] 环境部署完成。")
    return venv_dir


def download_modelscope_model(data_dir: Path, model_id: str) -> Path:
    """Download model snapshot from ModelScope into transactional staging and activate."""
    with model_operation_lock(data_dir, model_id):
        descriptor = get_model_descriptor(model_id)
        if not descriptor:
            raise ValueError(f"未知的 ModelScope 模型标识: {model_id}")

        target_dir = get_model_dir(data_dir, model_id)
        ok, _ = verify_model_integrity(target_dir, model_id)
        if ok:
            emit(f"模型 [{model_id}] 已存在且完整，跳过下载。")
            # Re-sanitize pre-existing weights so upgrades of the app also
            # neutralize configs downloaded by older versions.
            sanitize_model_config(target_dir)
            return target_dir

        staging_dir = get_staging_dir(data_dir, model_id)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.parent.mkdir(parents=True, exist_ok=True)

        emit(f"正在从 ModelScope (魔搭社区) 下载模型 [{descriptor.display_name}] (repo: {descriptor.repo_id}, revision: {descriptor.revision})...")

        rt_manager = get_runtime_manager(data_dir)
        dl_profile = rt_manager.get_modelscope_capable_runtime()
        if dl_profile is None:
            emit(f"准备下载前先安装基础运行环境 [{PROFILE_TORCH_CPU}]...")
            install_isolated_runtime(data_dir, PROFILE_TORCH_CPU)
            dl_profile = PROFILE_TORCH_CPU

        python_bin = rt_manager.get_python_bin(dl_profile)
        downloader_script = Path(__file__).resolve().parent / "privacy" / "workers" / "modelscope_downloader.py"
        cmd = [
            str(python_bin),
            str(downloader_script),
            "--repo-id", descriptor.repo_id,
            "--revision", descriptor.revision,
            "--target-dir", str(staging_dir),
            "--model-id", model_id,
        ]

        emit(f"正在通过隔离运行时 [{dl_profile}] 执行 ModelScope 快照下载...")
        env = build_runtime_env(data_dir)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip() or f"Downloader exit code {proc.returncode}"
            try:
                parsed = json.loads(proc.stdout.strip())
                if parsed.get("error"):
                    err_msg = f"{parsed.get('error_type', 'Error')}: {parsed.get('error')}"
            except Exception:
                pass
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise RuntimeError(f"ModelScope 下载失败: {err_msg}")

        ok, reason = verify_model_integrity(staging_dir, model_id)
        if not ok:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise RuntimeError(f"ModelScope 模型完整性校验未通过: {reason}")

        sanitize_model_config(staging_dir)

        metadata = {
            "provider": "modelscope",
            "model_id": descriptor.id,
            "repo_id": descriptor.repo_id,
            "revision": descriptor.revision,
            "license": descriptor.license,
            "installed_at": int(time.time()),
            "size_bytes": get_dir_size(staging_dir),
        }
        (staging_dir / ".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

        # Transactional atomic replacement
        if target_dir.exists():
            old_backup = data_dir / "models" / f"{model_id}.old.tmp"
            if old_backup.exists():
                shutil.rmtree(old_backup, ignore_errors=True)
            os.replace(target_dir, old_backup)
            os.replace(staging_dir, target_dir)
            shutil.rmtree(old_backup, ignore_errors=True)
        else:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_dir, target_dir)

        shutil.rmtree(staging_dir, ignore_errors=True)
        emit(f"模型 [{model_id}] 激活成功: {target_dir}")
        return target_dir


def import_local_model(data_dir: Path, model_id: str, source_path: Path) -> Tuple[bool, str]:
    """Import an existing model from a user-authorized path into app data directory."""
    with model_operation_lock(data_dir, model_id):
        valid_ok, valid_reason, resolved_src = validate_import_source_path(source_path, data_dir)
        if not valid_ok or resolved_src is None:
            return False, valid_reason

        target_dir = get_model_dir(data_dir, model_id)
        staging_dir = get_staging_dir(data_dir, model_id)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.parent.mkdir(parents=True, exist_ok=True)

        emit(f"开始从本地授权路径导入模型 [{model_id}]，源路径: {resolved_src}")

        # Copy files into temporary staging area (source remains untouched)
        if resolved_src.is_file():
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_src, staging_dir / resolved_src.name)
        else:
            shutil.copytree(resolved_src, staging_dir, symlinks=False)

        ok, reason = verify_model_integrity(staging_dir, model_id)
        if not ok:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return False, f"模型导入格式校验失败: {reason}"

        sanitize_model_config(staging_dir)

        descriptor = get_model_descriptor(model_id)
        metadata = {
            "provider": "manual_import",
            "model_id": model_id,
            "repo_id": descriptor.repo_id if descriptor else "custom",
            "license": descriptor.license if descriptor else "unknown",
            "imported_from": str(resolved_src),
            "installed_at": int(time.time()),
            "size_bytes": get_dir_size(staging_dir),
        }
        (staging_dir / ".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

        # Atomic swap
        if target_dir.exists():
            old_backup = data_dir / "models" / f"{model_id}.old.tmp"
            if old_backup.exists():
                shutil.rmtree(old_backup, ignore_errors=True)
            os.replace(target_dir, old_backup)
            os.replace(staging_dir, target_dir)
            shutil.rmtree(old_backup, ignore_errors=True)
        else:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_dir, target_dir)

        shutil.rmtree(staging_dir, ignore_errors=True)

        # Strict synthetic smoke test verification
        diag = DEVICE_MANAGER.probe_diagnostics()
        model_res = DEVICE_MANAGER.resolve_for_model(model_id)
        profile = model_res.get("runtime_profile")
        device = model_res.get("actual_device", "cpu")

        if not profile:
            req = DEVICE_MANAGER.get_requested_device()
            has_nv = bool(diag["hardware"].get("nvidia_available", False))
            supports_cuda = descriptor.supports_cuda if descriptor else True
            supports_cpu = descriptor.supports_cpu if descriptor else True

            target_p = PROFILE_TORCH_CUDA if (has_nv and req in ("auto", "cuda") and supports_cuda) else PROFILE_TORCH_CPU
            if target_p == PROFILE_TORCH_CPU and not supports_cpu:
                write_state(data_dir, "error", f"模型 [{model_id}] 仅支持 CUDA，但无可用 CUDA 环境", model_id)
                return False, f"模型 [{model_id}] 仅支持 CUDA，但无可用 CUDA 环境"
            try:
                write_state(data_dir, "installing", f"正在为本地导入模型准备 [{target_p}] 隔离运行环境...", model_id)
                install_isolated_runtime(data_dir, target_p)
                profile = target_p
                device = "cuda" if target_p == PROFILE_TORCH_CUDA else "cpu"
            except Exception as rt_exc:
                if target_p == PROFILE_TORCH_CUDA and req == "auto" and supports_cpu:
                    emit(f"CUDA 环境安装失败，自动回退 CPU: {rt_exc}")
                    try:
                        install_isolated_runtime(data_dir, PROFILE_TORCH_CPU)
                        profile = PROFILE_TORCH_CPU
                        device = "cpu"
                    except Exception as cpu_exc:
                        write_state(data_dir, "error", f"隔离运行环境准备失败: {cpu_exc}", model_id)
                        return False, f"隔离运行环境准备失败: {cpu_exc}"
                else:
                    write_state(data_dir, "error", f"隔离运行环境准备失败: {rt_exc}", model_id)
                    return False, f"隔离运行环境准备失败: {rt_exc}"

        write_state(data_dir, "installing", f"正在校验模型 [{model_id}] 专属运行依赖...", model_id)
        ensure_model_runtime_dependencies(data_dir, model_id, profile)

        write_state(data_dir, "testing", "正在执行端到端合成冒烟推理验证...", model_id)
        client = get_worker_client(data_dir)
        smoke_ok, smoke_err = client.run_smoke_test(
            model_id=model_id,
            model_path=target_dir,
            profile=profile,
            device=device,
        )
        if not smoke_ok:
            write_state(data_dir, "error", f"模型已导入但冒烟测试失败: {smoke_err}", model_id)
            emit(f"警告: 模型 [{model_id}] 冒烟推理未通过 ({smoke_err})，权重已保留在 {target_dir}。")
            return False, f"模型导入后冒烟推理未通过: {smoke_err}"

        write_state(data_dir, "ready", f"模型已就绪: {target_dir}", model_id)
        emit(f"模型 [{model_id}] 本地导入并通过冒烟验证成功。")
        return True, f"成功导入模型到 {target_dir}"


def uninstall_model(data_dir: Path, model_id: str) -> Tuple[bool, str]:
    """Uninstall private model weights cleanly without touching user shared source models."""
    with model_operation_lock(data_dir, model_id):
        try:
            from privacy.worker_client import get_worker_client
            get_worker_client(data_dir).stop_worker_for_model(model_id)
        except Exception as exc:
            emit(f"停止模型 worker 提示: {exc}")

        target_dir = get_model_dir(data_dir, model_id)
        if not target_dir.exists():
            return True, "模型原本未安装"

        try:
            shutil.rmtree(target_dir)
            write_state(data_dir, "not_installed", "模型已被管理员卸载", model_id=model_id)
            emit(f"模型 [{model_id}] 已成功卸载。")
            return True, f"模型 {model_id} 已卸载"
        except Exception as exc:
            return False, f"卸载模型失败: {exc}"


def main() -> int:
    """Entry point for CLI or background installer execution."""
    data_dir = Path(os.environ.get("APP_DATA_DIR", "/data")).resolve()
    action = sys.argv[1] if len(sys.argv) > 1 else "install"

    if action == "rebuild":
        profile = sys.argv[2] if len(sys.argv) > 2 else PROFILE_TORCH_CPU
        if profile not in (PROFILE_TORCH_CPU, PROFILE_TORCH_CUDA):
            emit(f"错误: 无效的运行时 Profile {profile}，必须是 {PROFILE_TORCH_CPU} 或 {PROFILE_TORCH_CUDA}")
            return 1
        ok, msg = rebuild_runtime(data_dir, profile)
        return 0 if ok else 1

    model_id = sys.argv[2] if len(sys.argv) > 2 else "gliner-pii-edge"

    descriptor = get_model_descriptor(model_id)
    if not descriptor:
        write_state(data_dir, "error", f"未知的模型标识: {model_id}", model_id)
        emit(f"错误: 未知的模型标识 {model_id}")
        return 1

    supports_cpu = descriptor.supports_cpu
    supports_cuda = descriptor.supports_cuda

    try:
        if action == "install":
            with model_operation_lock(data_dir, model_id):
                diag = DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
                req = DEVICE_MANAGER.get_requested_device()
                has_nv = bool(diag["hardware"].get("nvidia_available", False))

                target_profile: str
                target_device: str

                if req == "cuda":
                    if not supports_cuda:
                        raise RuntimeError(f"模型 [{model_id}] 不支持 CUDA 加速。")
                    if not has_nv:
                        raise RuntimeError("用户显式配置使用 NVIDIA CUDA，但主机未检测到可用 NVIDIA GPU 或驱动。")
                    target_profile = PROFILE_TORCH_CUDA
                    target_device = "cuda"
                    write_state(data_dir, "installing", f"正在准备 [{target_profile}] 隔离运行环境...", model_id)
                    install_isolated_runtime(data_dir, target_profile)

                elif req == "cpu":
                    if not supports_cpu:
                        raise RuntimeError(f"模型 [{model_id}] 仅支持 CUDA 运行，不支持 CPU 模式。")
                    target_profile = PROFILE_TORCH_CPU
                    target_device = "cpu"
                    write_state(data_dir, "installing", f"正在准备 [{target_profile}] 隔离运行环境...", model_id)
                    install_isolated_runtime(data_dir, target_profile)

                else:  # "auto"
                    if has_nv and supports_cuda:
                        write_state(data_dir, "installing", f"检测到 NVIDIA GPU，正在准备 [{PROFILE_TORCH_CUDA}] 隔离运行环境...", model_id)
                        try:
                            install_isolated_runtime(data_dir, PROFILE_TORCH_CUDA)
                            target_profile = PROFILE_TORCH_CUDA
                            target_device = "cuda"
                        except Exception as cuda_exc:
                            emit(f"CUDA 运行环境部署失败: {cuda_exc}")
                            if not supports_cpu:
                                raise RuntimeError(f"CUDA 运行时不可用，且模型 [{model_id}] 不支持 CPU: {cuda_exc}")
                            emit("自动降级至 CPU 运行环境...")
                            write_state(data_dir, "installing", f"CUDA 环境失败，正在回退准备 [{PROFILE_TORCH_CPU}] 运行环境...", model_id)
                            target_profile = PROFILE_TORCH_CPU
                            target_device = "cpu"
                            install_isolated_runtime(data_dir, PROFILE_TORCH_CPU)
                    else:
                        if not supports_cpu:
                            raise RuntimeError(f"主机未检测到可用 NVIDIA GPU，且模型 [{model_id}] 不支持 CPU。")
                        target_profile = PROFILE_TORCH_CPU
                        target_device = "cpu"
                        write_state(data_dir, "installing", f"正在准备 [{PROFILE_TORCH_CPU}] 隔离运行环境...", model_id)
                        install_isolated_runtime(data_dir, PROFILE_TORCH_CPU)

                # Model-specific runtime dependency contract: resolve extras
                # even when the shared runtime above was skipped as ready.
                write_state(data_dir, "installing", f"正在校验模型 [{model_id}] 专属运行依赖...", model_id)
                ensure_model_runtime_dependencies(data_dir, model_id, target_profile)

                target_dir = get_model_dir(data_dir, model_id)
                weights_ok, _ = verify_model_integrity(target_dir, model_id)
                if not weights_ok:
                    write_state(data_dir, "downloading", "正在从 ModelScope 下载模型权重...", model_id)
                checkpoint_dir = download_modelscope_model(data_dir, model_id)

                # Strict Synthetic smoke test
                write_state(data_dir, "testing", "正在执行端到端合成冒烟推理验证...", model_id)
                emit(f"正在通过隔离运行时 [{target_profile}] 执行冒烟推理验证...")
                client = get_worker_client(data_dir)
                smoke_ok, smoke_err = client.run_smoke_test(
                    model_id=model_id,
                    model_path=checkpoint_dir,
                    profile=target_profile,
                    device=target_device,
                )
                if not smoke_ok:
                    write_state(data_dir, "error", f"冒烟推理验证未通过: {smoke_err}", model_id)
                    emit(f"错误: 冒烟推理未通过 ({smoke_err})，模型权重已保留供排查。")
                    return 1

                write_state(data_dir, "ready", f"模型已就绪: {checkpoint_dir}", model_id)
                emit("安装流程全部完成。")
                return 0
        elif action == "repair":
            write_state(data_dir, "installing", f"正在检查并修复模型 [{model_id}] 运行环境...", model_id)
            ok, msg = repair_model_runtime(data_dir, model_id)
            return 0 if ok else 1
        elif action == "import":
            if len(sys.argv) < 4:
                emit("用法: python model_installer.py import <model_id> <source_path>")
                return 1
            source_path = Path(sys.argv[3]).resolve()
            ok, msg = import_local_model(data_dir, model_id, source_path)
            return 0 if ok else 1
        elif action == "uninstall":
            ok, msg = uninstall_model(data_dir, model_id)
            return 0 if ok else 1
        else:
            emit(f"未知操作: {action}")
            return 1
    except Exception as exc:
        write_state(data_dir, "error", str(exc), model_id)
        emit(f"执行失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

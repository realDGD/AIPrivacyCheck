"""Model lifecycle management for AI Privacy Check.

All officially supported models originate exclusively from ModelScope (魔搭社区).
Supports:
1. Online transactional download and installation via ModelScope.
2. Local manual import from user-authorized fnOS directories (e.g. data-share models)
   with strict authorization boundary validation, symlink escape protection,
   and atomic replacement without mutating user source files.
3. Scanning user-visible fnOS data-share models folder (AI 脱敏器/models).
4. Isolated runtime virtual environments (torch-cpu, torch-cuda, paddle-cpu, paddle-cuda)
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
    get_runtime_manager,
)
from privacy.worker_client import get_worker_client

_LOCK_STATE = threading.local()


@contextmanager
def model_operation_lock(data_dir: Path, model_id: str, non_blocking: bool = True):
    """Acquires a cross-process mutual exclusion file lock for a specific model operation.

    Uses fcntl.flock on ${DATA_DIR}/locks/{model_id}.lock.
    Re-entrant within the same thread.
    """
    locks_dir = Path(data_dir) / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (locks_dir / f"{model_id}.lock").resolve()

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


def emit(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def get_model_dir(data_dir: Path, model_id: str) -> Path:
    return data_dir / "models" / model_id


def get_staging_dir(data_dir: Path, model_id: str) -> Path:
    return data_dir / "models" / ".staging" / model_id


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
    return check_model_integrity(model_dir, model_id)


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
        if probe.get("verified", False):
            emit(f"运行时 [{profile}] 已就绪并通过验证，跳过安装。")
            return venv_dir

    emit(f"正在为 [{profile}] 创建隔离 Python 运行环境: {venv_dir}...")

    # Locate uv tool as required by environment rules
    uv_bin = shutil.which("uv")
    if not uv_bin:
        for p in ("/usr/local/bin/uv", "/opt/homebrew/bin/uv", os.path.expanduser("~/.cargo/bin/uv")):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                uv_bin = p
                break

    if not interp.is_file():
        if uv_bin:
            cmd = [uv_bin, "venv", str(venv_dir)]
            subprocess.run(cmd, check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    def run_install(*args: str) -> None:
        if uv_bin:
            cmd = [uv_bin, "pip", "install", "--python", str(interp), *args]
        else:
            pip_bin = str(venv_dir / "bin" / "pip")
            cmd = [pip_bin, "install", "--disable-pip-version-check", "--no-input", "--upgrade", *args]
        subprocess.run(cmd, check=True)

    emit(f"正在安装 [{profile}] 基础依赖 (modelscope, numpy, packaging, tqdm)...")
    run_install("modelscope", "numpy", "packaging", "tqdm", "-i", PYPI_MIRROR_URL)

    if profile == PROFILE_TORCH_CPU:
        emit(f"正在安装 [{profile}] PyTorch CPU 官方轮子...")
        run_install("torch", "--index-url", PYTORCH_CPU_INDEX)
        run_install("transformers", "accelerate", "gliner", "-i", PYPI_MIRROR_URL)
    elif profile == PROFILE_TORCH_CUDA:
        emit(f"正在安装 [{profile}] PyTorch CUDA (cu124) 官方轮子...")
        run_install("torch", "--index-url", PYTORCH_CUDA_INDEX)
        run_install("transformers", "accelerate", "gliner", "-i", PYPI_MIRROR_URL)

    # Run genuine probe verification via isolated interpreter
    emit(f"正在对 [{profile}] 运行环境执行真实子进程 Probe 验证...")
    probe_result = rt_manager.probe_profile(profile, force_refresh=True)
    if not probe_result.get("verified", False):
        err = probe_result.get("error") or "探针验证失败"
        emit(f"错误: 运行时 [{profile}] 验证未通过: {err}")
        raise RuntimeError(f"隔离运行时 [{profile}] 验证未通过: {err}")

    metadata = {
        "profile": profile,
        "installed_at": int(time.time()),
        "probe": probe_result,
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
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

                write_state(data_dir, "downloading", "正在从 ModelScope 下载模型权重...", model_id)
                checkpoint_dir = download_modelscope_model(data_dir, model_id)

                # Strict Synthetic smoke test
                write_state(data_dir, "testing", "正在执行端到端合成冒烟推理验证...", model_id)
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

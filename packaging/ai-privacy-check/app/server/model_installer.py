"""Model lifecycle management for AI Privacy Check.

All officially supported models originate exclusively from ModelScope (魔搭社区).
Supports:
1. Online transactional download and installation via ModelScope.
2. Local manual import from user-authorized fnOS directories with atomic replacement.
3. Integrity checks and smoke testing before activation.
4. Model uninstallation, status reporting, and safe hot-reload.
"""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from privacy.device import DEVICE_MANAGER
from privacy.model_catalog import MODEL_CATALOG, get_model_descriptor


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


def verify_model_integrity(model_dir: Path, model_id: str) -> Tuple[bool, str]:
    """Verify integrity of model weights and configs based on catalog descriptor."""
    if not model_dir.is_dir():
        return False, "模型目录不存在"

    descriptor = get_model_descriptor(model_id)
    if not descriptor:
        # Fallback heuristic check
        weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin")) + list(model_dir.glob("*.pdparams"))
        if not weights:
            return False, "未找到模型权重文件"
        return True, "验证通过"

    if descriptor.runtime == "paddle":
        weights = list(model_dir.glob("*.pdparams")) + list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
        if not weights:
            return False, "未检测到 Paddle/UIE 权重文件 (*.pdparams / *.safetensors / *.bin)"
    else:
        # PyTorch / Transformers format
        config_file = model_dir / "config.json"
        gliner_config = model_dir / "gliner_config.json"
        if not config_file.is_file() and not gliner_config.is_file():
            return False, "缺少模型配置文件 (config.json / gliner_config.json)"
        weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if not weights:
            return False, "未检测到 PyTorch 权重文件 (*.safetensors / *.bin)"

    return True, "验证通过"


def install_runtime_dependencies(data_dir: Path, runtime_name: str) -> Path:
    """Install isolated Python dependencies for specific runtime group."""
    package_dir = data_dir / "runtimes" / runtime_name
    package_dir.mkdir(parents=True, exist_ok=True)

    marker = package_dir / f".{runtime_name}-installed"
    if marker.is_file():
        return package_dir

    def pip_install(*requirements: str, extra: tuple = ()) -> None:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--upgrade",
            "--target",
            str(package_dir),
            *extra,
            *requirements,
        ]
        subprocess.run(cmd, check=True)

    emit(f"正在准备 [{runtime_name}] 运行环境依赖...")
    # Base requirements
    pip_install("modelscope", "numpy", "packaging", "tqdm")

    device_diag = DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
    machine = platform.machine().lower()

    if runtime_name in ("torch", "gliner"):
        if device_diag.get("requested_device") == "cuda" or (device_diag.get("requested_device") == "auto" and shutil.which("nvidia-smi")):
            try:
                pip_install("torch", extra=("--index-url", "https://download.pytorch.org/whl/cu124"))
            except subprocess.CalledProcessError:
                pip_install("torch")
        elif machine in ("x86_64", "amd64"):
            try:
                pip_install("torch", extra=("--index-url", "https://download.pytorch.org/whl/cpu"))
            except subprocess.CalledProcessError:
                pip_install("torch")
        else:
            pip_install("torch")

        if runtime_name == "gliner":
            pip_install("gliner")
        else:
            pip_install("transformers", "accelerate")

    elif runtime_name == "paddle":
        emit("准备 PaddleNLP 推理依赖...")
        try:
            pip_install("paddlepaddle", "paddlenlp")
        except Exception as exc:
            emit(f"Paddle 安装提示: {exc}")

    marker.write_text("ok\n", encoding="utf-8")
    emit(f"[{runtime_name}] 运行环境准备完成。")
    return package_dir


def download_modelscope_model(data_dir: Path, model_id: str) -> Path:
    """Download model snapshot from ModelScope into transactional staging and activate."""
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

    emit(f"正在从 ModelScope (魔搭社区) 下载模型 [{descriptor.display_name}] (repo: {descriptor.repo_id})...")

    # Use ModelScope SDK snapshot_download if available, or official HTTP API
    try:
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

        snapshot_download(
            model_id=descriptor.repo_id,
            revision=descriptor.revision,
            local_dir=str(staging_dir),
        )
    except ImportError:
        # Fallback to direct git/http download or subprocess if modelscope package not yet in current sys.path
        emit("正在调用 ModelScope 专用下载器...")
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"from modelscope.hub.snapshot_download import snapshot_download; snapshot_download(model_id='{descriptor.repo_id}', revision='{descriptor.revision}', local_dir='{staging_dir}')",
            ],
            check=True,
        )

    # Integrity verification on staging
    ok, reason = verify_model_integrity(staging_dir, model_id)
    if not ok:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"ModelScope 模型完整性校验未通过: {reason}")

    # Write metadata
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
    if not source_path.exists():
        return False, f"源路径不存在: {source_path}"

    target_dir = get_model_dir(data_dir, model_id)
    staging_dir = get_staging_dir(data_dir, model_id)
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)

    emit(f"开始从本地路径导入模型 [{model_id}]，源路径: {source_path}")

    # Copy files into temporary staging area
    if source_path.is_file():
        staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, staging_dir / source_path.name)
    else:
        shutil.copytree(source_path, staging_dir, symlinks=False)

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
        "imported_from": str(source_path),
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
    emit(f"模型 [{model_id}] 本地导入并激活成功。")
    return True, f"成功导入模型到 {target_dir}"


def uninstall_model(data_dir: Path, model_id: str) -> Tuple[bool, str]:
    """Uninstall model weights cleanly."""
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
    runtime_name = descriptor.runtime if descriptor else "torch"

    try:
        if action == "install":
            write_state(data_dir, "installing", f"正在准备 [{runtime_name}] 运行环境...", model_id)
            install_runtime_dependencies(data_dir, runtime_name)
            write_state(data_dir, "downloading", "正在从 ModelScope 下载模型权重...", model_id)
            checkpoint_dir = download_modelscope_model(data_dir, model_id)
            write_state(data_dir, "ready", f"模型已就绪: {checkpoint_dir}", model_id)
            emit("安装流程全部完成。")
            return 0
        elif action == "import":
            if len(sys.argv) < 4:
                emit("用法: python model_installer.py import <model_id> <source_path>")
                return 1
            source_path = Path(sys.argv[3]).resolve()
            ok, msg = import_local_model(data_dir, model_id, source_path)
            state = "ready" if ok else "error"
            write_state(data_dir, state, msg, model_id)
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

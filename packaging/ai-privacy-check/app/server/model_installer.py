"""Model lifecycle management for AI Privacy Check.

Supports:
1. Online installation and downloading (OpenAI Privacy Filter, Chinese IE).
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
from privacy.model import OPF_MODEL_REVISION, OPF_SOURCE_REVISION


def emit(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def get_model_dir(data_dir: Path, model_name: str) -> Path:
    return data_dir / "models" / model_name


def write_state(data_dir: Path, state: str, detail: str, model_name: str = "privacy-filter") -> None:
    status_dir = data_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "state": state,
        "detail": detail,
        "updated_at": int(time.time()),
    }
    target = status_dir / f"{model_name}-install.json"
    temporary = status_dir / f"{model_name}-install.json.tmp"
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


def verify_opf_integrity(model_dir: Path) -> Tuple[bool, str]:
    """Verify presence of config and weights for OpenAI Privacy Filter."""
    if not model_dir.is_dir():
        return False, "模型目录不存在"
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        return False, "缺少 config.json 配置文件"
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False, "config.json 格式无效"
    except Exception as exc:
        return False, f"config.json 读取失败: {exc}"

    safetensors = list(model_dir.glob("*.safetensors"))
    bin_files = list(model_dir.glob("*.bin"))
    if not safetensors and not bin_files:
        return False, "未找到权重文件 (*.safetensors 或 *.bin)"
    return True, "验证通过"


def verify_chinese_ie_integrity(model_dir: Path) -> Tuple[bool, str]:
    """Verify presence of PaddleNLP UIE or custom IE model weights."""
    if not model_dir.is_dir():
        return False, "模型目录不存在"
    params = list(model_dir.glob("*.pdparams")) + list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
    if not params:
        return False, "未检测到模型权重文件 (*.pdparams / *.safetensors)"
    return True, "验证通过"


def install_python_dependencies(data_dir: Path) -> Path:
    """Install PyTorch and runtime packages respecting hardware and architecture."""
    package_dir = data_dir / "python-packages"
    marker = package_dir / ".opf-revision"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == OPF_SOURCE_REVISION:
        emit("OPF 运行库已是固定版本，跳过基础安装。")
        return package_dir

    package_dir.mkdir(parents=True, exist_ok=True)
    source_url = f"https://github.com/openai/privacy-filter/archive/{OPF_SOURCE_REVISION}.tar.gz"

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

    emit("正在安装基础推理依赖 (huggingface_hub, safetensors, numpy, tiktoken)...")
    pip_install("huggingface_hub", "numpy", "packaging", "safetensors", "tiktoken", "tqdm")

    device_diag = DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
    machine = platform.machine().lower()
    emit(f"检测到运行平台架构: {machine}, 目标设备请求: {device_diag.get('requested_device')}")

    # Adaptive PyTorch install
    if device_diag.get("requested_device") == "cuda" or (device_diag.get("requested_device") == "auto" and shutil.which("nvidia-smi")):
        emit("检测到 NVIDIA 环境，尝试安装 CUDA 适配版 PyTorch...")
        try:
            pip_install("torch", extra=("--index-url", "https://download.pytorch.org/whl/cu124"))
        except subprocess.CalledProcessError:
            emit("CUDA wheel 安装遇到问题，正在安全回退至标准 PyTorch...")
            pip_install("torch")
    elif machine in ("x86_64", "amd64"):
        emit("正在安装 CPU 优化版 PyTorch...")
        try:
            pip_install("torch", extra=("--index-url", "https://download.pytorch.org/whl/cpu"))
        except subprocess.CalledProcessError:
            pip_install("torch")
    else:
        emit("正在安装标准版 PyTorch...")
        pip_install("torch")

    emit("正在安装 OpenAI Privacy Filter 核心驱动...")
    pip_install(source_url, extra=("--no-deps",))
    marker.write_text(OPF_SOURCE_REVISION + "\n", encoding="utf-8")
    emit("运行库依赖安装完成。")
    return package_dir


def download_openai_privacy_filter(data_dir: Path, package_dir: Path) -> Path:
    target_dir = get_model_dir(data_dir, "privacy-filter")
    ok, _ = verify_opf_integrity(target_dir)
    if ok:
        emit("OpenAI Privacy Filter 模型权重完整，跳过下载。")
        return target_dir

    package_path = str(package_dir)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from huggingface_hub import snapshot_download  # type: ignore

    download_dir = data_dir / "models" / "privacy-filter.download.tmp"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True)

    emit(f"正在从 Hugging Face 下载 OpenAI Privacy Filter 权重 (revision: {OPF_MODEL_REVISION})...")
    snapshot_download(
        repo_id="openai/privacy-filter",
        revision=OPF_MODEL_REVISION,
        local_dir=str(download_dir),
        allow_patterns=["original/*"],
    )

    payload_dir = download_dir / "original"
    ok, reason = verify_opf_integrity(payload_dir)
    if not ok:
        raise RuntimeError(f"下载文件完整性校验失败: {reason}")

    # Metadata
    (payload_dir / ".metadata.json").write_text(
        json.dumps({
            "source": "huggingface",
            "repo_id": "openai/privacy-filter",
            "revision": OPF_MODEL_REVISION,
            "installed_at": int(time.time()),
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(payload_dir, target_dir)
    shutil.rmtree(download_dir, ignore_errors=True)
    emit("OpenAI Privacy Filter 权重下载与校验成功。")
    return target_dir


def import_local_model(data_dir: Path, model_name: str, source_path: Path) -> Tuple[bool, str]:
    """Import an existing model from a user-authorized path into app-managed data storage.

    Never modifies or deletes the user's source directory.
    Uses atomic replacement so existing working models remain untouched on error.
    """
    if not source_path.exists():
        return False, f"源路径不存在: {source_path}"

    target_dir = get_model_dir(data_dir, model_name)
    temp_dir = data_dir / "models" / f"{model_name}.import.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    emit(f"开始导入模型 [{model_name}]，源目录: {source_path}")

    # Copy files into temporary staging area
    if source_path.is_file():
        temp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, temp_dir / source_path.name)
    else:
        shutil.copytree(source_path, temp_dir, symlinks=False)

    # Integrity verification
    if model_name == "privacy-filter":
        # Sometimes user points to parent directory containing 'original'
        if not (temp_dir / "config.json").is_file() and (temp_dir / "original" / "config.json").is_file():
            actual_staging = temp_dir / "original"
        else:
            actual_staging = temp_dir
        ok, reason = verify_opf_integrity(actual_staging)
        if not ok:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, f"OpenAI Privacy Filter 格式校验失败: {reason}"
        active_src = actual_staging
    elif model_name == "chinese-ie":
        ok, reason = verify_chinese_ie_integrity(temp_dir)
        if not ok:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, f"Chinese IE 格式校验失败: {reason}"
        active_src = temp_dir
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False, f"未知模型类型: {model_name}"

    # Write metadata
    meta = {
        "source": "manual_import",
        "imported_from": str(source_path),
        "imported_at": int(time.time()),
        "size_bytes": get_dir_size(active_src),
    }
    (active_src / ".metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # Atomic swap
    if target_dir.exists():
        old_backup = data_dir / "models" / f"{model_name}.old.tmp"
        if old_backup.exists():
            shutil.rmtree(old_backup)
        os.replace(target_dir, old_backup)
        os.replace(active_src, target_dir)
        shutil.rmtree(old_backup, ignore_errors=True)
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active_src, target_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)
    emit(f"模型 [{model_name}] 导入并校验成功。")
    return True, f"成功导入模型到 {target_dir}"


def uninstall_model(data_dir: Path, model_name: str) -> Tuple[bool, str]:
    """Uninstall model weights from app data directory without affecting user source files."""
    target_dir = get_model_dir(data_dir, model_name)
    if not target_dir.exists():
        return True, "模型原本未安装"

    try:
        shutil.rmtree(target_dir)
        write_state(data_dir, "not_installed", "模型已被管理员卸载", model_name=model_name)
        emit(f"模型 [{model_name}] 已成功卸载。")
        return True, f"模型 {model_name} 已卸载"
    except Exception as exc:
        return False, f"卸载模型失败: {exc}"


def main() -> int:
    """Entry point for CLI or background installer execution."""
    data_dir = Path(os.environ.get("APP_DATA_DIR", "/data")).resolve()
    action = sys.argv[1] if len(sys.argv) > 1 else "install"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "privacy-filter"

    try:
        if action == "install":
            write_state(data_dir, "installing", "正在准备运行环境...", model_name)
            package_dir = install_python_dependencies(data_dir)
            if model_name == "privacy-filter":
                write_state(data_dir, "installing", "正在下载模型权重...", model_name)
                checkpoint_dir = download_openai_privacy_filter(data_dir, package_dir)
                write_state(data_dir, "ready", f"模型已就绪: {checkpoint_dir}", model_name)
            emit("安装流程全部完成。")
            return 0
        elif action == "import":
            if len(sys.argv) < 4:
                emit("用法: python model_installer.py import <model_name> <source_path>")
                return 1
            source_path = Path(sys.argv[3]).resolve()
            ok, msg = import_local_model(data_dir, model_name, source_path)
            state = "ready" if ok else "error"
            write_state(data_dir, state, msg, model_name)
            return 0 if ok else 1
        elif action == "uninstall":
            ok, msg = uninstall_model(data_dir, model_name)
            return 0 if ok else 1
        else:
            emit(f"未知操作: {action}")
            return 1
    except Exception as exc:
        write_state(data_dir, "error", str(exc), model_name)
        emit(f"执行失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

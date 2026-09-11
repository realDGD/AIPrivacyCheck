"""Install the pinned OPF runtime and checkpoint into app-owned storage."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from privacy.model import OPF_MODEL_REVISION, OPF_SOURCE_REVISION


def emit(message: str) -> None:
    print(message, flush=True)


def write_state(data_dir: Path, state: str, detail: str) -> None:
    status_dir = data_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "detail": detail,
        "updated_at": int(time.time()),
    }
    target = status_dir / "model-install.json"
    temporary = status_dir / "model-install.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def install_packages(data_dir: Path) -> Path:
    package_dir = data_dir / "python-packages"
    marker = package_dir / ".opf-revision"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == OPF_SOURCE_REVISION:
        emit("OPF 运行库已是固定版本，跳过安装。")
        return package_dir
    package_dir.mkdir(parents=True, exist_ok=True)
    source_url = "https://github.com/openai/privacy-filter/archive/{}.tar.gz".format(OPF_SOURCE_REVISION)

    def pip_install(*requirements: str, extra: tuple = ()) -> None:
        subprocess.run(
            [
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
            ],
            check=True,
        )

    emit("正在安装 OPF 基础依赖……")
    pip_install("huggingface_hub", "numpy", "packaging", "safetensors", "tiktoken", "tqdm")
    emit("正在安装 CPU 版 PyTorch……")
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        pip_install("torch", extra=("--index-url", "https://download.pytorch.org/whl/cpu"))
    else:
        pip_install("torch")
    emit("正在安装固定版本的 OPF 运行库……")
    pip_install(source_url, extra=("--no-deps",))
    marker.write_text(OPF_SOURCE_REVISION + "\n", encoding="utf-8")
    emit("OPF 运行库安装完成。")
    return package_dir


def checkpoint_is_ready(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "config.json").is_file() and any(checkpoint_dir.glob("*.safetensors"))


def install_checkpoint(data_dir: Path, package_dir: Path) -> Path:
    model_root = data_dir / "models"
    checkpoint_dir = model_root / "privacy-filter"
    if checkpoint_is_ready(checkpoint_dir):
        emit("模型权重已存在，跳过下载。")
        return checkpoint_dir

    package_path = str(package_dir)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from huggingface_hub import snapshot_download

    model_root.mkdir(parents=True, exist_ok=True)
    download_dir = model_root / "privacy-filter.download"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True)
    emit("正在下载 OpenAI Privacy Filter 权重（约 2.8GB）……")
    snapshot_download(
        repo_id="openai/privacy-filter",
        revision=OPF_MODEL_REVISION,
        local_dir=str(download_dir),
        allow_patterns=["original/*"],
    )
    payload_dir = download_dir / "original"
    if not checkpoint_is_ready(payload_dir):
        raise RuntimeError("下载完成，但权重目录校验失败")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    os.replace(payload_dir, checkpoint_dir)
    (checkpoint_dir / ".revision").write_text(OPF_MODEL_REVISION + "\n", encoding="utf-8")
    shutil.rmtree(download_dir)
    emit("模型权重下载完成。")
    return checkpoint_dir


def main() -> int:
    data_dir = Path(os.environ.get("APP_DATA_DIR", "/data")).resolve()
    try:
        write_state(data_dir, "installing", "正在安装模型运行环境")
        package_dir = install_packages(data_dir)
        write_state(data_dir, "installing", "正在下载模型权重")
        checkpoint_dir = install_checkpoint(data_dir, package_dir)
        write_state(data_dir, "ready", "模型已就绪：{}".format(checkpoint_dir))
        emit("OpenAI Privacy Filter 已就绪。")
        return 0
    except Exception as exc:
        write_state(data_dir, "error", str(exc))
        emit("安装失败：{}".format(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

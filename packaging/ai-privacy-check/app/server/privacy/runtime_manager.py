"""Isolated Model Runtime Manager for AI Privacy Check.

Manages isolated virtual environments (venv) for PyTorch CPU and CUDA profiles:
  - torch-cpu
  - torch-cuda

Decoupled from control-plane Python. Control plane does not import torch/transformers/gliner/modelscope.
Workers and framework verification run exclusively within the profile's isolated interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .python_runtime import (
    MANAGED_PYTHON_VERSION,
    REQUIRED_PYTHON_CAPABILITIES,
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    probe_python_capabilities,
)


PYPI_MIRROR_URL = "https://mirrors.aliyun.com/pypi/simple/"
PYPI_OFFICIAL_URL = "https://pypi.org/simple"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"

PROFILE_TORCH_CPU = "torch-cpu"
PROFILE_TORCH_CUDA = "torch-cuda"

ALL_PROFILES = (
    PROFILE_TORCH_CPU,
    PROFILE_TORCH_CUDA,
)


@dataclass(frozen=True)
class RuntimeProfileDescriptor:
    profile: str
    framework: str  # "torch"
    device_target: str  # "cpu" or "cuda"
    display_name: str
    description: str


RUNTIME_PROFILES: Dict[str, RuntimeProfileDescriptor] = {
    PROFILE_TORCH_CPU: RuntimeProfileDescriptor(
        profile=PROFILE_TORCH_CPU,
        framework="torch",
        device_target="cpu",
        display_name="PyTorch CPU 运行时",
        description="适用于 SiameseUIE、GLiNER 与 MemPrivacy 模型的纯 CPU 轻量级推理环境。",
    ),
    PROFILE_TORCH_CUDA: RuntimeProfileDescriptor(
        profile=PROFILE_TORCH_CUDA,
        framework="torch",
        device_target="cuda",
        display_name="PyTorch CUDA 运行时",
        description="基于 NVIDIA CUDA 加速的 PyTorch 环境，提供高吞吐深度模型推理能力。",
    ),
}


class RuntimeManager:
    """Manages independent virtual environments for each model runtime profile."""

    def __init__(
        self,
        data_dir: Path,
        control_python: Optional[str] = None,
        command_runner: Optional[Callable[..., Tuple[int, str, str]]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.runtimes_dir = data_dir / "runtimes"
        self.control_python = control_python or sys.executable
        self._runner = command_runner or self._default_runner
        self._lock = threading.Lock()
        self._probe_cache: Dict[str, Dict[str, Any]] = {}
        self._probe_generation = 0

    @staticmethod
    def _default_runner(
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 15,
    ) -> Tuple[int, str, str]:
        try:
            res = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            return res.returncode, res.stdout, res.stderr
        except FileNotFoundError:
            return 127, "", f"command not found: {cmd[0] if cmd else ''}"
        except subprocess.TimeoutExpired:
            return 124, "", "command timed out"
        except Exception as exc:
            return 1, "", str(exc)

    def profile_dir(self, profile: str) -> Path:
        return self.runtimes_dir / profile

    def venv_dir(self, profile: str) -> Path:
        return self.profile_dir(profile) / "venv"

    def interpreter_path(self, profile: str) -> Optional[Path]:
        for cand in (
            self.venv_dir(profile) / "bin" / "python3",
            self.venv_dir(profile) / "bin" / "python",
            self.venv_dir(profile) / "Scripts" / "python.exe",
        ):
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
        return None

    def get_python_bin(self, profile: str) -> Path:
        interp = self.interpreter_path(profile)
        if interp:
            return interp
        return self.venv_dir(profile) / "bin" / "python3"

    def is_installed(self, profile: str) -> bool:
        interp = self.interpreter_path(profile)
        installed_flag = self.profile_dir(profile) / "installed.json"
        return interp is not None and installed_flag.is_file()

    def invalidate_probe_cache(self, profile: Optional[str] = None) -> None:
        """Thread-safely invalidates cached runtime probe results."""
        with self._lock:
            self._probe_generation += 1
            if profile is None:
                self._probe_cache.clear()
            else:
                self._probe_cache.pop(profile, None)

    def manifest_file(self, profile: str) -> Path:
        return self.profile_dir(profile) / "runtime-manifest.json"

    def adopt_legacy_runtime(self, profile: str) -> bool:
        """Adopts an existing healthy legacy virtual environment into schema v3."""
        interp = self.interpreter_path(profile)
        if not interp:
            return False
        manifest_path = self.manifest_file(profile)
        if manifest_path.is_file():
            return False
        py_cap = probe_python_capabilities(interp, runner=self._runner)
        if not py_cap.get("ok"):
            return False
        try:
            manifest = {
                "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
                "profile": profile,
                "python": {
                    "provider": "adopted-legacy",
                    "version": py_cap.get("version"),
                    "executable": str(interp),
                    "capabilities_verified": True,
                },
                "base_contract_version": 2,
                "created_by": "AIPrivacyCheck/0.6.11",
                "created_at": int(time.time()),
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception:
            return False

    def probe_profile(self, profile: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Probes an isolated runtime profile via its dedicated interpreter."""
        with self._lock:
            if not force_refresh and profile in self._probe_cache:
                return dict(self._probe_cache[profile])
            probe_generation = self._probe_generation

        descriptor = RUNTIME_PROFILES.get(profile)
        if not descriptor:
            return {"installed": False, "verified": False, "error": f"Unknown profile: {profile}"}

        interp = self.interpreter_path(profile)
        if not interp:
            status: Dict[str, Any] = {
                "profile": profile,
                "framework": descriptor.framework,
                "device_target": descriptor.device_target,
                "display_name": descriptor.display_name,
                "installed": False,
                "verified": False,
                "cuda_available": False,
                "framework_version": None,
                "cuda_version": None,
                "device_name": None,
                "device_count": 0,
                "interpreter": None,
                "python_runtime_ready": False,
                "python_runtime_source": "none",
                "python_runtime_version": None,
                "missing_python_capabilities": [],
                "runtime_rebuild_required": False,
                "base_packages_ready": False,
                "error": "运行时未安装",
            }
            with self._lock:
                if probe_generation == self._probe_generation:
                    self._probe_cache[profile] = status
            return status

        # Unified single-subprocess probe: verifies stdlib/native Python capabilities first,
        # then framework (torch/cuda).
        probe_code = (
            "import json, sys\n"
            "cap_names = ['lzma', '_lzma', 'bz2', '_bz2', 'ssl', '_ssl', 'sqlite3', '_sqlite3', 'ctypes', '_ctypes', 'zlib', 'hashlib', 'json', 'multiprocessing', 'subprocess', 'venv', 'ensurepip']\n"
            "missing_caps = []\n"
            "for m in cap_names:\n"
            "    try: __import__(m)\n"
            "    except Exception: missing_caps.append(m)\n"
            "py_ok = len(missing_caps) == 0\n"
            "base_prefix = getattr(sys, 'base_prefix', sys.prefix)\n"
            "py_source = 'uv-managed' if 'installations' in base_prefix else ('legacy-system-python' if any(base_prefix.startswith(p) for p in ('/usr', '/lib', '/bin', '/System', '/private')) else 'custom')\n"
            "res = {\n"
            "  'python_runtime_ready': py_ok,\n"
            "  'python_runtime_source': py_source,\n"
            "  'python_runtime_version': '.'.join(map(str, sys.version_info[:3])),\n"
            "  'missing_python_capabilities': missing_caps,\n"
            "}\n"
            "if py_ok:\n"
            "    try:\n"
            "        import torch\n"
            "        is_cuda = bool(torch.cuda.is_available())\n"
            "        res.update({\n"
            "          'framework_version': torch.__version__,\n"
            "          'cuda_version': getattr(torch.version, 'cuda', None),\n"
            "          'cuda_available': is_cuda,\n"
            "          'device_count': torch.cuda.device_count() if is_cuda else 0,\n"
            "          'device_name': torch.cuda.get_device_name(0) if is_cuda and torch.cuda.device_count() > 0 else None\n"
            "        })\n"
            "    except Exception as e:\n"
            "        res.update({'torch_error': str(e), 'cuda_available': False})\n"
            "print(json.dumps(res))\n"
        )

        cmd = [str(interp), "-c", probe_code]
        retcode, stdout, stderr = self._runner(cmd, timeout=10)
        if retcode != 0:
            err_msg = stderr.strip() or stdout.strip() or f"Probe failed with code {retcode}"
            status = {
                "profile": profile,
                "framework": descriptor.framework,
                "device_target": descriptor.device_target,
                "display_name": descriptor.display_name,
                "installed": True,
                "verified": False,
                "cuda_available": False,
                "framework_version": None,
                "cuda_version": None,
                "device_name": None,
                "device_count": 0,
                "interpreter": str(interp),
                "python_runtime_ready": False,
                "python_runtime_source": "unknown",
                "python_runtime_version": None,
                "missing_python_capabilities": list(REQUIRED_PYTHON_CAPABILITIES),
                "runtime_rebuild_required": True,
                "base_packages_ready": False,
                "error": f"运行时验证失败: {err_msg}",
            }
        else:
            try:
                data = json.loads(stdout.strip())
                py_ready = bool(data.get("python_runtime_ready", True))
                py_source = str(data.get("python_runtime_source", "uv-managed"))
                py_ver = data.get("python_runtime_version")
                missing_caps = list(data.get("missing_python_capabilities", []))

                manifest_path = self.manifest_file(profile)
                if manifest_path.is_file():
                    try:
                        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        rebuild_required = not py_ready or bool(manifest_data.get("schema_version", 1) < 3 and not py_ready)
                    except Exception:
                        rebuild_required = not py_ready
                else:
                    if py_ready:
                        self.adopt_legacy_runtime(profile)
                        rebuild_required = False
                    else:
                        rebuild_required = True

                if not py_ready:
                    status = {
                        "profile": profile,
                        "framework": descriptor.framework,
                        "device_target": descriptor.device_target,
                        "display_name": descriptor.display_name,
                        "installed": True,
                        "verified": False,
                        "cuda_available": False,
                        "framework_version": None,
                        "cuda_version": None,
                        "device_name": None,
                        "device_count": 0,
                        "interpreter": str(interp),
                        "python_runtime_ready": False,
                        "python_runtime_source": py_source,
                        "python_runtime_version": py_ver,
                        "missing_python_capabilities": missing_caps,
                        "runtime_rebuild_required": True,
                        "base_packages_ready": False,
                        "error": f"Python 原生能力缺失: {', '.join(missing_caps)}",
                    }
                else:
                    cuda_avail = bool(data.get("cuda_available", False))
                    fw_version = data.get("framework_version")
                    base_pkgs_ready = fw_version is not None
                    if descriptor.device_target == "cuda":
                        fw_verified = cuda_avail
                        err = None if cuda_avail else "框架已安装，但未检测到可用 CUDA 驱动与硬件。"
                    else:
                        fw_verified = base_pkgs_ready
                        err = None if base_pkgs_ready else data.get("torch_error", "PyTorch 基础框架未就绪")

                    verified = fw_verified and not rebuild_required
                    status = {
                        "profile": profile,
                        "framework": descriptor.framework,
                        "device_target": descriptor.device_target,
                        "display_name": descriptor.display_name,
                        "installed": True,
                        "verified": verified,
                        "cuda_available": cuda_avail,
                        "framework_version": fw_version,
                        "cuda_version": data.get("cuda_version"),
                        "device_name": data.get("device_name"),
                        "device_count": data.get("device_count", 0),
                        "interpreter": str(interp),
                        "python_runtime_ready": True,
                        "python_runtime_source": py_source,
                        "python_runtime_version": py_ver,
                        "missing_python_capabilities": [],
                        "runtime_rebuild_required": rebuild_required,
                        "base_packages_ready": base_pkgs_ready,
                        "error": err,
                    }
            except Exception as parse_exc:
                status = {
                    "profile": profile,
                    "framework": descriptor.framework,
                    "device_target": descriptor.device_target,
                    "display_name": descriptor.display_name,
                    "installed": True,
                    "verified": False,
                    "cuda_available": False,
                    "framework_version": None,
                    "cuda_version": None,
                    "device_name": None,
                    "device_count": 0,
                    "interpreter": str(interp),
                    "python_runtime_ready": False,
                    "python_runtime_source": "unknown",
                    "python_runtime_version": None,
                    "missing_python_capabilities": list(REQUIRED_PYTHON_CAPABILITIES),
                    "runtime_rebuild_required": True,
                    "base_packages_ready": False,
                    "error": f"无法解析验证输出: {parse_exc}",
                }

        with self._lock:
            if probe_generation == self._probe_generation:
                self._probe_cache[profile] = status
        return status

    def probe_all(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        results = {}
        for profile in ALL_PROFILES:
            results[profile] = self.probe_profile(profile, force_refresh=force_refresh)
        return results

    def best_runtime_for_framework(
        self,
        framework: str = "torch",
        prefer_cuda: bool = True,
        hardware_nvidia_available: bool = False,
    ) -> Optional[str]:
        """Resolves the best available runtime profile."""
        cuda_profile = PROFILE_TORCH_CUDA
        cpu_profile = PROFILE_TORCH_CPU

        cuda_status = self.probe_profile(cuda_profile)
        cpu_status = self.probe_profile(cpu_profile)

        if prefer_cuda and hardware_nvidia_available:
            if cuda_status.get("installed") and cuda_status.get("verified") and cuda_status.get("cuda_available"):
                return cuda_profile

        if cpu_status.get("installed") and cpu_status.get("verified"):
            return cpu_profile

        if cuda_status.get("installed") and cuda_status.get("verified") and cuda_status.get("cuda_available"):
            return cuda_profile

        return None

    def get_modelscope_capable_runtime(self, preferred_profile: Optional[str] = None) -> Optional[str]:
        """Returns any installed runtime that can execute ModelScope downloads."""
        if preferred_profile and self.is_installed(preferred_profile):
            return preferred_profile
        for prof in (PROFILE_TORCH_CUDA, PROFILE_TORCH_CPU):
            if self.is_installed(prof):
                return prof
        return None

    def run_in_runtime(
        self,
        profile: str,
        args: List[str],
        cwd: Optional[Path] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> Tuple[int, str, str]:
        """Executes a command using the isolated virtual environment interpreter."""
        interp = self.interpreter_path(profile)
        if not interp:
            return 127, "", f"Runtime {profile} is not installed or interpreter missing."

        cmd = [str(interp)] + args
        env = os.environ.copy()
        venv_root = str(self.venv_dir(profile))
        env["VIRTUAL_ENV"] = venv_root
        env["PATH"] = f"{venv_root}/bin:{env.get('PATH', '')}"
        if env_overrides:
            env.update(env_overrides)

        return self._runner(cmd, cwd=cwd, env=env, timeout=timeout)


_GLOBAL_MANAGER: Optional[RuntimeManager] = None


def get_runtime_manager(data_dir: Optional[Path] = None) -> RuntimeManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        if data_dir is None:
            data_dir = Path("/tmp")
        _GLOBAL_MANAGER = RuntimeManager(data_dir)
    elif data_dir is not None and _GLOBAL_MANAGER.data_dir != data_dir:
        _GLOBAL_MANAGER = RuntimeManager(data_dir)
    return _GLOBAL_MANAGER

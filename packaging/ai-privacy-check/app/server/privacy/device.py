"""Compute device and runtime diagnostics for AI Privacy Check.

Completely decouples Hardware Probe (NVIDIA GPU / driver via nvidia-smi)
from Framework Runtime Probes (isolated PyTorch / Paddle virtual environments).

Does not import torch or paddle into the control plane Python process.
Supports per-model / per-framework device arbitration with graceful CPU fallback.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hardware import HARDWARE_PROBE, HardwareProbe
from .runtime_manager import (
    PROFILE_PADDLE_CPU,
    PROFILE_PADDLE_CUDA,
    PROFILE_TORCH_CPU,
    PROFILE_TORCH_CUDA,
    RuntimeManager,
    get_runtime_manager,
)


class DeviceManager:
    """Thread-safe hardware & runtime resolution and diagnostics."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        hardware_probe: Optional[HardwareProbe] = None,
        runtime_manager: Optional[RuntimeManager] = None,
    ) -> None:
        self.data_dir = data_dir or Path(os.environ.get("TRIM_PKGVAR", "/tmp")) / "data"
        self._hw_probe = hardware_probe or HARDWARE_PROBE
        self._rt_manager = runtime_manager or get_runtime_manager(self.data_dir)
        self._lock = threading.Lock()
        self._diagnostics_cache: Optional[Dict[str, Any]] = None

    def set_data_dir(self, data_dir: Path) -> None:
        with self._lock:
            self.data_dir = data_dir
            self._rt_manager = get_runtime_manager(data_dir)
            self._diagnostics_cache = None

    def get_requested_device(self) -> str:
        device = os.environ.get("AI_PRIVACY_DEVICE") or "auto"
        device = device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            return "auto"
        return device

    def set_requested_device(self, device: str) -> str:
        device = device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            device = "auto"
        os.environ["AI_PRIVACY_DEVICE"] = device
        with self._lock:
            self._diagnostics_cache = None
        return device

    def probe_diagnostics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Inspects hardware and framework runtimes without importing heavy frameworks into control plane."""
        with self._lock:
            if not force_refresh and self._diagnostics_cache is not None:
                return dict(self._diagnostics_cache)

        requested = self.get_requested_device()
        hw_info = self._hw_probe.probe_nvidia()
        has_nvidia = bool(hw_info.get("nvidia_available", False))

        rt_info = self._rt_manager.probe_all(force_refresh=force_refresh)
        torch_cpu = rt_info.get(PROFILE_TORCH_CPU, {})
        torch_cuda = rt_info.get(PROFILE_TORCH_CUDA, {})
        paddle_cpu = rt_info.get(PROFILE_PADDLE_CPU, {})
        paddle_cuda = rt_info.get(PROFILE_PADDLE_CUDA, {})

        warnings: List[str] = []

        # Overall CUDA availability across any runtime
        any_cuda_ready = (
            torch_cuda.get("installed", False) and torch_cuda.get("verified", False) and torch_cuda.get("cuda_available", False)
        ) or (
            paddle_cuda.get("installed", False) and paddle_cuda.get("verified", False) and paddle_cuda.get("cuda_available", False)
        )

        if requested == "cuda" and not any_cuda_ready:
            if has_nvidia:
                warnings.append("用户显式配置使用 NVIDIA CUDA，但未安装或未就绪任何 CUDA 运行时，已安全回退到 CPU。")
            else:
                warnings.append("用户显式配置使用 NVIDIA CUDA，但主机未检测到可用 NVIDIA GPU 或驱动，已安全回退到 CPU。")

        # Top-level actual device compatibility field
        if requested == "cuda":
            actual_device = "cuda" if any_cuda_ready else "cpu"
        elif requested == "auto":
            actual_device = "cuda" if (has_nvidia and any_cuda_ready) else "cpu"
        else:
            actual_device = "cpu"

        # Model / framework specific device decisions
        model_devices: Dict[str, Dict[str, Any]] = {}
        for fw in ("torch", "paddle"):
            chosen_profile = self._rt_manager.best_runtime_for_framework(
                fw,
                prefer_cuda=(requested in ("auto", "cuda")),
                hardware_nvidia_available=has_nvidia,
            )
            model_devices[fw] = {
                "profile": chosen_profile,
                "device": "cuda" if (chosen_profile and chosen_profile.endswith("-cuda")) else "cpu",
                "ready": chosen_profile is not None,
            }

        diag = {
            "requested_device": requested,
            "actual_device": actual_device,
            "hardware": hw_info,
            "runtimes": {
                "torch_cpu": torch_cpu,
                "torch_cuda": torch_cuda,
                "paddle_cpu": paddle_cpu,
                "paddle_cuda": paddle_cuda,
            },
            "model_devices": model_devices,
            # Legacy compatibility fields for existing consumers
            "cuda_available": has_nvidia and any_cuda_ready,
            "cuda_device_count": hw_info.get("gpu_count", 0),
            "cuda_device_name": hw_info["gpus"][0]["name"] if hw_info.get("gpus") else None,
            "torch_version": torch_cuda.get("framework_version") or torch_cpu.get("framework_version"),
            "cuda_version": hw_info.get("driver_version"),
            "warnings": warnings,
        }

        with self._lock:
            self._diagnostics_cache = diag
        return dict(diag)

    def resolve_for_framework(self, framework: str) -> Tuple[str, Optional[str], List[str]]:
        """Resolves target device ('cuda' or 'cpu'), active runtime profile, and warnings for a specific framework."""
        diag = self.probe_diagnostics()
        requested = str(diag["requested_device"])
        has_nvidia = bool(diag["hardware"].get("nvidia_available", False))
        warnings = list(diag.get("warnings", []))

        fw_info = diag["model_devices"].get(framework, {})
        profile = fw_info.get("profile")
        dev = fw_info.get("device", "cpu")

        if requested == "cuda" and dev != "cuda":
            warnings.append(f"框架 {framework} 请求 CUDA 但无可用 CUDA 运行时，使用 CPU 运行。")

        return dev, profile, warnings

    def resolve(self, requested: Optional[str] = None) -> Tuple[str, List[str]]:
        """Generic resolve method for backward compatibility."""
        if requested:
            self.set_requested_device(requested)
        diag = self.probe_diagnostics()
        return str(diag["actual_device"]), list(diag.get("warnings", []))


DEVICE_MANAGER = DeviceManager()

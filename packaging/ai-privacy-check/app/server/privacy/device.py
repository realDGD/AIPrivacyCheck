"""Compute device and runtime diagnostics for AI Privacy Check.

Completely decouples Hardware Probe (NVIDIA GPU / driver via nvidia-smi)
from Framework Runtime Probes (isolated PyTorch virtual environments).

Does not import torch/transformers/gliner into the control plane Python process.
Supports persistent device preferences via SettingsStore and per-model capability resolution.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hardware import HARDWARE_PROBE, HardwareProbe
from .model_catalog import resolve_for_model as catalog_resolve_for_model
from .runtime_manager import (
    PROFILE_TORCH_CPU,
    PROFILE_TORCH_CUDA,
    RuntimeManager,
    get_runtime_manager,
)
from .settings_store import SettingsStore, get_settings_store


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
        self._settings_store = get_settings_store(self.data_dir)
        self._lock = threading.Lock()
        self._diagnostics_cache: Optional[Dict[str, Any]] = None

    def set_data_dir(self, data_dir: Path) -> None:
        with self._lock:
            self.data_dir = data_dir
            self._rt_manager = get_runtime_manager(data_dir)
            self._settings_store = get_settings_store(data_dir)
            self._diagnostics_cache = None

    def get_requested_device(self) -> str:
        env_dev = os.environ.get("AI_PRIVACY_DEVICE")
        if env_dev and env_dev.strip().lower() in ("auto", "cpu", "cuda"):
            return env_dev.strip().lower()
        stored = self._settings_store.get_requested_device()
        if stored and stored.strip().lower() in ("auto", "cpu", "cuda"):
            return stored.strip().lower()
        return "auto"

    def set_requested_device(self, device: str) -> str:
        device = device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            device = "auto"
        self._settings_store.set_requested_device(device)
        with self._lock:
            self._diagnostics_cache = None
        from .worker_client import get_worker_client
        get_worker_client(self.data_dir).stop_all()
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

        warnings: List[str] = []

        # Overall CUDA availability across any runtime
        any_cuda_ready = bool(
            torch_cuda.get("installed", False)
            and torch_cuda.get("verified", False)
            and torch_cuda.get("cuda_available", False)
        )

        if requested == "cuda" and not any_cuda_ready:
            warnings.append("已显式选择 NVIDIA CUDA，但当前 CUDA 运行环境不可用。相关模型将保持未就绪状态，请安装/修复 CUDA Runtime，或切换至 Auto/CPU。")

        # Top-level actual device compatibility field
        if requested == "cuda":
            actual_device = "cuda" if any_cuda_ready else "none"
        elif requested == "auto":
            actual_device = "cuda" if (has_nvidia and any_cuda_ready) else "cpu"
        else:
            actual_device = "cpu"

        # Model / framework specific device decisions
        if requested == "cuda":
            chosen_profile = PROFILE_TORCH_CUDA if any_cuda_ready else None
            model_devices: Dict[str, Dict[str, Any]] = {
                "torch": {
                    "profile": chosen_profile,
                    "device": "cuda" if chosen_profile else "none",
                    "ready": chosen_profile is not None,
                }
            }
        else:
            chosen_profile = self._rt_manager.best_runtime_for_framework(
                "torch",
                prefer_cuda=(requested == "auto"),
                hardware_nvidia_available=has_nvidia,
            )
            model_devices = {
                "torch": {
                    "profile": chosen_profile,
                    "device": "cuda" if (chosen_profile and chosen_profile.endswith("-cuda")) else "cpu",
                    "ready": chosen_profile is not None,
                }
            }

        diag = {
            "requested_device": requested,
            "actual_device": actual_device,
            "hardware": hw_info,
            "runtimes": {
                "torch_cpu": torch_cpu,
                "torch_cuda": torch_cuda,
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

    def resolve_for_model(self, model_id: str) -> Dict[str, Any]:
        """Resolves runtime profile, target device, and readiness for a specific model."""
        hw_info = self._hw_probe.probe_nvidia()
        has_nvidia = bool(hw_info.get("nvidia_available", False))
        requested = self.get_requested_device()
        return catalog_resolve_for_model(
            model_id=model_id,
            requested_device=requested,
            runtime_manager=self._rt_manager,
            hardware_nvidia_available=has_nvidia,
        )

    def resolve_for_framework(self, framework: str = "torch") -> Tuple[str, Optional[str], List[str]]:
        """Resolves target device ('cuda' or 'cpu'), active runtime profile, and warnings."""
        diag = self.probe_diagnostics()
        requested = str(diag["requested_device"])
        has_nvidia = bool(diag["hardware"].get("nvidia_available", False))
        warnings = list(diag.get("warnings", []))

        fw_info = diag["model_devices"].get(framework, {})
        profile = fw_info.get("profile")

        if requested == "cuda":
            if profile == PROFILE_TORCH_CUDA:
                return "cuda", profile, warnings
            warnings.append("已显式选择 NVIDIA CUDA，但当前 CUDA 运行环境不可用。相关模型保持未就绪。")
            return "none", None, warnings
        elif requested == "auto":
            if has_nvidia and profile == PROFILE_TORCH_CUDA:
                return "cuda", profile, warnings
            return "cpu", profile, warnings
        else:
            return "cpu", profile, warnings
    def resolve(self, requested: Optional[str] = None) -> Tuple[str, List[str]]:
        """Generic resolve method for backward compatibility."""
        if requested:
            self.set_requested_device(requested)
        diag = self.probe_diagnostics()
        return str(diag["actual_device"]), list(diag.get("warnings", []))


DEVICE_MANAGER = DeviceManager()

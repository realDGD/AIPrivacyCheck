"""Compute device management (Auto / CPU / NVIDIA CUDA) with lazy probing and graceful fallback."""

import os
import threading
from typing import Dict, List, Optional, Tuple


class DeviceManager:
    """Thread-safe device resolution and diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._diagnostics_cache: Optional[Dict[str, object]] = None

    def get_requested_device(self) -> str:
        device = os.environ.get("AI_PRIVACY_DEVICE") or os.environ.get("OPF_DEVICE") or "auto"
        device = device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            return "auto"
        return device

    def set_requested_device(self, device: str) -> str:
        device = device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            device = "auto"
        os.environ["AI_PRIVACY_DEVICE"] = device
        os.environ["OPF_DEVICE"] = device
        with self._lock:
            self._diagnostics_cache = None
        return device

    def probe_diagnostics(self, force_refresh: bool = False) -> Dict[str, object]:
        """Lazy inspection of PyTorch/CUDA environment without crashing if torch is absent."""
        with self._lock:
            if not force_refresh and self._diagnostics_cache is not None:
                return dict(self._diagnostics_cache)

        requested = self.get_requested_device()
        cuda_available = False
        cuda_device_count = 0
        cuda_device_name: Optional[str] = None
        torch_version: Optional[str] = None
        cuda_version: Optional[str] = None
        warnings: List[str] = []

        try:
            import torch  # type: ignore

            torch_version = getattr(torch, "__version__", None)
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device_count = torch.cuda.device_count()
                if cuda_device_count > 0:
                    cuda_device_name = torch.cuda.get_device_name(0)
                cuda_version = getattr(torch.version, "cuda", None)
        except (ImportError, Exception) as exc:
            warnings.append(f"PyTorch 环境未就绪或无法检测 CUDA: {exc}")

        actual_device = "cpu"
        if requested == "cuda":
            if cuda_available:
                actual_device = "cuda"
            else:
                warnings.append("请求使用 NVIDIA CUDA，但系统未检测到可用 GPU 或 CUDA 驱动，已安全回退到 CPU。")
                actual_device = "cpu"
        elif requested == "auto":
            actual_device = "cuda" if cuda_available else "cpu"
        else:
            actual_device = "cpu"

        diag = {
            "requested_device": requested,
            "actual_device": actual_device,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "warnings": warnings,
        }

        with self._lock:
            self._diagnostics_cache = diag
        return dict(diag)

    def resolve(self, requested: Optional[str] = None) -> Tuple[str, List[str]]:
        """Resolve actual device string and warnings."""
        if requested:
            self.set_requested_device(requested)
        diag = self.probe_diagnostics()
        return str(diag["actual_device"]), list(diag.get("warnings", []))


DEVICE_MANAGER = DeviceManager()

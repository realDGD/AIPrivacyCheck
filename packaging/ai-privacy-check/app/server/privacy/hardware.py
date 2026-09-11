"""Hardware probe for NVIDIA GPU and host driver detection.

Completely decoupled from Python framework runtimes (PyTorch / Paddle / Transformers).
Directly executes nvidia-smi to obtain authoritative host hardware telemetry.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple


class HardwareProbe:
    """Probes host hardware capabilities independently of Python frameworks."""

    def __init__(
        self,
        command_runner: Optional[Callable[[List[str]], Tuple[int, str, str]]] = None,
        nvidia_smi_path: Optional[str] = None,
    ) -> None:
        self._runner = command_runner or self._default_runner
        self._nvidia_smi_path = nvidia_smi_path

    @staticmethod
    def _default_runner(cmd: List[str]) -> Tuple[int, str, str]:
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            return res.returncode, res.stdout, res.stderr
        except FileNotFoundError:
            return 127, "", "command not found"
        except subprocess.TimeoutExpired:
            return 124, "", "command timed out"
        except Exception as exc:
            return 1, "", str(exc)

    def probe_nvidia(self) -> Dict[str, Any]:
        """Probes NVIDIA GPUs via nvidia-smi."""
        smi_bin = self._nvidia_smi_path or shutil.which("nvidia-smi")
        if not smi_bin:
            return {
                "nvidia_available": False,
                "gpu_count": 0,
                "gpus": [],
                "driver_version": None,
                "reason": "未找到 nvidia-smi；主机未安装或未加载英伟达显卡驱动。",
            }

        cmd = [
            smi_bin,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        retcode, stdout, stderr = self._runner(cmd)
        if retcode != 0:
            err_msg = stderr.strip() or stdout.strip() or f"nvidia-smi exited with code {retcode}"
            return {
                "nvidia_available": False,
                "gpu_count": 0,
                "gpus": [],
                "driver_version": None,
                "reason": f"英伟达显卡驱动状态异常: {err_msg}",
            }

        lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
        if not lines:
            return {
                "nvidia_available": False,
                "gpu_count": 0,
                "gpus": [],
                "driver_version": None,
                "reason": "nvidia-smi 未返回任何 GPU 信息。",
            }

        gpus: List[Dict[str, Any]] = []
        overall_driver_version: Optional[str] = None

        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if len(parts) > 0 else f"GPU-{idx}"
            driver = parts[1] if len(parts) > 1 else None
            mem_mb: Optional[int] = None
            if len(parts) > 2:
                try:
                    mem_mb = int(float(parts[2]))
                except (ValueError, TypeError):
                    mem_mb = None

            if overall_driver_version is None and driver:
                overall_driver_version = driver

            gpus.append(
                {
                    "index": idx,
                    "name": name,
                    "driver_version": driver,
                    "memory_total_mb": mem_mb,
                }
            )

        return {
            "nvidia_available": True,
            "gpu_count": len(gpus),
            "gpus": gpus,
            "driver_version": overall_driver_version,
            "reason": None,
        }


HARDWARE_PROBE = HardwareProbe()

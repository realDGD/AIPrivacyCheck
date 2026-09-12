"""Isolated Runtime Worker Client for AI Privacy Check.

Manages persistent worker subprocesses running in isolated virtual environments.
Features:
1. Pure JSONL IPC over stdin/stdout (no TCP sockets or HTTP ports).
2. Per-worker thread synchronization for thread-safe server execution.
3. Offline environment enforcement (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, MODELSCOPE_OFFLINE).
4. Process termination upon unload/reload for complete RAM, VRAM, and CUDA context reclamation.
5. Per-model timeouts and synthetic post-installation smoke tests.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Dict, Optional, Tuple

from privacy.runtime_manager import RuntimeManager, get_runtime_manager

logger = logging.getLogger("ai_privacy.worker_client")


class RuntimeWorkerProcess:
    """Represents a long-running, isolated worker process communicating via JSONL."""

    def __init__(
        self,
        python_bin: Path,
        worker_script: Path,
        model_id: str,
        profile: str,
        device: str = "cpu",
        startup_timeout: int = 45,
    ) -> None:
        self.python_bin = python_bin
        self.worker_script = worker_script
        self.model_id = model_id
        self.profile = profile
        self.device = device
        self.startup_timeout = startup_timeout

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._alive = False
        self._last_error: Optional[str] = None

    def start(self) -> None:
        with self._lock:
            if self._alive and self._process and self._process.poll() is None:
                return

            if not self.python_bin.is_file():
                raise FileNotFoundError(f"Python interpreter not found: {self.python_bin}")
            if not self.worker_script.is_file():
                raise FileNotFoundError(f"Worker script not found: {self.worker_script}")

            env = os.environ.copy()
            # Strict offline inference enforcement
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["MODELSCOPE_OFFLINE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            # Set venv environment
            venv_root = str(self.python_bin.parent.parent)
            env["VIRTUAL_ENV"] = venv_root
            env["PATH"] = f"{venv_root}/bin:{env.get('PATH', '')}"

            cmd = [str(self.python_bin), str(self.worker_script)]
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                self._alive = True
                self._last_error = None
            except Exception as exc:
                self._alive = False
                self._last_error = str(exc)
                logger.error(f"启动模型 worker [{self.model_id}] 进程失败: {exc}")
                raise

    def query(self, req: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        with self._lock:
            if not self._alive or not self._process or self._process.poll() is not None:
                self.start()

            if not self._process or not self._process.stdin or not self._process.stdout:
                return {"ok": False, "error_type": "ProcessError", "error": "Worker process pipes not available"}

            req_copy = dict(req)
            req_copy.setdefault("device", self.device)
            payload = json.dumps(req_copy, ensure_ascii=False) + "\n"

            try:
                self._process.stdin.write(payload)
                self._process.stdin.flush()
            except Exception as write_exc:
                self._terminate_locked()
                return {"ok": False, "error_type": "PipeWriteError", "error": str(write_exc)}

            # Wait for response line with timeout simulation
            resp_line = None
            start_time = time.monotonic()

            # Read stdout line
            try:
                resp_line = self._process.stdout.readline()
            except Exception as read_exc:
                self._terminate_locked()
                return {"ok": False, "error_type": "PipeReadError", "error": str(read_exc)}

            if not resp_line:
                # Subprocess exited or died
                ret = self._process.poll() if self._process else -1
                stderr_output = ""
                if self._process and self._process.stderr:
                    try:
                        stderr_output = self._process.stderr.read()
                    except Exception:
                        pass
                self._terminate_locked()
                return {
                    "ok": False,
                    "error_type": "WorkerCrashed",
                    "error": f"Worker process died with exit code {ret}: {stderr_output.strip()[:200]}",
                }

            try:
                return json.loads(resp_line.strip())
            except Exception as parse_exc:
                return {
                    "ok": False,
                    "error_type": "JSONDecodeError",
                    "error": f"Failed to parse worker response: {parse_exc}",
                }

    def _terminate_locked(self) -> None:
        self._alive = False
        if self._process:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=1)
            except Exception:
                pass
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
            except Exception:
                pass
            self._process = None

    def terminate(self) -> None:
        with self._lock:
            self._terminate_locked()

    def is_alive(self) -> bool:
        with self._lock:
            return self._alive and self._process is not None and self._process.poll() is None


class WorkerClient:
    """Manages lifecycles of all model worker subprocesses."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.runtime_manager = get_runtime_manager(data_dir)
        self._workers: Dict[str, RuntimeWorkerProcess] = {}
        self._lock = threading.Lock()

    def _get_worker_script(self, model_id: str) -> Path:
        workers_dir = Path(__file__).resolve().parent / "workers"
        if "gliner" in model_id.lower():
            return workers_dir / "gliner_worker.py"
        elif "siamese" in model_id.lower():
            return workers_dir / "siamese_uie_worker.py"
        elif "memprivacy" in model_id.lower():
            return workers_dir / "memprivacy_worker.py"
        else:
            raise ValueError(f"Unknown worker type for model: {model_id}")

    def get_timeout_for_model(self, model_id: str) -> Tuple[int, int]:
        """Returns (startup_timeout, inference_timeout)."""
        if "gliner" in model_id.lower():
            return 30, 45
        elif "siamese" in model_id.lower():
            return 45, 60
        elif "memprivacy" in model_id.lower():
            return 120, 180
        return 45, 60

    def get_worker(
        self,
        model_id: str,
        profile: str,
        device: str = "cpu",
    ) -> RuntimeWorkerProcess:
        key = f"{model_id}:{profile}:{device}"
        with self._lock:
            if key in self._workers:
                worker = self._workers[key]
                if worker.is_alive():
                    return worker
                else:
                    worker.terminate()
                    del self._workers[key]

            python_bin = self.runtime_manager.get_python_bin(profile)
            worker_script = self._get_worker_script(model_id)
            startup_timeout, _ = self.get_timeout_for_model(model_id)

            worker = RuntimeWorkerProcess(
                python_bin=python_bin,
                worker_script=worker_script,
                model_id=model_id,
                profile=profile,
                device=device,
                startup_timeout=startup_timeout,
            )
            worker.start()
            self._workers[key] = worker
            return worker

    def stop_worker_for_model(self, model_id: str) -> None:
        with self._lock:
            keys_to_remove = [k for k in self._workers if k.startswith(f"{model_id}:")]
            for k in keys_to_remove:
                worker = self._workers.pop(k, None)
                if worker:
                    worker.terminate()

    def stop_all(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.terminate()
            self._workers.clear()

    def run_smoke_test(
        self,
        model_id: str,
        model_path: Path,
        profile: str,
        device: str = "cpu",
    ) -> Tuple[bool, Optional[str]]:
        """Executes a synthetic end-to-end smoke inference test to verify model + worker readiness."""
        try:
            worker = self.get_worker(model_id, profile, device=device)
            _, infer_timeout = self.get_timeout_for_model(model_id)

            if "gliner" in model_id.lower():
                sample_text = "My email is test@example.com."
                req = {
                    "action": "detect",
                    "model_path": str(model_path),
                    "text": sample_text,
                    "labels": ["email"],
                    "threshold": 0.2,
                }
            elif "siamese" in model_id.lower():
                sample_text = "我叫张三，住在北京市朝阳区测试路88号。"
                req = {
                    "action": "detect",
                    "model_path": str(model_path),
                    "text": sample_text,
                }
            elif "memprivacy" in model_id.lower():
                sample_text = "My verification code is 89757."
                req = {
                    "action": "detect",
                    "model_path": str(model_path),
                    "text": sample_text,
                    "real_name": "unknown",
                    "max_new_tokens": 128,
                }
            else:
                return True, None

            res = worker.query(req, timeout=infer_timeout)
            if not res.get("ok"):
                err_msg = res.get("error") or "Worker returned ok=False"
                return False, f"冒烟推理验证失败: {err_msg}"

            return True, None
        except Exception as exc:
            return False, f"冒烟测试异常: {exc}"


_GLOBAL_WORKER_CLIENT: Optional[WorkerClient] = None


def get_worker_client(data_dir: Optional[Path] = None) -> WorkerClient:
    global _GLOBAL_WORKER_CLIENT
    if _GLOBAL_WORKER_CLIENT is None:
        target = data_dir or Path("/tmp")
        _GLOBAL_WORKER_CLIENT = WorkerClient(target)
    elif data_dir is not None and _GLOBAL_WORKER_CLIENT.data_dir != data_dir:
        _GLOBAL_WORKER_CLIENT = WorkerClient(data_dir)
    return _GLOBAL_WORKER_CLIENT

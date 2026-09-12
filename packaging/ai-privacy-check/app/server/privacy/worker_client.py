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

import collections
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Deque, Dict, Optional, Tuple

from privacy.runtime_manager import RuntimeManager, get_runtime_manager

logger = logging.getLogger("ai_privacy.worker_client")

STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_BUSY = "busy"
STATE_FAILED = "failed"


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
        self._pid: Optional[int] = None
        self._lock = threading.Lock()
        self.state: str = STATE_STOPPED
        self._alive: bool = False
        self._last_error: Optional[str] = None

        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stderr_buffer: Deque[str] = collections.deque(maxlen=100)
        self._stop_event = threading.Event()
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def _is_alive_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _is_ready_locked(self) -> bool:
        return self.state in (STATE_READY, STATE_BUSY) and self._is_alive_locked()

    def _stdout_reader(self, proc: subprocess.Popen) -> None:
        while not self._stop_event.is_set():
            try:
                if not proc.stdout:
                    break
                line = proc.stdout.readline()
                if not line:
                    # EOF: process terminated or closed stdout
                    self._stdout_queue.put("")
                    break
                self._stdout_queue.put(line)
            except Exception:
                self._stdout_queue.put("")
                break

    def _stderr_drainer(self, proc: subprocess.Popen) -> None:
        while not self._stop_event.is_set():
            try:
                if not proc.stderr:
                    break
                line = proc.stderr.readline()
                if not line:
                    break
                # Truncate each line to 500 chars to avoid memory bloat, strip sensitive data
                clean_line = line.strip()[:500]
                if clean_line:
                    self._stderr_buffer.append(clean_line)
            except Exception:
                break

    def _handshake_locked(self, timeout: int) -> Tuple[bool, Optional[str]]:
        """Sends startup ping and waits for pong to guarantee worker initialization."""
        ping_req = json.dumps({"action": "ping"}, ensure_ascii=False) + "\n"
        try:
            if not self._process or not self._process.stdin:
                return False, "Worker stdin not open"
            self._process.stdin.write(ping_req)
            self._process.stdin.flush()
        except Exception as exc:
            return False, f"Failed to write ping to worker: {exc}"

        try:
            resp_line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty:
            return False, f"Worker startup handshake timed out after {timeout}s"

        if not resp_line:
            ret = self._process.poll() if self._process else -1
            recent_err = " | ".join(list(self._stderr_buffer)[-3:])
            return False, f"Worker died during handshake (exit {ret}): {recent_err}"

        try:
            parsed = json.loads(resp_line.strip())
            if parsed.get("ok") and (parsed.get("status") == "pong" or parsed.get("pong") is True):
                return True, None
            return False, f"Unexpected handshake response: {resp_line.strip()[:200]}"
        except Exception as exc:
            return False, f"Invalid handshake JSON: {exc}"

    def _start_locked(self) -> None:
        if self._is_ready_locked():
            return

        if self._is_alive_locked():
            self._terminate_locked()

        if not self.python_bin.is_file():
            self.state = STATE_FAILED
            self._last_error = f"Python interpreter not found: {self.python_bin}"
            raise FileNotFoundError(self._last_error)
        if not self.worker_script.is_file():
            self.state = STATE_FAILED
            self._last_error = f"Worker script not found: {self.worker_script}"
            raise FileNotFoundError(self._last_error)

        self.state = STATE_STARTING
        self._stop_event.clear()
        # Empty queues/buffers
        while not self._stdout_queue.empty():
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                break
        self._stderr_buffer.clear()

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["MODELSCOPE_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

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
            self._pid = self._process.pid
            self._alive = True
            self._last_error = None
        except Exception as exc:
            self.state = STATE_FAILED
            self._alive = False
            self._last_error = str(exc)
            logger.error(f"启动模型 worker [{self.model_id}] 失败: {exc}")
            raise

        # Spawn background reader and drainer threads
        self._stdout_thread = threading.Thread(
            target=self._stdout_reader,
            args=(self._process,),
            name=f"WorkerStdout-{self.model_id}-{self._pid}",
            daemon=True,
        )
        self._stdout_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._stderr_drainer,
            args=(self._process,),
            name=f"WorkerStderr-{self.model_id}-{self._pid}",
            daemon=True,
        )
        self._stderr_thread.start()

        # Perform explicit startup handshake
        handshake_ok, handshake_err = self._handshake_locked(timeout=self.startup_timeout)
        if not handshake_ok:
            self._terminate_locked()
            self.state = STATE_FAILED
            self._last_error = handshake_err
            raise RuntimeError(f"Worker startup handshake failed for [{self.model_id}]: {handshake_err}")

        self.state = STATE_READY
        self._alive = True
        self._last_error = None

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _terminate_locked(self) -> None:
        self._stop_event.set()
        proc = self._process
        self._process = None
        self._pid = None
        self._alive = False
        self.state = STATE_STOPPED

        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
            except Exception:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    def terminate(self) -> None:
        with self._lock:
            self._terminate_locked()

    def is_alive(self) -> bool:
        with self._lock:
            return self._is_alive_locked()

    def is_ready(self) -> bool:
        with self._lock:
            return self._is_ready_locked()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "alive": self._is_alive_locked(),
                "ready": self._is_ready_locked(),
                "state": self.state,
                "pid": self._pid,
                "profile": self.profile,
                "device": self.device,
                "last_error": self._last_error,
            }

    def _query_raw_locked(self, req: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        if not self._is_alive_locked() or not self._process or not self._process.stdin:
            return {"ok": False, "error_type": "ProcessError", "error": "Worker process not running"}

        self.state = STATE_BUSY
        req_copy = dict(req)
        req_copy.setdefault("device", self.device)
        payload = json.dumps(req_copy, ensure_ascii=False) + "\n"

        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except Exception as write_exc:
            self._terminate_locked()
            self.state = STATE_FAILED
            self._last_error = str(write_exc)
            return {"ok": False, "error_type": "PipeWriteError", "error": str(write_exc)}

        try:
            resp_line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty:
            # Enforce true timeout: terminate hanging worker
            self._terminate_locked()
            self.state = STATE_FAILED
            self._last_error = f"Worker inference timed out after {timeout}s"
            logger.warning(f"模型 Worker [{self.model_id}] 执行超时 ({timeout}s)，已强制终止。")
            return {
                "ok": False,
                "error_type": "WorkerTimeout",
                "error": f"Worker inference timed out after {timeout}s",
            }

        if not resp_line:
            # Process terminated or closed pipe unexpectedly
            ret = self._process.poll() if self._process else -1
            recent_err = " | ".join(list(self._stderr_buffer)[-3:])
            self._terminate_locked()
            self.state = STATE_FAILED
            self._last_error = f"Worker died (exit {ret}): {recent_err}"
            return {
                "ok": False,
                "error_type": "WorkerCrashed",
                "error": f"Worker process died unexpectedly with exit code {ret}: {recent_err}",
            }

        try:
            data = json.loads(resp_line.strip())
            self.state = STATE_READY
            return data
        except Exception as json_exc:
            self.state = STATE_READY
            return {
                "ok": False,
                "error_type": "InvalidJSONResponse",
                "error": f"Failed to parse worker response: {json_exc}",
            }

    def _query_with_retry_locked(self, req: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        if not self._is_ready_locked():
            self._start_locked()

        res = self._query_raw_locked(req, timeout)
        # Automatic 1-attempt restart on unexpected crash / pipe disconnect (NOT on timeout)
        if res.get("error_type") in ("PipeWriteError", "WorkerCrashed"):
            logger.warning(f"Worker [{self.model_id}] 发生崩溃或管道断开，正在尝试自动恢复重启...")
            self._terminate_locked()
            try:
                self._start_locked()
                return self._query_raw_locked(req, timeout)
            except Exception as restart_exc:
                return {
                    "ok": False,
                    "error_type": "WorkerRestartFailed",
                    "error": f"Worker crashed and automatic restart failed: {restart_exc}",
                }

        return res

    def query(self, req: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        with self._lock:
            return self._query_with_retry_locked(req, timeout)


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
                if worker.is_ready():
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

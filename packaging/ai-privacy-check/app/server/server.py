#!/usr/bin/env python3
"""Dependency-free HTTP and Unix Stream Socket service for AI Privacy Check."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from privacy import PrivacyService
from privacy.device import DEVICE_MANAGER
from privacy.model_catalog import get_model_descriptor, list_all_models
from privacy.worker_client import get_worker_client
import model_installer


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
DATA_DIR = Path(os.environ.get("APP_DATA_DIR") or (os.environ.get("TRIM_PKGVAR", "/tmp") + "/data")).resolve()
BASE_PATH = os.environ.get("APP_BASE_PATH", "").rstrip("/")
MAX_BODY_BYTES = 2 * 1024 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
DEVICE_MANAGER.set_data_dir(DATA_DIR)
PRIVACY = PrivacyService(DATA_DIR)


class ModelLifecycleController:
    """Manages online download, local manual import, uninstall, and device switching."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.log_file = data_dir / "status" / "model-install.log"
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def status(self) -> Dict[str, object]:
        status_dir = self.data_dir / "status"

        # Read status files
        install_states: Dict[str, object] = {}
        if status_dir.is_dir():
            for sf in status_dir.glob("*-install.json"):
                try:
                    install_states[sf.stem.replace("-install", "")] = json.loads(sf.read_text(encoding="utf-8"))
                except Exception:
                    pass

        with self._lock:
            running = self._process is not None

        device_diag = DEVICE_MANAGER.probe_diagnostics()
        catalog_items = [m.to_dict() for m in list_all_models()]

        return {
            "registry": PRIVACY.registry.status(),
            "catalog": catalog_items,
            "installing": running,
            "install_states": install_states,
            "device": device_diag,
            "shared_dirs": self.get_shared_dirs(),
            "shared_candidates": self.scan_shared(),
            "log_tail": self._log_tail(),
        }

    def scan_shared(self) -> List[Dict[str, Any]]:
        return model_installer.scan_shared_models_directory(self.data_dir)

    def get_shared_dirs(self) -> List[str]:
        return [str(d) for d in model_installer.get_shared_models_dirs()]

    def _log_tail(self, lines: int = 16) -> List[str]:
        if not self.log_file.is_file():
            return []
        try:
            content = self.log_file.read_text(encoding="utf-8", errors="replace")
            return content.splitlines()[-lines:]
        except OSError:
            return []

    def start_install(self, model_name: str = "gliner-pii-edge") -> bool:
        with self._lock:
            if self._process is not None:
                return False
            try:
                with model_installer.model_operation_lock(self.data_dir, model_name, non_blocking=True):
                    pass
            except RuntimeError:
                return False
            status_dir = self.data_dir / "status"
            status_dir.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_file.open("ab", buffering=0)
            env = os.environ.copy()
            from privacy.runtime_env import build_runtime_env
            env = build_runtime_env(self.data_dir, base_env=env)
            env["APP_DATA_DIR"] = str(self.data_dir)
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [sys.executable, str(APP_DIR / "model_installer.py"), "install", model_name],
                cwd=str(APP_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            process = self._process

        def wait_for_install() -> None:
            try:
                process.wait()
                log_handle.close()
                DEVICE_MANAGER.invalidate_runtime_state()
                if process.returncode == 0:
                    try:
                        DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
                    finally:
                        PRIVACY.reset_models()
            finally:
                if not log_handle.closed:
                    log_handle.close()
                with self._lock:
                    if self._process is process:
                        self._process = None

        threading.Thread(target=wait_for_install, daemon=True).start()
        return True

    def start_repair(self, model_name: str) -> bool:
        with self._lock:
            if self._process is not None:
                return False
            try:
                with model_installer.model_operation_lock(self.data_dir, model_name, non_blocking=True):
                    pass
            except RuntimeError:
                return False
            status_dir = self.data_dir / "status"
            status_dir.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_file.open("ab", buffering=0)
            env = os.environ.copy()
            from privacy.runtime_env import build_runtime_env
            env = build_runtime_env(self.data_dir, base_env=env)
            env["APP_DATA_DIR"] = str(self.data_dir)
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [sys.executable, str(APP_DIR / "model_installer.py"), "repair", model_name],
                cwd=str(APP_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            process = self._process

        def wait_for_repair() -> None:
            try:
                process.wait()
                log_handle.close()
                DEVICE_MANAGER.invalidate_runtime_state()
                model_installer.clear_dependency_probe_cache()
                if process.returncode == 0:
                    try:
                        DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
                    finally:
                        PRIVACY.reset_models()
            finally:
                if not log_handle.closed:
                    log_handle.close()
                with self._lock:
                    if self._process is process:
                        self._process = None

        threading.Thread(target=wait_for_repair, daemon=True).start()
        return True

    def start_rebuild(self, profile: str) -> bool:
        if profile not in (runtime_manager.PROFILE_TORCH_CPU, runtime_manager.PROFILE_TORCH_CUDA):
            return False
        with self._lock:
            if self._process is not None:
                return False
            try:
                with model_installer.runtime_operation_lock(self.data_dir, profile, non_blocking=True):
                    pass
            except RuntimeError:
                return False
            status_dir = self.data_dir / "status"
            status_dir.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_file.open("ab", buffering=0)
            from privacy.python_runtime import build_uv_env
            env = build_uv_env(self.data_dir, base_env=os.environ.copy())
            env["APP_DATA_DIR"] = str(self.data_dir)
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [sys.executable, str(APP_DIR / "model_installer.py"), "rebuild", profile],
                cwd=str(APP_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            process = self._process

        def wait_for_rebuild() -> None:
            try:
                process.wait()
                log_handle.close()
                DEVICE_MANAGER.invalidate_runtime_state()
                model_installer.clear_dependency_probe_cache()
                if process.returncode == 0:
                    try:
                        DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
                    finally:
                        PRIVACY.reset_models()
            finally:
                if not log_handle.closed:
                    log_handle.close()
                with self._lock:
                    if self._process is process:
                        self._process = None

        threading.Thread(target=wait_for_rebuild, daemon=True).start()
        return True

    def import_model(self, model_name: str, source_path: str) -> Tuple[bool, str]:
        path = Path(source_path).resolve()
        ok, msg = model_installer.import_local_model(self.data_dir, model_name, path)
        DEVICE_MANAGER.invalidate_runtime_state()
        if ok:
            try:
                DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
            finally:
                PRIVACY.reset_models()
        return ok, msg

    def uninstall_model(self, model_name: str) -> Tuple[bool, str]:
        ok, msg = model_installer.uninstall_model(self.data_dir, model_name)
        DEVICE_MANAGER.invalidate_cache()
        if ok:
            PRIVACY.reset_models()
        return ok, msg

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


INSTALLER = ModelLifecycleController(DATA_DIR)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "AIPrivacyCheck/0.6.13"

    def log_message(self, fmt: str, *args) -> None:
        safe_path = urlsplit(self.path).path
        client = str(self.client_address[0]) if isinstance(self.client_address, tuple) and self.client_address else "local"
        sys.stderr.write(f"{client} - {self.command} {safe_path}\n")

    def _route(self) -> Optional[str]:
        path = unquote(urlsplit(self.path).path)
        if BASE_PATH:
            if path == BASE_PATH:
                return "/"
            if not path.startswith(BASE_PATH + "/"):
                return None
            path = path[len(BASE_PATH):]
        return path or "/"

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors *",
        )

    def _json(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, object]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("请求必须使用 application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length 无效") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("请求正文为空或超过 2MB")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON 格式无效") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON 顶层必须是对象")
        return value

    def _is_admin(self) -> bool:
        if not os.environ.get("GATEWAY_SOCKET"):
            return True
        return self.headers.get("X-Trim-Isadmin", "").lower() == "true"

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if BASE_PATH and request_path == BASE_PATH:
            self.send_response(HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Location", BASE_PATH + "/")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self._security_headers()
            self.end_headers()
            return

        route = self._route()
        if route is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "路径不存在"})
            return

        if route == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "version": "0.6.13",
                    "base_path": BASE_PATH,
                    "capabilities": PRIVACY.capabilities(),
                },
            )
            return

        if route == "/api/model/status":
            payload = INSTALLER.status()
            payload["is_admin"] = self._is_admin()
            payload["disk_hint_gb"] = "4-8"
            self._json(HTTPStatus.OK, payload)
            return

        if route == "/api/model/shared/scan":
            self._json(
                HTTPStatus.OK,
                {
                    "shared_dirs": INSTALLER.get_shared_dirs(),
                    "candidates": INSTALLER.scan_shared(),
                },
            )
            return

        if route == "/api/device":
            self._json(HTTPStatus.OK, DEVICE_MANAGER.probe_diagnostics(force_refresh=True))
            return

        self._serve_static(route)

    def do_POST(self) -> None:
        route = self._route()
        if route == "/api/detect":
            try:
                payload = self._read_json()
                result = PRIVACY.detect(
                    text=payload.get("text"),
                    use_model=bool(payload.get("use_model", False)),
                    slots=payload.get("slots"),
                    policy_level=payload.get("policy_level"),
                )
                self._json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "检测服务暂时不可用，请稍后重试"})
            return

        if route == "/api/model/runtime/repair":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以修复模型运行环境"})
                return
            try:
                payload = self._read_json()
            except Exception:
                payload = {}
            model_name = str(payload.get("model", "")).strip()
            if not model_name:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "缺少 model 参数"})
                return
            if not get_model_descriptor(model_name):
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"未知模型标识: {model_name}"})
                return
            started = INSTALLER.start_repair(model_name)
            status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
            self._json(
                status,
                {
                    "ok": started,
                    "started": started,
                    "message": f"已开始修复模型 [{model_name}] 的运行环境" if started else "已有其他任务正在执行",
                    "status": INSTALLER.status(),
                },
            )
            return

        if route in ("/api/runtime/rebuild", "/api/model/runtime/rebuild"):
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以重建运行环境"})
                return
            try:
                payload = self._read_json()
            except Exception:
                payload = {}
            profile = str(payload.get("profile", runtime_manager.PROFILE_TORCH_CPU)).strip()
            if profile not in (runtime_manager.PROFILE_TORCH_CPU, runtime_manager.PROFILE_TORCH_CUDA):
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"未知运行时 Profile: {profile}"})
                return
            started = INSTALLER.start_rebuild(profile)
            status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
            self._json(
                status,
                {
                    "ok": started,
                    "started": started,
                    "profile": profile,
                    "message": f"已开始重建运行时环境 [{profile}]" if started else "已有其他任务正在执行",
                    "status": INSTALLER.status(),
                },
            )
            return

        if route == "/api/model/install":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以安装模型"})
                return
            try:
                payload = self._read_json()
            except Exception:
                payload = {}
            model_name = str(payload.get("model", "gliner-pii-edge")).strip()
            if not get_model_descriptor(model_name):
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"未知模型标识: {model_name}"})
                return
            started = INSTALLER.start_install(model_name)
            status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
            self._json(status, {"started": started, "status": INSTALLER.status()})
            return

        if route == "/api/model/import":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以导入模型"})
                return
            try:
                payload = self._read_json()
                model_name = str(payload.get("model", "gliner-pii-edge")).strip()
                if not get_model_descriptor(model_name):
                    raise ValueError(f"未知模型标识: {model_name}")
                source_path = str(payload.get("source_path", "")).strip()
                if not source_path:
                    raise ValueError("必须指定模型导入路径 source_path")
                ok, message = INSTALLER.import_model(model_name, source_path)
                status = HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST
                self._json(status, {"ok": ok, "message": message, "status": INSTALLER.status()})
            except RuntimeError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"导入失败: {exc}"})
            return

        if route == "/api/model/shared/import":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以导入模型"})
                return
            try:
                payload = self._read_json()
                model_name = str(payload.get("model", "")).strip()
                if not get_model_descriptor(model_name):
                    raise ValueError(f"未知模型标识: {model_name}")
                source_path = str(payload.get("source_path", "")).strip()
                if not model_name or not source_path:
                    raise ValueError("必须指定 model 和 source_path")
                ok, message = INSTALLER.import_model(model_name, source_path)
                status = HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST
                self._json(status, {"ok": ok, "message": message, "status": INSTALLER.status()})
            except RuntimeError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"导入失败: {exc}"})
            return

        if route == "/api/model/uninstall":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以卸载模型"})
                return
            try:
                payload = self._read_json()
                model_name = str(payload.get("model", "gliner-pii-edge")).strip()
                if not get_model_descriptor(model_name):
                    raise ValueError(f"未知模型标识: {model_name}")
                ok, message = INSTALLER.uninstall_model(model_name)
                self._json(HTTPStatus.OK, {"ok": ok, "message": message, "status": INSTALLER.status()})
            except RuntimeError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"卸载失败: {exc}"})
            return

        if route == "/api/model/reload":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以重载模型"})
                return
            PRIVACY.reset_models()
            self._json(HTTPStatus.OK, {"ok": True, "status": INSTALLER.status()})
            return

        if route == "/api/model/slot/toggle":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以配置模型槽位"})
                return
            try:
                payload = self._read_json()
                slot = str(payload.get("slot", "")).strip()
                enabled = bool(payload.get("enabled", True))
                PRIVACY.registry.set_slot_enabled(slot, enabled)
                self._json(HTTPStatus.OK, {"ok": True, "status": INSTALLER.status()})
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if route == "/api/model/select":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以切换激活模型"})
                return
            try:
                payload = self._read_json()
                slot = str(payload.get("slot", "")).strip()
                model_id = str(payload.get("model", "")).strip()
                if not get_model_descriptor(model_id):
                    raise ValueError(f"未知模型标识: {model_id}")
                ok = PRIVACY.registry.set_active_model(slot, model_id)
                if not ok:
                    raise ValueError(f"无法将模型 {model_id} 分配给槽位 {slot}")
                self._json(HTTPStatus.OK, {"ok": True, "status": INSTALLER.status()})
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if route == "/api/device/select":
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以切换计算设备"})
                return
            try:
                payload = self._read_json()
                device = str(payload.get("device", "auto"))
                actual = DEVICE_MANAGER.set_requested_device(device)
                get_worker_client(DATA_DIR).stop_all()
                PRIVACY.reset_models()
                self._json(HTTPStatus.OK, {"ok": True, "device": DEVICE_MANAGER.probe_diagnostics()})
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._json(HTTPStatus.NOT_FOUND, {"error": "路径不存在"})

    def _serve_static(self, route: str) -> None:
        relative = "index.html" if route == "/" else route.lstrip("/")
        candidate = (WEB_DIR / relative).resolve()
        try:
            candidate.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "路径不存在"})
            return
        if not candidate.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "路径不存在"})
            return
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if candidate.name == "index.html" else "public, max-age=3600")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)


class ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    address_family = socket.AF_UNIX
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        socketserver.UnixStreamServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = 0


def run() -> None:
    gateway_socket = os.environ.get("GATEWAY_SOCKET", "").strip()
    if gateway_socket:
        socket_path = Path(gateway_socket)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
        server = ThreadingUnixHTTPServer(str(socket_path), AppHandler)
        os.chmod(socket_path, 0o660)
        destination = f"unix://{socket_path}"
    else:
        host = os.environ.get("APP_HOST", "127.0.0.1")
        port = int(os.environ.get("APP_PORT", "8976"))
        server = ThreadingHTTPServer((host, port), AppHandler)
        destination = f"http://{host}:{port}"

    def request_shutdown(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    print(f"AI Privacy Check native service listening on {destination}", flush=True)
    try:
        server.serve_forever()
    finally:
        INSTALLER.stop()
        get_worker_client(DATA_DIR).stop_all()
        server.server_close()
        if gateway_socket:
            socket_path = Path(gateway_socket)
            if socket_path.exists() or socket_path.is_socket():
                socket_path.unlink()


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Dependency-free HTTP service for AI Privacy Check."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import signal
import threading
import time
from typing import Dict, Optional
from urllib.parse import unquote, urlsplit

from privacy import PrivacyService


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/data")).resolve()
BASE_PATH = os.environ.get("APP_BASE_PATH", "").rstrip("/")
MAX_BODY_BYTES = 2 * 1024 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
PRIVACY = PrivacyService(DATA_DIR)


class ModelInstallController:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.state_file = data_dir / "status" / "model-install.json"
        self.log_file = data_dir / "status" / "model-install.log"
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def status(self) -> Dict[str, object]:
        model_status = PRIVACY.model.status()
        install_status: Dict[str, object] = {}
        if self.state_file.is_file():
            try:
                install_status = json.loads(self.state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                install_status = {"state": "unknown", "detail": "安装状态文件无法读取"}
        with self._lock:
            running = self._process is not None and self._process.poll() is None
        if running:
            model_status["state"] = "installing"
            model_status["ready"] = False
        model_status["install"] = install_status
        model_status["log_tail"] = self._log_tail()
        return model_status

    def _log_tail(self, lines: int = 14) -> list:
        if not self.log_file.is_file():
            return []
        try:
            content = self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return content.splitlines()[-lines:]

    def start(self) -> bool:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return False
            status_dir = self.data_dir / "status"
            status_dir.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_file.open("ab", buffering=0)
            env = os.environ.copy()
            env["APP_DATA_DIR"] = str(self.data_dir)
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [sys.executable, str(APP_DIR / "model_installer.py")],
                cwd=str(APP_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            process = self._process

        def wait_for_install() -> None:
            process.wait()
            log_handle.close()
            if process.returncode == 0:
                PRIVACY.model.reset()

        threading.Thread(target=wait_for_install, daemon=True).start()
        return True

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


INSTALLER = ModelInstallController(DATA_DIR)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "AIPrivacyCheck/0.2"

    def log_message(self, fmt: str, *args) -> None:
        # Never log request bodies or query values; paths are stripped to avoid
        # accidental disclosure if a client puts text in a query string.
        safe_path = urlsplit(self.path).path
        if isinstance(self.client_address, tuple) and self.client_address:
            client = str(self.client_address[0])
        else:
            client = "local"
        sys.stderr.write("%s - %s %s\n" % (client, self.command, safe_path))

    def _route(self) -> Optional[str]:
        path = unquote(urlsplit(self.path).path)
        if BASE_PATH:
            if path == BASE_PATH:
                return "/"
            if not path.startswith(BASE_PATH + "/"):
                return None
            path = path[len(BASE_PATH) :]
        return path or "/"

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'",
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
                    "version": "0.2.0",
                    "base_path": BASE_PATH,
                    "capabilities": PRIVACY.capabilities(),
                    "model": PRIVACY.model.status(),
                },
            )
            return
        if route == "/api/model/status":
            payload = INSTALLER.status()
            payload["is_admin"] = self._is_admin()
            payload["disk_hint_gb"] = "4-8"
            self._json(HTTPStatus.OK, payload)
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
                )
                self._json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "检测服务暂时不可用，请稍后重试"})
            return
        if route == "/api/model/install":
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请求必须使用 application/json"})
                return
            if not self._is_admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "只有 fnOS 管理员可以安装模型"})
                return
            started = INSTALLER.start()
            status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
            self._json(status, {"started": started, "model": INSTALLER.status()})
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
        destination = "unix://{}".format(socket_path)
    else:
        host = os.environ.get("APP_HOST", "127.0.0.1")
        port = int(os.environ.get("APP_PORT", "8976"))
        server = ThreadingHTTPServer((host, port), AppHandler)
        destination = "http://{}:{}".format(host, port)
    def request_shutdown(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    print("AI Privacy Check native service listening on {}".format(destination), flush=True)
    try:
        server.serve_forever()
    finally:
        INSTALLER.stop()
        server.server_close()
        if gateway_socket:
            socket_path = Path(gateway_socket)
            if socket_path.exists() or socket_path.is_socket():
                socket_path.unlink()


if __name__ == "__main__":
    run()

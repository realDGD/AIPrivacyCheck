"""Runtime download sources, mirror policies, TLS trust modes, and error classification.

This module centralizes:
- Layered TLS trust strategy (uv-native -> system-certs -> explicit-ca -> fail-closed)
- PyPI package indexes (Cernet primary, Official PyPI fallback)
- PyTorch wheel indexes (SJTUG fixed mirror primary, Official PyTorch fallback)
- Managed CPython sources (Cernet mirror primary, Official fallback)
- Reasonable timeout budgets (mirror fast-fail in 20-40s, official full timeout)
- Comprehensive failure classification (TLS_TRUST, TIMEOUT, DNS, CONNECTION, HTTP_404, HTTP_429, HTTP_5XX, INTEGRITY, LOCAL_IO, etc.)
- User-friendly error messages and URL sanitization
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse, urlunparse

# Managed CPython (Astral python-build-standalone)
MANAGED_PYTHON_VERSION: str = "3.12.9"
CERNET_PYTHON_INSTALL_MIRROR: str = "https://mirrors.cernet.edu.cn/python-build-standalone"
OFFICIAL_PYTHON_INSTALL_SOURCE: str = "official"

# Supply Chain B: PyPI Packages
CERNET_PYPI_INDEX: str = "https://mirrors.cernet.edu.cn/pypi/web/simple"
OFFICIAL_PYPI_INDEX: str = "https://pypi.org/simple"

# Supply Chain C: PyTorch Wheels (Pinned 2.6.0)
# Primary: SJTUG fixed PyTorch mirror (Verified: torch 2.6.0 cu124 & cpu available)
SJTUG_TORCH_INDEX_CUDA: str = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/"
SJTUG_TORCH_INDEX_CPU: str = "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/"

# Fallback: Official PyTorch wheel indexes
OFFICIAL_TORCH_INDEX_CUDA: str = "https://download.pytorch.org/whl/cu124"
OFFICIAL_TORCH_INDEX_CPU: str = "https://download.pytorch.org/whl/cpu"

# Backward-compatibility aliases
CERNET_TORCH_INDEX_CUDA: str = SJTUG_TORCH_INDEX_CUDA
CERNET_TORCH_INDEX_CPU: str = SJTUG_TORCH_INDEX_CPU

# Universal flag constant for system certificates mode
UV_SYSTEM_CERTS: str = "true"

# Timeout budgets in seconds
MIRROR_HTTP_TIMEOUT: int = 25
MIRROR_CONNECT_TIMEOUT: int = 10
MIRROR_HTTP_RETRIES: int = 1

OFFICIAL_HTTP_TIMEOUT: int = 60
OFFICIAL_CONNECT_TIMEOUT: int = 15
OFFICIAL_HTTP_RETRIES: int = 2

# TLS Modes
TLS_MODE_UV_NATIVE: str = "uv-native"
TLS_MODE_SYSTEM_CERTS: str = "system-certs"
TLS_MODE_EXPLICIT_CA: str = "explicit-ca"

TLS_MODES: Tuple[str, ...] = (
    TLS_MODE_UV_NATIVE,
    TLS_MODE_SYSTEM_CERTS,
    TLS_MODE_EXPLICIT_CA,
)

# Standard candidate locations for system PEM CA bundles
STANDARD_CA_BUNDLE_PATHS: Tuple[str, ...] = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem",
    "/etc/ssl/ca-bundle.pem",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
)

# Failure category constants
FAIL_TLS_TRUST: str = "TLS_TRUST"
FAIL_TIMEOUT: str = "TIMEOUT"
FAIL_DNS: str = "DNS"
FAIL_CONNECTION: str = "CONNECTION"
FAIL_HTTP_404: str = "HTTP_404"
FAIL_HTTP_429: str = "HTTP_429"
FAIL_HTTP_5XX: str = "HTTP_5XX"
FAIL_INTEGRITY: str = "INTEGRITY"
FAIL_LOCAL_IO: str = "LOCAL_IO"
FAIL_VERSION: str = "VERSION"
FAIL_ARCHITECTURE: str = "ARCHITECTURE"
FAIL_UNKNOWN: str = "UNKNOWN"


class InstallResult(str):
    """String subclass holding download source name, successful TLS mode, and attempt telemetry."""

    source: str
    tls_mode: str
    attempts: List[Dict[str, Any]]

    def __new__(
        cls,
        source: str,
        tls_mode: str,
        attempts: Optional[List[Dict[str, Any]]] = None,
    ):
        instance = super().__new__(cls, source)
        instance.source = source
        instance.tls_mode = tls_mode
        instance.attempts = list(attempts or [])
        return instance


def find_explicit_ca_bundle() -> Optional[str]:
    """Finds a valid, non-empty, regular PEM CA bundle on the system if available.

    Never creates dummy files, never downloads unverified CA files.
    """
    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(env_var)
        if val and os.path.isfile(val) and os.path.getsize(val) > 0:
            return val

    for path in STANDARD_CA_BUNDLE_PATHS:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path

    return None


def classify_uv_failure(text: str) -> str:
    """Classifies a uv/pip failure output into a standardized diagnostic category.

    Order of evaluation is critical: integrity and local I/O errors must be
    detected before generic connection or timeout errors.
    """
    if not text:
        return FAIL_UNKNOWN
    lowered = text.lower()

    # 1. Supply-chain integrity violations or archive corruption (FAIL CLOSED)
    integrity_patterns = [
        "sha256 mismatch",
        "hash mismatch",
        "checksum mismatch",
        "digest mismatch",
        "archive corruption",
        "corrupt",
        "capability contract: fail",
        "capability contract fail",
    ]
    if any(pat in lowered for pat in integrity_patterns):
        return FAIL_INTEGRITY

    # 2. Local filesystem, disk space, or permission errors (No mirror fallback)
    local_io_patterns = [
        "no space left on device",
        "permission denied",
        "read-only file system",
        "read-only filesystem",
        "disk quota exceeded",
    ]
    if any(pat in lowered for pat in local_io_patterns):
        return FAIL_LOCAL_IO

    # 3. TLS certificate or trust chain verification errors
    tls_patterns = [
        "unknownissuer",
        "invalid peer certificate",
        "certificate verify failed",
        "certificate_verify_failed",
        "unable to get local issuer",
        "self signed certificate",
        "tlsv1 alert",
        "ssl: cert",
        "ssl_error",
        "ca certificate",
    ]
    if any(pat in lowered for pat in tls_patterns):
        return FAIL_TLS_TRUST

    # 4. Timeout errors
    timeout_patterns = [
        "timed out",
        "timeout",
        "operation timed out",
        "read timeout",
        "connection timeout",
        "connect timeout",
        "elapsed:",
    ]
    if any(pat in lowered for pat in timeout_patterns):
        return FAIL_TIMEOUT

    # 5. DNS / Name resolution errors
    dns_patterns = [
        "name resolution",
        "could not resolve host",
        "getaddrinfo failed",
        "nodename nor servname provided",
    ]
    if any(pat in lowered for pat in dns_patterns):
        return FAIL_DNS

    # 6. HTTP Status codes
    if "404 not found" in lowered or "code: 404" in lowered or "status: 404" in lowered:
        return FAIL_HTTP_404
    if "429" in lowered and ("too many requests" in lowered or "rate limit" in lowered):
        return FAIL_HTTP_429
    if any(code in lowered for code in ("500 internal", "502 bad gateway", "503 service unavailable", "504 gateway timeout")):
        return FAIL_HTTP_5XX

    # 7. Connection / Network errors
    connection_patterns = [
        "connection refused",
        "connection reset",
        "broken pipe",
        "network is unreachable",
        "network unreachable",
        "failed to fetch",
        "client error (connect)",
        "error sending request for url",
        "tunnel error",
    ]
    if any(pat in lowered for pat in connection_patterns):
        return FAIL_CONNECTION

    # 8. Version / Architecture incompatibility
    if "unexpected architecture" in lowered or "incompatible architecture" in lowered:
        return FAIL_ARCHITECTURE
    if "wrong version" in lowered or "no matching distribution" in lowered:
        return FAIL_VERSION

    return FAIL_UNKNOWN


def is_tls_error(text: str) -> bool:
    """Detects whether an error output indicates an HTTPS/TLS certificate verification failure."""
    return classify_uv_failure(text) == FAIL_TLS_TRUST


def is_integrity_or_corruption_error(text: str) -> bool:
    """Detects supply-chain integrity violations or corruption that MUST FAIL CLOSED (never fallback)."""
    return classify_uv_failure(text) == FAIL_INTEGRITY


def is_local_io_error(text: str) -> bool:
    """Detects local disk or permission errors that mirror changes cannot resolve."""
    return classify_uv_failure(text) == FAIL_LOCAL_IO


def is_retryable_network_error(text: str) -> bool:
    """Determines whether an error is a transient network/TLS or mirror failure permitting fallback.

    Integrity violations and local I/O errors are strictly excluded.
    """
    category = classify_uv_failure(text)
    return category in (
        FAIL_TLS_TRUST,
        FAIL_TIMEOUT,
        FAIL_DNS,
        FAIL_CONNECTION,
        FAIL_HTTP_404,
        FAIL_HTTP_429,
        FAIL_HTTP_5XX,
    )


def format_user_friendly_network_error(err_msg: str) -> str:
    """Formats network or TLS errors into actionable, non-cryptic error messages for the fnOS user."""
    cat = classify_uv_failure(err_msg)
    if cat == FAIL_TLS_TRUST:
        detail = f": {err_msg.strip()}" if err_msg else ""
        return (
            f"运行环境下载失败：无法验证 HTTPS 证书链 (TLS Trust Failure{detail})。\n"
            "已按顺序尝试 uv 默认 CA、系统 CA 根证书及显式 CA Bundle 均无法通过验证。\n"
            "请检查 fnOS 系统 CA 根证书 (/etc/ssl/certs)、系统时间或网络代理/证书拦截配置。"
        )
    if cat == FAIL_LOCAL_IO:
        return (
            "运行环境构建失败：本地存储空间不足或权限受限。\n"
            "请检查 fnOS 磁盘剩余空间及应用数据目录读写权限。"
        )
    if cat == FAIL_INTEGRITY:
        return (
            "运行环境下载失败：依赖包完整性校验未通过 (SHA mismatch / Archive corruption)。\n"
            "为保障供应链安全与系统可靠性，已阻断安装并拒绝回退。"
        )
    return f"运行环境下载失败: {err_msg}"


def sanitize_url_for_logging(url: str) -> str:
    """Sanitizes sensitive elements (credentials, query parameters, fragments) from URLs before logging."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        sanitized = urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
        return sanitized
    except Exception:
        return re.sub(r"://[^@]+@", "://***:***@", url)


def get_pypi_index_url(use_mirror: bool = True) -> str:
    """Returns primary Cernet PyPI mirror or official PyPI index URL."""
    return CERNET_PYPI_INDEX if use_mirror else OFFICIAL_PYPI_INDEX


def get_torch_index_url(profile: str, use_mirror: bool = True) -> str:
    """Returns primary SJTUG PyTorch mirror or official PyTorch download index URL."""
    if profile == "torch-cuda":
        return SJTUG_TORCH_INDEX_CUDA if use_mirror else OFFICIAL_TORCH_INDEX_CUDA
    return SJTUG_TORCH_INDEX_CPU if use_mirror else OFFICIAL_TORCH_INDEX_CPU


def build_attempt_env(
    base_env: Mapping[str, str],
    tls_mode: str,
    is_mirror: bool = False,
    ca_bundle_path: Optional[str] = None,
    explicit_ca_bundle: Optional[str] = None,
) -> Dict[str, str]:
    """Applies TLS mode configuration and timeout budget to an execution environment."""
    env = dict(base_env)
    effective_ca = ca_bundle_path or explicit_ca_bundle

    # 1. TLS configuration
    if tls_mode == TLS_MODE_UV_NATIVE:
        env.pop("UV_SYSTEM_CERTS", None)
        env.pop("SSL_CERT_FILE", None)
    elif tls_mode == TLS_MODE_SYSTEM_CERTS:
        env["UV_SYSTEM_CERTS"] = "true"
        env.pop("SSL_CERT_FILE", None)
    elif tls_mode == TLS_MODE_EXPLICIT_CA:
        env.pop("UV_SYSTEM_CERTS", None)
        if effective_ca:
            env["SSL_CERT_FILE"] = str(effective_ca)
    else:
        env.pop("UV_SYSTEM_CERTS", None)

    # 2. Timeout budget configuration
    if is_mirror:
        env["UV_HTTP_TIMEOUT"] = str(MIRROR_HTTP_TIMEOUT)
        env["UV_HTTP_CONNECT_TIMEOUT"] = str(MIRROR_CONNECT_TIMEOUT)
        env["UV_HTTP_RETRIES"] = str(MIRROR_HTTP_RETRIES)
    else:
        env["UV_HTTP_TIMEOUT"] = str(OFFICIAL_HTTP_TIMEOUT)
        env["UV_HTTP_CONNECT_TIMEOUT"] = str(OFFICIAL_CONNECT_TIMEOUT)
        env["UV_HTTP_RETRIES"] = str(OFFICIAL_HTTP_RETRIES)

    return env

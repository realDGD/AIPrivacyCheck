"""Runtime download sources, Cernet/MirrorZ mirror policies, and TLS configurations.

This module centralizes:
- Official Astral musl uv environment flags (UV_SYSTEM_CERTS)
- Cernet/MirrorZ PyPI, Managed Python, and PyTorch wheel mirrors
- Official fallback endpoints
- Network error classification (retryable vs fail-closed integrity errors)
- URL sanitization for telemetry and user-facing error formatting
"""

from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

# Universal uv environment flag ensuring system CA certificate store is used
UV_SYSTEM_CERTS: str = "true"

# Supply Chain A: Managed CPython (Astral python-build-standalone)
# Verified Cernet mirror for python-build-standalone
CERNET_PYTHON_INSTALL_MIRROR: str = "https://mirrors.cernet.edu.cn/python-build-standalone"
OFFICIAL_PYTHON_INSTALL_SOURCE: str = "official"

# Supply Chain B: PyPI Packages
CERNET_PYPI_INDEX: str = "https://mirrors.cernet.edu.cn/pypi/web/simple"
OFFICIAL_PYPI_INDEX: str = "https://pypi.org/simple"

# Supply Chain C: PyTorch Wheels (Pinned 2.6.0)
CERNET_TORCH_INDEX_CUDA: str = "https://mirrors.cernet.edu.cn/pytorch/whl/cu124"
OFFICIAL_TORCH_INDEX_CUDA: str = "https://download.pytorch.org/whl/cu124"

CERNET_TORCH_INDEX_CPU: str = "https://mirrors.cernet.edu.cn/pytorch/whl/cpu"
OFFICIAL_TORCH_INDEX_CPU: str = "https://download.pytorch.org/whl/cpu"


def is_tls_error(text: str) -> bool:
    """Detects whether an error output indicates an HTTPS/TLS certificate verification failure."""
    if not text:
        return False
    lowered = text.lower()
    patterns = [
        "unknownissuer",
        "invalid peer certificate",
        "certificate verify failed",
        "certificate_verify_failed",
        "unable to get local issuer certificate",
        "self signed certificate",
        "tlsv1 alert",
        "ssl: cert",
        "ssl_error",
        "ca certificate",
    ]
    return any(pat in lowered for pat in patterns)


def is_integrity_or_corruption_error(text: str) -> bool:
    """Detects supply-chain integrity violations or corruption that MUST FAIL CLOSED (never fallback)."""
    if not text:
        return False
    lowered = text.lower()
    patterns = [
        "sha256 mismatch",
        "hash mismatch",
        "checksum mismatch",
        "digest mismatch",
        "archive corruption",
        "corrupt",
        "unexpected architecture",
        "invalid executable",
        "wrong version",
        "capability contract: fail",
        "capability contract fail",
    ]
    return any(pat in lowered for pat in patterns)


def is_retryable_network_error(text: str) -> bool:
    """Determines whether an error is a transient network or mirror failure permitting official fallback.

    Integrity violations or corruption errors are strictly excluded to avoid masking attacks or broken mirrors.
    """
    if not text:
        return False
    if is_integrity_or_corruption_error(text):
        return False

    if is_tls_error(text):
        return True

    lowered = text.lower()
    retryable_patterns = [
        "failed to fetch",
        "connection refused",
        "connection reset",
        "connection timed out",
        "timed out",
        "timeout",
        "temporary failure",
        "name resolution",
        "could not resolve host",
        "network is unreachable",
        "network unreachable",
        "broken pipe",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "rate limit",
        "404 not found",
        "404",
        "429",
        "500 internal",
        "502",
        "503",
        "504",
    ]
    return any(pat in lowered for pat in retryable_patterns)


def format_user_friendly_network_error(err_msg: str) -> str:
    """Formats network or TLS errors into actionable, non-cryptic error messages for the fnOS user."""
    if is_tls_error(err_msg):
        return (
            "运行环境下载失败：无法验证 HTTPS 证书链 (TLS UnknownIssuer)。\n"
            "请检查 fnOS 系统 CA 根证书 (/etc/ssl/certs)、网络时间、或网络代理/证书拦截配置。"
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
    """Returns primary Cernet PyTorch mirror or official PyTorch download index URL."""
    if profile == "torch-cuda":
        return CERNET_TORCH_INDEX_CUDA if use_mirror else OFFICIAL_TORCH_INDEX_CUDA
    return CERNET_TORCH_INDEX_CPU if use_mirror else OFFICIAL_TORCH_INDEX_CPU

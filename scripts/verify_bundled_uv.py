#!/usr/bin/env python3
"""Verifies the integrity, ELF architecture, and SHA-256 checksums of bundled uv binaries.

Called during FPK packaging (scripts/build_fpk.sh) and preflight self-checks.
Fails hard if any required architecture binary is missing or corrupted.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Dict, NamedTuple, Optional


class BundledUvSpec(NamedTuple):
    arch_name: str
    relative_path: str
    expected_elf_machine: int  # e_machine in ELF header
    expected_sha256: str
    expected_version: str


EXPECTED_UV_VERSION = "0.12.13"

BUNDLED_UV_SPECS: Dict[str, BundledUvSpec] = {
    "linux-x86_64": BundledUvSpec(
        arch_name="linux-x86_64",
        relative_path="packaging/ai-privacy-check/app/bin/linux-x86_64/uv",
        expected_elf_machine=0x3E,  # EM_X86_64
        expected_sha256="37a89bb1ffa013a95f81f888ef445fef056eb877ac994b713158cbe185809ac9",
        expected_version=EXPECTED_UV_VERSION,
    ),
    "linux-aarch64": BundledUvSpec(
        arch_name="linux-aarch64",
        relative_path="packaging/ai-privacy-check/app/bin/linux-aarch64/uv",
        expected_elf_machine=0xB7,  # EM_AARCH64
        expected_sha256="9ea448a9d8534ec5143dc1328a7a3e865391f851115e2ca5b266fdd6c8a1790e",
        expected_version=EXPECTED_UV_VERSION,
    ),
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_elf_header(path: Path) -> Optional[int]:
    """Inspects ELF header and returns e_machine value, or None if not an ELF binary."""
    try:
        with open(path, "rb") as f:
            header = f.read(20)
        if len(header) < 20:
            return None
        magic, ei_class, ei_data, ei_version, osabi, abiversion, pad, e_type, e_machine = struct.unpack(
            "<4sBBBBB7sHH", header
        )
        if magic != b"\x7fELF" or ei_class != 2:  # 64-bit ELF
            return None
        return e_machine
    except Exception:
        return None


def verify_spec(root_dir: Path, spec: BundledUvSpec) -> None:
    bin_path = (root_dir / spec.relative_path).resolve()
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing bundled uv binary for {spec.arch_name}: {bin_path}")

    if not os.access(bin_path, os.X_OK):
        raise PermissionError(f"Bundled uv binary for {spec.arch_name} is not executable: {bin_path}")

    elf_machine = inspect_elf_header(bin_path)
    if elf_machine is None:
        raise ValueError(f"Bundled uv binary for {spec.arch_name} is not a valid 64-bit ELF executable: {bin_path}")

    if elf_machine != spec.expected_elf_machine:
        raise ValueError(
            f"Bundled uv binary for {spec.arch_name} architecture mismatch: "
            f"expected machine={hex(spec.expected_elf_machine)}, found={hex(elf_machine)}"
        )

    actual_sha256 = compute_sha256(bin_path)
    if actual_sha256 != spec.expected_sha256:
        raise ValueError(
            f"Bundled uv binary for {spec.arch_name} SHA-256 checksum mismatch:\n"
            f"  Expected: {spec.expected_sha256}\n"
            f"  Actual:   {actual_sha256}"
        )

    # If running on Linux matching the target architecture, test execution directly
    if sys.platform.startswith("linux"):
        import platform
        mach = platform.machine().lower()
        is_native = (
            (spec.arch_name == "linux-x86_64" and mach in ("x86_64", "amd64")) or
            (spec.arch_name == "linux-aarch64" and mach in ("aarch64", "arm64"))
        )
        if is_native:
            res = subprocess.run([str(bin_path), "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Execution check failed for {spec.arch_name}: {res.stderr.strip()}")
            if spec.expected_version not in res.stdout:
                raise ValueError(
                    f"Version mismatch for {spec.arch_name}: expected {spec.expected_version}, got {res.stdout.strip()}"
                )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Verifying bundled uv binaries in {repo_root}...")

    errors = []
    for arch, spec in BUNDLED_UV_SPECS.items():
        try:
            verify_spec(repo_root, spec)
            print(f"  [OK] {arch}: ELF verified (machine={hex(spec.expected_elf_machine)}), SHA-256 match")
        except Exception as exc:
            errors.append(f"  [FAIL] {arch}: {exc}")

    # Verify third-party notices packaging
    for notice_rel in (
        "packaging/ai-privacy-check/THIRD_PARTY_NOTICES.md",
        "packaging/ai-privacy-check/app/THIRD_PARTY_NOTICES.md",
    ):
        notice_path = repo_root / notice_rel
        if not notice_path.is_file():
            errors.append(f"  [FAIL] Missing package notice: {notice_rel}")
        else:
            text = notice_path.read_text(encoding="utf-8")
            if "Astral" not in text or "uv" not in text:
                errors.append(f"  [FAIL] Notice file {notice_rel} does not mention Astral / uv")
            else:
                print(f"  [OK] Notice verified: {notice_rel}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("\nBundled uv / notice verification FAILED!", file=sys.stderr)
        return 1

    print("All bundled uv binaries and license notices verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

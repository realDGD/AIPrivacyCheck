"""Tests for uv-managed Python runtime foundation and transactional rebuild.

Covers:
1. Native capability probe (complete vs missing _lzma, _ssl, _sqlite3).
2. Legacy runtime adoption without rebuild when healthy (schema v3 written without recreating venv).
3. Legacy runtime incompatibility sets runtime_rebuild_required and repairable.
4. Disk space preflight check fails fast without mutating state.
5. Profile concurrency lock prevents concurrent rebuild operations.
6. Transactional staging rollback on failure leaves existing venv intact.
7. Multi-model dependency aggregation installs union of requirements in staging venv.
8. Zero model weight mutation and zero downloads during rebuild (checksums unchanged).
9. Atomic swap to final venv with schema v3 manifest and smoke test verification.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import MagicMock, patch

# Set up import path
PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
import sys
sys.path.insert(0, str(SERVER_DIR))

from model_installer import (
    clear_dependency_probe_cache,
    rebuild_runtime,
    repair_model_runtime,
    runtime_operation_lock,
    check_disk_space_for_rebuild,
    run_install_pypi_with_fallback,
    run_install_torch_with_fallback,
)
from privacy.chinese_ie import ChineseIEDetector
from privacy.detectors import GLiNERDetector
from privacy.device import DeviceManager, DEVICE_MANAGER
from privacy.model_catalog import MODEL_CATALOG, get_model_descriptor
from privacy.python_runtime import (
    MANAGED_PYTHON_VERSION,
    PROBE_BASE_PACKAGES,
    PYTHON_CAPABILITY_CONTRACT,
    REQUIRED_PYTHON_CAPABILITIES,
    UV_VERSION,
    UV_X86_64_SHA256,
    UV_AARCH64_SHA256,
    _VERIFIED_BUNDLED_UV_PATHS,
    clear_verified_uv_cache,
    build_uv_env,
    ensure_managed_python,
    ensure_managed_python_info,
    find_uv,
    find_uv_info,
    get_python_installations_dir,
    get_uv_cache_dir,
    probe_base_runtime_contract,
    probe_python_capabilities,
    verify_bundled_uv,
)
from privacy.runtime_sources import (
    UV_SYSTEM_CERTS,
    CERNET_PYPI_INDEX,
    OFFICIAL_PYPI_INDEX,
    SJTUG_TORCH_INDEX_CPU,
    OFFICIAL_TORCH_INDEX_CPU,
    SJTUG_TORCH_INDEX_CUDA,
    OFFICIAL_TORCH_INDEX_CUDA,
    CERNET_TORCH_INDEX_CPU,
    CERNET_TORCH_INDEX_CUDA,
    CERNET_PYTHON_INSTALL_MIRROR,
    TLS_MODE_UV_NATIVE,
    TLS_MODE_SYSTEM_CERTS,
    TLS_MODE_EXPLICIT_CA,
    build_attempt_env,
    classify_uv_failure,
    is_tls_error,
    is_integrity_or_corruption_error,
    is_local_io_error,
    is_retryable_network_error,
    format_user_friendly_network_error,
    get_pypi_index_url,
    get_torch_index_url,
)
from privacy.runtime_manager import (
    RuntimeManager,
    PROFILE_TORCH_CPU,
    PROFILE_TORCH_CUDA,
    get_runtime_manager,
)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class ManagedPythonRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        DEVICE_MANAGER.set_data_dir(self.data_dir)
        clear_dependency_probe_cache()

    def tearDown(self):
        clear_dependency_probe_cache()
        self.temp_dir.cleanup()

    def _create_mock_installed_model(self, model_id: str) -> Path:
        m_dir = self.data_dir / "models" / model_id
        m_dir.mkdir(parents=True, exist_ok=True)
        if model_id == "siamese-uie":
            (m_dir / "configuration.json").write_text('{"model_type": "uie"}', encoding="utf-8")
            (m_dir / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")
            (m_dir / "vocab.txt").write_text("token1\ntoken2\n", encoding="utf-8")
            (m_dir / "pytorch_model.bin").write_text("fake-uie-weights-12345", encoding="utf-8")
        elif model_id == "gliner-pii-edge":
            (m_dir / "gliner_config.json").write_text('{"model_type": "gliner"}', encoding="utf-8")
            (m_dir / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
            (m_dir / "model.safetensors").write_text("fake-gliner-weights-67890", encoding="utf-8")
        elif model_id == "memprivacy-1.7b-rl":
            (m_dir / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
            (m_dir / "model.safetensors").write_text("fake-mem-weights-99999", encoding="utf-8")
        return m_dir

    def _create_mock_runtime_venv(self, profile: str = PROFILE_TORCH_CPU, schema_version: int = 3, missing_caps: List[str] = None) -> Path:
        rt_dir = self.data_dir / "runtimes" / profile
        venv_bin = rt_dir / "venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        py_bin = venv_bin / "python"

        missing = missing_caps or []
        caps_dict = {cap: (cap not in missing) for cap in REQUIRED_PYTHON_CAPABILITIES}
        is_ok = len(missing) == 0

        py_bin.write_text(
            '#!/bin/sh\n'
            'echo \'{"ok": ' + str(is_ok).lower() + ', "version": "3.12.9", "base_prefix": "/usr", "capabilities": {}, "missing": ' + json.dumps(missing) + ', "python_runtime_ready": ' + str(is_ok).lower() + ', "python_runtime_source": "legacy", "python_runtime_version": "3.12.9", "missing_python_capabilities": ' + json.dumps(missing) + ', "base_packages_ready": ' + str(is_ok).lower() + ', "missing_base_packages": [], "base_contract_violations": [], "packages": {"torch": "2.6.0", "transformers": "4.51.0", "modelscope": "1.20.0", "gliner": "0.2.0"}, "framework_version": "2.6.0+cpu", "torch_version": "2.6.0+cpu", "cuda_available": false}\'\n'
        )
        py_bin.chmod(0o755)

        manifest = {
            "schema_version": schema_version,
            "profile": profile,
            "created_at": 1741000000,
            "python_interpreter": str(py_bin),
            "framework_version": "2.6.0+cpu",
        }
        if schema_version >= 3:
            manifest.update({
                "python_runtime_source": "managed" if not missing else "legacy-system-python",
                "python_runtime_version": "3.12.9",
                "capabilities": [c for c in REQUIRED_PYTHON_CAPABILITIES if c not in missing],
                "capabilities_verified_at": 1741000000,
            })
        (rt_dir / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return py_bin

    def _create_rebuild_runner(self, recorded_commands: List[List[str]] = None, pip_installs: List[List[str]] = None, fail_pip: bool = False, cuda_available: bool = False):
        def runner(cmd, **kwargs):
            cmd_list = [str(c) for c in cmd]
            is_cuda = cuda_available or any("torch-cuda" in c for c in cmd_list)
            if recorded_commands is not None:
                recorded_commands.append(cmd_list)
            if "pip" in cmd_list and "install" in cmd_list:
                if pip_installs is not None:
                    pip_installs.append(cmd_list)
                if fail_pip:
                    return 1, "", "Simulated network failure while downloading wheels"
                return 0, "", ""
            if len(cmd_list) >= 3 and cmd_list[1] == "python":
                if "find" in cmd_list:
                    py_path = self.data_dir / "python" / "installations" / "cpython-3.12.9-linux-x86_64" / "bin" / "python3"
                    if py_path.is_file():
                        return 0, str(py_path), ""
                    return 1, "", "not found"
                if "install" in cmd_list:
                    py_dir = self.data_dir / "python" / "installations" / "cpython-3.12.9-linux-x86_64" / "bin"
                    py_dir.mkdir(parents=True, exist_ok=True)
                    py_bin = py_dir / "python3"
                    py_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                    py_bin.chmod(0o755)
                    return 0, "", ""
            if len(cmd_list) >= 2 and cmd_list[1] == "venv":
                staging_path = Path(cmd_list[4])
                s_bin = staging_path / "bin"
                s_bin.mkdir(parents=True, exist_ok=True)
                s_py = s_bin / "python"
                s_cuda = "true" if (is_cuda or "torch-cuda" in str(staging_path)) else "false"
                s_py.write_text(
                    '#!/bin/sh\n'
                    f'echo \'{{"ok": true, "version": "3.12.9", "base_prefix": "/app/python/installations/cpython", "capabilities": {{}}, "missing": [], "python_runtime_ready": true, "python_runtime_source": "uv-managed", "python_runtime_version": "3.12.9", "missing_python_capabilities": [], "base_packages_ready": true, "missing_base_packages": [], "base_contract_violations": [], "packages": {{"torch": "2.6.0", "transformers": "4.51.0", "modelscope": "1.20.0", "gliner": "0.2.0"}}, "framework_version": "2.6.0", "torch_version": "2.6.0", "cuda_available": {s_cuda}}}\'\n'
                )
                s_py.chmod(0o755)
                return 0, "", ""
            if "-c" in cmd_list:
                code = cmd_list[cmd_list.index("-c") + 1]
                if "BASE_PACKAGES" in code:
                    return 0, json.dumps({
                        "base_packages_ready": True,
                        "missing_base_packages": [],
                        "base_contract_violations": [],
                        "packages": {
                            "torch": "2.6.0",
                            "transformers": "4.51.0",
                            "modelscope": "1.20.0",
                            "gliner": "0.2.0",
                            "numpy": "1.26.4",
                            "packaging": "24.2",
                            "tqdm": "4.66.5",
                            "accelerate": "0.34.2",
                        },
                    }), ""
                if "CAPABILITIES" in code:
                    caps_dict = {cap: True for cap in REQUIRED_PYTHON_CAPABILITIES}
                    return 0, json.dumps({
                        "ok": True,
                        "version": "3.12.9",
                        "base_prefix": "/app/python/installations/cpython",
                        "capabilities": caps_dict,
                        "missing": [],
                    }), ""
                if "cap_names" in code:
                    return 0, json.dumps({
                        "python_runtime_ready": True,
                        "python_runtime_source": "uv-managed",
                        "python_runtime_version": "3.12.9",
                        "missing_python_capabilities": [],
                        "base_packages_ready": True,
                        "missing_base_packages": [],
                        "base_contract_violations": [],
                        "packages": {
                            "torch": "2.6.0",
                            "transformers": "4.51.0",
                            "modelscope": "1.20.0",
                            "gliner": "0.2.0",
                        },
                        "framework_version": "2.6.0",
                        "torch_version": "2.6.0",
                        "cuda_available": is_cuda,
                    }), ""
                if "DEPENDENCY_PROBE_MARKER" in code or "specs_json" in code or "metadata.version" in code:
                    return 0, json.dumps({"ok": True, "missing": [], "error": None}), ""
            return 0, "", ""
        return runner

    def test_probe_python_capabilities_success_and_failure(self):
        """probe_python_capabilities accurately distinguishes complete vs broken environments."""
        # 1. Complete environment
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write(
                '#!/bin/sh\n'
                'echo \'{"ok": true, "version": "3.12.9", "base_prefix": "/app/python/installations/cpython", "capabilities": {}, "missing": []}\'\n'
            )
            mock_py = Path(tf.name)
        mock_py.chmod(0o755)
        try:
            report = probe_python_capabilities(mock_py)
            self.assertTrue(report["ok"])
            self.assertEqual(report["version"], "3.12.9")
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["source"], "uv-managed")
        finally:
            mock_py.unlink(missing_ok=True)

        # 2. Missing _lzma and _ssl
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write(
                '#!/bin/sh\n'
                'echo \'{"ok": false, "version": "3.10.12", "base_prefix": "/usr", "capabilities": {"_lzma": false, "_ssl": false}, "missing": ["_lzma", "_ssl"]}\'\n'
            )
            mock_broken_py = Path(tf.name)
        mock_broken_py.chmod(0o755)
        try:
            report = probe_python_capabilities(mock_broken_py)
            self.assertFalse(report["ok"])
            self.assertEqual(report["missing"], ["_lzma", "_ssl"])
            self.assertEqual(report["source"], "legacy-system-python")
        finally:
            mock_broken_py.unlink(missing_ok=True)

        # 3. Non-existent interpreter
        missing_py = Path("/nonexistent/python/path")
        report = probe_python_capabilities(missing_py)
        self.assertFalse(report["ok"])
        self.assertIn("解释器不存在", report["error"])

    def test_legacy_runtime_adoption_without_rebuild_when_healthy(self):
        """A healthy legacy runtime (schema_version=1 or 2) is seamlessly adopted to schema_version=3 without rebuilding."""
        py_bin = self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=1, missing_caps=[])
        rt_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        manifest_file = rt_dir / "runtime-manifest.json"

        # Initially schema_version = 1
        initial_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(initial_manifest.get("schema_version"), 1)

        # Probe profile via RuntimeManager
        rt_mgr = get_runtime_manager(self.data_dir)
        probe_result = rt_mgr.probe_profile(PROFILE_TORCH_CPU, force_refresh=True)

        self.assertTrue(probe_result["installed"])
        self.assertTrue(probe_result["python_runtime_ready"])
        self.assertFalse(probe_result["runtime_rebuild_required"])

        # Manifest must have been upgraded in place to schema_version=3
        updated_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(updated_manifest.get("schema_version"), 3)
        self.assertIn("capabilities", updated_manifest)
        self.assertIn("capabilities_verified_at", updated_manifest)

    def test_legacy_runtime_missing_capabilities_triggers_rebuild_required(self):
        """A legacy runtime missing _lzma must flag runtime_rebuild_required and mark detector repairable."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2, missing_caps=["_lzma"])

        rt_mgr = get_runtime_manager(self.data_dir)
        probe_result = rt_mgr.probe_profile(PROFILE_TORCH_CPU, force_refresh=True)

        self.assertTrue(probe_result["installed"])
        self.assertFalse(probe_result["python_runtime_ready"])
        self.assertTrue(probe_result["runtime_rebuild_required"])
        self.assertIn("_lzma", probe_result["missing_python_capabilities"])

        # SiameseUIE Detector should report model_ready=False and repairable=True
        det = ChineseIEDetector(self.data_dir, active_model_id="siamese-uie")
        status = det.status()
        self.assertTrue(status["installed"])
        self.assertFalse(status["model_ready"])
        self.assertTrue(status["runtime_rebuild_required"])
        self.assertTrue(status["repairable"])

    def test_rebuild_runtime_disk_space_preflight_failure(self):
        """Rebuild fails fast if disk free space is below the safety threshold (2GB CPU / 4.5GB CUDA)."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=3)

        # Mock shutil.disk_usage returning 500 MB free
        low_disk = shutil._ntuple_diskusage(total=100 * 1024**3, used=99.5 * 1024**3, free=500 * 1024**2)
        with patch("shutil.disk_usage", return_value=low_disk):
            ok, msg = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU)
            self.assertFalse(ok)
            self.assertIn("空间不足", msg)

        # Verify no staging directory was created
        profile_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        staging_dirs = list(profile_dir.glob("venv.rebuild-*"))
        self.assertEqual(len(staging_dirs), 0)

    def test_rebuild_runtime_concurrency_locking(self):
        """Concurrent rebuild operations on the same profile are blocked by runtime_operation_lock."""
        lock_error: List[str] = []
        with runtime_operation_lock(self.data_dir, PROFILE_TORCH_CPU):
            def try_acquire():
                try:
                    with runtime_operation_lock(self.data_dir, PROFILE_TORCH_CPU, non_blocking=True):
                        pass
                except RuntimeError as exc:
                    lock_error.append(str(exc))

            t = threading.Thread(target=try_acquire)
            t.start()
            t.join()

        self.assertEqual(len(lock_error), 1)
        self.assertIn("正在执行环境部署或重建操作", lock_error[0])

    def test_rebuild_runtime_staging_rollback_on_failure(self):
        """On staging failure, staging directory is deleted and existing venv remains 100% intact."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2, missing_caps=["_lzma"])

        # Place a sentinel marker in the existing venv
        venv_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU / "venv"
        sentinel_file = venv_dir / "intact_marker.txt"
        sentinel_file.write_text("existing-venv-marker-12345", encoding="utf-8")

        failing_runner = self._create_rebuild_runner(fail_pip=True)

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer._find_uv", return_value="/usr/bin/uv"):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    ok, msg = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=failing_runner)

        self.assertFalse(ok)
        self.assertIn("Simulated network failure", msg)

        # Verify existing venv still has its intact marker
        self.assertTrue(sentinel_file.is_file())
        self.assertEqual(sentinel_file.read_text(encoding="utf-8"), "existing-venv-marker-12345")

        # Verify no staging directory remains
        profile_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        staging_dirs = list(profile_dir.glob("venv.rebuild-*"))
        self.assertEqual(len(staging_dirs), 0)

    def test_rebuild_runtime_zero_model_weight_mutation_and_no_downloads(self):
        """Rebuild must NEVER re-download or modify model weights in ${DATA_DIR}/models/*."""
        uie_dir = self._create_mock_installed_model("siamese-uie")
        gliner_dir = self._create_mock_installed_model("gliner-pii-edge")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2, missing_caps=["_lzma"])

        # Record pre-rebuild file checksums
        uie_hashes_before = {f.name: _file_hash(f) for f in uie_dir.iterdir() if f.is_file()}
        gliner_hashes_before = {f.name: _file_hash(f) for f in gliner_dir.iterdir() if f.is_file()}

        installed_commands: List[List[str]] = []
        runner = self._create_rebuild_runner(recorded_commands=installed_commands)

        # Patch download functions to guarantee zero downloads occur
        with patch("model_installer.download_modelscope_model", side_effect=AssertionError("download_modelscope_model was called during rebuild!")):
            with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
                with patch("model_installer._find_uv", return_value="/usr/bin/uv"):
                    with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                        with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                            ok, msg = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertTrue(ok)

        # Verify weights were not touched (100% identical hashes)
        uie_hashes_after = {f.name: _file_hash(f) for f in uie_dir.iterdir() if f.is_file()}
        gliner_hashes_after = {f.name: _file_hash(f) for f in gliner_dir.iterdir() if f.is_file()}
        self.assertEqual(uie_hashes_before, uie_hashes_after)
        self.assertEqual(gliner_hashes_before, gliner_hashes_after)

    def test_rebuild_runtime_aggregates_dependencies_across_installed_models(self):
        """Rebuild must aggregate model-specific dependencies across all installed models for that profile."""
        self._create_mock_installed_model("siamese-uie")  # requires 'addict'
        self._create_mock_installed_model("gliner-pii-edge")  # requires 'gliner>=0.1.13'
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2)

        executed_pip_installs: List[List[str]] = []
        runner = self._create_rebuild_runner(pip_installs=executed_pip_installs)

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer._find_uv", return_value="/usr/bin/uv"):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                        ok, msg = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertTrue(ok)

        # Check that both addict and datasets from siamese-uie were requested in staging
        installed_args = [arg for cmd in executed_pip_installs for arg in cmd]
        self.assertIn("addict", installed_args)
        self.assertIn("datasets", installed_args)
        # Check base dependencies include gliner
        self.assertIn("gliner", installed_args)

    def test_rebuild_runtime_atomic_swap_and_manifest_v3(self):
        """Atomic swap promotes staging venv to final venv and writes schema v3 manifest."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2, missing_caps=["_lzma"])

        runner = self._create_rebuild_runner()

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer._find_uv", return_value="/usr/bin/uv"):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                        ok, msg = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertTrue(ok)

        # Manifest must be schema_version 3 with managed source
        rt_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        manifest = json.loads((rt_dir / "runtime-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["python_runtime_source"], "managed")
        self.assertEqual(manifest["python_runtime_version"], "3.12.9")
        self.assertEqual(len(manifest["capabilities"]), len(REQUIRED_PYTHON_CAPABILITIES))

        # Runtime probe must report ready
        rt_mgr = get_runtime_manager(self.data_dir)
        probe = rt_mgr.probe_profile(PROFILE_TORCH_CPU, force_refresh=True)
        self.assertTrue(probe["python_runtime_ready"])
        self.assertEqual(probe["missing_python_capabilities"], [])

        # Venv directory must exist and contain python
        final_py = rt_dir / "venv" / "bin" / "python"
        self.assertTrue(final_py.is_file())

    # ==========================================
    # v0.6.12 Mid-Swap Rollback & Readiness Tests
    # ==========================================

    def test_mid_swap_failure_case1_staging_to_final_fails_restores_old_runtime(self):
        """Case 1: old -> backup PASS, staging -> final FAIL -> old runtime restored."""
        self._create_mock_installed_model("siamese-uie")
        old_py = self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2)
        rt_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        venv_dir = rt_dir / "venv"
        marker = venv_dir / "old_runtime_sentinel.txt"
        marker.write_text("old-runtime-sentinel-12345", encoding="utf-8")

        runner = self._create_rebuild_runner()

        orig_replace = os.replace
        def broken_replace(src, dst):
            # Fail when promoting staging venv to venv_dir
            if "venv.rebuild-" in str(src) and str(dst) == str(venv_dir):
                raise OSError("Disk write I/O error during promotion")
            return orig_replace(src, dst)

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer.find_uv_info", return_value=("/usr/bin/uv", "bundled")):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                        with patch("os.replace", side_effect=broken_replace):
                            ok, err = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertFalse(ok)
        self.assertIn("Disk write I/O error during promotion", err)
        # Old runtime must be fully restored!
        self.assertTrue(venv_dir.is_dir())
        self.assertTrue(marker.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "old-runtime-sentinel-12345")

    def test_mid_swap_failure_case2_manifest_write_fails_removes_broken_and_restores_old(self):
        """Case 2: promote PASS, manifest write FAIL -> broken new runtime removed, old runtime restored."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2)
        rt_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        venv_dir = rt_dir / "venv"
        marker = venv_dir / "old_runtime_sentinel.txt"
        marker.write_text("old-runtime-sentinel-54321", encoding="utf-8")

        runner = self._create_rebuild_runner()

        orig_write_text = Path.write_text
        def broken_write_text(path_obj, *args, **kwargs):
            if path_obj.name == "runtime-manifest.json":
                raise IOError("Read-only filesystem error when writing manifest")
            return orig_write_text(path_obj, *args, **kwargs)

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer.find_uv_info", return_value=("/usr/bin/uv", "bundled")):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                        with patch.object(Path, "write_text", broken_write_text):
                            ok, err = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertFalse(ok)
        self.assertIn("Read-only filesystem error", err)
        # Old runtime restored
        self.assertTrue(venv_dir.is_dir())
        self.assertTrue(marker.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "old-runtime-sentinel-54321")

    def test_mid_swap_failure_case3_final_probe_fails_removes_broken_and_restores_old(self):
        """Case 3: promote PASS, final probe FAIL -> broken new runtime removed, old runtime restored."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2)
        rt_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        venv_dir = rt_dir / "venv"
        marker = venv_dir / "old_runtime_sentinel.txt"
        marker.write_text("old-runtime-sentinel-final-probe", encoding="utf-8")

        runner = self._create_rebuild_runner()

        rt_mgr = get_runtime_manager(self.data_dir)
        def failing_final_probe(profile, force_refresh=False):
            return {
                "profile": profile,
                "installed": True,
                "verified": False,
                "runtime_rebuild_required": True,
                "error": "Promoted venv crashed on import check",
            }

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer.find_uv_info", return_value=("/usr/bin/uv", "bundled")):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                        with patch.object(rt_mgr, "probe_profile", side_effect=failing_final_probe):
                            ok, err = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertFalse(ok)
        self.assertIn("Promoted venv crashed on import check", err)
        # Old runtime restored
        self.assertTrue(venv_dir.is_dir())
        self.assertTrue(marker.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "old-runtime-sentinel-final-probe")

    def test_mid_swap_failure_case4_rollback_itself_fails_preserves_backup_and_reports_critical(self):
        """Case 4: rollback itself fails -> preserve backup, raise critical recovery error, do NOT delete backup."""
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CPU, schema_version=2)
        rt_dir = self.data_dir / "runtimes" / PROFILE_TORCH_CPU
        venv_dir = rt_dir / "venv"
        marker = venv_dir / "old_runtime_sentinel.txt"
        marker.write_text("old-runtime-critical-test", encoding="utf-8")

        runner = self._create_rebuild_runner()

        orig_replace = os.replace
        def failing_rollback_replace(src, dst):
            if "venv.old." in str(src) and str(dst) == str(venv_dir):
                raise OSError("Permission denied on rollback rename")
            return orig_replace(src, dst)

        rt_mgr = get_runtime_manager(self.data_dir)
        def failing_probe(profile, force_refresh=False):
            return {"installed": True, "verified": False, "runtime_rebuild_required": True, "error": "Crash"}

        with patch("model_installer.find_uv", return_value="/usr/bin/uv"):
            with patch("model_installer.find_uv_info", return_value=("/usr/bin/uv", "bundled")):
                with patch("model_installer.ensure_managed_python", return_value=Path("/mock/py")):
                    with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                        with patch.object(rt_mgr, "probe_profile", side_effect=failing_probe):
                            with patch("os.replace", side_effect=failing_rollback_replace):
                                ok, err = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertFalse(ok)
        self.assertIn("CRITICAL RECOVERY ERROR", err)
        # Backup MUST be preserved and not deleted!
        backups = list(rt_dir.glob("venv.old.*"))
        self.assertTrue(len(backups) >= 1)
        self.assertTrue((backups[0] / "old_runtime_sentinel.txt").is_file())

    def test_base_ml_contract_transformers_version_violation(self):
        """Transformers 5.x violates >=4.51,<5 constraint and flags base_packages_ready=False."""
        def runner(cmd, **kwargs):
            return 0, json.dumps({
                "base_packages_ready": False,
                "missing_base_packages": [],
                "base_contract_violations": ["transformers==5.16.1 violates >=4.51,<5"],
                "packages": {
                    "torch": "2.6.0",
                    "transformers": "5.16.1",
                    "modelscope": "1.20.0",
                    "gliner": "0.2.0",
                    "numpy": "1.26.4",
                    "packaging": "24.2",
                    "tqdm": "4.66.5",
                    "accelerate": "0.34.2",
                },
            }), ""

        mock_py = self.data_dir / "mock_py"
        mock_py.write_text("#!/bin/sh\n", encoding="utf-8")
        mock_py.chmod(0o755)

        res = probe_base_runtime_contract(mock_py, runner=runner)
        self.assertFalse(res["base_packages_ready"])
        self.assertIn("transformers==5.16.1 violates >=4.51,<5", res["base_contract_violations"])

    def test_base_ml_contract_missing_modelscope(self):
        """Missing modelscope flags base_packages_ready=False."""
        def runner(cmd, **kwargs):
            return 0, json.dumps({
                "base_packages_ready": False,
                "missing_base_packages": ["modelscope"],
                "base_contract_violations": [],
                "packages": {
                    "torch": "2.6.0",
                    "transformers": "4.51.0",
                    "gliner": "0.2.0",
                },
            }), ""

        mock_py = self.data_dir / "mock_py"
        mock_py.write_text("#!/bin/sh\n", encoding="utf-8")
        mock_py.chmod(0o755)

        res = probe_base_runtime_contract(mock_py, runner=runner)
        self.assertFalse(res["base_packages_ready"])
        self.assertIn("modelscope", res["missing_base_packages"])

    def test_bundled_uv_selected_when_system_path_has_no_uv(self):
        """Bundled uv is prioritized and selected when system PATH contains no uv."""
        fake_app_dir = self.data_dir / "fake_app"
        bin_dir = fake_app_dir / "bin" / "linux-x86_64"
        bin_dir.mkdir(parents=True, exist_ok=True)
        fake_uv = bin_dir / "uv"
        fake_uv.write_text("#!/bin/sh\necho uv 0.12.13\n", encoding="utf-8")
        fake_uv.chmod(0o755)

        def runner(cmd, **kwargs):
            return 0, f"uv {UV_VERSION} (bundled)\n", ""

        uv_path, uv_source = find_uv_info(
            app_dir=fake_app_dir,
            runner=runner,
            target_machine="x86_64",
            target_platform="linux",
        )
        self.assertEqual(uv_source, "bundled")
        self.assertEqual(Path(uv_path).resolve(), fake_uv.resolve())

    def test_bundled_uv_corrupt_fails_clearly_and_does_not_mutate_runtime(self):
        """Corrupt bundled uv raises RuntimeError on find_uv_info and prevents silent fallback."""
        fake_app_dir = self.data_dir / "fake_app"
        bin_dir = fake_app_dir / "bin" / "linux-x86_64"
        bin_dir.mkdir(parents=True, exist_ok=True)
        fake_uv = bin_dir / "uv"
        fake_uv.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake_uv.chmod(0o755)

        def runner(cmd, **kwargs):
            return 1, "", "Segmentation fault"

        with self.assertRaises(RuntimeError) as ctx:
            find_uv_info(
                app_dir=fake_app_dir,
                runner=runner,
                target_machine="x86_64",
                target_platform="linux",
                strict_bundled_check=True,
            )
        self.assertIn("Bundled uv binary invalid", str(ctx.exception))

    def test_architecture_mismatch_fails_clearly_before_mutation(self):
        """Unsupported architecture raises RuntimeError before any runtime mutation."""
        with self.assertRaises(RuntimeError) as ctx:
            find_uv_info(
                target_machine="riscv64",
                target_platform="linux",
            )
        self.assertIn("Unsupported architecture", str(ctx.exception))

    def test_real_fnos_scenario_regression(self):
        """Exact reproduction of fnOS blocker:
        - System PATH has NO uv
        - Bundled uv exists in app/bin/linux-x86_64/uv
        - Legacy torch-cuda has missing _lzma
        - SiameseUIE and GLiNER are installed
        - Repair triggers rebuild with bundled uv, restores all dependencies, 0 weight download.
        """
        self._create_mock_installed_model("siamese-uie")
        self._create_mock_installed_model("gliner-pii-edge")
        uie_weights = self.data_dir / "models" / "siamese-uie" / "pytorch_model.bin"
        gliner_weights = self.data_dir / "models" / "gliner-pii-edge" / "model.safetensors"
        uie_hash_before = _file_hash(uie_weights)
        gliner_hash_before = _file_hash(gliner_weights)

        # Create broken legacy torch-cuda venv missing _lzma
        self._create_mock_runtime_venv(profile=PROFILE_TORCH_CUDA, schema_version=2, missing_caps=["_lzma"])

        # Create bundled uv binary in app/bin
        fake_app_dir = self.data_dir / "app"
        bundled_uv_bin = fake_app_dir / "bin" / "linux-x86_64" / "uv"
        bundled_uv_bin.parent.mkdir(parents=True, exist_ok=True)
        bundled_uv_bin.write_text("#!/bin/sh\necho uv 0.12.13\n", encoding="utf-8")
        bundled_uv_bin.chmod(0o755)

        DEVICE_MANAGER.set_requested_device("cuda")

        executed_commands: List[List[str]] = []
        runner = self._create_rebuild_runner(recorded_commands=executed_commands)

        with patch("shutil.which", return_value=None):  # System PATH has NO uv!
            with patch("platform.machine", return_value="x86_64"):
                with patch("sys.platform", "linux"):
                    with patch("privacy.python_runtime.verify_bundled_uv", return_value=(True, None)):
                        with patch.object(DEVICE_MANAGER._hw_probe, "probe_nvidia", return_value={"nvidia_available": True}):
                            with patch("privacy.worker_client.WorkerClient.run_smoke_test", return_value=(True, "ok")):
                                # Execute repair
                                ok, msg = repair_model_runtime(self.data_dir, "siamese-uie", command_runner=runner)

        self.assertTrue(ok)
        self.assertIn("运行环境修复完成", msg)

        # Check weights are completely untouched
        self.assertEqual(_file_hash(uie_weights), uie_hash_before)
        self.assertEqual(_file_hash(gliner_weights), gliner_hash_before)

        # Check manifest schema v3 was created
        manifest_path = self.data_dir / "runtimes" / PROFILE_TORCH_CUDA / "runtime-manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["profile"], PROFILE_TORCH_CUDA)

        # Detector status must be model_ready = True
        with patch.object(DEVICE_MANAGER._hw_probe, "probe_nvidia", return_value={"nvidia_available": True}):
            det = ChineseIEDetector(self.data_dir)
            st = det.status()
            self.assertTrue(st["model_ready"])
            self.assertTrue(st["python_runtime_ready"])
            self.assertFalse(st["runtime_rebuild_required"])
            self.assertTrue(st["base_packages_ready"])

    def test_uv_env_default_native_and_layered_tls(self):
        """build_uv_env defaults to uv-native (no UV_SYSTEM_CERTS), while build_attempt_env sets layered TLS."""
        env = build_uv_env(self.data_dir)
        self.assertNotIn("UV_SYSTEM_CERTS", env)
        self.assertTrue(str(self.data_dir) in env.get("UV_PYTHON_INSTALL_DIR", ""))
        self.assertTrue(str(self.data_dir) in env.get("UV_CACHE_DIR", ""))

        # Layered TLS Mode A: uv-native
        env_native = build_attempt_env(env, TLS_MODE_UV_NATIVE, is_mirror=True)
        self.assertNotIn("UV_SYSTEM_CERTS", env_native)
        self.assertEqual(env_native.get("UV_HTTP_TIMEOUT"), "25")
        self.assertEqual(env_native.get("UV_HTTP_RETRIES"), "1")

        # Layered TLS Mode B: system-certs
        env_system = build_attempt_env(env, TLS_MODE_SYSTEM_CERTS, is_mirror=False)
        self.assertEqual(env_system.get("UV_SYSTEM_CERTS"), "true")
        self.assertEqual(env_system.get("UV_HTTP_TIMEOUT"), "60")
        self.assertEqual(env_system.get("UV_HTTP_RETRIES"), "2")

        # Layered TLS Mode C: explicit-ca with fake cert file
        fake_ca = self.data_dir / "fake_ca.crt"
        fake_ca.write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
        env_explicit = build_attempt_env(env, TLS_MODE_EXPLICIT_CA, is_mirror=True, explicit_ca_bundle=str(fake_ca))
        self.assertEqual(env_explicit.get("SSL_CERT_FILE"), str(fake_ca))

    def test_cernet_pypi_primary(self):
        """Primary PyPI package installations query Cernet mirror endpoint."""
        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            return 0, "Successfully installed\n", ""

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        source = run_install_pypi_with_fallback(
            runner=fake_runner,
            uv_bin="/bin/uv",
            interp=interp,
            venv_dir=venv,
            packages=["modelscope", "numpy"],
            env=env,
        )
        self.assertEqual(source, "cernet")
        self.assertEqual(source.source, "cernet")
        self.assertEqual(source.tls_mode, "uv-native")
        self.assertEqual(len(recorded_cmds), 1)
        self.assertIn("-i", recorded_cmds[0])
        idx = recorded_cmds[0].index("-i")
        self.assertEqual(recorded_cmds[0][idx + 1], CERNET_PYPI_INDEX)

    def test_official_pypi_fallback_on_network_error(self):
        """When Cernet mirror fails with timeout or disconnect, system falls back to official PyPI."""
        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            if CERNET_PYPI_INDEX in cmd:
                return 1, "", "error: timed out after 25000ms\n"
            return 0, "Successfully installed from official\n", ""

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        source = run_install_pypi_with_fallback(
            runner=fake_runner,
            uv_bin="/bin/uv",
            interp=interp,
            venv_dir=venv,
            packages=["gliner"],
            env=env,
        )
        self.assertEqual(source, "official")
        self.assertEqual(len(recorded_cmds), 2)
        self.assertIn(CERNET_PYPI_INDEX, recorded_cmds[0])
        self.assertIn(OFFICIAL_PYPI_INDEX, recorded_cmds[1])

    def test_tls_mode_layered_retry_success(self):
        """When native TLS fails with UnknownIssuer, system retries with system-certs and succeeds."""
        recorded_attempts = []
        def fake_runner(cmd, env=None, **kwargs):
            recorded_attempts.append(dict(env or {}))
            if not env or not env.get("UV_SYSTEM_CERTS"):
                return 1, "", "error: invalid peer certificate: UnknownIssuer\n"
            return 0, "Successfully installed using system certs\n", ""

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        source = run_install_pypi_with_fallback(
            runner=fake_runner,
            uv_bin="/bin/uv",
            interp=interp,
            venv_dir=venv,
            packages=["gliner"],
            env=env,
        )
        self.assertEqual(source, "cernet")
        self.assertEqual(source.tls_mode, "system-certs")
        self.assertEqual(len(recorded_attempts), 2)
        self.assertNotIn("UV_SYSTEM_CERTS", recorded_attempts[0])
        self.assertEqual(recorded_attempts[1].get("UV_SYSTEM_CERTS"), "true")

    def test_sjtug_torch_primary_and_index_separation(self):
        """PyTorch installation queries SJTUG mirror with index separation."""
        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            return 0, "Successfully installed torch\n", ""

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        source = run_install_torch_with_fallback(
            runner=fake_runner,
            uv_bin="/bin/uv",
            interp=interp,
            venv_dir=venv,
            profile=PROFILE_TORCH_CPU,
            env=env,
        )
        self.assertEqual(source, "sjtug")
        self.assertEqual(source.source, "sjtug")
        self.assertEqual(source.tls_mode, "uv-native")
        self.assertEqual(len(recorded_cmds), 1)
        cmd = recorded_cmds[0]
        self.assertIn("--index", cmd)
        self.assertIn("--default-index", cmd)
        self.assertIn("--index-strategy", cmd)
        idx_strategy = cmd.index("--index-strategy")
        self.assertEqual(cmd[idx_strategy + 1], "first-index")
        idx_torch = cmd.index("--index")
        self.assertEqual(cmd[idx_torch + 1], SJTUG_TORCH_INDEX_CPU)
        idx_default = cmd.index("--default-index")
        self.assertEqual(cmd[idx_default + 1], CERNET_PYPI_INDEX)

    def test_sjtug_torch_fallback_to_official(self):
        """When SJTUG mirror fails with 404 or connection error, falls back to official PyTorch."""
        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            if SJTUG_TORCH_INDEX_CPU in cmd:
                return 1, "", "error: 404 Not Found\n"
            return 0, "Successfully installed torch from official\n", ""

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        source = run_install_torch_with_fallback(
            runner=fake_runner,
            uv_bin="/bin/uv",
            interp=interp,
            venv_dir=venv,
            profile=PROFILE_TORCH_CPU,
            env=env,
        )
        self.assertEqual(source, "official")
        self.assertEqual(len(recorded_cmds), 2)
        self.assertIn(SJTUG_TORCH_INDEX_CPU, recorded_cmds[0])
        self.assertIn(OFFICIAL_TORCH_INDEX_CPU, recorded_cmds[1])

    def test_local_io_failure_fails_closed(self):
        """Local disk full or permission error must FAIL CLOSED immediately without fallback."""
        raw_err = "error: failed to write to disk: No space left on device (os error 28)"
        self.assertTrue(is_local_io_error(raw_err))
        self.assertFalse(is_retryable_network_error(raw_err))

        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            return 1, "", raw_err

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        with self.assertRaises(RuntimeError) as ctx:
            run_install_pypi_with_fallback(
                runner=fake_runner,
                uv_bin="/bin/uv",
                interp=interp,
                venv_dir=venv,
                packages=["any-pkg"],
                env=env,
            )
        self.assertIn("不可换源重试", str(ctx.exception))
        self.assertEqual(len(recorded_cmds), 1)

    def test_manifest_schema_v3_records_tls_modes(self):
        """Manifest schema v3 and probe_profile record tls_modes and download_sources."""
        rt_mgr = RuntimeManager(self.data_dir)
        profile_dir = rt_mgr.profile_dir(PROFILE_TORCH_CPU)
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = profile_dir / "runtime-manifest.json"
        manifest_file.write_text(json.dumps({
            "schema_version": 3,
            "profile": PROFILE_TORCH_CPU,
            "download_sources": {
                "managed_python": "existing-local",
                "pypi": "cernet",
                "torch": "sjtug",
            },
            "tls_modes": {
                "pypi": "uv-native",
                "torch": "system-certs",
            },
        }), encoding="utf-8")

        mock_interp = profile_dir / "venv" / "bin" / "python3"
        mock_interp.parent.mkdir(parents=True, exist_ok=True)
        mock_interp.write_text("#!/bin/sh\n", encoding="utf-8")
        mock_interp.chmod(0o755)

        def fake_runner(cmd, **kwargs):
            return 0, json.dumps({
                "python_runtime_ready": True,
                "framework_version": "2.6.0",
                "packages": {"torch": "2.6.0", "transformers": "4.51.0", "gliner": "0.2.14"},
            }), ""

        rt_mgr._runner = fake_runner
        probe = rt_mgr.probe_profile(PROFILE_TORCH_CPU, force_refresh=True)
        self.assertEqual(probe.get("download_sources", {}).get("torch"), "sjtug")
        self.assertEqual(probe.get("tls_modes", {}).get("torch"), "system-certs")
        self.assertEqual(probe.get("tls_modes", {}).get("pypi"), "uv-native")

    def test_tls_unknown_issuer_is_retryable(self):
        """TLS UnknownIssuer error is classified as retryable network error and formatted with actionable advice."""
        raw_err = "failed to fetch wheel: invalid peer certificate: UnknownIssuer (curl 60)"
        self.assertTrue(is_tls_error(raw_err))
        self.assertTrue(is_retryable_network_error(raw_err))
        self.assertFalse(is_integrity_or_corruption_error(raw_err))
        msg = format_user_friendly_network_error(raw_err)
        self.assertIn("UnknownIssuer", msg)
        self.assertIn("CA 根证书", msg)

    def test_sha_mismatch_is_not_retryable(self):
        """Supply-chain SHA mismatch or corruption must FAIL CLOSED immediately without fallback."""
        raw_err = "hash mismatch for package foo: expected 1234, got 5678"
        self.assertTrue(is_integrity_or_corruption_error(raw_err))
        self.assertFalse(is_retryable_network_error(raw_err))

        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            return 1, "", raw_err

        interp = self.data_dir / "python"
        venv = self.data_dir / "venv"
        env = build_uv_env(self.data_dir)
        with self.assertRaises(RuntimeError) as ctx:
            run_install_pypi_with_fallback(
                runner=fake_runner,
                uv_bin="/bin/uv",
                interp=interp,
                venv_dir=venv,
                packages=["corrupt-pkg"],
                env=env,
            )
        self.assertIn("不可降级重试", str(ctx.exception))
        self.assertEqual(len(recorded_cmds), 1)

    def test_managed_python_existing_install_is_reused(self):
        """When Managed Python 3.12.9 already exists and passes capability contract, reuse it (0 bytes downloaded)."""
        install_dir = get_python_installations_dir(self.data_dir)
        cpython_dir = install_dir / f"cpython-{MANAGED_PYTHON_VERSION}-x86_64"
        py_bin = cpython_dir / "bin" / "python3"
        py_bin.parent.mkdir(parents=True, exist_ok=True)
        py_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        py_bin.chmod(0o755)

        recorded_cmds = []
        def fake_runner(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            if "find" in cmd:
                return 0, f"{py_bin}\n", ""
            if "-c" in cmd:
                return 0, json.dumps({
                    "ok": True,
                    "version": MANAGED_PYTHON_VERSION,
                    "base_prefix": str(cpython_dir),
                    "missing": [],
                    "capabilities": {}
                }), ""
            if "install" in cmd:
                raise AssertionError("Should not execute python install when valid managed Python exists!")
            return 0, "", ""

        resolved, source = ensure_managed_python_info(
            data_dir=self.data_dir,
            uv_bin="/bin/uv",
            runner=fake_runner,
        )
        self.assertEqual(resolved, py_bin.resolve())
        self.assertEqual(source, "existing-local")
        self.assertFalse(any("install" in cmd for cmd in recorded_cmds))

    def test_python_mirror_fallback(self):
        """Managed Python installation falls back to official source on Cernet network failure, fails closed on corruption."""
        install_dir = get_python_installations_dir(self.data_dir)
        target_bin = install_dir / f"cpython-{MANAGED_PYTHON_VERSION}" / "bin" / "python3"

        # Scenario 1: Retryable network error -> official fallback succeeds
        recorded_envs = []
        def fallback_runner(cmd, env=None, **kwargs):
            if env:
                recorded_envs.append(dict(env))
            if "find" in cmd:
                if len(recorded_envs) >= 2:
                    target_bin.parent.mkdir(parents=True, exist_ok=True)
                    target_bin.write_text("#!/bin/sh\n")
                    target_bin.chmod(0o755)
                    return 0, f"{target_bin}\n", ""
                return 1, "", "not found"
            if "-c" in cmd:
                return 0, json.dumps({
                    "ok": True,
                    "version": MANAGED_PYTHON_VERSION,
                    "base_prefix": str(install_dir),
                    "missing": [],
                    "capabilities": {}
                }), ""
            if "install" in cmd:
                if env and env.get("UV_PYTHON_INSTALL_MIRROR") == CERNET_PYTHON_INSTALL_MIRROR:
                    return 1, "", "error: connection reset by peer (cernet)\n"
                return 0, "Installed from official\n", ""
            return 0, "", ""

        resolved, src = ensure_managed_python_info(
            data_dir=self.data_dir,
            uv_bin="/bin/uv",
            runner=fallback_runner,
        )
        self.assertEqual(src, "official")

        # Scenario 2: Corruption/SHA mismatch -> Fail closed without fallback
        def corrupt_runner(cmd, env=None, **kwargs):
            if "find" in cmd:
                return 1, "", "not found"
            if "install" in cmd:
                return 1, "", "archive corruption: checksum mismatch\n"
            return 0, "", ""

        shutil.rmtree(install_dir, ignore_errors=True)
        with self.assertRaises(RuntimeError) as ctx:
            ensure_managed_python_info(
                data_dir=self.data_dir,
                uv_bin="/bin/uv",
                runner=corrupt_runner,
            )
        self.assertIn("完整性校验失败", str(ctx.exception))

    def test_runtime_metadata_remains_consistent_after_failure(self):
        """Metadata consistency transaction across swap failure, probe failure, and success."""
        rt_mgr = get_runtime_manager(self.data_dir)
        profile_dir = rt_mgr.profile_dir(PROFILE_TORCH_CPU)
        profile_dir.mkdir(parents=True, exist_ok=True)
        venv_dir = rt_mgr.venv_dir(PROFILE_TORCH_CPU)
        venv_dir.mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
        (venv_dir / "bin" / "python").chmod(0o755)

        old_manifest_data = {
            "schema_version": 3,
            "profile": PROFILE_TORCH_CPU,
            "state": "original_state_v1",
        }
        old_installed_data = {
            "profile": PROFILE_TORCH_CPU,
            "version": "original_installed_v1",
        }
        manifest_file = profile_dir / "runtime-manifest.json"
        installed_file = profile_dir / "installed.json"
        manifest_file.write_text(json.dumps(old_manifest_data), encoding="utf-8")
        installed_file.write_text(json.dumps(old_installed_data), encoding="utf-8")

        # Case B: Final probe failure triggers transaction rollback to old venv and old metadata
        runner = self._create_rebuild_runner()
        with patch.object(rt_mgr, "probe_profile", return_value={"verified": False, "error": "Simulated hardware smoke fail"}):
            with patch("shutil.which", return_value="/bin/uv"):
                ok, err = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertFalse(ok)
        self.assertIn("Simulated hardware smoke fail", err)

        # Metadata must be perfectly preserved as original, no leftover .tmp
        self.assertTrue(manifest_file.is_file())
        self.assertTrue(installed_file.is_file())
        self.assertFalse((profile_dir / "runtime-manifest.json.tmp").exists())
        self.assertFalse((profile_dir / "installed.json.tmp").exists())
        cur_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(cur_manifest.get("state"), "original_state_v1")
        cur_installed = json.loads(installed_file.read_text(encoding="utf-8"))
        self.assertEqual(cur_installed.get("version"), "original_installed_v1")

        # Case C: Clean success writes download_sources telemetry and schema v3
        with patch.object(rt_mgr, "probe_profile", return_value={
            "verified": True,
            "python_runtime_ready": True,
            "runtime_rebuild_required": False,
            "packages": {"torch": "2.6.0", "transformers": "4.51.0"},
            "cuda_available": False,
        }):
            with patch("shutil.which", return_value="/bin/uv"):
                ok, msg = rebuild_runtime(self.data_dir, PROFILE_TORCH_CPU, command_runner=runner)

        self.assertTrue(ok)
        self.assertFalse((profile_dir / "runtime-manifest.json.tmp").exists())
        final_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(final_manifest["schema_version"], 3)
        self.assertIn("download_sources", final_manifest)
        self.assertIn("python", final_manifest["download_sources"])
        self.assertIn("pypi", final_manifest["download_sources"])
        self.assertIn("torch", final_manifest["download_sources"])

    def test_third_party_notice_is_packaged(self):
        """THIRD_PARTY_NOTICES.md is packaged at both package root and app root with Astral uv notices."""
        pkg_root_notice = PROJECT_DIR / "packaging" / "ai-privacy-check" / "THIRD_PARTY_NOTICES.md"
        app_notice = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "THIRD_PARTY_NOTICES.md"
        self.assertTrue(pkg_root_notice.is_file(), f"Missing {pkg_root_notice}")
        self.assertTrue(app_notice.is_file(), f"Missing {app_notice}")

        content1 = pkg_root_notice.read_text(encoding="utf-8")
        content2 = app_notice.read_text(encoding="utf-8")
        self.assertIn("Astral", content1)
        self.assertIn("uv", content1)
        self.assertIn("MIT License", content1)
        self.assertIn("Apache License", content1)
        self.assertEqual(content1, content2)

    def test_bundled_uv_runtime_sha_verified(self):
        """Bundled uv SHA-256 integrity is checked at runtime and cached in-process."""
        clear_verified_uv_cache()
        real_x86_uv = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "bin" / "linux-x86_64" / "uv"
        real_aarch64_uv = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "bin" / "linux-aarch64" / "uv"

        def mock_runner(cmd, timeout=10):
            return 0, f"uv {UV_VERSION}\n", ""

        # 1. Real x86_64 binary matches pinned SHA
        self.assertNotIn(str(real_x86_uv.resolve()), _VERIFIED_BUNDLED_UV_PATHS)
        ok, err = verify_bundled_uv(real_x86_uv, runner=mock_runner, verify_sha=True)
        self.assertTrue(ok, err)
        self.assertIn(str(real_x86_uv.resolve()), _VERIFIED_BUNDLED_UV_PATHS)

        # 2. Subsequent call hits in-process cache
        ok2, err2 = verify_bundled_uv(real_x86_uv, runner=mock_runner, verify_sha=True)
        self.assertTrue(ok2, err2)

        # 3. Corrupted binary fails closed with SHA-256 mismatch
        corrupt_bin = self.data_dir / "fake_corrupt_bin" / "linux-x86_64" / "uv"
        corrupt_bin.parent.mkdir(parents=True, exist_ok=True)
        corrupt_bin.write_text("#!/bin/sh\necho corrupted\n", encoding="utf-8")
        corrupt_bin.chmod(0o755)

        ok_bad, err_bad = verify_bundled_uv(corrupt_bin, runner=mock_runner, verify_sha=True)
        self.assertFalse(ok_bad)
        self.assertIn("SHA-256 mismatch", err_bad)


if __name__ == "__main__":
    unittest.main()

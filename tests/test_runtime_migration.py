"""Tests for runtime migration and model-specific dependency drift detection and repair.

Verifies:
Case A: model installed = true, base runtime ready = true, dependency missing (e.g. addict)
        -> installed=true, runtime_ready=true, dependencies_ready=false, detector_ready=false, repairable=true
Case B: click repair -> ensure_model_runtime_dependencies() -> missing dependencies installed -> smoke pass -> ready=true
Case C: model weights already installed -> download_modelscope_model is NEVER CALLED during repair
Case D: intact torch runtime -> torch is never reinstalled and venv is never recreated
Case E: app restart not required -> detector status reflects ready=true within same process instance
"""

import json
import os
from pathlib import Path
import tempfile
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
    probe_model_runtime_dependencies,
    repair_model_runtime,
)
from privacy.chinese_ie import ChineseIEDetector
from privacy.detectors import GLiNERDetector
from privacy.device import DeviceManager
from privacy.model_catalog import SLOT_CHINESE_IE, SLOT_GENERAL_PII, get_model_descriptor
from privacy.registry import DetectorRegistry


class RuntimeMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        clear_dependency_probe_cache()

    def tearDown(self):
        clear_dependency_probe_cache()
        self.temp_dir.cleanup()

    def _create_mock_siamese_uie(self) -> Path:
        m_dir = self.data_dir / "models" / "siamese-uie"
        m_dir.mkdir(parents=True, exist_ok=True)
        (m_dir / "configuration.json").write_text("{}", encoding="utf-8")
        (m_dir / "config.json").write_text("{}", encoding="utf-8")
        (m_dir / "vocab.txt").write_text("dummy", encoding="utf-8")
        (m_dir / "pytorch_model.bin").write_text("dummy", encoding="utf-8")
        return m_dir

    def _create_mock_runtime(self, profile: str = "torch-cpu") -> Path:
        rt_dir = self.data_dir / "runtimes" / profile
        venv_bin = rt_dir / "venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        py_bin = venv_bin / "python"
        py_bin.write_text(
            '#!/bin/sh\n'
            'echo \'{"python_runtime_ready": true, "python_runtime_source": "uv-managed", "python_runtime_version": "3.12.9", "missing_python_capabilities": [], "framework_version": "2.6.0+cpu", "cuda_available": false}\'\n'
        )
        py_bin.chmod(0o755)
        manifest = {
            "schema_version": 3,
            "profile": profile,
            "created_at": 1741500000,
            "python_runtime_source": "managed",
            "python_runtime_version": "3.12.9",
            "python_interpreter": str(py_bin),
            "managed_python_path": "/fake/managed/python",
            "capabilities": [
                "lzma", "_lzma", "bz2", "_bz2", "ssl", "_ssl", "sqlite3", "_sqlite3",
                "ctypes", "_ctypes", "zlib", "hashlib", "json", "multiprocessing",
                "subprocess", "venv", "ensurepip"
            ],
            "capabilities_verified_at": 1741500000,
            "framework_version": "2.6.0+cpu",
        }
        (rt_dir / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return py_bin

    def test_case_a_dependency_drift_detection(self):
        """Case A: model installed, base runtime ready, but addict missing.
        Should report installed=True, base_runtime_ready=True, model_dependencies_ready=False,
        missing_dependencies=['addict'], model_ready=False, repairable=True.
        """
        self._create_mock_siamese_uie()
        self._create_mock_runtime("torch-cpu")

        def mock_runner(cmd, **kwargs):
            if "AIPRIVACY_DEPENDENCY_PROBE" in cmd[2]:
                return 0, json.dumps({"ok": True, "missing": ["addict"], "error": None}), ""
            return 0, "", ""

        det = ChineseIEDetector(self.data_dir, active_model_id="siamese-uie")

        with patch("privacy.chinese_ie.DEVICE_MANAGER.resolve_for_model") as mock_resolve, \
             patch("model_installer._default_dependency_runner", side_effect=mock_runner):
            mock_resolve.return_value = {
                "ready": True,
                "runtime_profile": "torch-cpu",
                "actual_device": "cpu",
            }

            status = det.status()
            self.assertTrue(status["installed"])
            self.assertTrue(status["base_runtime_ready"])
            self.assertFalse(status["model_dependencies_ready"])
            self.assertIn("addict", status["missing_dependencies"])
            self.assertFalse(status["model_ready"])
            self.assertTrue(status["repairable"])

    def test_case_b_repair_installs_dependencies_and_reaches_ready(self):
        """Case B: repair_model_runtime installs missing dependencies, passes smoke test,
        and transitions model to ready=True.
        """
        self._create_mock_siamese_uie()
        self._create_mock_runtime("torch-cpu")

        installed_packages = []
        is_repaired = [False]

        def mock_runner(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "pip" in cmd_str and "install" in cmd_str:
                for arg in cmd:
                    if isinstance(arg, str) and not arg.startswith("-") and not arg.startswith("/") and not arg.startswith("http"):
                        installed_packages.append(arg)
                is_repaired[0] = True
                return 0, "Successfully installed", ""
            if "AIPRIVACY_DEPENDENCY_PROBE" in cmd_str:
                specs = json.loads(cmd[3])
                if "addict" in specs:
                    missing = [] if is_repaired[0] else ["addict"]
                else:
                    missing = []
                return 0, json.dumps({"ok": True, "missing": missing, "error": None}), ""
            return 0, "", ""

        with patch("model_installer._default_dependency_runner", side_effect=mock_runner), \
             patch("model_installer.get_worker_client") as mock_gwc, \
             patch("model_installer.DEVICE_MANAGER.get_requested_device", return_value="cpu"), \
             patch("model_installer.DEVICE_MANAGER.probe_diagnostics", return_value={"hardware": {}}):
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (True, None)
            mock_gwc.return_value = mock_client

            ok, msg = repair_model_runtime(self.data_dir, "siamese-uie")
            self.assertTrue(ok)
            self.assertEqual(msg, "运行环境修复完成")
            self.assertIn("addict", installed_packages)
            mock_client.run_smoke_test.assert_called_once()

    def test_case_c_model_weights_never_downloaded_during_repair(self):
        """Case C: Model weights are already present; download_modelscope_model must NEVER be called."""
        self._create_mock_siamese_uie()
        self._create_mock_runtime("torch-cpu")

        def mock_runner(cmd, **kwargs):
            return 0, json.dumps({"ok": True, "missing": [], "error": None}), ""

        with patch("model_installer._default_dependency_runner", side_effect=mock_runner), \
             patch("model_installer.download_modelscope_model") as mock_download, \
             patch("model_installer.get_worker_client") as mock_gwc, \
             patch("model_installer.DEVICE_MANAGER.get_requested_device", return_value="cpu"), \
             patch("model_installer.DEVICE_MANAGER.probe_diagnostics", return_value={"hardware": {}}):
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (True, None)
            mock_gwc.return_value = mock_client

            ok, msg = repair_model_runtime(self.data_dir, "siamese-uie")
            self.assertTrue(ok)
            mock_download.assert_not_called()

    def test_case_d_torch_never_reinstalled_and_venv_not_rebuilt(self):
        """Case D: PyTorch is never reinstalled and venv is never recreated when runtime exists."""
        self._create_mock_siamese_uie()
        self._create_mock_runtime("torch-cpu")

        pip_install_calls = []
        is_repaired = [False]

        def mock_runner(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "pip" in cmd_str and "install" in cmd_str:
                pip_install_calls.append(cmd)
                is_repaired[0] = True
                return 0, "Installed", ""
            if "AIPRIVACY_DEPENDENCY_PROBE" in cmd_str:
                specs = json.loads(cmd[3])
                if "addict" in specs:
                    missing = [] if is_repaired[0] else ["addict"]
                else:
                    missing = []
                return 0, json.dumps({"ok": True, "missing": missing, "error": None}), ""
            return 0, "", ""

        with patch("model_installer._default_dependency_runner", side_effect=mock_runner), \
             patch("model_installer.install_isolated_runtime") as mock_install_runtime, \
             patch("model_installer.get_worker_client") as mock_gwc, \
             patch("model_installer.DEVICE_MANAGER.get_requested_device", return_value="cpu"), \
             patch("model_installer.DEVICE_MANAGER.probe_diagnostics", return_value={"hardware": {}}):
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (True, None)
            mock_gwc.return_value = mock_client

            ok, msg = repair_model_runtime(self.data_dir, "siamese-uie")
            self.assertTrue(ok)
            mock_install_runtime.assert_not_called()
            for call in pip_install_calls:
                call_str = " ".join(str(c) for c in call)
                self.assertNotIn("torch==", call_str)
                self.assertNotIn("torch>=", call_str)
                self.assertNotIn("torch<", call_str)

    def test_case_e_no_app_restart_required_transitions_ready(self):
        """Case E: Within the same python process, detector status updates to ready=True after repair."""
        self._create_mock_siamese_uie()
        self._create_mock_runtime("torch-cpu")

        is_repaired = [False]

        def mock_runner(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "pip" in cmd_str and "install" in cmd_str:
                is_repaired[0] = True
                return 0, "", ""
            if "AIPRIVACY_DEPENDENCY_PROBE" in cmd_str:
                specs = json.loads(cmd[3])
                if "addict" in specs:
                    missing = [] if is_repaired[0] else ["addict"]
                else:
                    missing = []
                return 0, json.dumps({"ok": True, "missing": missing, "error": None}), ""
            return 0, "", ""

        det = ChineseIEDetector(self.data_dir, active_model_id="siamese-uie")

        with patch("privacy.chinese_ie.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "runtime_profile": "torch-cpu", "actual_device": "cpu"}), \
             patch("model_installer._default_dependency_runner", side_effect=mock_runner), \
             patch("model_installer.DEVICE_MANAGER.get_requested_device", return_value="cpu"), \
             patch("model_installer.DEVICE_MANAGER.probe_diagnostics", return_value={"hardware": {}}), \
             patch("model_installer.get_worker_client") as mock_gwc:
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (True, None)
            mock_gwc.return_value = mock_client

            # Initial status: needs repair
            s_before = det.status()
            self.assertFalse(s_before["model_ready"])
            self.assertTrue(s_before["repairable"])

            # Execute repair
            ok, _ = repair_model_runtime(self.data_dir, "siamese-uie")
            self.assertTrue(ok)

            # In the same detector instance, next status check is immediately ready!
            s_after = det.status()
            self.assertTrue(s_after["model_ready"])
            self.assertFalse(s_after["repairable"])
            self.assertEqual(s_after["engine"], "siamese_uie")


class ServerRepairEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_start_repair_invokes_repair_action(self):
        from server import ModelLifecycleController
        controller = ModelLifecycleController(self.data_dir)

        with patch("subprocess.Popen") as mock_popen, \
             patch("model_installer.model_operation_lock"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.wait = MagicMock(side_effect=lambda: time.sleep(0.2))
            mock_popen.return_value = mock_proc

            started = controller.start_repair("siamese-uie")
            self.assertTrue(started)
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            self.assertIn("repair", cmd)
            self.assertIn("siamese-uie", cmd)

            # Second call while running should return False
            started_second = controller.start_repair("siamese-uie")
            self.assertFalse(started_second)


if __name__ == "__main__":
    unittest.main()

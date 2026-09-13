"""v0.6.5 regression tests: base runtime contract + pre-load security gate.

Covers the three confirmed v0.6.4 gaps:
1. Historical runtimes (e.g. transformers 5.16.1) skip reinstall because the
   torch-only probe verifies; the Base Runtime Contract must migrate them
   in place without touching torch or the venv.
2. Models installed by older versions still carry `allow_remote` in
   configuration.json: the security gate must neutralize them before ANY
   worker load, not only during install.
3. Production and the benchmark harness must share one GLiNER threshold.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

import model_installer as mi  # noqa: E402
from privacy.model_security import ModelSecurityGateError, gate_model_security  # noqa: E402
from privacy.model_catalog import get_model_descriptor  # noqa: E402
from privacy.runtime_manager import RuntimeManager  # noqa: E402


class BaseRuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.manager = RuntimeManager(self.data_dir)
        self.profile = "torch-cuda"
        venv_bin = self.manager.venv_dir(self.profile) / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        self.python_bin = venv_bin / "python"
        self.python_bin.write_text("#!/bin/sh\n")
        self.python_bin.chmod(0o755)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_transformers_5_runtime_is_migrated_without_torch_reinstall(self):
        """verified runtime + transformers 5.16.1 -> repair pin only."""
        runner = RecordingRunner(missing_first_probe=["transformers>=4.51,<5"])
        result = mi.ensure_base_runtime_contract(
            self.data_dir, self.profile, command_runner=runner
        )

        self.assertTrue(result["satisfied"])
        self.assertEqual(result["missing_initial"], ["transformers>=4.51,<5"])
        self.assertEqual(len(runner.install_cmds), 1)
        install_cmd = runner.install_cmds[0]
        self.assertIn("transformers>=4.51,<5", install_cmd)
        torch_args = [a for a in install_cmd if str(a).strip() == "torch"]
        self.assertEqual(torch_args, [], "Base Contract 迁移不得重装 PyTorch")
        # Probe must target the isolated interpreter
        self.assertEqual(runner.probe_cmds[0][0], str(self.python_bin))

    def test_satisfied_runtime_skips_repair(self):
        runner = RecordingRunner(missing_first_probe=[])
        result = mi.ensure_base_runtime_contract(
            self.data_dir, self.profile, command_runner=runner
        )
        self.assertTrue(result["satisfied"])
        self.assertEqual(runner.install_cmds, [])

    def test_missing_torch_refuses_in_place_repair(self):
        runner = RecordingRunner(missing_first_probe=["torch"])
        with self.assertRaises(RuntimeError) as ctx:
            mi.ensure_base_runtime_contract(
                self.data_dir, self.profile, command_runner=runner
            )
        self.assertIn("PyTorch", str(ctx.exception))
        self.assertEqual(runner.install_cmds, [], "缺 torch 不得走增量修复")

    def test_verified_skip_path_invokes_contract(self):
        """The 'runtime verified, skip install' path must still run the contract."""
        manager = MagicMock()
        manager.profile_dir.return_value = self.manager.profile_dir(self.profile)
        manager.venv_dir.return_value = self.manager.venv_dir(self.profile)
        manager.probe_profile.return_value = {"verified": True}
        (self.manager.profile_dir(self.profile) / "installed.json").write_text("{}")

        with patch.object(mi, "get_runtime_manager", return_value=manager), \
             patch.object(mi, "ensure_base_runtime_contract") as mock_contract:
            mi.install_isolated_runtime(self.data_dir, self.profile)
            mock_contract.assert_called_once_with(self.data_dir, self.profile)

    def test_base_contract_and_model_contract_stay_separate(self):
        """Base requirements are shared; model-specific deps live in the catalog."""
        self.assertIn("transformers>=4.51,<5", mi.BASE_RUNTIME_REQUIREMENTS)
        self.assertIn("torch", mi.BASE_RUNTIME_REQUIREMENTS)
        self.assertNotIn("addict", mi.BASE_RUNTIME_REQUIREMENTS)
        siamese = get_model_descriptor("siamese-uie").runtime_dependencies
        self.assertIn("addict", siamese)
        for dep in siamese:
            self.assertNotIn(dep, mi.BASE_RUNTIME_REQUIREMENTS)


class RecordingRunner:
    """Answers dependency probes and records install invocations."""

    def __init__(self, missing_first_probe):
        self.missing_first_probe = list(missing_first_probe)
        self.install_cmds = []
        self.probe_cmds = []

    def __call__(self, cmd, cwd=None, env=None, timeout=15):
        joined = " ".join(str(part) for part in cmd)
        if mi.DEPENDENCY_PROBE_MARKER in joined:
            self.probe_cmds.append(cmd)
            missing, self.missing_first_probe = self.missing_first_probe, []
            return 0, json.dumps({"ok": True, "missing": missing, "error": None}), ""
        if "install" in cmd[:6]:
            self.install_cmds.append(cmd)
            return 0, "", ""
        return 0, "", ""


class PreLoadSecurityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name) / "siamese-uie"
        self.model_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_config(self, payload):
        (self.model_dir / "configuration.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_legacy_model_with_allow_remote_is_sanitized(self):
        """v0.6.3-era installed model: allow_remote removed on first load."""
        self._write_config({"task": "siamese-uie", "allow_remote": True})
        result = gate_model_security(self.model_dir)
        self.assertEqual(result["status"], "sanitized")
        on_disk = json.loads((self.model_dir / "configuration.json").read_text(encoding="utf-8"))
        self.assertNotIn("allow_remote", on_disk)
        self.assertEqual(on_disk["task"], "siamese-uie")

    def test_plugins_declaration_is_sanitized_too(self):
        self._write_config({"task": "x", "plugins": ["some-repo"]})
        result = gate_model_security(self.model_dir)
        self.assertEqual(result["status"], "sanitized")
        on_disk = json.loads((self.model_dir / "configuration.json").read_text(encoding="utf-8"))
        self.assertNotIn("plugins", on_disk)

    def test_clean_model_is_untouched(self):
        self._write_config({"task": "siamese-uie", "framework": "pytorch"})
        result = gate_model_security(self.model_dir)
        self.assertEqual(result["status"], "safe")

    def test_missing_configuration_is_safe(self):
        result = gate_model_security(self.model_dir)
        self.assertEqual(result["status"], "safe")

    def test_corrupt_remote_code_config_is_rejected(self):
        (self.model_dir / "configuration.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(ModelSecurityGateError):
            gate_model_security(self.model_dir)

    def test_gliner_detect_gates_model_directory(self):
        """The detect path must run the gate before issuing worker queries."""
        from privacy.detectors import GLiNERDetector

        data_dir = Path(self.temp_dir.name) / "data"
        model_dir = data_dir / "models" / "gliner-pii-edge"
        model_dir.mkdir(parents=True)
        (model_dir / "gliner_config.json").write_text("{}")
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_text("dummy")
        (model_dir / "tokenizer.json").write_text("{}")
        (model_dir / "configuration.json").write_text(
            json.dumps({"allow_remote": True}), encoding="utf-8"
        )

        detector = GLiNERDetector(data_dir)

        from privacy.device import DEVICE_MANAGER

        fake_client = MagicMock()
        fake_worker = MagicMock()
        fake_worker.query.return_value = {"ok": True, "entities": []}
        fake_client.get_worker.return_value = fake_worker
        fake_client.get_timeout_for_model.return_value = (10, 10)
        fake_client.cuda_execution_session.return_value.__enter__ = MagicMock()
        fake_client.cuda_execution_session.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(DEVICE_MANAGER, "resolve_for_model",
                          return_value={"ready": True, "runtime_profile": "torch-cpu", "actual_device": "cpu"}), \
             patch("privacy.detectors.get_worker_client", return_value=fake_client):
            detector.detect("hello world")

        on_disk = json.loads((model_dir / "configuration.json").read_text(encoding="utf-8"))
        self.assertNotIn("allow_remote", on_disk, "worker 加载前必须净化远程代码声明")
        self.assertTrue(fake_worker.query.called)

    def test_install_sanitize_delegates_to_gate(self):
        self._write_config({"task": "siamese-uie", "allow_remote": True})
        changed, detail = mi.sanitize_model_config(self.model_dir)
        self.assertTrue(changed)
        self.assertIn("allow_remote", detail)
        on_disk = json.loads((self.model_dir / "configuration.json").read_text(encoding="utf-8"))
        self.assertNotIn("allow_remote", on_disk)


class ThresholdSingleSourceTests(unittest.TestCase):
    def test_production_and_harness_share_one_threshold(self):
        from privacy.detectors import GLiNERDetector

        self.assertEqual(GLiNERDetector.GLINER_DEFAULT_THRESHOLD, 0.50)
        harness_src = (PROJECT_DIR / "scripts" / "benchmark_models.py").read_text(encoding="utf-8")
        self.assertIn("GLiNERDetector.GLINER_DEFAULT_THRESHOLD", harness_src)
        self.assertNotIn("threshold=0.4", harness_src)
        production_src = (SERVER_DIR / "privacy" / "detectors.py").read_text(encoding="utf-8")
        self.assertNotIn('"threshold": 0.4', production_src)
        self.assertNotIn('"threshold": 0.55,', production_src)


if __name__ == "__main__":
    unittest.main()

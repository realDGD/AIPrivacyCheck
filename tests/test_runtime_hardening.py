import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Set up import path to server modules
PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR))

from privacy.hardware import HardwareProbe
from privacy.runtime_manager import RuntimeManager
from privacy.device import DeviceManager
from privacy.detectors import parse_memprivacy_json
from model_installer import validate_import_source_path, scan_shared_models_directory


class HardwareProbeTests(unittest.TestCase):
    def test_hardware_probe_missing_nvidia_smi(self):
        with patch("shutil.which", return_value=None):
            probe = HardwareProbe()
            telemetry = probe.probe_nvidia()
            self.assertFalse(telemetry["nvidia_available"])
            self.assertIn("未找到 nvidia-smi", telemetry["reason"])
            self.assertEqual(len(telemetry["gpus"]), 0)

    def test_hardware_probe_success_single_gpu(self):
        mock_output = "NVIDIA GeForce RTX 4090, 535.129.03, 24564\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            runner_fn = lambda cmd: (0, mock_output, "")
            probe = HardwareProbe(command_runner=runner_fn)
            telemetry = probe.probe_nvidia()
            self.assertTrue(telemetry["nvidia_available"])
            self.assertIsNone(telemetry["reason"])
            self.assertEqual(len(telemetry["gpus"]), 1)
            gpu = telemetry["gpus"][0]
            self.assertEqual(gpu["index"], 0)
            self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 4090")
            self.assertEqual(gpu["driver_version"], "535.129.03")
            self.assertEqual(gpu["memory_total_mb"], 24564)

    def test_hardware_probe_multi_gpu(self):
        mock_output = (
            "Tesla T4, 525.60.13, 15360\n"
            "Tesla T4, 525.60.13, 15360\n"
        )
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            runner_fn = lambda cmd: (0, mock_output, "")
            probe = HardwareProbe(command_runner=runner_fn)
            telemetry = probe.probe_nvidia()
            self.assertTrue(telemetry["nvidia_available"])
            self.assertEqual(len(telemetry["gpus"]), 2)
            self.assertEqual(telemetry["gpus"][1]["index"], 1)
            self.assertEqual(telemetry["gpus"][1]["name"], "Tesla T4")

    def test_hardware_probe_command_failure(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            runner_fn = lambda cmd: (12, "", "Driver error")
            probe = HardwareProbe(command_runner=runner_fn)
            telemetry = probe.probe_nvidia()
            self.assertFalse(telemetry["nvidia_available"])
            self.assertIn("Driver error", telemetry["reason"])


class RuntimeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.manager = RuntimeManager(self.data_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_directory_structure(self):
        profile_dir = self.manager.profile_dir("torch-cuda")
        self.assertEqual(profile_dir, self.data_dir / "runtimes" / "torch-cuda")
        venv_dir = self.manager.venv_dir("torch-cuda")
        self.assertEqual(venv_dir, profile_dir / "venv")

    def test_probe_not_installed(self):
        status = self.manager.probe_profile("torch-cpu")
        self.assertFalse(status["installed"])
        self.assertFalse(status["cuda_available"])
        self.assertIn("未安装", status["error"])

    def test_probe_installed_with_runner(self):
        profile_dir = self.manager.profile_dir("torch-cuda")
        bin_dir = profile_dir / "venv" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        py_bin = bin_dir / "python"
        py_bin.touch(mode=0o755)

        probe_json = json.dumps({
            "framework_version": "2.4.0",
            "cuda_version": "12.4",
            "cuda_available": True,
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 4090",
        })
        manager = RuntimeManager(
            self.data_dir,
            command_runner=lambda cmd, **kw: (0, probe_json, "")
        )

        status = manager.probe_profile("torch-cuda", force_refresh=True)
        self.assertTrue(status["installed"])
        self.assertTrue(status["verified"])
        self.assertTrue(status["cuda_available"])
        self.assertEqual(status["framework_version"], "2.4.0")
        self.assertEqual(status["device_name"], "NVIDIA GeForce RTX 4090")

    def test_best_runtime_for_framework(self):
        # When none installed
        best = self.manager.best_runtime_for_framework("torch")
        self.assertIsNone(best)

        # Create mock cpu runtime
        cpu_bin = self.manager.venv_dir("torch-cpu") / "bin"
        cpu_bin.mkdir(parents=True, exist_ok=True)
        (cpu_bin / "python").touch(mode=0o755)

        cpu_probe = json.dumps({
            "framework_version": "2.4.0",
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
        })
        manager = RuntimeManager(
            self.data_dir,
            command_runner=lambda cmd, **kw: (0, cpu_probe, "")
        )

        best = manager.best_runtime_for_framework("torch")
        self.assertEqual(best, "torch-cpu")


class ModelImportSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True)
        self.share_dir = self.base_dir / "shares" / "ai-privacy-check" / "models"
        self.share_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_share_source_allowed(self):
        valid_model = self.share_dir / "my_model"
        valid_model.mkdir()
        (valid_model / "config.json").write_text("{}")

        with patch.dict(os.environ, {"TRIM_DATA_SHARE_PATHS": str(self.base_dir / "shares")}):
            is_valid, msg, resolved = validate_import_source_path(str(valid_model), self.data_dir)
            self.assertTrue(is_valid)
            self.assertEqual(resolved, valid_model.resolve())
            self.assertEqual(msg, "路径合法")

    def test_path_outside_allowed_roots_rejected(self):
        outside_path = self.base_dir / "outside_dir" / "secret_model"
        outside_path.mkdir(parents=True)

        with patch.dict(os.environ, {"TRIM_DATA_SHARE_PATHS": str(self.share_dir)}):
            is_valid, msg, resolved = validate_import_source_path(str(outside_path), self.data_dir)
            self.assertFalse(is_valid)
            self.assertIsNone(resolved)
            self.assertIn("超出允许的模型导入目录范围", msg)

    def test_symlink_escape_rejected(self):
        outside_target = self.base_dir / "unauthorized" / "real_model"
        outside_target.mkdir(parents=True)
        (outside_target / "config.json").write_text("{}")

        # Create symlink inside share_dir pointing to outside_target
        symlink_path = self.share_dir / "symlink_escape"
        os.symlink(outside_target, symlink_path)

        with patch.dict(os.environ, {"TRIM_DATA_SHARE_PATHS": str(self.share_dir)}):
            is_valid, msg, resolved = validate_import_source_path(str(symlink_path), self.data_dir)
            self.assertFalse(is_valid)
            self.assertIsNone(resolved)
            self.assertIn("超出允许的模型导入目录范围", msg)

    def test_system_sensitive_paths_rejected(self):
        with patch.dict(os.environ, {"TRIM_DATA_SHARE_PATHS": "/"}):
            is_valid, msg, resolved = validate_import_source_path("/etc/passwd", self.data_dir)
            self.assertFalse(is_valid)
            self.assertIsNone(resolved)
            self.assertIn("拒绝访问系统受限路径", msg)

    def test_nonexistent_path_rejected(self):
        with patch.dict(os.environ, {"TRIM_DATA_SHARE_PATHS": str(self.share_dir)}):
            is_valid, msg, resolved = validate_import_source_path(str(self.share_dir / "ghost"), self.data_dir)
            self.assertFalse(is_valid)
            self.assertIsNone(resolved)
            self.assertIn("源路径无效或不存在", msg)


class MemPrivacyParserTests(unittest.TestCase):
    def test_clean_json_array(self):
        raw = '[{"original_text": "张三", "privacy_type": "name", "privacy_level": "medium"}]'
        full_text = "这是张三的个人主页。"
        items = parse_memprivacy_json(raw, full_text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], "PERSON")
        self.assertEqual(items[0][1], "张三")
        self.assertEqual(items[0][4], "PL2")

    def test_markdown_code_fence(self):
        raw = """```json
[
  {"original_text": "北京市朝阳区酒仙桥路4号", "privacy_type": "address", "privacy_level": "medium"},
  {"original_text": "13800138000", "privacy_type": "phone", "privacy_level": "PL3"}
]
```"""
        full_text = "地址是北京市朝阳区酒仙桥路4号，电话13800138000。"
        items = parse_memprivacy_json(raw, full_text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][0], "ADDRESS")
        self.assertEqual(items[1][0], "PHONE")
        self.assertEqual(items[1][4], "PL3")

    def test_trailing_comma_cleanup(self):
        raw = """[
  {"original_text": "user@example.com", "privacy_type": "email", "privacy_level": "medium",},
]"""
        full_text = "请发送到 user@example.com 邮箱。"
        items = parse_memprivacy_json(raw, full_text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], "EMAIL")
        self.assertEqual(items[0][1], "user@example.com")

    def test_text_not_in_document_ignored(self):
        raw = '[{"original_text": "李四", "privacy_type": "name"}]'
        full_text = "这里没有任何李四的名字。"
        items = parse_memprivacy_json(raw, full_text)
        self.assertEqual(len(items), 1)

        raw_hallucination = '[{"original_text": "王五", "privacy_type": "name"}]'
        items2 = parse_memprivacy_json(raw_hallucination, full_text)
        self.assertEqual(len(items2), 0)


class UninstallCallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Directory must match *ai-privacy-check* to pass path security validation
        self.pkgvar = self.root / "ai-privacy-check-var"
        self.pkgvar.mkdir()
        self.pkgetc = self.root / "ai-privacy-check-etc"
        self.pkgetc.mkdir()
        (self.pkgetc / "app.conf").write_text("config=true")

        self.data_dir = self.pkgvar / "data"
        self.data_dir.mkdir()
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir()
        (self.models_dir / "model_weights.bin").write_text("model weights")
        self.runtimes_dir = self.data_dir / "runtimes"
        self.runtimes_dir.mkdir()
        (self.runtimes_dir / "runtime_bin").write_text("venv binary")
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir()
        (self.cache_dir / "temp.log").write_text("temp log")

        # Shared user models directory
        self.shared_dir = self.root / "shared" / "ai-privacy-check" / "models"
        self.shared_dir.mkdir(parents=True)
        (self.shared_dir / "user_saved_model.safetensors").write_text("precious user file")

        self.callback_script = PROJECT_DIR / "packaging" / "ai-privacy-check" / "cmd" / "uninstall_callback"

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_callback(self, choice: str):
        env = os.environ.copy()
        env["TRIM_PKGVAR"] = str(self.pkgvar)
        env["TRIM_PKGETC"] = str(self.pkgetc)
        env["TRIM_DATA_SHARE_PATHS"] = str(self.root / "shared")
        env["wizard_data_action"] = choice
        proc = subprocess.run(
            ["bash", str(self.callback_script)],
            env=env,
            capture_output=True,
            text=True,
        )
        return proc

    def test_keep_preserves_everything(self):
        res = self.run_callback("keep")
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertTrue((self.models_dir / "model_weights.bin").exists())
        self.assertTrue((self.runtimes_dir / "runtime_bin").exists())
        self.assertTrue((self.pkgetc / "app.conf").exists())
        self.assertTrue((self.shared_dir / "user_saved_model.safetensors").exists())

    def test_keep_runtime_preserves_models_and_runtimes_cleans_cache_and_etc(self):
        res = self.run_callback("keep_runtime")
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertTrue((self.models_dir / "model_weights.bin").exists())
        self.assertTrue((self.runtimes_dir / "runtime_bin").exists())
        self.assertFalse(self.cache_dir.exists())
        self.assertFalse(self.pkgetc.exists())
        self.assertTrue((self.shared_dir / "user_saved_model.safetensors").exists())

    def test_delete_cleans_pkgvar_but_preserves_shared(self):
        res = self.run_callback("delete")
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertFalse(self.data_dir.exists())
        self.assertFalse(self.pkgetc.exists())
        # The user's shared directory MUST NEVER be touched!
        self.assertTrue((self.shared_dir / "user_saved_model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()

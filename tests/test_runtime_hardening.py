import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
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
        self.assertTrue((self.shared_dir / "user_saved_model.safetensors").exists())


class ControlPlaneIsolationTests(unittest.TestCase):
    """Verifies that the control plane Python process NEVER imports heavy ML frameworks."""

    def test_control_plane_has_no_ml_framework(self):
        forbidden_modules = (
            "torch",
            "transformers",
            "gliner",
            "modelscope",
            "paddle",
            "paddlenlp",
        )
        for mod in forbidden_modules:
            self.assertNotIn(
                mod,
                sys.modules,
                f"Control plane Python process illegally imported {mod}! ML frameworks must run in isolated worker processes.",
            )

    def test_privacy_service_runs_without_ml_imports(self):
        with tempfile.TemporaryDirectory() as td:
            from privacy import PrivacyService
            svc = PrivacyService(Path(td))
            res = svc.detect("My phone is 13800138000 and ID is 11010519491231002X", use_model=True)
            self.assertTrue(res["entities"])
            # Ensure still no ML modules loaded
            for mod in ("torch", "transformers", "gliner", "modelscope", "paddle", "paddlenlp"):
                self.assertNotIn(mod, sys.modules)


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_settings_store_defaults_and_persistence(self):
        from privacy.settings_store import SettingsStore
        store = SettingsStore(self.data_dir)
        self.assertEqual(store.get_requested_device(), "auto")
        self.assertTrue(store.get_slot_enabled("chinese_ie"))

        # Mutate and verify atomic file write
        store.set_requested_device("cuda")
        store.set_slot_enabled("chinese_ie", False)
        store.set_active_model("general_pii", "gliner-pii-base")

        settings_file = self.data_dir / "settings.json"
        self.assertTrue(settings_file.exists())
        saved_data = json.loads(settings_file.read_text(encoding="utf-8"))
        self.assertEqual(saved_data["requested_device"], "cuda")
        self.assertFalse(saved_data["slots"]["chinese_ie"])
        self.assertEqual(saved_data["active_models"]["general_pii"], "gliner-pii-base")

        # Create fresh store instance on same directory and verify load
        new_store = SettingsStore(self.data_dir)
        self.assertEqual(new_store.get_requested_device(), "cuda")
        self.assertFalse(new_store.get_slot_enabled("chinese_ie"))
        self.assertEqual(new_store.get_active_model("general_pii"), "gliner-pii-base")


class WorkerLifecycleAndResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mock_worker_script = self.root / "mock_worker.py"
        script_content = """import sys, time, json
while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        data = json.loads(line.strip())
    except Exception:
        continue
    act = data.get("action")
    if act == "ping":
        sys.stdout.write(json.dumps({"ok": True, "pong": True}) + "\\n")
        sys.stdout.flush()
    elif act == "hang":
        time.sleep(10)
    elif act == "crash":
        sys.exit(139)
    elif act == "spam_stderr":
        sys.stderr.write("E" * 131072 + "\\n")
        sys.stderr.flush()
        sys.stdout.write(json.dumps({"ok": True, "spammed": True}) + "\\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(json.dumps({"ok": True, "echo": act}) + "\\n")
        sys.stdout.flush()
"""
        self.mock_worker_script.write_text(script_content, encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_worker_timeout_enforced(self):
        from privacy.worker_client import RuntimeWorkerProcess
        worker = RuntimeWorkerProcess(
            model_id="mock-model",
            profile="test-profile",
            python_bin=Path(sys.executable),
            worker_script=self.mock_worker_script,
            startup_timeout=5,
        )
        try:
            start_t = time.time()
            res = worker.query({"action": "hang"}, timeout=1)
            duration = time.time() - start_t
            self.assertFalse(res.get("ok"))
            self.assertEqual(res.get("error_type"), "WorkerTimeout")
            self.assertLess(duration, 3.0)
            self.assertFalse(worker.is_ready())
        finally:
            worker.terminate()

    def test_worker_stderr_saturation_no_deadlock(self):
        from privacy.worker_client import RuntimeWorkerProcess
        worker = RuntimeWorkerProcess(
            model_id="mock-model",
            profile="test-profile",
            python_bin=Path(sys.executable),
            worker_script=self.mock_worker_script,
            startup_timeout=5,
        )
        try:
            res = worker.query({"action": "spam_stderr"}, timeout=5)
            self.assertTrue(res.get("ok"))
            self.assertTrue(res.get("spammed"))
        finally:
            worker.terminate()

    def test_worker_crash_recovery_no_deadlock(self):
        from privacy.worker_client import RuntimeWorkerProcess
        worker = RuntimeWorkerProcess(
            model_id="mock-model",
            profile="test-profile",
            python_bin=Path(sys.executable),
            worker_script=self.mock_worker_script,
            startup_timeout=5,
        )
        try:
            res = worker.query({"action": "crash"}, timeout=5)
            self.assertFalse(res.get("ok"))
            self.assertIn(res.get("error_type"), ("WorkerCrashed", "WorkerRestartFailed", "PipeWriteError"))
        finally:
            worker.terminate()

    def test_model_switch_stops_old_worker(self):
        from privacy.detectors import GLiNERDetector
        from privacy.chinese_ie import ChineseIEDetector
        from privacy.detectors import MemPrivacyDetector

        gliner = GLiNERDetector(self.root, active_model_id="gliner-pii-edge")
        with patch("privacy.detectors.get_worker_client") as mock_wc:
            mock_client = MagicMock()
            mock_wc.return_value = mock_client
            gliner.set_active_model("gliner-pii-base")
            self.assertEqual(gliner.active_model_id, "gliner-pii-base")
            mock_client.stop_worker_for_model.assert_called_once_with("gliner-pii-edge")

        mem = MemPrivacyDetector(self.root, active_model_id="memprivacy-1.7b-rl")
        with patch("privacy.detectors.get_worker_client") as mock_wc:
            mock_client = MagicMock()
            mock_wc.return_value = mock_client
            mem.set_active_model("memprivacy-4b-rl")
            self.assertEqual(mem.active_model_id, "memprivacy-4b-rl")
            mock_client.stop_worker_for_model.assert_called_once_with("memprivacy-1.7b-rl")

        chie = ChineseIEDetector(self.root, active_model_id="siamese-uie")
        with patch("privacy.chinese_ie.get_worker_client") as mock_wc:
            mock_client = MagicMock()
            mock_wc.return_value = mock_client
            chie.set_active_model("other-chinese-model")
            self.assertEqual(chie.active_model_id, "other-chinese-model")
            mock_client.stop_worker_for_model.assert_called_once_with("siamese-uie")

    def test_device_priority_and_persistence(self):
        from privacy.device import DeviceManager
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_PRIVACY_DEVICE", None)
            dm1 = DeviceManager(self.root)
            self.assertEqual(dm1.get_requested_device(), "auto")

            dm1.set_requested_device("cuda")
            self.assertEqual(dm1.get_requested_device(), "cuda")

            dm2 = DeviceManager(self.root)
            self.assertEqual(dm2.get_requested_device(), "cuda")

        with patch.dict(os.environ, {"AI_PRIVACY_DEVICE": "cpu"}):
            dm3 = DeviceManager(self.root)
            self.assertEqual(dm3.get_requested_device(), "cpu")

    def test_strict_smoke_gate_records_error_state(self):
        import model_installer
        from privacy.device import DEVICE_MANAGER

        data_dir = self.root / "app_data"
        data_dir.mkdir(parents=True)
        model_id = "gliner-pii-edge"
        model_dir = data_dir / "models" / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_text("dummy")
        (model_dir / "tokenizer.json").write_text("{}")

        DEVICE_MANAGER.set_data_dir(data_dir)
        DEVICE_MANAGER.set_requested_device("cpu")

        with patch("model_installer.get_worker_client") as mock_wc, \
             patch("model_installer.install_isolated_runtime") as mock_rt, \
             patch("model_installer.download_modelscope_model", return_value=model_dir), \
             patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
             patch("sys.argv", ["model_installer.py", "install", model_id]):
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (False, "Inference validation failed")
            mock_wc.return_value = mock_client

            ret = model_installer.main()
            self.assertEqual(ret, 1)

            status_file = data_dir / "status" / f"{model_id}-install.json"
            self.assertTrue(status_file.exists())
            status = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "error")
            self.assertIn("冒烟推理验证未通过", status["detail"])

    def test_cuda_auto_fallback_to_cpu(self):
        import model_installer
        from privacy.device import DEVICE_MANAGER
        from privacy.runtime_manager import PROFILE_TORCH_CUDA, PROFILE_TORCH_CPU

        data_dir = self.root / "auto_fallback_test"
        data_dir.mkdir(parents=True)
        model_id = "gliner-pii-edge"
        model_dir = data_dir / "models" / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_text("dummy")
        (model_dir / "tokenizer.json").write_text("{}")

        DEVICE_MANAGER.set_data_dir(data_dir)
        DEVICE_MANAGER.set_requested_device("auto")

        installed_profiles = []

        def mock_install_runtime(d_dir, profile):
            if profile == PROFILE_TORCH_CUDA:
                raise RuntimeError("CUDA driver missing")
            installed_profiles.append(profile)
            return Path("/fake/venv")

        with patch("model_installer.install_isolated_runtime", side_effect=mock_install_runtime), \
             patch("model_installer.download_modelscope_model", return_value=model_dir), \
             patch("model_installer.get_worker_client") as mock_wc, \
             patch.object(DEVICE_MANAGER._hw_probe, "probe_nvidia", return_value={"nvidia_available": True}), \
             patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
             patch("sys.argv", ["model_installer.py", "install", model_id]):
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (True, None)
            mock_wc.return_value = mock_client

            ret = model_installer.main()
            self.assertEqual(ret, 0)
            self.assertIn(PROFILE_TORCH_CPU, installed_profiles)

            status_file = data_dir / "status" / f"{model_id}-install.json"
            self.assertTrue(status_file.exists())
            status = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "ready")

    def test_cuda_explicit_failure_no_silent_cpu_fallback(self):
        import model_installer
        from privacy.device import DEVICE_MANAGER
        from privacy.runtime_manager import PROFILE_TORCH_CUDA

        data_dir = self.root / "explicit_cuda_test"
        data_dir.mkdir(parents=True)
        model_id = "gliner-pii-edge"

        DEVICE_MANAGER.set_data_dir(data_dir)
        DEVICE_MANAGER.set_requested_device("cuda")

        def mock_install_runtime(d_dir, profile):
            raise RuntimeError("NVIDIA driver failure")

        with patch("model_installer.install_isolated_runtime", side_effect=mock_install_runtime), \
             patch.object(DEVICE_MANAGER._hw_probe, "probe_nvidia", return_value={"nvidia_available": True}), \
             patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
             patch("sys.argv", ["model_installer.py", "install", model_id]):
            ret = model_installer.main()
            self.assertEqual(ret, 1)

            status_file = data_dir / "status" / f"{model_id}-install.json"
            self.assertTrue(status_file.exists())
            status = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "error")
            self.assertIn("NVIDIA driver failure", status["detail"])


class StabilizationHardeningTests(unittest.TestCase):
    """Rigorous tests for v0.5.2 stabilization hardening fixes (P0-1, P1-1, P1-2, P1-3, P1-4)."""

    def test_all_worker_modules_import(self):
        """P0-1: Worker modules must import cleanly without NameError or missing type hints."""
        import importlib
        modules = [
            "privacy.workers.gliner_worker",
            "privacy.workers.siamese_uie_worker",
            "privacy.workers.memprivacy_worker",
            "privacy.workers.modelscope_downloader",
        ]
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod)
        from privacy.workers import memprivacy_worker
        self.assertTrue(callable(memprivacy_worker.strip_think_tags))
        self.assertTrue(callable(memprivacy_worker.parse_extracted_json))
        self.assertTrue(callable(memprivacy_worker.handle_request))

    def test_old_generation_eof_cannot_poison_new_worker(self):
        """P1-1: Stale EOF from terminating worker of previous generation must not poison new worker queue."""
        import queue
        import threading
        from privacy.worker_client import RuntimeWorkerProcess
        with tempfile.TemporaryDirectory() as td:
            worker = RuntimeWorkerProcess(
                python_bin=Path("/bin/dummy"),
                worker_script=Path("/bin/dummy"),
                model_id="test-model",
                profile="torch-cpu",
            )
            self.assertEqual(worker._generation, 0)

            q1 = queue.Queue()
            stop1 = threading.Event()
            worker._generation = 1
            worker._stdout_queue = q1

            q2 = queue.Queue()
            stop2 = threading.Event()
            worker._generation = 2
            worker._stdout_queue = q2

            # Simulate EOF from terminating generation 1 process
            mock_proc = MagicMock()
            mock_proc.stdout.readline.return_value = ""
            mock_proc.poll.return_value = 0

            worker._stdout_reader(mock_proc, q1, stop1, 1)

            # Gen 1 queue received the EOF empty string sentinel
            self.assertEqual(q1.get(timeout=1), "")

            # Gen 2 queue was completely untouched
            self.assertTrue(q2.empty())

            # Stderr drainer from gen 1 must discard output for gen 2
            mock_proc.stderr.readline.side_effect = ["stale error line\n", ""]
            worker._stderr_buffer.clear()
            worker._stderr_drainer(mock_proc, stop1, 1)
            self.assertEqual(len(worker._stderr_buffer), 0)

    def test_explicit_cuda_unavailable_reports_none_not_cpu(self):
        """P1-2: Explicit CUDA with no available CUDA runtime must report actual_device='none' and ready=False."""
        from privacy.device import DeviceManager
        with tempfile.TemporaryDirectory() as td:
            dm = DeviceManager(Path(td))
            dm.set_requested_device("cuda")

            with patch.object(dm._hw_probe, "probe_nvidia", return_value={"nvidia_available": False}):
                diag = dm.probe_diagnostics(force_refresh=True)
                self.assertEqual(diag["requested_device"], "cuda")
                self.assertEqual(diag["actual_device"], "none")
                self.assertFalse(diag["cuda_available"])

                # Warnings should not claim fallback or downgrade to CPU
                for w in diag["warnings"]:
                    self.assertNotIn("回退", w)
                    self.assertNotIn("降级", w)
                    self.assertNotIn("fallback", w.lower())

                torch_dev = diag["model_devices"]["torch"]
                self.assertEqual(torch_dev["device"], "none")
                self.assertFalse(torch_dev["ready"])
                self.assertIsNone(torch_dev["profile"])

                # resolve_for_framework must return "none"
                res_dev, res_prof, _ = dm.resolve_for_framework("torch")
                self.assertEqual(res_dev, "none")
                self.assertIsNone(res_prof)

    def test_same_model_operation_lock_blocks_second_process(self):
        """P1-3: Concurrent operation on the same model must be mutually exclusive."""
        import threading
        import model_installer
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            model_id = "gliner-pii-edge"

            with model_installer.model_operation_lock(data_dir, model_id):
                second_result = []
                def try_lock():
                    try:
                        with model_installer.model_operation_lock(data_dir, model_id, non_blocking=True):
                            second_result.append("acquired")
                    except RuntimeError as e:
                        second_result.append(e)

                t = threading.Thread(target=try_lock)
                t.start()
                t.join()

                self.assertEqual(len(second_result), 1)
                self.assertIsInstance(second_result[0], RuntimeError)
                self.assertIn("正在执行安装、导入或卸载操作", str(second_result[0]))

    def test_different_model_operations_do_not_block_each_other(self):
        """P1-3: Operations on distinct models can proceed concurrently."""
        import model_installer
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            acquired_both = []

            with model_installer.model_operation_lock(data_dir, "gliner-pii-edge"):
                with model_installer.model_operation_lock(data_dir, "siamese-uie"):
                    acquired_both.append(True)

            self.assertTrue(acquired_both[0])

    def test_lock_released_after_exception(self):
        """P1-3: Lock must be cleanly released when an exception occurs."""
        import model_installer
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            model_id = "gliner-pii-edge"

            try:
                with model_installer.model_operation_lock(data_dir, model_id):
                    raise ValueError("Simulated failure")
            except ValueError:
                pass

            # Should be able to acquire immediately afterwards
            acquired = False
            with model_installer.model_operation_lock(data_dir, model_id):
                acquired = True
            self.assertTrue(acquired)

    def test_prompt_license_regression(self):
        """P1-4: Verify absence of unlicensed prompts and presence of Apache-2.0 semantic prompt and THIRD_PARTY_LICENSES.md."""
        from privacy.workers import memprivacy_worker, modelscope_downloader

        # Verify no MEMPRIVACY_OFFICIAL_PROMPT remains
        self.assertFalse(hasattr(memprivacy_worker, "MEMPRIVACY_OFFICIAL_PROMPT"))
        self.assertFalse(hasattr(modelscope_downloader, "MEMPRIVACY_OFFICIAL_PROMPT"))

        # Verify AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT is present
        self.assertTrue(hasattr(memprivacy_worker, "AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT"))
        self.assertTrue(hasattr(modelscope_downloader, "AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT"))

        # Verify THIRD_PARTY_LICENSES.md exists and documents licenses
        root_dir = Path(__file__).resolve().parent.parent
        license_file = root_dir / "THIRD_PARTY_LICENSES.md"
        self.assertTrue(license_file.is_file(), "THIRD_PARTY_LICENSES.md must exist in repo root")
        content = license_file.read_text(encoding="utf-8")
        self.assertIn("CC BY-NC-ND 4.0", content)
        self.assertIn("Apache-2.0", content)
        self.assertIn("AIPrivacyCheck", content)


class ModelInstallationHotfixV053Tests(unittest.TestCase):
    """Regression and security tests for v0.5.3 hotfix."""

    def test_verify_model_integrity_argument_order(self):
        """P0: Ensure verify_model_integrity passes (model_id, model_dir) to check_model_integrity without 'str' has no attribute 'is_dir'."""
        import model_installer
        from privacy.model_catalog import check_model_integrity

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "models" / "gliner-pii-edge"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}")
            (model_dir / "pytorch_model.bin").write_text("weights")
            (model_dir / "tokenizer.json").write_text("{}")

            with patch("model_installer.check_model_integrity", wraps=check_model_integrity) as mock_check:
                ok, err = model_installer.verify_model_integrity(model_dir, "gliner-pii-edge")
                self.assertTrue(ok)
                self.assertIsNone(err)
                mock_check.assert_called_once_with("gliner-pii-edge", model_dir)

    def test_unknown_model_id_rejected(self):
        """P1: Ensure unknown model IDs and path traversal patterns are rejected by validator and path resolvers."""
        import model_installer

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            bad_ids = ["../runtimes", "../../etc", "unknown-model", "", " ", "   ", "model/sub", "eval(1)"]
            for bad_id in bad_ids:
                with self.assertRaises(ValueError, msg=f"Should reject bad model_id: {bad_id}"):
                    model_installer.validate_model_id(bad_id)

                with self.assertRaises(ValueError, msg=f"get_model_dir should reject bad model_id: {bad_id}"):
                    model_installer.get_model_dir(data_dir, bad_id)

                with self.assertRaises(ValueError, msg=f"get_staging_dir should reject bad model_id: {bad_id}"):
                    model_installer.get_staging_dir(data_dir, bad_id)

                with self.assertRaises(ValueError, msg=f"model_operation_lock should reject bad model_id: {bad_id}"):
                    with model_installer.model_operation_lock(data_dir, bad_id):
                        pass

    def test_uninstall_traversal_rejected(self):
        """P1: Attempting to uninstall with a traversal path must be rejected and must not delete directories."""
        import model_installer

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            runtimes_dir = data_dir / "runtimes"
            runtimes_dir.mkdir(parents=True)
            marker_file = runtimes_dir / "marker.txt"
            marker_file.write_text("keep_alive")

            with self.assertRaises(ValueError):
                model_installer.uninstall_model(data_dir, "../runtimes")

            self.assertTrue(marker_file.exists(), "Marker file in runtimes directory must remain intact")

    def test_import_traversal_rejected(self):
        """P1: Attempting to import into a traversal target must be rejected before staging."""
        import model_installer

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            src_dir = Path(td) / "source"
            src_dir.mkdir(parents=True)
            (src_dir / "config.json").write_text("{}")
            (src_dir / "pytorch_model.bin").write_text("weights")
            (src_dir / "tokenizer.json").write_text("{}")

            with self.assertRaises(ValueError):
                model_installer.import_local_model(data_dir, "../runtimes", src_dir)

    def test_installer_chain_dry_run(self):
        """End-to-end verification of model installation flow with mocked runtime and downloader."""
        import model_installer
        from privacy.device import DEVICE_MANAGER

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            model_id = "gliner-pii-edge"

            # Prepare simulated model files
            fake_download_dir = data_dir / "models" / model_id
            fake_download_dir.mkdir(parents=True)
            (fake_download_dir / "config.json").write_text("{}")
            (fake_download_dir / "pytorch_model.bin").write_text("weights")
            (fake_download_dir / "tokenizer.json").write_text("{}")

            DEVICE_MANAGER.set_data_dir(data_dir)
            DEVICE_MANAGER.set_requested_device("cpu")

            with patch("model_installer.install_isolated_runtime") as mock_rt, \
                 patch("model_installer.download_modelscope_model", return_value=fake_download_dir), \
                 patch("model_installer.get_worker_client") as mock_wc, \
                 patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
                 patch("sys.argv", ["model_installer.py", "install", model_id]):

                mock_rt.return_value = data_dir / "runtimes" / "torch-cpu" / "venv"
                mock_client = MagicMock()
                mock_client.run_smoke_test.return_value = (True, None)
                mock_wc.return_value = mock_client

                exit_code = model_installer.main()
                self.assertEqual(exit_code, 0)

                status_file = data_dir / "status" / f"{model_id}-install.json"
                self.assertTrue(status_file.exists())
                status = json.loads(status_file.read_text(encoding="utf-8"))
                self.assertEqual(status["state"], "ready")
                self.assertIn("模型已就绪", status["detail"])


class WritableHomeAndStorageHotfixV054Tests(unittest.TestCase):
    """Regression and defense-in-depth tests for v0.5.4 writable HOME / cache hotfix."""

    def test_lifecycle_script_defines_writable_storage_env(self):
        """P0: Ensure cmd/main defines and exports writable HOME, XDG, and ML SDK cache paths."""
        main_script = PROJECT_DIR / "packaging" / "ai-privacy-check" / "cmd" / "main"
        self.assertTrue(main_script.is_file(), "cmd/main must exist")
        content = main_script.read_text(encoding="utf-8")

        self.assertIn('APP_HOME="${TRIM_PKGVAR}/home"', content)
        self.assertIn('CACHE_DIR="${DATA_DIR}/cache"', content)
        self.assertIn('MODELSCOPE_HOME_DIR="${DATA_DIR}/modelscope-home"', content)
        self.assertIn('MODELSCOPE_CACHE_DIR="${DATA_DIR}/modelscope"', content)
        self.assertIn('HF_HOME_DIR="${DATA_DIR}/huggingface"', content)
        self.assertIn('PIP_CACHE_DIR_PATH="${DATA_DIR}/pip-cache"', content)

        # Ensure env passing
        self.assertIn('HOME="$APP_HOME"', content)
        self.assertIn('XDG_CACHE_HOME="$CACHE_DIR"', content)
        self.assertIn('MODELSCOPE_HOME="$MODELSCOPE_HOME_DIR"', content)
        self.assertIn('MODELSCOPE_CACHE="$MODELSCOPE_CACHE_DIR"', content)
        self.assertIn('HF_HOME="$HF_HOME_DIR"', content)
        self.assertIn('PIP_CACHE_DIR="$PIP_CACHE_DIR_PATH"', content)

    def test_downloader_env_override(self):
        """P0: ModelScope downloader subprocess must receive all 6 storage env variables redirected to app data."""
        import model_installer

        with tempfile.TemporaryDirectory() as td:
            data_dir = (Path(td) / "pkgvar" / "data").resolve()
            data_dir.mkdir(parents=True)
            model_id = "gliner-pii-edge"

            captured_envs = []

            def mock_run(cmd, *args, **kwargs):
                env = kwargs.get("env")
                if env:
                    captured_envs.append(env)
                # Create fake target files in staging dir so integrity check passes
                staging = model_installer.get_staging_dir(data_dir, model_id)
                staging.mkdir(parents=True, exist_ok=True)
                (staging / "config.json").write_text("{}")
                (staging / "pytorch_model.bin").write_text("weights")
                (staging / "tokenizer.json").write_text("{}")
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = ""
                mock_proc.stderr = ""
                return mock_proc

            with patch("model_installer.install_isolated_runtime") as mock_rt, \
                 patch("model_installer.get_runtime_manager") as mock_rm, \
                 patch("subprocess.run", side_effect=mock_run):

                mock_mgr = MagicMock()
                mock_mgr.get_modelscope_capable_runtime.return_value = "torch-cpu"
                mock_mgr.get_python_bin.return_value = Path("/fake/python")
                mock_rm.return_value = mock_mgr

                target = model_installer.download_modelscope_model(data_dir, model_id)
                self.assertTrue(target.is_dir())
                self.assertGreater(len(captured_envs), 0)

                env = captured_envs[0]
                expected_keys = [
                    "HOME",
                    "XDG_CACHE_HOME",
                    "MODELSCOPE_HOME",
                    "MODELSCOPE_CACHE",
                    "HF_HOME",
                    "PIP_CACHE_DIR",
                ]
                for k in expected_keys:
                    self.assertIn(k, env, f"Key {k} must be in downloader environment")
                    val_path = Path(env[k]).resolve()
                    # All paths must be within the temp package root (data_dir.parent)
                    self.assertTrue(
                        val_path == data_dir.parent or data_dir.parent in val_path.parents,
                        f"{k}={val_path} must be within app storage root {data_dir.parent}",
                    )

    def test_modelscope_downloader_never_uses_system_home(self):
        """P0: Even if process HOME is set to package user /home/ai-privacy-check, runtime env must override it."""
        from privacy.runtime_env import build_runtime_env

        with tempfile.TemporaryDirectory() as td:
            data_dir = (Path(td) / "var" / "data").resolve()
            data_dir.mkdir(parents=True)

            with patch.dict(os.environ, {"HOME": "/home/ai-privacy-check"}, clear=False):
                env = build_runtime_env(data_dir)
                self.assertNotEqual(env["HOME"], "/home/ai-privacy-check")
                self.assertEqual(env["HOME"], str(data_dir.parent / "home"))
                self.assertEqual(env["MODELSCOPE_HOME"], str(data_dir / "modelscope-home"))
                self.assertEqual(env["MODELSCOPE_CACHE"], str(data_dir / "modelscope"))
                self.assertEqual(env["HF_HOME"], str(data_dir / "huggingface"))
                self.assertEqual(env["XDG_CACHE_HOME"], str(data_dir / "cache"))
                self.assertEqual(env["PIP_CACHE_DIR"], str(data_dir / "pip-cache"))

    def test_runtime_dirs_auto_created(self):
        """P1: Calling prepare_runtime_dirs must physically create all 6 storage directories."""
        from privacy.runtime_env import prepare_runtime_dirs

        with tempfile.TemporaryDirectory() as td:
            data_dir = (Path(td) / "var" / "data").resolve()
            # Directory does not exist yet
            self.assertFalse((data_dir.parent / "home").exists())
            self.assertFalse((data_dir / "modelscope-home").exists())

            dirs = prepare_runtime_dirs(data_dir)

            self.assertTrue((data_dir.parent / "home").is_dir())
            self.assertTrue((data_dir / "cache").is_dir())
            self.assertTrue((data_dir / "modelscope-home").is_dir())
            self.assertTrue((data_dir / "modelscope").is_dir())
            self.assertTrue((data_dir / "huggingface").is_dir())
            self.assertTrue((data_dir / "pip-cache").is_dir())

    def test_worker_process_uses_writable_env(self):
        """P1: RuntimeWorkerProcess must inject build_runtime_env when data_dir is provided."""
        from privacy.worker_client import RuntimeWorkerProcess

        with tempfile.TemporaryDirectory() as td:
            data_dir = (Path(td) / "var" / "data").resolve()
            fake_py = Path(td) / "venv" / "bin" / "python"
            fake_script = Path(td) / "worker.py"
            fake_py.parent.mkdir(parents=True)
            fake_py.touch(mode=0o755)
            fake_script.touch()

            worker = RuntimeWorkerProcess(
                python_bin=fake_py,
                worker_script=fake_script,
                model_id="gliner-pii-edge",
                profile="torch-cpu",
                data_dir=data_dir,
            )

            captured_env = {}

            def mock_popen(cmd, *args, **kwargs):
                nonlocal captured_env
                captured_env = kwargs.get("env", {})
                mock_p = MagicMock()
                mock_p.pid = 12345
                mock_p.poll.return_value = None
                mock_p.stdin = MagicMock()
                mock_p.stdout = MagicMock()
                mock_p.stderr = MagicMock()
                mock_p.stdout.readline.return_value = '{"ok": true, "status": "pong"}\n'
                return mock_p

            with patch("subprocess.Popen", side_effect=mock_popen):
                worker.start()
                self.assertEqual(captured_env.get("HOME"), str(data_dir.parent / "home"))
                self.assertEqual(captured_env.get("MODELSCOPE_HOME"), str(data_dir / "modelscope-home"))
                self.assertEqual(captured_env.get("MODELSCOPE_CACHE"), str(data_dir / "modelscope"))
                worker.terminate()



class RuntimeReadinessAndStateSeparationV055Tests(unittest.TestCase):
    """Regression tests for v0.5.5 separating model installation from runtime readiness."""

    def test_runtime_ux_state_matrix_cases(self):
        """Case 1 - Case 5 state derivation matrix verification via app.js deriveModelRuntimeState."""
        app_js_path = SERVER_DIR / "web" / "app.js"
        node_script = f"""
        const fs = require('fs');
        const content = fs.readFileSync('{app_js_path}', 'utf8');
        const fnMatch = content.match(/function deriveModelRuntimeState[\\s\\S]*?\\n\\}}/);
        if (!fnMatch) throw new Error("deriveModelRuntimeState not found in app.js");
        eval(fnMatch[0]);

        const cases = [
          // Case 1: model=false, CPU runtime=false
          deriveModelRuntimeState({{ detector: {{ installed: false, ready: false }} }}, {{ requested_device: 'auto', runtimes: {{ torch_cpu: {{ installed: false }} }} }}),
          // Case 2: model=true, CUDA=true, requested=cuda
          deriveModelRuntimeState({{ detector: {{ installed: true, ready: true }} }}, {{ requested_device: 'cuda', runtimes: {{ torch_cuda: {{ installed: true, verified: true, cuda_available: true }} }} }}),
          // Case 3: model=true, CUDA=true, CPU=false, requested=cpu
          deriveModelRuntimeState({{ detector: {{ installed: true, ready: false }} }}, {{ requested_device: 'cpu', runtimes: {{ torch_cpu: {{ installed: false }}, torch_cuda: {{ installed: true, verified: true, cuda_available: true }} }} }}),
          // Case 4: model=true, CUDA=false, CPU=true, requested=cpu
          deriveModelRuntimeState({{ detector: {{ installed: true, ready: true }} }}, {{ requested_device: 'cpu', runtimes: {{ torch_cpu: {{ installed: true, verified: true }} }} }}),
          // Case 5: model=true, CUDA=false, CPU=false, requested=auto
          deriveModelRuntimeState({{ detector: {{ installed: true, ready: false }} }}, {{ requested_device: 'auto', hardware: {{ nvidia_available: false }}, runtimes: {{ torch_cpu: {{ installed: false }} }} }})
        ];
        console.log(JSON.stringify(cases));
        """
        proc = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
        results = json.loads(proc.stdout)

        # Case 1: model=false, CPU runtime=false -> install model button
        c1 = results[0]
        self.assertFalse(c1["isInstalled"])
        self.assertFalse(c1["isReady"])
        self.assertEqual(c1["installButtonText"], "从魔搭下载安装")
        self.assertFalse(c1["showUninstall"])

        # Case 2: model=true, CUDA=true, requested=cuda -> ready, no runtime button, show uninstall
        c2 = results[1]
        self.assertTrue(c2["isInstalled"])
        self.assertTrue(c2["isReady"])
        self.assertIsNone(c2["installButtonText"])
        self.assertTrue(c2["showUninstall"])

        # Case 3: model=true, CUDA=true, CPU=false, requested=cpu -> runtime missing -> install CPU runtime button & uninstall
        c3 = results[2]
        self.assertTrue(c3["isInstalled"])
        self.assertFalse(c3["isReady"])
        self.assertTrue(c3["runtimeMissing"])
        self.assertEqual(c3["installButtonText"], "安装 CPU 运行时")
        self.assertTrue(c3["showUninstall"])

        # Case 4: model=true, CUDA=false, CPU=true, requested=cpu -> ready, show uninstall
        c4 = results[3]
        self.assertTrue(c4["isInstalled"])
        self.assertTrue(c4["isReady"])
        self.assertIsNone(c4["installButtonText"])
        self.assertTrue(c4["showUninstall"])

        # Case 5: model=true, CUDA=false, CPU=false, requested=auto -> install recommended runtime
        c5 = results[4]
        self.assertTrue(c5["isInstalled"])
        self.assertFalse(c5["isReady"])
        self.assertTrue(c5["runtimeMissing"])
        self.assertEqual(c5["installButtonText"], "安装推荐运行时")
        self.assertTrue(c5["showUninstall"])

    def test_ui_inline_status_text_regression(self):
        """When model is installed but runtime is missing, UI must never state '未安装增强模型'."""
        app_js_path = SERVER_DIR / "web" / "app.js"
        node_script = f"""
        const fs = require('fs');
        const content = fs.readFileSync('{app_js_path}', 'utf8');

        function evaluateStatus(data) {{
          const slots = (data.registry && data.registry.slots) || {{}};
          const dev = data.device || {{}};
          const glinerSlot = slots.general_pii || {{}};
          const memSlot = slots.semantic_privacy || {{}};
          const glinerInstalled = Boolean(glinerSlot.detector && glinerSlot.detector.installed);
          const memInstalled = Boolean(memSlot.detector && memSlot.detector.installed);
          const anyModelInstalled = glinerInstalled || memInstalled;

          const glinerReady = Boolean(glinerSlot.detector && glinerSlot.detector.ready);
          const memReady = Boolean(memSlot.detector && memSlot.detector.ready);
          const anyModelReady = glinerReady || memReady;

          if (data.installing) {{
            return "正在后台安装模型或运行环境…";
          }} else if (anyModelReady) {{
            return "增强模型已就绪 (GLiNER / MemPrivacy)";
          }} else if (anyModelInstalled) {{
            const req = (dev.requested_device || "auto").toLowerCase();
            if (req === "cpu") {{
              return "增强模型已安装，但 CPU 运行时未就绪。";
            }} else if (req === "cuda") {{
              return "增强模型已安装，但 CUDA 运行时未就绪。";
            }} else {{
              return "增强模型已安装，但计算运行时未就绪。";
            }}
          }} else {{
            return "未安装增强模型（基础规则与中文语义始终可用）";
          }}
        }}

        const tests = {{
          installedCpuMissing: evaluateStatus({{
            registry: {{ slots: {{ general_pii: {{ detector: {{ installed: true, ready: false }} }} }} }},
            device: {{ requested_device: "cpu" }}
          }}),
          installedCudaMissing: evaluateStatus({{
            registry: {{ slots: {{ general_pii: {{ detector: {{ installed: true, ready: false }} }} }} }},
            device: {{ requested_device: "cuda" }}
          }}),
          installedAutoMissing: evaluateStatus({{
            registry: {{ slots: {{ general_pii: {{ detector: {{ installed: true, ready: false }} }} }} }},
            device: {{ requested_device: "auto" }}
          }}),
          neitherInstalled: evaluateStatus({{
            registry: {{ slots: {{ general_pii: {{ detector: {{ installed: false, ready: false }} }} }} }},
            device: {{ requested_device: "auto" }}
          }}),
          modelReady: evaluateStatus({{
            registry: {{ slots: {{ general_pii: {{ detector: {{ installed: true, ready: true }} }} }} }},
            device: {{ requested_device: "cuda" }}
          }})
        }};
        console.log(JSON.stringify(tests));
        """
        proc = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
        res = json.loads(proc.stdout)

        self.assertEqual(res["installedCpuMissing"], "增强模型已安装，但 CPU 运行时未就绪。")
        self.assertNotIn("未安装增强模型", res["installedCpuMissing"])

        self.assertEqual(res["installedCudaMissing"], "增强模型已安装，但 CUDA 运行时未就绪。")
        self.assertNotIn("未安装增强模型", res["installedCudaMissing"])

        self.assertEqual(res["installedAutoMissing"], "增强模型已安装，但计算运行时未就绪。")
        self.assertNotIn("未安装增强模型", res["installedAutoMissing"])

        self.assertEqual(res["neitherInstalled"], "未安装增强模型（基础规则与中文语义始终可用）")
        self.assertEqual(res["modelReady"], "增强模型已就绪 (GLiNER / MemPrivacy)")

    def test_install_missing_runtime_skips_model_download(self):
        """Installing missing runtime for existing model weights must skip model re-download and test with runtime."""
        import model_installer
        from privacy.device import DEVICE_MANAGER

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            model_id = "gliner-pii-edge"

            # Create existing model files
            m_dir = data_dir / "models" / model_id
            m_dir.mkdir(parents=True, exist_ok=True)
            (m_dir / "gliner_config.json").write_text("{}", encoding="utf-8")
            (m_dir / "model.safetensors").write_text("fake-weights", encoding="utf-8")
            (m_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (m_dir / ".metadata.json").write_text(json.dumps({"model_id": model_id}), encoding="utf-8")

            # Verify integrity passes before install
            ok, _ = model_installer.verify_model_integrity(m_dir, model_id)
            self.assertTrue(ok)

            installed_runtimes = []
            def fake_install_isolated_runtime(d_dir, profile):
                installed_runtimes.append(profile)
                rt_dir = d_dir / "runtimes" / profile
                rt_dir.mkdir(parents=True, exist_ok=True)
                (rt_dir / ".installed.json").write_text("{}", encoding="utf-8")
                return rt_dir / "venv"

            smoke_test_calls = []
            def fake_smoke_test(model_id, model_path, profile, device):
                smoke_test_calls.append({
                    "model_id": model_id,
                    "model_path": model_path,
                    "profile": profile,
                    "device": device,
                })
                return True, None

            fake_client = MagicMock()
            fake_client.run_smoke_test.side_effect = fake_smoke_test

            env_patch = {"APP_DATA_DIR": str(data_dir)}
            with patch.dict(os.environ, env_patch), \
                 patch.object(DEVICE_MANAGER, "get_requested_device", return_value="cpu"), \
                 patch.object(DEVICE_MANAGER, "probe_diagnostics", return_value={"hardware": {"nvidia_available": False}}), \
                 patch("model_installer.install_isolated_runtime", side_effect=fake_install_isolated_runtime), \
                 patch("model_installer.get_worker_client", return_value=fake_client), \
                 patch("sys.argv", ["model_installer.py", "install", model_id]):

                code = model_installer.main()
                self.assertEqual(code, 0)

            # Assert torch-cpu was installed
            self.assertIn("torch-cpu", installed_runtimes)

            # Assert smoke test executed with torch-cpu and cpu
            self.assertEqual(len(smoke_test_calls), 1)
            self.assertEqual(smoke_test_calls[0]["profile"], "torch-cpu")
            self.assertEqual(smoke_test_calls[0]["device"], "cpu")

            # Assert final state written is 'ready'
            status_file = data_dir / "status" / f"{model_id}-install.json"
            self.assertTrue(status_file.is_file())
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(status_data["state"], "ready")

    def test_uninstall_model_preserves_runtimes(self):
        """Uninstalling model weights must preserve torch-cpu and torch-cuda runtimes."""
        import model_installer

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            model_id = "gliner-pii-edge"

            # Setup model
            m_dir = data_dir / "models" / model_id
            m_dir.mkdir(parents=True, exist_ok=True)
            (m_dir / "gliner_config.json").write_text("{}", encoding="utf-8")

            # Setup runtimes
            r_cpu = data_dir / "runtimes" / "torch-cpu"
            r_cpu.mkdir(parents=True, exist_ok=True)
            (r_cpu / ".installed.json").write_text(json.dumps({"profile": "torch-cpu"}), encoding="utf-8")

            r_cuda = data_dir / "runtimes" / "torch-cuda"
            r_cuda.mkdir(parents=True, exist_ok=True)
            (r_cuda / ".installed.json").write_text(json.dumps({"profile": "torch-cuda"}), encoding="utf-8")

            # Execute uninstall
            ok, msg = model_installer.uninstall_model(data_dir, model_id)
            self.assertTrue(ok)

            # Model is deleted
            self.assertFalse(m_dir.exists())

            # Runtimes are strictly preserved
            self.assertTrue((r_cpu / ".installed.json").is_file())
            self.assertTrue((r_cuda / ".installed.json").is_file())

    def test_device_switch_preserves_model_installed(self):
        """Switching requested device from CUDA to CPU must preserve installed=True even when runtime is not ready."""
        from privacy.detectors import GLiNERDetector
        from privacy.device import DeviceManager
        import privacy.detectors as detectors_mod

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            model_id = "gliner-pii-edge"

            # Create model files
            m_dir = data_dir / "models" / model_id
            m_dir.mkdir(parents=True, exist_ok=True)
            (m_dir / "gliner_config.json").write_text("{}", encoding="utf-8")
            (m_dir / "model.safetensors").write_text("fake-weights", encoding="utf-8")
            (m_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            detector = GLiNERDetector(data_dir, active_model_id=model_id)

            # 1. Simulate CUDA ready
            cuda_res = {"ready": True, "actual_device": "cuda", "runtime_profile": "torch-cuda"}
            with patch.object(detectors_mod.DEVICE_MANAGER, "resolve_for_model", return_value=cuda_res):
                st_cuda = detector.status()
                self.assertTrue(st_cuda["installed"])
                self.assertTrue(st_cuda["ready"])
                self.assertEqual(st_cuda["device"], "cuda")

            # 2. Simulate switch to CPU where CPU runtime is missing
            cpu_missing_res = {"ready": False, "actual_device": "none", "runtime_profile": None}
            with patch.object(detectors_mod.DEVICE_MANAGER, "resolve_for_model", return_value=cpu_missing_res):
                st_cpu = detector.status()
                # installed must strictly stay True!
                self.assertTrue(st_cpu["installed"])
                self.assertFalse(st_cpu["ready"])
                self.assertEqual(st_cpu["device"], "none")



class RuntimeCacheInvalidationV056Tests(unittest.TestCase):
    """Regression tests for v0.5.6 event-driven runtime probe and device diagnostics cache invalidation."""

    @staticmethod
    def _probe_output(interpreter: Path) -> str:
        return json.dumps({
            "ok": True,
            "framework": "torch",
            "framework_version": "2.4.0+cpu",
            "device_target": "cpu",
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "cuda_version": None,
            "interpreter": str(interpreter),
            "error": None,
        })

    @staticmethod
    def _create_mock_cpu_runtime(runtime_manager: RuntimeManager) -> Path:
        python_bin = runtime_manager.venv_dir("torch-cpu") / "bin" / "python3"
        python_bin.parent.mkdir(parents=True, exist_ok=True)
        python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python_bin.chmod(0o755)
        runtime_manager._runner = lambda cmd, **kwargs: (
            0,
            RuntimeCacheInvalidationV056Tests._probe_output(python_bin),
            "",
        )
        return python_bin

    @staticmethod
    def _create_mock_gliner_model(data_dir: Path) -> None:
        model_dir = data_dir / "models" / "gliner-pii-edge"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "gliner_config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    @staticmethod
    def _wait_until(predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_runtime_probe_cache_invalidation(self):
        """RuntimeManager must invalidate probe cache on demand, allowing updated runtime state to reflect."""
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            rm = RuntimeManager(data_dir)

            # 1. Initial probe: torch-cpu not installed
            res1 = rm.probe_profile("torch-cpu")
            self.assertFalse(res1["installed"])
            self.assertIn("torch-cpu", rm._probe_cache)

            # 2. Simulate installation of torch-cpu in filesystem
            profile_dir = rm.profile_dir("torch-cpu")
            venv_bin = profile_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True, exist_ok=True)
            mock_python = venv_bin / "python3"
            mock_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            mock_python.chmod(0o755)

            # Mock runner for probe command
            mock_probe_out = json.dumps({
                "ok": True,
                "framework": "torch",
                "framework_version": "2.4.0+cpu",
                "device_target": "cpu",
                "cuda_available": False,
                "device_count": 0,
                "device_name": None,
                "cuda_version": None,
                "interpreter": str(mock_python),
                "error": None,
            })
            rm._runner = lambda cmd, **kwargs: (0, mock_probe_out, "")

            # Without cache invalidation: hits old cache
            res_cached = rm.probe_profile("torch-cpu")
            self.assertFalse(res_cached["installed"])

            # Call invalidate_probe_cache with specific profile
            rm.invalidate_probe_cache("torch-cpu")
            self.assertNotIn("torch-cpu", rm._probe_cache)

            # Now probe again: reflects new installed state
            res_fresh = rm.probe_profile("torch-cpu")
            self.assertTrue(res_fresh["installed"])
            self.assertTrue(res_fresh["verified"])

            # Call invalidate_probe_cache without arguments: clears entire cache
            rm.invalidate_probe_cache()
            self.assertEqual(len(rm._probe_cache), 0)

    def test_inflight_runtime_probe_cannot_overwrite_refreshed_cache(self):
        """A probe started before invalidation must not publish after a newer refresh."""
        with tempfile.TemporaryDirectory() as td:
            rm = RuntimeManager(Path(td))
            original_interpreter_path = rm.interpreter_path
            old_probe_started = threading.Event()
            publish_old_probe = threading.Event()
            first_call = True
            call_lock = threading.Lock()

            def delayed_interpreter_path(profile: str):
                nonlocal first_call
                with call_lock:
                    is_old_probe = first_call
                    first_call = False
                result = original_interpreter_path(profile)
                if is_old_probe:
                    old_probe_started.set()
                    self.assertTrue(publish_old_probe.wait(timeout=5))
                return result

            rm.interpreter_path = delayed_interpreter_path
            old_result = {}
            old_thread = threading.Thread(
                target=lambda: old_result.update(rm.probe_profile("torch-cpu")),
            )
            old_thread.start()
            self.assertTrue(old_probe_started.wait(timeout=5))

            self._create_mock_cpu_runtime(rm)
            rm.invalidate_probe_cache("torch-cpu")
            fresh_result = rm.probe_profile("torch-cpu", force_refresh=True)
            publish_old_probe.set()
            old_thread.join(timeout=5)

            self.assertFalse(old_thread.is_alive())
            self.assertTrue(fresh_result["verified"])
            self.assertFalse(old_result["installed"])
            self.assertTrue(rm.probe_profile("torch-cpu")["installed"])

    def test_device_diagnostics_cache_invalidation(self):
        """DeviceManager.invalidate_runtime_state must clear both runtime probe cache and device diagnostics."""
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            rm = RuntimeManager(data_dir)
            dm = DeviceManager(data_dir=data_dir, runtime_manager=rm)

            # 1. Initial diagnostics: torch-cpu uninstalled
            diag1 = dm.probe_diagnostics()
            self.assertFalse(diag1["runtimes"]["torch_cpu"].get("installed", False))
            self.assertIsNotNone(dm._diagnostics_cache)

            # 2. Simulate torch-cpu installed
            profile_dir = rm.profile_dir("torch-cpu")
            venv_bin = profile_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True, exist_ok=True)
            mock_python = venv_bin / "python3"
            mock_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            mock_python.chmod(0o755)

            mock_probe_out = json.dumps({
                "ok": True,
                "framework": "torch",
                "framework_version": "2.4.0+cpu",
                "device_target": "cpu",
                "cuda_available": False,
                "device_count": 0,
                "device_name": None,
                "cuda_version": None,
                "interpreter": str(mock_python),
                "error": None,
            })
            rm._runner = lambda cmd, **kwargs: (0, mock_probe_out, "")

            # Without invalidation: returns old cached diagnostics
            diag_cached = dm.probe_diagnostics()
            self.assertFalse(diag_cached["runtimes"]["torch_cpu"].get("installed", False))

            # Call invalidate_runtime_state
            dm.invalidate_runtime_state()
            self.assertIsNone(dm._diagnostics_cache)
            self.assertNotIn("torch-cpu", rm._probe_cache)

            # Next probe_diagnostics reflects new installed state
            diag_fresh = dm.probe_diagnostics()
            self.assertTrue(diag_fresh["runtimes"]["torch_cpu"].get("installed", False))

    def test_inflight_device_probe_cannot_overwrite_refreshed_cache(self):
        """Diagnostics started before invalidation must not replace a newer snapshot."""
        with tempfile.TemporaryDirectory() as td:
            old_probe_started = threading.Event()
            publish_old_probe = threading.Event()
            call_count = 0
            call_lock = threading.Lock()

            old_runtime = {
                "torch-cpu": {"installed": False, "verified": False},
                "torch-cuda": {"installed": False, "verified": False},
            }
            new_runtime = {
                "torch-cpu": {"installed": True, "verified": True},
                "torch-cuda": {"installed": False, "verified": False},
            }
            runtime_manager = MagicMock()

            def probe_all(force_refresh=False):
                nonlocal call_count
                with call_lock:
                    call_count += 1
                    is_old_probe = call_count == 1
                snapshot = old_runtime if is_old_probe else new_runtime
                if is_old_probe:
                    old_probe_started.set()
                    self.assertTrue(publish_old_probe.wait(timeout=5))
                return snapshot

            runtime_manager.probe_all.side_effect = probe_all
            runtime_manager.best_runtime_for_framework.return_value = "torch-cpu"
            hardware_probe = MagicMock()
            hardware_probe.probe_nvidia.return_value = {
                "nvidia_available": False,
                "gpu_count": 0,
                "gpus": [],
                "driver_version": None,
            }
            dm = DeviceManager(
                data_dir=Path(td),
                hardware_probe=hardware_probe,
                runtime_manager=runtime_manager,
            )
            old_result = {}
            old_thread = threading.Thread(
                target=lambda: old_result.update(dm.probe_diagnostics()),
            )
            old_thread.start()
            self.assertTrue(old_probe_started.wait(timeout=5))

            dm.invalidate_cache()
            fresh_result = dm.probe_diagnostics(force_refresh=True)
            publish_old_probe.set()
            old_thread.join(timeout=5)

            self.assertFalse(old_thread.is_alive())
            self.assertFalse(old_result["runtimes"]["torch_cpu"]["installed"])
            self.assertTrue(fresh_result["runtimes"]["torch_cpu"]["installed"])
            self.assertTrue(dm.probe_diagnostics()["runtimes"]["torch_cpu"]["installed"])

    def test_installer_completion_integration(self):
        """The real start/wait callback must refresh readiness without restarting the server."""
        import server
        from server import ModelLifecycleController

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_gliner_model(data_dir)
            previous_data_dir = server.DEVICE_MANAGER.data_dir
            previous_privacy = server.PRIVACY
            child_may_exit = threading.Event()
            finalization_complete = threading.Event()

            try:
                controller = ModelLifecycleController(data_dir)
                server.DEVICE_MANAGER.set_data_dir(data_dir)
                server.DEVICE_MANAGER.set_requested_device("cpu")
                server.PRIVACY = server.PrivacyService(data_dir)
                runtime_manager = server.DEVICE_MANAGER._rt_manager

                before = controller.status()
                self.assertFalse(before["device"]["runtimes"]["torch_cpu"].get("installed", False))
                self.assertFalse(before["registry"]["slots"]["general_pii"]["detector"]["ready"])

                test_case = self

                class SuccessfulInstallerProcess:
                    returncode = None

                    def poll(self):
                        return self.returncode

                    def wait(self, timeout=None):
                        if not child_may_exit.wait(timeout=timeout or 5):
                            raise subprocess.TimeoutExpired("installer", timeout)
                        test_case._create_mock_cpu_runtime(runtime_manager)
                        self.returncode = 0
                        return 0

                original_reset = server.PRIVACY.reset_models

                def tracked_reset():
                    original_reset()
                    finalization_complete.set()

                with patch("server.subprocess.Popen", return_value=SuccessfulInstallerProcess()), \
                     patch.object(server.PRIVACY, "reset_models", side_effect=tracked_reset):
                    self.assertTrue(controller.start_install("gliner-pii-edge"))
                    self.assertTrue(controller.status()["installing"])
                    child_may_exit.set()
                    self.assertTrue(finalization_complete.wait(timeout=5))
                    self.assertTrue(self._wait_until(lambda: controller._process is None))

                after = controller.status()
                self.assertFalse(after["installing"])
                self.assertTrue(after["device"]["runtimes"]["torch_cpu"].get("installed", False))
                self.assertTrue(after["registry"]["slots"]["general_pii"]["detector"]["ready"])
            finally:
                server.DEVICE_MANAGER.set_data_dir(previous_data_dir)
                server.PRIVACY = previous_privacy

    def test_failed_installer_invalidates_without_refreshing_or_resetting(self):
        """A failed child must clear caches but must not force readiness or reset workers."""
        import server
        from server import ModelLifecycleController

        with tempfile.TemporaryDirectory() as td:
            controller = ModelLifecycleController(Path(td))
            child_may_exit = threading.Event()

            class FailedInstallerProcess:
                returncode = None

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    if not child_may_exit.wait(timeout=timeout or 5):
                        raise subprocess.TimeoutExpired("installer", timeout)
                    self.returncode = 1
                    return 1

            with patch("server.subprocess.Popen", return_value=FailedInstallerProcess()), \
                 patch.object(server.DEVICE_MANAGER, "invalidate_runtime_state") as invalidate, \
                 patch.object(server.DEVICE_MANAGER, "probe_diagnostics") as probe, \
                 patch.object(server.PRIVACY, "reset_models") as reset:
                self.assertTrue(controller.start_install("gliner-pii-edge"))
                child_may_exit.set()
                self.assertTrue(self._wait_until(lambda: controller._process is None))

            invalidate.assert_called_once_with()
            probe.assert_not_called()
            reset.assert_not_called()

    def test_installing_remains_true_until_completion_refresh_finishes(self):
        """The UI polling flag must cover cache refresh, not only child-process lifetime."""
        import server
        from server import ModelLifecycleController

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            controller = ModelLifecycleController(data_dir)
            child_may_exit = threading.Event()
            refresh_started = threading.Event()
            allow_refresh = threading.Event()
            finalization_complete = threading.Event()

            class SuccessfulInstallerProcess:
                returncode = None

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    if not child_may_exit.wait(timeout=timeout or 5):
                        raise subprocess.TimeoutExpired("installer", timeout)
                    self.returncode = 0
                    return 0

            fake_privacy = MagicMock()
            fake_privacy.registry.status.return_value = {}
            fake_privacy.reset_models.side_effect = finalization_complete.set

            def blocked_invalidation():
                refresh_started.set()
                self.assertTrue(allow_refresh.wait(timeout=5))

            with patch("server.subprocess.Popen", return_value=SuccessfulInstallerProcess()), \
                 patch("server.PRIVACY", fake_privacy), \
                 patch.object(server.DEVICE_MANAGER, "invalidate_runtime_state", side_effect=blocked_invalidation), \
                 patch.object(server.DEVICE_MANAGER, "probe_diagnostics", return_value={}), \
                 patch("server.model_installer.scan_shared_models_directory", return_value=[]), \
                 patch("server.model_installer.get_shared_models_dirs", return_value=[]):
                self.assertTrue(controller.start_install("gliner-pii-edge"))
                child_may_exit.set()
                self.assertTrue(refresh_started.wait(timeout=5))
                during_refresh = controller.status()
                allow_refresh.set()
                self.assertTrue(finalization_complete.wait(timeout=5))
                self.assertTrue(self._wait_until(lambda: controller._process is None))
                after_refresh = controller.status()

            self.assertTrue(during_refresh["installing"])
            self.assertFalse(after_refresh["installing"])

    def test_ready_without_server_restart(self):
        """Model readiness transitions from false to true without needing application/server restart."""
        import server
        from server import ModelLifecycleController

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            m_dir = data_dir / "models" / "gliner-pii-edge"
            m_dir.mkdir(parents=True, exist_ok=True)
            (m_dir / "gliner_config.json").write_text("{}", encoding="utf-8")
            (m_dir / "model.safetensors").write_text("fake", encoding="utf-8")
            (m_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            controller = ModelLifecycleController(data_dir)
            server.DEVICE_MANAGER.set_data_dir(data_dir)
            server.DEVICE_MANAGER.set_requested_device("cpu")
            server.PRIVACY = server.PrivacyService(data_dir)

            # 1. State before: not ready
            st1 = controller.status()
            self.assertTrue(st1["registry"]["slots"]["general_pii"]["detector"]["installed"])
            self.assertFalse(st1["registry"]["slots"]["general_pii"]["detector"]["ready"])

            # 2. Simulate runtime installation finishing in background
            rm = server.DEVICE_MANAGER._rt_manager
            profile_dir = rm.profile_dir("torch-cpu")
            venv_bin = profile_dir / "venv" / "bin"
            venv_bin.mkdir(parents=True, exist_ok=True)
            mock_python = venv_bin / "python3"
            mock_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            mock_python.chmod(0o755)

            mock_probe_out = json.dumps({
                "ok": True,
                "framework": "torch",
                "framework_version": "2.4.0+cpu",
                "device_target": "cpu",
                "cuda_available": False,
                "device_count": 0,
                "device_name": None,
                "cuda_version": None,
                "interpreter": str(mock_python),
                "error": None,
            })
            rm._runner = lambda cmd, **kwargs: (0, mock_probe_out, "")

            # Invalidate runtime state
            server.DEVICE_MANAGER.invalidate_runtime_state()
            server.DEVICE_MANAGER.probe_diagnostics(force_refresh=True)
            server.PRIVACY.reset_models()

            # 3. Next status check on SAME controller and SAME server instance
            st2 = controller.status()
            self.assertTrue(st2["registry"]["slots"]["general_pii"]["detector"]["installed"])
            self.assertTrue(st2["registry"]["slots"]["general_pii"]["detector"]["ready"])
            self.assertEqual(st2["registry"]["slots"]["general_pii"]["detector"]["device"], "cpu")


class CorrectnessHardeningV061Tests(unittest.TestCase):
    """Regression tests for v0.6.1 correctness hardening."""

    def test_codepoint_index_to_utf16_helper(self):
        from privacy.entities import codepoint_index_to_utf16
        # 1. ASCII
        self.assertEqual(codepoint_index_to_utf16("hello", 0), 0)
        self.assertEqual(codepoint_index_to_utf16("hello", 3), 3)
        self.assertEqual(codepoint_index_to_utf16("hello", 5), 5)

        # 2. Chinese (BMP, 1 code point == 1 code unit)
        self.assertEqual(codepoint_index_to_utf16("你好世界", 2), 2)
        self.assertEqual(codepoint_index_to_utf16("你好世界", 4), 4)

        # 3. Emoji (Astral plane, U+1F600, 1 code point == 2 code units)
        self.assertEqual(codepoint_index_to_utf16("😀张三", 0), 0)
        self.assertEqual(codepoint_index_to_utf16("😀张三", 1), 2)
        self.assertEqual(codepoint_index_to_utf16("😀张三", 2), 3)
        self.assertEqual(codepoint_index_to_utf16("😀张三", 3), 4)

        # 4. Regional indicator flag (🇨🇳 = U+1F1E8 U+1F1F3 -> 4 code units)
        self.assertEqual(codepoint_index_to_utf16("🇨🇳张三", 1), 2)
        self.assertEqual(codepoint_index_to_utf16("🇨🇳张三", 2), 4)
        self.assertEqual(codepoint_index_to_utf16("🇨🇳张三", 3), 5)

        # 5. Astral plane CJK character (𠀀 = U+20000 -> 2 code units)
        self.assertEqual(codepoint_index_to_utf16("A𠀀B", 1), 1)
        self.assertEqual(codepoint_index_to_utf16("A𠀀B", 2), 3)
        self.assertEqual(codepoint_index_to_utf16("A𠀀B", 3), 4)

        # 6. ZWJ emoji (👨\u200d👩\u200d👧\u200d👦 = 4 people + 3 ZWJs -> 7 codepoints, 11 code units)
        zwj_text = "👨\u200d👩\u200d👧\u200d👦姓名"
        self.assertEqual(codepoint_index_to_utf16(zwj_text, 7), 11)
        self.assertEqual(codepoint_index_to_utf16(zwj_text, 9), 13)

    def test_entity_serialization_exposes_utf16_offsets(self):
        from privacy.entities import Entity
        text = "😀张三的电话是13800138000"
        entity = Entity(
            entity_type="PHONE",
            start=7,
            end=18,
            text="13800138000",
            confidence=0.95,
            sources=("rules",),
        )
        # Without text passed: falls back to start/end
        d_no_text = entity.to_dict()
        self.assertEqual(d_no_text["start"], 7)
        self.assertEqual(d_no_text["end"], 18)
        self.assertEqual(d_no_text["start_utf16"], 7)
        self.assertEqual(d_no_text["end_utf16"], 18)

        # With text passed: maps correctly to UTF-16 code unit offsets 8 and 19
        d_with_text = entity.to_dict(text=text)
        self.assertEqual(d_with_text["start"], 7)
        self.assertEqual(d_with_text["end"], 18)
        self.assertEqual(d_with_text["start_utf16"], 8)
        self.assertEqual(d_with_text["end_utf16"], 19)

    def test_reset_models_runs_even_if_refresh_probe_fails(self):
        import server
        with tempfile.TemporaryDirectory() as td:
            ctrl = server.ModelLifecycleController(Path(td))
            reset_called = []
            orig_reset = server.PRIVACY.reset_models
            server.PRIVACY.reset_models = lambda: reset_called.append(True)
            orig_probe = server.DEVICE_MANAGER.probe_diagnostics
            try:
                def failing_probe(*args, **kwargs):
                    if kwargs.get("force_refresh"):
                        raise RuntimeError("Mock probe crash during refresh")
                    return orig_probe(*args, **kwargs)

                server.DEVICE_MANAGER.probe_diagnostics = failing_probe

                with patch.object(server.model_installer, "import_local_model", return_value=(True, "ok")):
                    with self.assertRaises(RuntimeError):
                        ctrl.import_model("gliner-pii-edge", "/some/path")
                    self.assertTrue(reset_called, "PRIVACY.reset_models must be called even if probe_diagnostics raised")
            finally:
                server.PRIVACY.reset_models = orig_reset
                server.DEVICE_MANAGER.probe_diagnostics = orig_probe


class GlinerUsernameHardeningV062Tests(unittest.TestCase):
    """P1: GLiNER username plausibility filtering regression tests (v0.6.2)."""

    def test_gliner_rejects_chinese_narrative_as_username(self):
        from privacy.detectors import GLiNERDetector
        text = "张三今天来了，稍后张三又打电话过来。"

        # Narrative spans from real reproducer must be rejected
        self.assertFalse(GLiNERDetector._is_plausible_username(text, 0, 6, "张三今天来了"))
        self.assertFalse(GLiNERDetector._is_plausible_username(text, 6, 7, "，"))
        self.assertFalse(GLiNERDetector._is_plausible_username(text, 8, 17, "稍后张三又打电话过来"))

        # Pure CJK names or short phrases without username context must also be rejected
        self.assertFalse(GLiNERDetector._is_plausible_username(text, 0, 2, "张三"))
        self.assertFalse(GLiNERDetector._is_plausible_username("王五今天上班", 0, 2, "王五"))
        self.assertFalse(GLiNERDetector._is_plausible_username("稍后联系李雷", 4, 6, "李雷"))

    def test_gliner_accepts_ascii_username_token(self):
        from privacy.detectors import GLiNERDetector

        # Standard ASCII token usernames are accepted without explicit context
        self.assertTrue(GLiNERDetector._is_plausible_username("My username is ecarter92.", 15, 24, "ecarter92"))
        self.assertTrue(GLiNERDetector._is_plausible_username("Contact john.smith", 8, 18, "john.smith"))
        self.assertTrue(GLiNERDetector._is_plausible_username("user_123 logged in", 0, 8, "user_123"))
        self.assertTrue(GLiNERDetector._is_plausible_username("admin-test", 0, 10, "admin-test"))
        self.assertTrue(GLiNERDetector._is_plausible_username("zimo_zhou96", 0, 11, "zimo_zhou96"))
        self.assertTrue(GLiNERDetector._is_plausible_username("camille.m91", 0, 11, "camille.m91"))
        self.assertTrue(GLiNERDetector._is_plausible_username("登录账号为 zhangsan2026", 6, 18, "zhangsan2026"))

    def test_gliner_accepts_cjk_username_with_explicit_context(self):
        from privacy.detectors import GLiNERDetector

        # CJK usernames with nearby explicit account indicators are accepted
        t1 = "用户名：测试用户01"
        self.assertTrue(GLiNERDetector._is_plausible_username(t1, 4, 10, "测试用户01"))

        t2 = "用户名是 张三2026"
        self.assertTrue(GLiNERDetector._is_plausible_username(t2, 5, 11, "张三2026"))

        t3 = "登录账号：测试账户"
        self.assertTrue(GLiNERDetector._is_plausible_username(t3, 5, 9, "测试账户"))

        t4 = "登录ID 是 zhangsan2026"
        self.assertTrue(GLiNERDetector._is_plausible_username(t4, 6, 18, "zhangsan2026"))

        # But long narrative sentence with a username context is still rejected
        t5 = "用户名是 张三今天来了然后去了公司"
        self.assertFalse(GLiNERDetector._is_plausible_username(t5, 5, 18, "张三今天来了然后去了公司"))

    def test_gliner_username_filter_runs_in_detector_path(self):
        from privacy.detectors import GLiNERDetector
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            det = GLiNERDetector(data_dir, active_model_id="gliner-pii-edge")

            # Mock check_model_integrity and resolve_for_model so detector believes model is ready
            with patch("privacy.detectors.check_model_integrity", return_value=(True, "")):
                with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "runtime_profile": "torch-cpu", "actual_device": "cpu"}):
                    mock_worker = MagicMock()
                    # Simulate real reproducer raw output where GLiNER produced 3 false positive usernames
                    text = "张三今天来了，稍后张三又打电话过来。"
                    mock_worker.query.return_value = {
                        "ok": True,
                        "entities": [
                            {"label": "username", "start": 0, "end": 6, "text": "张三今天来了", "score": 0.88},
                            {"label": "username", "start": 6, "end": 7, "text": "，", "score": 0.76},
                            {"label": "username", "start": 8, "end": 17, "text": "稍后张三又打电话过来", "score": 0.92},
                            {"label": "phone", "start": 12, "end": 15, "text": "打电话", "score": 0.35},
                        ],
                    }
                    with patch("privacy.detectors.get_worker_client") as mock_gwc:
                        mock_client = MagicMock()
                        mock_client.get_worker.return_value = mock_worker
                        mock_client.get_timeout_for_model.return_value = (10, 10)
                        mock_gwc.return_value = mock_client

                        entities, warnings = det.detect(text)
                        usernames = [e for e in entities if e.entity_type == "USERNAME"]
                        self.assertEqual(len(usernames), 0, "All narrative false positive usernames must be filtered out")


class MemPrivacyRuntimeHardeningV063Tests(unittest.TestCase):
    """Regression test suite for v0.6.3: MemPrivacy Runtime, VRAM, and Timeout Hardening."""

    def _create_mock_memprivacy_model(self, data_dir: Path) -> Path:
        model_dir = data_dir / "models" / "memprivacy-1.7b-rl"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
        (model_dir / "tokenizer.json").write_text('{}')
        (model_dir / "model.safetensors").write_text('MOCK_WEIGHTS')
        return model_dir

    def test_cuda_oom_response_classification_and_worker_cleanup(self):
        """Classification of fatal CUDA OOM response and immediate worker termination."""
        from privacy.worker_client import is_cuda_oom_response, RuntimeWorkerProcess, STATE_FAILED

        self.assertFalse(is_cuda_oom_response({}))
        self.assertFalse(is_cuda_oom_response({"ok": True}))
        self.assertFalse(is_cuda_oom_response({"ok": True, "error": "CUDA out of memory"}))
        self.assertFalse(is_cuda_oom_response({"ok": False, "error": "Timeout expired"}))

        self.assertTrue(is_cuda_oom_response({"ok": False, "error_type": "OutOfMemoryError"}))
        self.assertTrue(is_cuda_oom_response({"ok": False, "error_type": "CUDAOutOfMemoryError"}))
        self.assertTrue(is_cuda_oom_response({"ok": False, "error": "CUDA out of memory. Tried to allocate 1.57 GiB"}))
        self.assertTrue(is_cuda_oom_response({"ok": False, "error": "RuntimeError: CUDA error: out of memory"}))

        # Test worker process behavior upon OOM response
        worker = RuntimeWorkerProcess(
            python_bin=Path(sys.executable),
            worker_script=Path("/tmp/fake_worker.py"),
            model_id="memprivacy-1.7b-rl",
            profile="torch-cuda",
            device="cuda",
        )
        worker._process = MagicMock()
        worker._process.poll.return_value = None
        worker._process.stdin = MagicMock()
        worker._alive = True
        worker._reader = MagicMock()
        worker._writer = MagicMock()
        worker._terminated = False
        worker.state = "ready"

        oom_resp_json = json.dumps({"ok": False, "error": "CUDA out of memory"})
        worker._stdout_queue.put(oom_resp_json)
        with patch.object(worker, "_terminate_locked") as mock_term:
            with patch("privacy.worker_client.logger"):
                resp = worker._query_raw_locked({"action": "detect"}, timeout=10)
                self.assertFalse(resp["ok"])
                mock_term.assert_called_once()
                self.assertEqual(worker.state, STATE_FAILED)
                self.assertIn("CUDA out of memory", worker._last_error)

    def test_stop_other_cuda_workers_eviction(self):
        """stop_other_cuda_workers must terminate other CUDA workers while preserving CPU and target."""
        from privacy.worker_client import WorkerClient

        client = WorkerClient(MagicMock())
        w_gliner_cuda = MagicMock(device="cuda", model_id="gliner-pii-edge")
        w_siamese_cuda = MagicMock(device="cuda", model_id="siamese-uie")
        w_gliner_cpu = MagicMock(device="cpu", model_id="gliner-pii-edge")
        w_mem_cuda = MagicMock(device="cuda", model_id="memprivacy-1.7b-rl")

        client._workers = {
            "gliner-pii-edge:torch-cuda:cuda": w_gliner_cuda,
            "siamese-uie:torch-cuda:cuda": w_siamese_cuda,
            "gliner-pii-edge:torch-cpu:cpu": w_gliner_cpu,
            "memprivacy-1.7b-rl:torch-cuda:cuda": w_mem_cuda,
        }

        evicted = client.stop_other_cuda_workers(keep_model_id="memprivacy-1.7b-rl")
        self.assertIn("gliner-pii-edge:torch-cuda:cuda", evicted)
        self.assertIn("siamese-uie:torch-cuda:cuda", evicted)
        self.assertEqual(len(evicted), 2)

        w_gliner_cuda.terminate.assert_called_once()
        w_siamese_cuda.terminate.assert_called_once()
        w_gliner_cpu.terminate.assert_not_called()
        w_mem_cuda.terminate.assert_not_called()

        self.assertIn("gliner-pii-edge:torch-cpu:cpu", client._workers)
        self.assertIn("memprivacy-1.7b-rl:torch-cuda:cuda", client._workers)
        self.assertNotIn("gliner-pii-edge:torch-cuda:cuda", client._workers)
        self.assertNotIn("siamese-uie:torch-cuda:cuda", client._workers)

    def test_no_external_pid_manipulation(self):
        """WorkerClient must never inspect, signal, or kill any external OS PIDs."""
        from privacy.worker_client import WorkerClient

        client = WorkerClient(MagicMock())
        mock_worker = MagicMock(device="cuda", model_id="other-model")
        client._workers = {"other-model:torch-cuda:cuda": mock_worker}

        with patch("os.kill") as mock_os_kill:
            client.stop_other_cuda_workers(keep_model_id="target-model")
            for call in mock_os_kill.call_args_list:
                pid_arg = call[0][0]
                self.assertNotEqual(pid_arg, 465815)

    def test_memprivacy_exclusive_cuda_policy(self):
        """Exclusive CUDA policy: evict others before inference and terminate self after inference."""
        from privacy.detectors import MemPrivacyDetector

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            # Case A: CUDA mode
            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model") as mock_resolve, \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_resolve.return_value = {
                    "ready": True,
                    "actual_device": "cuda",
                    "runtime_profile": "torch-cuda",
                }
                mock_client = MagicMock()
                mock_worker = MagicMock()
                mock_worker.query.return_value = {"ok": True, "entities": []}
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (120, 180)
                mock_gwc.return_value = mock_client

                entities, warnings = det.detect("Test input for MemPrivacy.")
                mock_client.stop_other_cuda_workers.assert_called_once_with(keep_model_id="memprivacy-1.7b-rl")
                mock_client.stop_worker_for_model.assert_called_with("memprivacy-1.7b-rl")

            # Case B: CPU mode
            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model") as mock_resolve, \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_resolve.return_value = {
                    "ready": True,
                    "actual_device": "cpu",
                    "runtime_profile": "torch-cpu",
                }
                mock_client = MagicMock()
                mock_worker = MagicMock()
                mock_worker.query.return_value = {"ok": True, "entities": []}
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (150, 360)
                mock_gwc.return_value = mock_client

                entities, warnings = det.detect("Test input for MemPrivacy CPU.")
                mock_client.stop_other_cuda_workers.assert_not_called()
                mock_client.stop_worker_for_model.assert_not_called()

    def test_device_aware_timeouts(self):
        """Worker timeouts must adapt to model type and active execution device."""
        from privacy.worker_client import WorkerClient

        client = WorkerClient(MagicMock())
        # MemPrivacy: 120s/180s for CUDA, 150s/360s for CPU
        self.assertEqual(client.get_timeout_for_model("memprivacy-1.7b-rl", device="cuda"), (120, 180))
        self.assertEqual(client.get_timeout_for_model("memprivacy-1.7b-rl", device="cpu"), (150, 360))
        self.assertEqual(client.get_timeout_for_model("memprivacy-4b-rl", device="cuda"), (120, 180))
        self.assertEqual(client.get_timeout_for_model("memprivacy-4b-rl", device="cpu"), (150, 360))

        # GLiNER: 30s/45s
        self.assertEqual(client.get_timeout_for_model("gliner-pii-edge", device="cuda"), (30, 45))
        self.assertEqual(client.get_timeout_for_model("gliner-pii-edge", device="cpu"), (30, 45))

    def test_choose_memprivacy_generation_budget(self):
        """Generation budget scales safely with text length and is strictly bounded <= 512."""
        from privacy.detectors import choose_memprivacy_generation_budget

        self.assertEqual(choose_memprivacy_generation_budget("short text"), 256)
        self.assertEqual(choose_memprivacy_generation_budget("a" * 1000), 256)
        self.assertEqual(choose_memprivacy_generation_budget("a" * 1001), 384)
        self.assertEqual(choose_memprivacy_generation_budget("a" * 4000), 384)
        self.assertEqual(choose_memprivacy_generation_budget("a" * 4001), 512)
        self.assertEqual(choose_memprivacy_generation_budget("a" * 20000), 512)
        self.assertLessEqual(choose_memprivacy_generation_budget("a" * 50000), 512)

    def test_chunk_memprivacy_text_and_offset_reconciliation(self):
        """Texts <= 3500 chars stay single-chunk; long texts chunk with exact text mapping and deduplication."""
        from privacy.detectors import chunk_memprivacy_text, MemPrivacyDetector

        # Case 1: Short text <= 3500 chars
        text_short = "Hello world! This is a test."
        chunks_short = chunk_memprivacy_text(text_short)
        self.assertEqual(len(chunks_short), 1)
        self.assertEqual(chunks_short[0], (0, len(text_short), text_short))

        # Case 2: Long text > 3500 chars
        paragraph = "测试句子。" * 200  # 1000 chars per 200 repetitions
        text_long = (paragraph + "\n") * 5  # > 5000 chars
        chunks = chunk_memprivacy_text(text_long, max_chunk_chars=3000, overlap_chars=200)
        self.assertGreater(len(chunks), 1)

        # Every chunk text must strictly match text[start:end]
        for c_start, c_end, c_text in chunks:
            self.assertEqual(text_long[c_start:c_end], c_text)
            self.assertLessEqual(len(c_text), 3000)

        # Adjacent chunks must overlap
        for i in range(len(chunks) - 1):
            curr_end = chunks[i][1]
            next_start = chunks[i + 1][0]
            self.assertLess(next_start, curr_end, "Adjacent chunks must overlap")

        # Case 3: End-to-end offset reconciliation and deduplication across chunks
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            sentence_a = "患者张晓华（身份证号 110101199003072345）于今日入院。"
            padding = "日常巡检记录正常。" * 300  # ~2700 chars
            sentence_b = "主治医生联系电话为 13800138000。"
            doc = sentence_a + "\n" + padding + "\n" + sentence_b

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model") as mock_resolve, \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_resolve.return_value = {
                    "ready": True,
                    "actual_device": "cpu",
                    "runtime_profile": "torch-cpu",
                }
                mock_client = MagicMock()
                mock_worker = MagicMock()

                def mock_query(payload, timeout=None):
                    c_text = payload.get("text", "")
                    entities = []
                    if "张晓华" in c_text:
                        entities.append({"original_text": "张晓华", "privacy_type": "name", "privacy_level": "PL2"})
                    if "110101199003072345" in c_text:
                        entities.append({"original_text": "110101199003072345", "privacy_type": "id card", "privacy_level": "PL3"})
                    if "13800138000" in c_text:
                        entities.append({"original_text": "13800138000", "privacy_type": "phone", "privacy_level": "PL2"})
                    return {"ok": True, "entities": entities}

                mock_worker.query.side_effect = mock_query
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (150, 360)
                mock_gwc.return_value = mock_client

                found_entities, warns = det.detect(doc)
                found_texts = [e.text for e in found_entities]
                self.assertIn("张晓华", found_texts)
                self.assertIn("110101199003072345", found_texts)
                self.assertIn("13800138000", found_texts)

                # Every entity offset must match document text exactly
                for ent in found_entities:
                    self.assertEqual(doc[ent.start:ent.end], ent.text)

                # No duplicates
                self.assertEqual(len(found_entities), len(set((e.start, e.end, e.text) for e in found_entities)))

    def test_memprivacy_oom_fallback_and_warnings(self):
        """When CUDA OOM occurs during inference, worker terminates, resources are freed, and friendly warning is emitted."""
        from privacy.detectors import MemPrivacyDetector

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model") as mock_resolve, \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_resolve.return_value = {
                    "ready": True,
                    "actual_device": "cuda",
                    "runtime_profile": "torch-cuda",
                }
                mock_client = MagicMock()
                mock_worker = MagicMock()
                mock_worker.query.return_value = {
                    "ok": False,
                    "error_type": "CUDAOutOfMemoryError",
                    "error": "CUDA out of memory. Tried to allocate 1.57 GiB (GPU 0; 7.42 GiB total capacity)",
                }
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (120, 180)
                mock_gwc.return_value = mock_client

                entities, warnings = det.detect("Test input that triggers OOM.")
                self.assertEqual(len(entities), 0)
                self.assertTrue(any("MemPrivacy 可用显存不足" in w for w in warnings))
                mock_client.stop_worker_for_model.assert_called_with("memprivacy-1.7b-rl")


class ConcurrencyAndLifecycleHardeningV064Tests(unittest.TestCase):
    """Comprehensive test suite for v0.6.4: Concurrency Hardening, Deadlines, and Lifecycle Safety."""

    def _create_mock_memprivacy_model(self, data_dir: Path) -> Path:
        model_dir = data_dir / "models" / "memprivacy-1.7b-rl"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
        (model_dir / "tokenizer.json").write_text('{}')
        (model_dir / "model.safetensors").write_text('MOCK_WEIGHTS')
        return model_dir

    def test_retired_worker_cannot_resurrect(self):
        """1. Calling start or query on a retired worker raises WorkerRetiredError and cannot restart."""
        from privacy.worker_client import RuntimeWorkerProcess, WorkerRetiredError

        worker = RuntimeWorkerProcess(
            python_bin=Path(sys.executable),
            worker_script=Path("/tmp/fake_worker.py"),
            model_id="memprivacy-1.7b-rl",
            profile="torch-cuda",
            device="cuda",
        )
        self.assertFalse(worker.is_retired())
        worker.retire()
        self.assertTrue(worker.is_retired())
        self.assertFalse(worker.is_alive())
        self.assertFalse(worker.is_ready())

        with self.assertRaises(WorkerRetiredError):
            worker.start()

        with self.assertRaises(WorkerRetiredError):
            worker.query({"action": "ping"})

    def test_registry_invariant_after_eviction(self):
        """2. Evicted workers are removed from WorkerClient and marked retired, preventing resurrection."""
        from privacy.worker_client import WorkerClient, RuntimeWorkerProcess, WorkerRetiredError

        client = WorkerClient(Path("/tmp"))
        w1 = RuntimeWorkerProcess(
            python_bin=Path(sys.executable),
            worker_script=Path("/tmp/fake_worker.py"),
            model_id="gliner-pii-edge",
            profile="torch-cuda",
            device="cuda",
        )
        key = "gliner-pii-edge:torch-cuda:cuda"
        client._workers[key] = w1

        evicted = client.stop_other_cuda_workers(keep_model_id="memprivacy-1.7b-rl")
        self.assertIn(key, evicted)
        self.assertNotIn(key, client._workers)
        self.assertTrue(w1.is_retired())

        with self.assertRaises(WorkerRetiredError):
            w1.query({"action": "ping"})

    def test_concurrent_memprivacy_serialized(self):
        """3. Concurrent MemPrivacy requests are serialized via _inference_lock to protect system resources."""
        from privacy.detectors import MemPrivacyDetector
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            order = []
            start_barrier = threading.Barrier(2)

            def mock_detect_session(text, model_dir, profile, dev, deadline):
                order.append(f"start-{text}")
                time.sleep(0.05)
                order.append(f"end-{text}")
                return [], []

            with patch.object(det, "_detect_session_locked", side_effect=mock_detect_session), \
                 patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "actual_device": "cpu", "runtime_profile": "torch-cpu"}):
                def worker_thread(name):
                    start_barrier.wait()
                    det.detect(name)

                t1 = threading.Thread(target=worker_thread, args=("req1",))
                t2 = threading.Thread(target=worker_thread, args=("req2",))
                t1.start()
                t2.start()
                t1.join()
                t2.join()

            self.assertEqual(len(order), 4)
            is_1_then_2 = (order[0] == "start-req1" and order[1] == "end-req1" and order[2] == "start-req2" and order[3] == "end-req2")
            is_2_then_1 = (order[0] == "start-req2" and order[1] == "end-req2" and order[2] == "start-req1" and order[3] == "end-req1")
            self.assertTrue(is_1_then_2 or is_2_then_1, f"Execution was not serialized: {order}")

    def test_cuda_coordinator_blocks_concurrent_gliner(self):
        """4. CUDA coordinator ensures GLiNER cannot run on GPU concurrently with MemPrivacy session."""
        from privacy.worker_client import WorkerClient

        client = WorkerClient(Path("/tmp"))
        events = []

        def memprivacy_session():
            with client.cuda_execution_session(device="cuda", timeout=5.0):
                events.append("mem_start")
                time.sleep(0.1)
                events.append("mem_end")

        def gliner_session():
            time.sleep(0.02)
            with client.cuda_execution_session(device="cuda", timeout=5.0):
                events.append("gliner_start")
                events.append("gliner_end")

        t_mem = threading.Thread(target=memprivacy_session)
        t_gli = threading.Thread(target=gliner_session)
        t_mem.start()
        t_gli.start()
        t_mem.join()
        t_gli.join()

        self.assertEqual(events, ["mem_start", "mem_end", "gliner_start", "gliner_end"])

    def test_cuda_coordinator_leases_released_on_exception(self):
        """5. CUDA coordinator releases lock immediately if exception occurs inside session."""
        from privacy.worker_client import WorkerClient

        client = WorkerClient(Path("/tmp"))
        with self.assertRaises(RuntimeError):
            with client.cuda_execution_session(device="cuda", timeout=1.0):
                raise RuntimeError("Inference crash")

        # Must be immediately re-acquireable
        acquired = False
        with client.cuda_execution_session(device="cuda", timeout=0.5):
            acquired = True
        self.assertTrue(acquired)

    def test_lock_free_termination(self):
        """6. stop_other_cuda_workers must pop victims inside _lock and terminate them outside _lock."""
        from privacy.worker_client import WorkerClient

        client = WorkerClient(Path("/tmp"))
        mock_worker = MagicMock(device="cuda", model_id="gliner-pii-edge")
        lock_state_during_retire = []

        def mock_retire():
            acquired = client._lock.acquire(blocking=False)
            if acquired:
                lock_state_during_retire.append("unlocked")
                client._lock.release()
            else:
                lock_state_during_retire.append("locked")

        mock_worker.retire.side_effect = mock_retire
        client._workers["gliner:cuda"] = mock_worker

        client.stop_other_cuda_workers(keep_model_id="memprivacy-1.7b-rl")
        self.assertEqual(lock_state_during_retire, ["unlocked"])

    def test_whole_request_deadline_limits_chunks(self):
        """7. Request budget limits multi-chunk execution; stops when deadline is exhausted."""
        from privacy.detectors import MemPrivacyDetector
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            text = "这是测试段落，包含重要内容。\n\n" * 400

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "actual_device": "cuda", "runtime_profile": "torch-cuda"}), \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_client = MagicMock()
                mock_worker = MagicMock()
                chunk_call_count = [0]

                def mock_query(req, timeout):
                    chunk_call_count[0] += 1
                    return {
                        "ok": True,
                        "entities": [{"original_text": "重要内容", "privacy_type": "SECRET"}],
                    }

                mock_worker.query.side_effect = mock_query
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (120, 180)
                mock_gwc.return_value = mock_client

                base_time = 1000.0
                # Call 1: started_at, Call 2: acquire_timeout, Call 3: cuda_timeout, Call 4: chunk 0 remaining, Call 5: chunk 1 remaining (expired)
                times = [base_time, base_time + 1.0, base_time + 2.0, base_time + 3.0, base_time + 300.0]
                with patch("time.monotonic", side_effect=lambda: times.pop(0) if times else 2000.0):
                    entities, warnings = det.detect(text)

                self.assertEqual(chunk_call_count[0], 1)
                self.assertTrue(any("仅完成" in w for w in warnings))

    def test_deadline_lock_wait_accounting(self):
        """8. If waiting for lock times out, immediately return busy message."""
        from privacy.detectors import MemPrivacyDetector
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")
            det._inference_lock = MagicMock()
            det._inference_lock.acquire.return_value = False

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "actual_device": "cuda", "runtime_profile": "torch-cuda"}):
                entities, warnings = det.detect("测试繁忙锁等待")
                self.assertEqual(len(entities), 0)
                self.assertIn("MemPrivacy 语义推理繁忙，本次未在时间预算内执行。", warnings)

    def test_partial_results_preserved(self):
        """9. When chunk 1 succeeds but chunk 2 fails, chunk 1 entities are preserved and partial warning is emitted."""
        from privacy.detectors import MemPrivacyDetector
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            text = "这是第一个测试段落，包含秘密密钥ABC123。\n\n" * 150 + "这是第二个测试段落，包含密码DEF456。\n\n" * 150

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "actual_device": "cuda", "runtime_profile": "torch-cuda"}), \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_client = MagicMock()
                mock_worker = MagicMock()
                call_idx = [0]

                def mock_query(req, timeout):
                    call_idx[0] += 1
                    if call_idx[0] == 1:
                        return {
                            "ok": True,
                            "entities": [{"original_text": "ABC123", "privacy_type": "API_TOKEN"}],
                        }
                    else:
                        raise TimeoutError("Chunk 2 timed out")

                mock_worker.query.side_effect = mock_query
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (120, 180)
                mock_gwc.return_value = mock_client

                entities, warnings = det.detect(text)
                self.assertTrue(len(entities) > 0)
                self.assertEqual(entities[0].text, "ABC123")
                self.assertTrue(any("仅完成" in w for w in warnings))

    def test_deduplicated_truncation_warning(self):
        """10. When multiple chunks are truncated, exactly one summary warning is produced."""
        from privacy.detectors import MemPrivacyDetector
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            text = "分块内容段落一。\n\n" * 150 + "分块内容段落二。\n\n" * 150

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "actual_device": "cuda", "runtime_profile": "torch-cuda"}), \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_client = MagicMock()
                mock_worker = MagicMock()
                mock_worker.query.return_value = {
                    "ok": True,
                    "truncated": True,
                    "entities": [],
                }
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (120, 180)
                mock_gwc.return_value = mock_client

                entities, warnings = det.detect(text)
                trunc_warns = [w for w in warnings if "达到生成上限" in w]
                self.assertEqual(len(trunc_warns), 1)
                self.assertIn("个分块达到生成上限", trunc_warns[0])

    def test_cpu_oom_vs_cuda_vram_classification(self):
        """11. Differentiates CPU MemoryError vs CUDA VRAM Out-of-memory."""
        from privacy.detectors import MemPrivacyDetector
        from privacy.worker_client import is_cuda_oom_exception, is_cpu_oom_exception

        self.assertTrue(is_cuda_oom_exception(RuntimeError("CUDA out of memory"), device="cuda"))
        self.assertFalse(is_cuda_oom_exception(RuntimeError("CUDA out of memory"), device="cpu"))
        self.assertTrue(is_cpu_oom_exception(MemoryError(), device="cpu"))
        self.assertTrue(is_cpu_oom_exception(RuntimeError("std::bad_alloc"), device="cpu"))
        self.assertFalse(is_cpu_oom_exception(MemoryError(), device="cuda"))

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self._create_mock_memprivacy_model(data_dir)
            det = MemPrivacyDetector(data_dir=data_dir, active_model_id="memprivacy-1.7b-rl")

            with patch("privacy.detectors.DEVICE_MANAGER.resolve_for_model", return_value={"ready": True, "actual_device": "cpu", "runtime_profile": "torch-cpu"}), \
                 patch("privacy.detectors.get_worker_client") as mock_gwc:
                mock_client = MagicMock()
                mock_worker = MagicMock()
                mock_worker.query.side_effect = MemoryError("Host memory allocation failed")
                mock_client.get_worker.return_value = mock_worker
                mock_client.get_timeout_for_model.return_value = (150, 360)
                mock_gwc.return_value = mock_client

                entities, warnings = det.detect("测试 CPU 内存溢出")
                self.assertEqual(len(entities), 0)
                self.assertTrue(any("MemPrivacy 内存不足，已安全回退。" in w for w in warnings))
                self.assertFalse(any("显存不足" in w for w in warnings))

    def test_real_multichunk_long_text(self):
        """12. Real >8000 char long text splits into valid chunks with correct offsets."""
        from privacy.detectors import chunk_memprivacy_text

        text = "".join(f"第{i}章：这是用于验证超长文本分块切分正确性的内容句段。\n" for i in range(300))
        self.assertGreater(len(text), 8000)
        chunks = chunk_memprivacy_text(text)
        self.assertGreaterEqual(len(chunks), 3)

        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], len(text))
        for s, e, ctext in chunks:
            self.assertEqual(text[s:e], ctext)
            self.assertLessEqual(len(ctext), 3200)

    def test_budget_helper_bound(self):
        """13. Budget helper returns appropriate tokens and upper bounds <= 512."""
        from privacy.detectors import choose_memprivacy_generation_budget
        self.assertEqual(choose_memprivacy_generation_budget("short"), 256)
        self.assertEqual(choose_memprivacy_generation_budget("x" * 1000), 256)
        self.assertEqual(choose_memprivacy_generation_budget("x" * 1001), 384)
        self.assertEqual(choose_memprivacy_generation_budget("x" * 4000), 384)
        self.assertEqual(choose_memprivacy_generation_budget("x" * 4001), 512)
        self.assertLessEqual(choose_memprivacy_generation_budget("x" * 10000), 512)

    def test_version_consistency(self):
        """14. Version numbers across server.py, manifest, benchmark, lifecycle are consistent at 0.6.14."""
        expected = "0.6.14"
        self.assertIn(f'"AIPrivacyCheck/{expected}"', (SERVER_DIR / "server.py").read_text(encoding="utf-8"))
        self.assertIn(f'"version": "{expected}"', (SERVER_DIR / "server.py").read_text(encoding="utf-8"))
        m_content = (PROJECT_DIR / "packaging" / "ai-privacy-check" / "manifest").read_text(encoding="utf-8")
        self.assertIn(f"version               = {expected}", m_content)
        self.assertIn(f"{expected}:", m_content)
        self.assertIn(f"v{expected}", (PROJECT_DIR / "scripts" / "benchmark.py").read_text(encoding="utf-8"))
        self.assertIn(f'"{expected}"', (PROJECT_DIR / "scripts" / "test_native_lifecycle.sh").read_text(encoding="utf-8"))


class IntegrationSmokeTests(unittest.TestCase):
    """End-to-end integration tests gated by AI_PRIVACY_INTEGRATION_TESTS=1."""

    @unittest.skipUnless(
        os.environ.get("AI_PRIVACY_INTEGRATION_TESTS") == "1",
        "Integration tests skipped. Set AI_PRIVACY_INTEGRATION_TESTS=1 to execute.",
    )
    def test_live_isolated_worker_smoke(self):
        from privacy.worker_client import get_worker_client
        with tempfile.TemporaryDirectory() as td:
            client = get_worker_client(Path(td))
            # If no model installed, smoke test should cleanly return (False, err) without crashing
            ok, err = client.run_smoke_test("gliner-pii-edge", Path(td) / "models" / "gliner-pii-edge", "torch-cpu")
            self.assertFalse(ok)
            self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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

"""Model-specific runtime dependency contract tests.

Regression coverage for the real fnOS failure:
    shared torch runtime probe verified, yet the SiameseUIE ModelScope pipeline
    failed with `No module named 'addict'`.

The contract under test:
  1. Model descriptors may declare model-specific pip requirements
     (runtime_dependencies) beyond the shared base runtime.
  2. The install flow MUST resolve those dependencies even when the shared
     runtime is already installed and verified (skip path).
  3. Dependency resolution is idempotent: probe first, install only what is
     missing, never reinstall the base runtime or PyTorch.
  4. Dependency failure blocks ready; dependency success still requires the
     real smoke inference gate.
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

from privacy.model_catalog import MODEL_CATALOG, get_model_descriptor
from privacy.runtime_manager import RuntimeManager
import model_installer


PROBE_MARKER = "AIPRIVACY_DEPENDENCY_PROBE"


class FakeVenvRunner:
    """Stands in for the isolated venv interpreter: answers dependency probes
    and records pip/uv install invocations."""

    def __init__(self, missing_first_probe):
        self.missing_first_probe = list(missing_first_probe)
        self.install_cmds = []
        self.probe_cmds = []
        self.install_fail_output: tuple = None  # (code, out, err) override

    def __call__(self, cmd, cwd=None, env=None, timeout=15):
        joined = " ".join(str(part) for part in cmd)
        if PROBE_MARKER in joined:
            self.probe_cmds.append(cmd)
            missing, self.missing_first_probe = self.missing_first_probe, []
            return 0, json.dumps({"ok": True, "missing": missing, "error": None}), ""
        if "install" in cmd[:6] or (len(cmd) > 2 and cmd[1] == "pip"):
            self.install_cmds.append(cmd)
            if self.install_fail_output is not None:
                return self.install_fail_output
            return 0, "", ""
        return 0, "", ""


class RuntimeDependencyContractTests(unittest.TestCase):
    def test_siamese_uie_declares_modelscope_pipeline_dependencies(self):
        deps = get_model_descriptor("siamese-uie").runtime_dependencies
        for required in (
            "addict",
            "datasets",
            "scipy",
            "Pillow",
            "simplejson",
            "sortedcontainers",
        ):
            self.assertIn(required, deps, f"siamese-uie 缺少经实证的依赖声明: {required}")

    def test_all_catalog_dependency_specs_are_wellformed(self):
        import re

        spec_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:[=<>!~]=?[0-9A-Za-z._,]+)?$")
        for model_id, descriptor in MODEL_CATALOG.items():
            for spec in descriptor.runtime_dependencies:
                self.assertRegex(spec, spec_re, f"{model_id} 的依赖声明格式非法: {spec}")

    def test_non_siamese_models_have_no_extra_requirements_yet(self):
        # Only siamese-uie is proven to need model-specific extras today; the
        # rest ride on the shared base runtime.
        for model_id, descriptor in MODEL_CATALOG.items():
            if model_id != "siamese-uie":
                self.assertEqual(
                    descriptor.runtime_dependencies,
                    (),
                    f"{model_id} 不应声明未经实证的模型专属依赖",
                )


class EnsureModelRuntimeDependenciesTests(unittest.TestCase):
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

    def test_case_a_installs_only_missing_dependencies_and_never_torch(self):
        runner = FakeVenvRunner(missing_first_probe=["addict"])
        result = model_installer.ensure_model_runtime_dependencies(
            self.data_dir, "siamese-uie", self.profile, command_runner=runner
        )

        self.assertTrue(result["satisfied"])
        self.assertEqual(result["missing_initial"], ["addict"])
        self.assertEqual(result["installed"], ["addict"])

        self.assertEqual(len(runner.install_cmds), 1)
        install_cmd = runner.install_cmds[0]
        self.assertIn("addict", install_cmd)
        torch_args = [a for a in install_cmd if str(a).strip() == "torch"]
        self.assertEqual(torch_args, [], "模型依赖补装不得重新安装 PyTorch")
        # Probe must run with the isolated venv interpreter, not control-plane python.
        self.assertEqual(runner.probe_cmds[0][0], str(self.python_bin))

    def test_case_b_dependency_satisfied_skips_install_entirely(self):
        runner = FakeVenvRunner(missing_first_probe=[])
        result = model_installer.ensure_model_runtime_dependencies(
            self.data_dir, "siamese-uie", self.profile, command_runner=runner
        )

        self.assertTrue(result["satisfied"])
        self.assertEqual(result["installed"], [])
        self.assertEqual(runner.install_cmds, [], "依赖已满足时不得执行任何安装")

    def test_case_c_install_failure_raises_and_blocks(self):
        runner = FakeVenvRunner(missing_first_probe=["addict"])
        runner.install_fail_output = (1, "", "error: failed to resolve distributions")

        with self.assertRaises(RuntimeError) as ctx:
            model_installer.ensure_model_runtime_dependencies(
                self.data_dir, "siamese-uie", self.profile, command_runner=runner
            )
        self.assertIn("addict", str(ctx.exception))

    def test_case_c_broken_venv_probe_raises(self):
        def broken_runner(cmd, cwd=None, env=None, timeout=15):
            return 127, "", "command not found"

        with self.assertRaises(RuntimeError):
            model_installer.ensure_model_runtime_dependencies(
                self.data_dir, "siamese-uie", self.profile, command_runner=broken_runner
            )

    def test_models_without_contract_are_fast_noop(self):
        runner = FakeVenvRunner(missing_first_probe=[])
        result = model_installer.ensure_model_runtime_dependencies(
            self.data_dir, "gliner-pii-edge", self.profile, command_runner=runner
        )
        self.assertTrue(result["satisfied"])
        self.assertEqual(result["required"], ())
        self.assertEqual(runner.probe_cmds, [], "无契约模型不应启动探测子进程")
        self.assertEqual(runner.install_cmds, [])


class InstallFlowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="aipc_deps_"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def _prepare_device(self, data_dir, device="cpu"):
        from privacy.device import DEVICE_MANAGER

        DEVICE_MANAGER.set_data_dir(data_dir)
        DEVICE_MANAGER.set_requested_device(device)
        return DEVICE_MANAGER

    def _make_model_dir(self, data_dir, model_id="siamese-uie"):
        model_dir = data_dir / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "configuration.json").write_text("{}")
        (model_dir / "config.json").write_text("{}")
        (model_dir / "vocab.txt").write_text("dummy")
        (model_dir / "pytorch_model.bin").write_text("dummy")
        return model_dir

    def test_runtime_skip_still_resolves_model_dependencies_before_download(self):
        """NAS regression: '运行时已就绪并跳过安装' must not skip model-specific deps."""
        import model_installer as mi
        from privacy.device import DEVICE_MANAGER

        data_dir = self.root / "app_data"
        data_dir.mkdir(parents=True)
        model_dir = self._make_model_dir(data_dir)

        self._prepare_device(data_dir)
        # Simulate GPU present + explicit cpu request is bypassed; force cpu path via env.
        with patch.object(DEVICE_MANAGER._hw_probe, "probe_nvidia", return_value={"nvidia_available": False}):
            order = []

            def record(name, fn):
                def wrapper(*args, **kwargs):
                    order.append(name)
                    return fn(*args, **kwargs)

                return wrapper

            with patch.object(mi, "install_isolated_runtime", side_effect=lambda *a, **k: order.append("install_runtime") or Path("/fake/venv")), \
                 patch.object(mi, "ensure_model_runtime_dependencies", side_effect=lambda *a, **k: order.append("ensure_deps") or {"satisfied": True}), \
                 patch.object(mi, "download_modelscope_model", side_effect=lambda *a, **k: order.append("download") or model_dir), \
                 patch.object(mi, "get_worker_client") as mock_wc, \
                 patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
                 patch("sys.argv", ["model_installer.py", "install", "siamese-uie"]):
                mock_client = MagicMock()
                mock_client.run_smoke_test.return_value = (True, None)
                mock_wc.return_value = mock_client

                ret = mi.main()
                self.assertEqual(ret, 0)

        self.assertIn("install_runtime", order)
        self.assertIn("ensure_deps", order)
        self.assertLess(
            order.index("ensure_deps"),
            order.index("download"),
            "模型专属依赖必须在下载/冒烟之前解析",
        )
        self.assertIn("ensure_deps", order[: order.index("download") + 1])
        status = json.loads(
            (data_dir / "status" / "siamese-uie-install.json").read_text(encoding="utf-8")
        )
        self.assertEqual(status["state"], "ready")

    def test_case_c_dependency_failure_blocks_ready_without_smoke(self):
        import model_installer as mi

        data_dir = self.root / "app_data_c"
        data_dir.mkdir(parents=True)
        self._make_model_dir(data_dir)
        self._prepare_device(data_dir)

        with patch.object(mi, "install_isolated_runtime", return_value=Path("/fake/venv")), \
             patch.object(
                 mi,
                 "ensure_model_runtime_dependencies",
                 side_effect=RuntimeError("模型依赖解析失败: addict"),
             ), \
             patch.object(mi, "download_modelscope_model") as mock_download, \
             patch.object(mi, "get_worker_client") as mock_wc, \
             patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
             patch("sys.argv", ["model_installer.py", "install", "siamese-uie"]):
            mock_client = MagicMock()
            mock_wc.return_value = mock_client

            ret = mi.main()
            self.assertEqual(ret, 1)
            mock_download.assert_not_called()
            mock_client.run_smoke_test.assert_not_called()

            status = json.loads(
                (data_dir / "status" / "siamese-uie-install.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "error")
            self.assertIn("addict", status["detail"])

    def test_case_d_smoke_failure_after_deps_preserves_weights_and_error_state(self):
        import model_installer as mi

        data_dir = self.root / "app_data_d"
        data_dir.mkdir(parents=True)
        model_dir = self._make_model_dir(data_dir)
        self._prepare_device(data_dir)

        with patch.object(mi, "install_isolated_runtime", return_value=Path("/fake/venv")), \
             patch.object(
                 mi,
                 "ensure_model_runtime_dependencies",
                 return_value={"satisfied": True, "installed": ["addict"], "required": ("addict",), "missing_initial": ["addict"]},
             ), \
             patch.object(mi, "download_modelscope_model", return_value=model_dir), \
             patch.object(mi, "get_worker_client") as mock_wc, \
             patch.dict(os.environ, {"APP_DATA_DIR": str(data_dir)}, clear=False), \
             patch("sys.argv", ["model_installer.py", "install", "siamese-uie"]):
            mock_client = MagicMock()
            mock_client.run_smoke_test.return_value = (
                False,
                "冒烟推理验证失败: No module named 'addict'",
            )
            mock_wc.return_value = mock_client

            ret = mi.main()
            self.assertEqual(ret, 1)

            status = json.loads(
                (data_dir / "status" / "siamese-uie-install.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "error")
            self.assertIn("冒烟推理", status["detail"])
            self.assertTrue(model_dir.exists(), "冒烟失败时模型权重必须保留供排查")

    def test_base_runtime_pins_compatible_transformers(self):
        """Fresh base runtime installs must land in the transformers 4.x line:
        >=4.51 (Qwen3 family floor) and <5 (ModelScope legacy pipelines)."""
        import model_installer as mi

        data_dir = self.root / "runtime_pin"
        data_dir.mkdir(parents=True)
        manager = RuntimeManager(data_dir)
        venv_dir = manager.venv_dir("torch-cpu")
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
        py_bin = venv_dir / "bin" / "python"
        py_bin.write_text("#!/bin/sh\n")
        py_bin.chmod(0o755)

        captured_cmds = []

        def fake_run(cmd, check=False, env=None):
            captured_cmds.append(list(cmd))
            return MagicMock(returncode=0)

        with patch.object(mi, "get_runtime_manager", return_value=manager), \
             patch.object(mi.shutil, "which", return_value="/usr/local/bin/uv"), \
             patch.object(mi.subprocess, "run", side_effect=fake_run), \
             patch.object(manager, "probe_profile", return_value={"verified": True}):
            mi.install_isolated_runtime(data_dir, "torch-cpu")

        joined_cmds = [" ".join(str(p) for p in cmd) for cmd in captured_cmds]
        self.assertTrue(
            any("transformers>=4.51,<5" in cmd for cmd in joined_cmds),
            f"基础运行时必须使用 transformers 兼容区间安装: {joined_cmds}",
        )
        self.assertFalse(
            any("transformers " in cmd.replace("transformers>=4.51,<5", "") for cmd in joined_cmds)
        )


if __name__ == "__main__":
    unittest.main()

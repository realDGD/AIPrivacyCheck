"""Tests for Selective Model Downloader (v0.6.6).

Verifies:
1. Forbidden models (MemPrivacy, AIguard, Qwen, RANER) cannot be downloaded.
2. If model is missing and download_missing=False, download is refused (MODEL NOT INSTALLED).
3. File categorization strictly isolates required PyTorch inference files and rejects ONNX/docs.
4. Existing complete model files are reused and not re-downloaded.
"""

from pathlib import Path
import shutil
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys_path = PROJECT_DIR / "scripts"
import sys
sys.path.insert(0, str(sys_path))

from selective_downloader import (
    BENCHMARK_MODEL_CATALOG,
    RemoteFileInfo,
    categorize_files,
    ensure_selective_model,
)


class SelectiveDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_forbidden_models_blocked(self):
        for model in ("memprivacy-1.7b", "openai-privacy-filter", "aiguard", "qwen3.5-0.8b", "raner"):
            ok, msg, manifest = ensure_selective_model(model, self.tmp_dir, download_missing=True)
            self.assertFalse(ok)
            self.assertIn("FORBIDDEN", msg)
            self.assertIsNone(manifest)

    def test_download_missing_false_skips(self):
        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertFalse(ok)
        self.assertIn("MODEL NOT INSTALLED", msg)
        self.assertIsNone(manifest)

    def test_file_categorization(self):
        spec = BENCHMARK_MODEL_CATALOG["gliner-pii-edge"]
        mock_files = [
            RemoteFileInfo(".gitattributes", 2185, "blob"),
            RemoteFileInfo("README.md", 15896, "blob"),
            RemoteFileInfo("configuration.json", 78, "blob"),
            RemoteFileInfo("gliner_config.json", 4316, "blob"),
            RemoteFileInfo("pytorch_model.bin", 180735778, "blob"),
            RemoteFileInfo("special_tokens_map.json", 694, "blob"),
            RemoteFileInfo("tokenizer.json", 3583593, "blob"),
            RemoteFileInfo("tokenizer_config.json", 21214, "blob"),
            RemoteFileInfo("onnx/model.onnx", 181078966, "blob"),
            RemoteFileInfo("onnx/model_fp16.onnx", 90845497, "blob"),
        ]
        required, optional, rejected = categorize_files(
            mock_files, spec["required_files"], spec["forbidden_files_prefix"]
        )
        req_paths = {f.path for f in required}
        rej_paths = {f.path for f in rejected}

        self.assertEqual(len(required), 6)
        self.assertIn("pytorch_model.bin", req_paths)
        self.assertIn("gliner_config.json", req_paths)
        self.assertIn("tokenizer.json", req_paths)

        self.assertIn(".gitattributes", rej_paths)
        self.assertIn("README.md", rej_paths)
        self.assertIn("onnx/model.onnx", rej_paths)
        self.assertIn("onnx/model_fp16.onnx", rej_paths)

    def test_existing_complete_model_is_reused(self):
        target = self.tmp_dir / "gliner-pii-edge"
        target.mkdir()
        spec = BENCHMARK_MODEL_CATALOG["gliner-pii-edge"]
        for fname in spec["required_files"]:
            (target / fname).write_text("mock content", encoding="utf-8")

        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertTrue(ok)
        self.assertIn("already installed and complete", msg)


if __name__ == "__main__":
    unittest.main()

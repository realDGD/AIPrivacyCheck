"""Tests for Selective Model Downloader (v0.6.8).

Verifies:
1. Forbidden models (MemPrivacy, AIguard, Qwen, RANER, etc.) cannot be downloaded.
2. If model is missing and download_missing=False, download is refused (MODEL NOT INSTALLED).
3. File categorization strictly isolates required PyTorch inference files and rejects ONNX/docs.
4. Existing complete model files with valid download-manifest.json are reused and verified.
5. Existing model files without manifest or with mismatched hashes are flagged as CORRUPTED.
6. Temporary download files (.tmp_download) are always cleaned up on failure.
7. Selective repair re-downloads only corrupted files when download_missing=True.
"""

from pathlib import Path
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys_path = PROJECT_DIR / "scripts"
import sys
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from selective_downloader import (
    BLOCKED_MODELS,
    DOWNLOADABLE_MODEL_CATALOG,
    RemoteFileInfo,
    categorize_files,
    download_single_file,
    ensure_selective_model,
    sha256_file,
    verify_existing_model_integrity,
)


class SelectiveDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_mock_installed_model(self, model_name: str = "gliner-pii-edge"):
        """Helper to create a fully valid model directory with matching manifest."""
        target = self.tmp_dir / model_name
        target.mkdir(parents=True, exist_ok=True)
        spec = DOWNLOADABLE_MODEL_CATALOG[model_name]
        files_record = []
        for fname in spec["required_files"]:
            fpath = target / fname
            content = f"content for {fname}".encode("utf-8")
            fpath.write_bytes(content)
            files_record.append({
                "path": fname,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        manifest = {
            "model_id": spec["repo_id"],
            "model_name": model_name,
            "revision": spec.get("revision", "master"),
            "files": files_record,
        }
        (target / "download-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return target, manifest

    def test_forbidden_models_blocked(self):
        for model in sorted(BLOCKED_MODELS):
            ok, msg, manifest = ensure_selective_model(model, self.tmp_dir, download_missing=True)
            self.assertFalse(ok, f"Model {model} should be blocked")
            self.assertIn("FORBIDDEN", msg)
            self.assertIsNone(manifest)

    def test_download_missing_false_skips_when_not_installed(self):
        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertFalse(ok)
        self.assertIn("MODEL NOT INSTALLED", msg)
        self.assertIsNone(manifest)

    def test_file_categorization(self):
        spec = DOWNLOADABLE_MODEL_CATALOG["gliner-pii-edge"]
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
            mock_files, spec["required_files"], spec.get("forbidden_files_prefix", ())
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

    def test_files_exist_with_manifest_reused(self):
        target, _ = self._create_mock_installed_model("gliner-pii-edge")
        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertTrue(ok)
        self.assertIn("already installed and complete", msg)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["model_name"], "gliner-pii-edge")

    def test_files_exist_but_no_manifest_unverified(self):
        target = self.tmp_dir / "gliner-pii-edge"
        target.mkdir()
        spec = DOWNLOADABLE_MODEL_CATALOG["gliner-pii-edge"]
        for fname in spec["required_files"]:
            (target / fname).write_text("mock content", encoding="utf-8")

        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertFalse(ok)
        self.assertIn("UNVERIFIED MODEL DIRECTORY", msg)
        self.assertIsNone(manifest)

    def test_files_exist_but_hash_wrong_corrupted(self):
        target, _ = self._create_mock_installed_model("gliner-pii-edge")
        # Tamper with pytorch_model.bin
        tampered_file = target / "pytorch_model.bin"
        tampered_file.write_bytes(b"tampered content")

        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertFalse(ok)
        self.assertIn("MODEL CORRUPTED", msg)
        self.assertIn("pytorch_model.bin", msg)
        self.assertIsNone(manifest)

    def test_one_file_corrupted_download_missing_false_fails(self):
        target, _ = self._create_mock_installed_model("gliner-pii-edge")
        (target / "gliner_config.json").unlink()

        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=False)
        self.assertFalse(ok)
        self.assertIn("MODEL CORRUPTED", msg)
        self.assertIn("gliner_config.json", msg)
        self.assertIsNone(manifest)

    @patch("selective_downloader.query_modelscope_repo_files")
    @patch("selective_downloader.download_single_file")
    def test_one_file_corrupted_download_missing_true_repairs(
        self, mock_download, mock_query
    ):
        target, orig_manifest = self._create_mock_installed_model("gliner-pii-edge")
        # Corrupt one file
        (target / "gliner_config.json").write_bytes(b"corrupted")

        spec = DOWNLOADABLE_MODEL_CATALOG["gliner-pii-edge"]
        mock_query.return_value = [
            RemoteFileInfo(fname, 100, "blob") for fname in spec["required_files"]
        ]
        mock_download.return_value = (True, 100, "dummy_hash")

        ok, msg, manifest = ensure_selective_model("gliner-pii-edge", self.tmp_dir, download_missing=True)
        self.assertTrue(ok)
        # Should only download the corrupted file, NOT all files
        self.assertEqual(mock_download.call_count, 1)
        args = mock_download.call_args[0]
        self.assertEqual(args[2], "gliner_config.json")

    def test_tmp_download_cleaned_up_on_failure(self):
        target_path = self.tmp_dir / "models" / "fake_file.bin"
        temp_path = target_path.with_name(target_path.name + ".tmp_download")

        # Mock urllib.request.urlopen to raise an error during read
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.side_effect = ConnectionResetError("Connection dropped")
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(ConnectionResetError):
                download_single_file(
                    repo_id="test/repo",
                    revision="master",
                    remote_path="fake_file.bin",
                    target_path=target_path,
                    expected_size=1024,
                )

            # Temp file must have been deleted
            self.assertFalse(temp_path.exists(), "temp_download file should be cleaned up on error")
            self.assertFalse(target_path.exists(), "target_path should not exist on error")

    def test_selective_downloader_imports_in_current_python(self):
        """Section 7: True subprocess import gate for benchmark scripts."""
        import subprocess
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(sys_path)!r}); "
            "import selective_downloader; "
            "import benchmark_cache; "
            "import benchmark_scoring; "
            "import model_integrity; "
            "print('ALL_BENCHMARK_MODULES_IMPORTED')"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("ALL_BENCHMARK_MODULES_IMPORTED", res.stdout)

        # Check Python 3.12 if available on host
        py312 = shutil.which("python3.12")
        if py312:
            res312 = subprocess.run([py312, "-c", code], capture_output=True, text=True, check=True)
            self.assertIn("ALL_BENCHMARK_MODULES_IMPORTED", res312.stdout)
        else:
            # Documented: Not Executed: Python 3.12 interpreter unavailable
            pass

    def test_manifest_path_traversal_rejected(self):
        """Section 9: Manifest containing '../' or absolute paths must be rejected."""
        target = self.tmp_dir / "gliner-pii-edge-traversal"
        target.mkdir(parents=True, exist_ok=True)
        manifest = {
            "files": [
                {"path": "../evil.bin", "size": 10, "sha256": "0" * 64},
                {"path": "/etc/passwd", "size": 10, "sha256": "1" * 64},
            ]
        }
        (target / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        ok, msg, _, _ = verify_existing_model_integrity(target)
        self.assertFalse(ok)
        self.assertIn("path traversal forbidden", msg)

    def test_manifest_missing_sha_rejected(self):
        """Section 8: Manifest entries without sha256 must be rejected."""
        target = self.tmp_dir / "gliner-pii-edge-nosha"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"content")
        manifest = {
            "files": [
                {"path": "model.bin", "size": 7},  # missing sha256
            ]
        }
        (target / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        ok, msg, _, _ = verify_existing_model_integrity(target)
        self.assertFalse(ok)
        self.assertIn("invalid manifest schema", msg)

    def test_manifest_negative_size_rejected(self):
        """Section 8: Manifest entries with negative size must be rejected."""
        target = self.tmp_dir / "gliner-pii-edge-negsize"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"content")
        manifest = {
            "files": [
                {"path": "model.bin", "size": -5, "sha256": "0" * 64},
            ]
        }
        (target / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        ok, msg, _, _ = verify_existing_model_integrity(target)
        self.assertFalse(ok)
        self.assertIn("invalid 'size'", msg)


if __name__ == "__main__":
    unittest.main()

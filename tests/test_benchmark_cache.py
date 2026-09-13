"""Tests for Raw Prediction Cache Architecture (v0.6.6).

Verifies:
1. Cache hit on exact match of all key components.
2. Cache miss when corpus SHA-256 changes.
3. Cache miss when label order changes.
4. Cache miss when model revision changes.
5. Cache security gate: non-synthetic corpus is rejected from persistence.
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys_path = PROJECT_DIR / "scripts"
import sys
sys.path.insert(0, str(sys_path))

from benchmark_cache import (
    AmbiguousCacheError,
    BenchmarkPredictionCache,
    CacheCorruptedError,
    LegacyUnverifiedCacheError,
    ModelIntegrityError,
    SecurityError,
    compute_model_files_hash,
)


class BenchmarkPredictionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.cache = BenchmarkPredictionCache(self.tmp_dir / "cache")
        self.fixture = self.tmp_dir / "privacy_benchmark_v2_100.jsonl"
        self.fixture.write_text(json.dumps({"id": "case_1", "text": "张三测试"}) + "\n", encoding="utf-8")
        self.model_dir = self.tmp_dir / "mock_model"
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cache_hit_on_exact_key(self):
        config = {
            "minimum_threshold": 0.30,
            "label_order": ["person", "organization"],
        }
        key = self.cache.build_cache_key(
            corpus_path=self.fixture,
            model_id="gliner-pii-edge",
            revision="master",
            model_dir=self.model_dir,
            inference_config=config,
            runtime_meta={"python_version": "3.12"},
        )
        sample_preds = [{"id": "case_1", "entities": [{"type": "CN_NAME", "start": 0, "end": 2, "score": 0.85}]}]
        cdir = self.cache.save(key, sample_preds, {"f1": 0.8})

        # Query cache
        hit, cached_path = self.cache.has_valid_cache(key)
        self.assertTrue(hit)
        self.assertEqual(cdir, cached_path)

        loaded_cfg, loaded_preds, loaded_m = self.cache.load(cached_path)
        self.assertEqual(loaded_cfg["model_id"], "gliner-pii-edge")
        self.assertEqual(len(loaded_preds), 1)
        self.assertEqual(loaded_preds[0]["id"], "case_1")
        self.assertEqual(loaded_m["f1"], 0.8)

    def test_cache_miss_on_corpus_change(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        # Modify fixture content
        self.fixture.write_text(json.dumps({"id": "case_1", "text": "李四测试"}) + "\n", encoding="utf-8")
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config)

        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit)

    def test_cache_miss_on_label_order_change(self):
        config1 = {"minimum_threshold": 0.30, "label_order": ["person", "organization"]}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config1)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        config2 = {"minimum_threshold": 0.30, "label_order": ["organization", "person"]}
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config2)

        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit)

    def test_cache_miss_on_revision_change(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "v2", self.model_dir, config)
        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit)

    def test_cache_security_gate_refuses_non_synthetic_corpus(self):
        user_input_file = self.tmp_dir / "user_production_input.jsonl"
        user_input_file.write_text(json.dumps({"text": "真实用户隐私数据"}) + "\n", encoding="utf-8")
        key = self.cache.build_cache_key(user_input_file, "gliner-pii-edge", "master", self.model_dir, {})

        with self.assertRaises(SecurityError):
            self.cache.save(key, [{"id": "1", "entities": []}])

    def test_cache_miss_on_runtime_change(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        rt1 = {"python_version": "3.12.1", "torch_version": "2.14.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt1)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        rt2 = {"python_version": "3.13.0", "torch_version": "2.14.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt2)
        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit, "Cache must miss when python_version changes")

    def test_cache_miss_on_torch_change(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        rt1 = {"python_version": "3.12.1", "torch_version": "2.14.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt1)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        rt2 = dict(rt1, torch_version="2.15.0")
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt2)
        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit, "Cache must miss when torch_version changes")

    def test_cache_miss_on_transformers_change(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        rt1 = {"python_version": "3.12.1", "torch_version": "2.14.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt1)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        rt2 = dict(rt1, transformers_version="4.40.0")
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt2)
        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit, "Cache must miss when transformers_version changes")

    def test_cache_miss_on_gliner_change(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        rt1 = {"python_version": "3.12.1", "torch_version": "2.14.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}
        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt1)
        self.cache.save(key1, [{"id": "case_1", "entities": []}])

        rt2 = dict(rt1, gliner_version="0.3.0")
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt2)
        hit, _ = self.cache.has_valid_cache(key2)
        self.assertFalse(hit, "Cache must miss when gliner_version changes")

    def test_same_size_different_content_changes_model_fingerprint(self):
        from benchmark_cache import compute_model_files_hash
        dir_a = self.tmp_dir / "model_a"
        dir_b = self.tmp_dir / "model_b"
        dir_a.mkdir()
        dir_b.mkdir()

        # Same filename and same exact size (10 bytes), but different bytes
        (dir_a / "model.bin").write_bytes(b"0123456789")
        (dir_b / "model.bin").write_bytes(b"abcdefghij")

        hash_a = compute_model_files_hash(dir_a)
        hash_b = compute_model_files_hash(dir_b)
        self.assertNotEqual(hash_a, hash_b, "Different content with same size must yield different fingerprints")

    def test_mtime_change_does_not_change_model_fingerprint(self):
        import os
        import time
        from benchmark_cache import compute_model_files_hash

        test_dir = self.tmp_dir / "model_mtime"
        test_dir.mkdir()
        model_file = test_dir / "model.bin"
        model_file.write_bytes(b"fixed_model_weights_bytes_12345")

        hash1 = compute_model_files_hash(test_dir)
        # Shift mtime by 1000 seconds
        new_time = time.time() - 10000
        os.utime(model_file, (new_time, new_time))
        hash2 = compute_model_files_hash(test_dir)

        self.assertEqual(hash1, hash2, "Mtime change must not affect model fingerprint")

    def test_cache_signatures_can_coexist(self):
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        rt1 = {"python_version": "3.12.1", "torch_version": "2.14.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}
        rt2 = {"python_version": "3.12.1", "torch_version": "2.15.0", "transformers_version": "5.16.1", "gliner_version": "0.2.29"}

        key1 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt1)
        key2 = self.cache.build_cache_key(self.fixture, "gliner-pii-edge", "master", self.model_dir, config, runtime_meta=rt2)

        self.assertNotEqual(key1["cache_signature"], key2["cache_signature"])

        dir1 = self.cache.save(key1, [{"id": "case_1", "entities": [{"text": "v1"}]}])
        dir2 = self.cache.save(key2, [{"id": "case_1", "entities": [{"text": "v2"}]}])

        self.assertNotEqual(dir1, dir2)
        self.assertTrue(dir1.is_dir())
        self.assertTrue(dir2.is_dir())

        hit1, found1 = self.cache.has_valid_cache(key1)
        hit2, found2 = self.cache.has_valid_cache(key2)

        self.assertTrue(hit1)
        self.assertTrue(hit2)
        self.assertEqual(found1, dir1)
        self.assertEqual(found2, dir2)

    def _create_model_with_manifest(self, dir_name: str, files_dict: Dict[str, bytes]) -> Path:
        mdir = self.tmp_dir / dir_name
        mdir.mkdir(parents=True, exist_ok=True)
        records = []
        import hashlib
        for fname, data in files_dict.items():
            fpath = mdir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(data)
            records.append({
                "path": fname,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        manifest = {
            "model_id": "test/model",
            "revision": "master",
            "files": records,
        }
        (mdir / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return mdir

    def test_manifest_backed_fingerprint_verifies_actual_content(self):
        """Test 1: Manifest + files match -> fingerprint stable & cache hits."""
        mdir = self._create_model_with_manifest("model_valid", {
            "config.json": b'{"model": "test"}',
            "weights.bin": b"valid_weights_12345",
        })
        fp1 = compute_model_files_hash(mdir)
        fp2 = compute_model_files_hash(mdir)
        self.assertEqual(fp1, fp2)

        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", mdir, config)
        self.cache.save(key, [{"id": "case_1", "entities": []}])
        hit, _ = self.cache.has_valid_cache(key)
        self.assertTrue(hit)

    def test_manifest_model_file_tamper_fails_integrity(self):
        """Test 2: Manifest unchanged, 1 byte of model file modified -> integrity failure & cache cannot hit."""
        mdir = self._create_model_with_manifest("model_tamper", {
            "config.json": b'{"model": "test"}',
            "weights.bin": b"valid_weights_12345",
        })
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", mdir, config)
        self.cache.save(key, [{"id": "case_1", "entities": []}])

        # Tamper with 1 byte in weights.bin without updating manifest
        tampered_weights = bytearray(b"valid_weights_12345")
        tampered_weights[0] = ord("X")
        (mdir / "weights.bin").write_bytes(bytes(tampered_weights))

        # Must raise ModelIntegrityError when computing hash
        with self.assertRaises(ModelIntegrityError):
            compute_model_files_hash(mdir)

        # Must raise ModelIntegrityError when attempting to build cache key
        with self.assertRaises(ModelIntegrityError):
            self.cache.build_cache_key(self.fixture, "test-model", "master", mdir, config)

        # When allow_corrupted=True, returns corrupted key that yields cache miss
        corrupted_key = self.cache.build_cache_key(self.fixture, "test-model", "master", mdir, config, allow_corrupted=True)
        hit, _ = self.cache.has_valid_cache(corrupted_key)
        self.assertFalse(hit, "Tampered model file must never hit cache")

    def test_manifest_same_size_content_corruption_rejected(self):
        """Test 3: Model file size unchanged, content changed, manifest unchanged -> rejected."""
        mdir = self._create_model_with_manifest("model_same_sz", {
            "weights.bin": b"1234567890",
        })
        # Overwrite with different bytes of same exact length (10 bytes)
        (mdir / "weights.bin").write_bytes(b"abcdefghij")

        with self.assertRaises(ModelIntegrityError):
            compute_model_files_hash(mdir)

    def test_manifest_missing_sha_rejected(self):
        """Test 4: Manifest entry missing sha256 -> rejected as untrusted manifest."""
        mdir = self.tmp_dir / "model_no_sha"
        mdir.mkdir()
        (mdir / "weights.bin").write_bytes(b"data")
        manifest = {
            "files": [{"path": "weights.bin", "size": 4}],  # missing sha256
        }
        (mdir / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(ModelIntegrityError):
            compute_model_files_hash(mdir)

    def test_manifest_missing_required_file_rejected(self):
        """Test 5: Manifest points to a non-existent file -> rejected."""
        mdir = self.tmp_dir / "model_missing_file"
        mdir.mkdir()
        manifest = {
            "files": [{"path": "non_existent.bin", "size": 100, "sha256": "0" * 64}],
        }
        (mdir / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(ModelIntegrityError):
            compute_model_files_hash(mdir)

    def test_no_manifest_falls_back_to_content_hash(self):
        """Test 6: No manifest -> fallback to streaming content hash."""
        import os
        import time
        mdir = self.tmp_dir / "model_no_manifest"
        mdir.mkdir()
        wfile = mdir / "weights.bin"
        wfile.write_bytes(b"raw_weights_content_54321")

        h1 = compute_model_files_hash(mdir)
        # Touch mtime
        new_time = time.time() - 50000
        os.utime(wfile, (new_time, new_time))
        h2 = compute_model_files_hash(mdir)
        self.assertEqual(h1, h2, "Mtime change must not alter streaming fingerprint")

        # Change content
        wfile.write_bytes(b"altered_weights_content_54321")
        h3 = compute_model_files_hash(mdir)
        self.assertNotEqual(h1, h3, "Content change must alter streaming fingerprint")

    def test_parent_cache_dir_with_multiple_signatures_is_ambiguous(self):
        """Section 18: Loading from parent directory with >1 signature must raise AmbiguousCacheError."""
        config1 = {"minimum_threshold": 0.30, "label_order": ["person"]}
        config2 = {"minimum_threshold": 0.30, "label_order": ["organization"]}
        key1 = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config1)
        key2 = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config2)

        dir1 = self.cache.save(key1, [{"id": "case_1", "entities": [{"text": "v1"}]}])
        dir2 = self.cache.save(key2, [{"id": "case_1", "entities": [{"text": "v2"}]}])

        parent_dir = dir1.parent
        self.assertEqual(parent_dir, dir2.parent)

        # Loading from parent_dir must fail because there are 2 signatures
        with self.assertRaises(AmbiguousCacheError) as ctx:
            self.cache.load(parent_dir)
        self.assertIn("multiple signature directories found", str(ctx.exception))

        # Explicitly loading from either signature directory must succeed
        cfg1, preds1, _ = self.cache.load(dir1)
        self.assertEqual(cfg1["inference_config"]["label_order"], ["person"])
        cfg2, preds2, _ = self.cache.load(dir2)
        self.assertEqual(cfg2["inference_config"]["label_order"], ["organization"])

    def test_prediction_cache_content_tamper_rejected(self):
        """Section 19: Tampering with predictions.jsonl triggers CacheCorruptedError."""
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config)
        preds = [
            {"id": "case_1", "entities": [{"text": "Alice", "type": "CN_NAME"}]},
            {"id": "case_2", "entities": [{"text": "Bob", "type": "CN_NAME"}]},
        ]
        cdir = self.cache.save(key, preds)

        # Load valid cache
        _, loaded, _ = self.cache.load(cdir)
        self.assertEqual(len(loaded), 2)

        # Tamper with predictions.jsonl content
        preds_file = cdir / "predictions.jsonl"
        lines = preds_file.read_text(encoding="utf-8").splitlines()
        # Alter entity in first line
        altered = lines[0].replace("Alice", "Mallory")
        preds_file.write_text(altered + "\n" + lines[1] + "\n", encoding="utf-8")

        with self.assertRaises(CacheCorruptedError) as ctx:
            self.cache.load(cdir)
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_cache_without_manifest_is_not_valid(self):
        """P2 Test: has_valid_cache returns False when cache-manifest.json is missing."""
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config)
        preds = [{"id": "case_1", "entities": [{"text": "Alice", "type": "CN_NAME"}]}]
        cdir = self.cache.save(key, preds)

        # Cache is valid initially
        hit, _ = self.cache.has_valid_cache(key)
        self.assertTrue(hit)

        # Remove cache-manifest.json
        (cdir / "cache-manifest.json").unlink()

        # Cache must no longer be considered valid
        hit2, _ = self.cache.has_valid_cache(key)
        self.assertFalse(hit2, "Missing cache-manifest.json must invalidate cache hit")

    def test_load_cache_without_manifest_rejected(self):
        """P2 Test: load rejects cache lacking cache-manifest.json with LegacyUnverifiedCacheError."""
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config)
        preds = [{"id": "case_1", "entities": [{"text": "Alice", "type": "CN_NAME"}]}]
        cdir = self.cache.save(key, preds)

        (cdir / "cache-manifest.json").unlink()

        # Default load must raise LegacyUnverifiedCacheError
        with self.assertRaises(LegacyUnverifiedCacheError) as ctx:
            self.cache.load(cdir)
        self.assertIn("missing mandatory cache-manifest.json", str(ctx.exception))

        # Explicit override succeeds
        cfg, loaded_preds, _ = self.cache.load(cdir, allow_legacy_unverified=True)
        self.assertEqual(len(loaded_preds), 1)

    def test_prediction_cache_checksum_tamper_rejected(self):
        """P2 Test: predictions.jsonl checksum mismatch with cache-manifest raises CacheCorruptedError."""
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config)
        preds = [{"id": "case_1", "entities": [{"text": "Alice", "type": "CN_NAME"}]}]
        cdir = self.cache.save(key, preds)

        # Tamper manifest checksum
        cman = json.loads((cdir / "cache-manifest.json").read_text(encoding="utf-8"))
        cman["predictions_sha256"] = "0" * 64
        (cdir / "cache-manifest.json").write_text(json.dumps(cman), encoding="utf-8")

        with self.assertRaises(CacheCorruptedError) as ctx:
            self.cache.load(cdir)
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_prediction_cache_count_tamper_rejected(self):
        """P2 Test: prediction_count mismatch with cache-manifest raises CacheCorruptedError."""
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config)
        preds = [
            {"id": "case_1", "entities": [{"text": "Alice", "type": "CN_NAME"}]},
            {"id": "case_2", "entities": [{"text": "Bob", "type": "CN_NAME"}]},
        ]
        cdir = self.cache.save(key, preds)

        # Tamper count in manifest
        cman = json.loads((cdir / "cache-manifest.json").read_text(encoding="utf-8"))
        cman["prediction_count"] = 999
        (cdir / "cache-manifest.json").write_text(json.dumps(cman), encoding="utf-8")

        with self.assertRaises(CacheCorruptedError) as ctx:
            self.cache.load(cdir)
        self.assertIn("prediction count mismatch", str(ctx.exception))

    def test_has_valid_cache_rejects_tampered_predictions(self):
        """P2 Test: has_valid_cache returns False if predictions.jsonl was modified."""
        config = {"minimum_threshold": 0.30, "label_order": ["person"]}
        key = self.cache.build_cache_key(self.fixture, "test-model", "master", self.model_dir, config)
        preds = [{"id": "case_1", "entities": [{"text": "Alice", "type": "CN_NAME"}]}]
        cdir = self.cache.save(key, preds)

        hit, _ = self.cache.has_valid_cache(key)
        self.assertTrue(hit)

        # Tamper predictions.jsonl file
        (cdir / "predictions.jsonl").write_text(
            json.dumps({"id": "case_1", "entities": [{"text": "Mallory", "type": "CN_NAME"}]}) + "\n",
            encoding="utf-8",
        )

        hit_tampered, _ = self.cache.has_valid_cache(key)
        self.assertFalse(hit_tampered, "has_valid_cache must reject tampered predictions file")


if __name__ == "__main__":
    unittest.main()

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

from benchmark_cache import BenchmarkPredictionCache, SecurityError


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


if __name__ == "__main__":
    unittest.main()

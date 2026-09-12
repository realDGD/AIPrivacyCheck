"""SiameseUIE worker output-normalization regression tests.

Captures the REAL ModelScope 1.40 output shape observed during the v0.6.4
runtime dependency fix:

    {"output": [[{"type": "地理位置", "span": "北京市朝阳区测试路88号",
                  "offset": [7, 19]}]]}

i.e. one inner list per schema key with half-open `offset` pairs. The worker
previously expected flat lists with start/end and silently returned zero
entities for the real shape.
"""

import sys
from pathlib import Path
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server"
sys.path.insert(0, str(SERVER_DIR / "privacy" / "workers"))
sys.path.insert(0, str(SERVER_DIR))

from siamese_uie_worker import extract_entities_from_output  # noqa: E402

TEXT = "我叫张三，住在北京市朝阳区测试路88号。"


class SiameseUieOutputExtractionTests(unittest.TestCase):
    def test_real_modelscope_140_nested_output_with_offsets(self):
        raw = {
            "output": [
                [
                    {"type": "地理位置", "span": "北京市朝阳区测试路88号", "offset": [7, 19]},
                ],
                [
                    {"type": "人物", "span": "张三", "offset": [2, 4]},
                ],
            ]
        }
        entities = extract_entities_from_output(raw, TEXT)
        by_label = {(e["label"], e["text"]): (e["start"], e["end"]) for e in entities}
        self.assertEqual(by_label[("地理位置", "北京市朝阳区测试路88号")], (7, 19))
        self.assertEqual(by_label[("人物", "张三")], (2, 4))
        self.assertTrue(all(TEXT[e["start"]:e["end"]] == e["text"] for e in entities))

    def test_flat_list_with_start_end_still_supported(self):
        raw = [{"type": "人物", "span": "张三", "start": 2, "end": 4, "probability": 0.98}]
        entities = extract_entities_from_output(raw, TEXT)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["start"], 2)
        self.assertEqual(entities[0]["score"], 0.98)

    def test_flat_output_dict(self):
        raw = {"output": [{"type": "人物", "span": "张三", "offset": [2, 4]}]}
        entities = extract_entities_from_output(raw, TEXT)
        self.assertEqual([(e["label"], e["start"], e["end"]) for e in entities], [("人物", 2, 4)])

    def test_label_keyed_dict(self):
        raw = {"人物": [{"span": "张三", "start": 2, "end": 4}]}
        entities = extract_entities_from_output(raw, TEXT)
        self.assertEqual([(e["label"], e["text"]) for e in entities], [("人物", "张三")])

    def test_offsets_are_half_open_and_verified_against_text(self):
        raw = {"output": [[{"type": "地理位置", "span": "测试路", "offset": [13, 16]}]]}
        entities = extract_entities_from_output(raw, TEXT)
        self.assertEqual(entities[0]["text"], TEXT[13:16])

    def test_invalid_offsets_fall_back_to_cursor_lookup(self):
        raw = {"output": [[{"type": "人物", "span": "张三", "offset": [99, 101]}]]}
        entities = extract_entities_from_output(raw, TEXT)
        self.assertEqual([(e["start"], e["end"]) for e in entities], [(2, 4)])


if __name__ == "__main__":
    unittest.main()

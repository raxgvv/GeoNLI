"""Fast SAM3 post-processing checks independent of loading the checkpoint."""
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path(__file__).resolve().parents[2] / "models" / "sam3"
sys.path.insert(0, str(SAM3_ROOT))
from models.sam3.unified_inference import get_prediction


class FakeProcessor:
    def __init__(self, scores):
        self.scores = torch.tensor(scores)

    def set_image(self, _image):
        return {}

    def set_text_prompt(self, *, state, prompt):
        self.prompt = prompt
        masks = torch.zeros((2, 32, 32), dtype=torch.float32)
        masks[0, 4:20, 5:22] = 1.0
        masks[1, 22:31, 23:31] = 1.0
        return {"masks": masks, "scores": self.scores}


class Sam3PostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "image.png"
        Image.new("RGB", (32, 32), "white").save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_raw_count_is_separate_from_thresholded_detections(self):
        results, masks, raw_count = get_prediction(
            None, FakeProcessor([0.75, 0.49]), self.image_path, "airplane", "cpu",
            score_threshold=0.5, min_area=20,
        )
        self.assertEqual(raw_count, 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["score"], 0.75)
        self.assertEqual(masks.shape[0], 2)
        self.assertEqual(len(results[0]["obb_corners"]), 4)

    def test_low_confidence_raw_candidates_remain_a_valid_empty_result(self):
        results, _masks, raw_count = get_prediction(
            None, FakeProcessor([0.42, 0.31]), self.image_path, "runway", "cpu",
            score_threshold=0.5, min_area=20,
        )
        self.assertEqual(raw_count, 2)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()

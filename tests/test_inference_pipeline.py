import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import VOYADataset
from inference import build_feature_sequence_from_image
from realtime.mediapipe_runtime import KEYPOINT_DIM


class InferencePipelineTests(unittest.TestCase):

    def test_build_feature_sequence_from_image_shape(self):
        """build_feature_sequence_from_image phải trả về tensor (60, 1605)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)

            seq = build_feature_sequence_from_image(image_path, input_dim=1605, sequence_length=60)

            self.assertEqual(seq.shape, (60, 1605))
            self.assertTrue(torch.isfinite(seq).all())

    def test_keypoint_dim_constant(self):
        """KEYPOINT_DIM phải đúng bằng 1605."""
        self.assertEqual(KEYPOINT_DIM, 1605)

    def test_voya_dataset_basic(self):
        """VOYADataset phải trả về (seq, label) đúng shape và kiểu."""
        rng = np.random.default_rng(0)
        num_classes = 5
        samples = [
            {"sequences": rng.standard_normal((60, 1605)).astype("float32").tolist(), "labels": i % num_classes}
            for i in range(10)
        ]
        ds = VOYADataset(samples, input_dim=1605, sequence_length=60, num_classes=num_classes)

        self.assertEqual(len(ds), 10)
        seq, label = ds[0]
        self.assertEqual(seq.shape, (60, 1605))
        self.assertEqual(seq.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.long)
        self.assertTrue(0 <= label.item() < num_classes)

    def test_voya_dataset_label_out_of_range(self):
        """VOYADataset phải raise ValueError khi label vượt num_classes."""
        rng = np.random.default_rng(0)
        samples = [{"sequences": rng.standard_normal((60, 1605)).astype("float32").tolist(), "labels": 999}]
        ds = VOYADataset(samples, num_classes=5)
        with self.assertRaises(ValueError):
            ds[0]

    def test_voya_dataset_missing_sequences(self):
        """VOYADataset phải raise KeyError khi sample không có 'sequences'."""
        ds = VOYADataset([{"labels": 0}], num_classes=5)
        with self.assertRaises(KeyError):
            ds[0]


if __name__ == "__main__":
    unittest.main()

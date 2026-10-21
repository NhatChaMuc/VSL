import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint import save_best_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def test_best_checkpoint_updates_only_when_better(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            model = torch.nn.Linear(2, 2)

            updated1 = save_best_checkpoint(model, path, score=0.80, epoch=1)
            self.assertTrue(updated1)
            self.assertTrue(path.exists())

            updated2 = save_best_checkpoint(model, path, score=0.70, epoch=2)
            self.assertFalse(updated2)

            updated3 = save_best_checkpoint(model, path, score=0.90, epoch=3)
            self.assertTrue(updated3)

            metadata_path = path.with_suffix(".json")
            self.assertTrue(metadata_path.exists())
            metadata = metadata_path.read_text(encoding="utf-8")
            self.assertIn("0.9", metadata)


if __name__ == "__main__":
    unittest.main()

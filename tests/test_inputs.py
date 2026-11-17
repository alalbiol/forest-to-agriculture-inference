"""Input handling and packaged-file checks; no checkpoint needed.

    python -m unittest discover -s tests
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f2a.model import CLASSES, load_head, preprocess, probabilities, top9  # noqa: E402
from inference import read_chip  # noqa: E402


class InputTest(unittest.TestCase):
    def test_preprocess_resizes_and_normalises(self):
        with np.load(next((ROOT / 'samples/inputs').glob('*.npz'))) as saved:
            tensor = preprocess(saved['imagery'])
        self.assertEqual(tuple(tensor.shape), (14, 224, 224))
        self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_rejects_wrong_band_count(self):
        with self.assertRaises(ValueError):
            preprocess(np.zeros((13, 44, 44), np.float32))

    def test_rejects_non_finite(self):
        stack = np.zeros((14, 44, 44), np.float32)
        stack[3, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            preprocess(stack)

    def test_refuses_chips_with_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gappy.npz'
            qa = np.ones((2, 44, 44), np.float32)
            qa[1, :10] = 0  # 23% of Sentinel-2 pixels unobserved
            np.savez(path, imagery=np.ones((14, 44, 44), np.float32), qa=qa)
            with self.assertRaises(ValueError):
                read_chip(path, allow_gaps=False)
            self.assertEqual(read_chip(path, allow_gaps=True).shape, (14, 44, 44))

    def test_head_gives_probabilities(self):
        head = load_head()
        p = probabilities(*head, np.zeros((768, 14, 14), np.float16))
        self.assertEqual(p.shape, (len(CLASSES), 14, 14))
        np.testing.assert_allclose(p.sum(0), 1, atol=1e-5)

    def test_top9(self):
        self.assertAlmostEqual(top9(np.arange(196.0)), float(np.arange(187, 196).mean()))

    def test_every_sample_has_both_years(self):
        reference = json.loads((ROOT / 'samples/reference/scores.json').read_text())
        for sample in reference['samples']:
            for year in reference['years']:
                self.assertTrue((ROOT / 'samples/inputs' / f'{sample["chip_id"]}_{year}.npz').is_file())


if __name__ == '__main__':
    unittest.main()

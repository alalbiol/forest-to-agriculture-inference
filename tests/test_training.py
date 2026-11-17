"""Smoke tests for the training code (CPU, a few seconds each).

    python -m unittest discover -s tests
The head-pipeline test needs the backbone checkpoint and is skipped without it.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/backbone'))
from f2a.model import DEFAULT_BACKBONE, load_backbone  # noqa: E402
import train as backbone_training  # noqa: E402


class BackboneTrainingTest(unittest.TestCase):
    def test_one_step_checkpoint_and_export(self):
        config = backbone_training.Config(ROOT / 'training/backbone/finetune_terrafm_early_fusion.yaml')
        config.PRETRAINED_MODEL_PATH = None
        config.MIXED_PRECISION = False
        config.NUM_EPOCHS = 1
        with tempfile.TemporaryDirectory() as tmp:
            config.SAVE_DIR = Path(tmp)
            torch.manual_seed(0)
            model = backbone_training.create_model(config)
            loader = [(torch.randn(2, 14, 224, 224), torch.tensor([[1, 0] * 9 + [1], [0, 1] * 9 + [0]]))]
            optimizer = backbone_training.create_optimizer(model, config)
            scheduler = backbone_training.create_scheduler(optimizer, config, len(loader))
            criterion = backbone_training.create_criterion(config)
            loss, _ = backbone_training.train_epoch(model, loader, criterion, optimizer, scheduler, None,
                                                    'cpu', config, 0, lambda values: None)
            self.assertTrue(np.isfinite(loss))
            metrics = backbone_training.validate(model, loader, criterion, 'cpu', 0, config)
            backbone_training.save_checkpoint(model, optimizer, 0, metrics, config, 'best_model.pth')
            exported = Path(tmp) / 'backbone.pt'
            subprocess.run([sys.executable, 'scripts/export_checkpoint.py', '--source',
                            str(Path(tmp) / 'best_model.pth'), '--any-source', '--output', str(exported)],
                           cwd=ROOT, check=True, capture_output=True)
            load_backbone(exported)  # strict load into the inference architecture
            self.assertTrue(exported.with_suffix('.json').is_file())


@unittest.skipUnless(DEFAULT_BACKBONE.is_file(), 'backbone checkpoint not downloaded')
class HeadPipelineTest(unittest.TestCase):
    def test_extract_train_select_infer(self):
        reference = json.loads((ROOT / 'samples/reference/scores.json').read_text())
        chips = [s['chip_id'] for s in reference['samples']]
        quadrants = np.zeros((44, 44), np.int64)
        quadrants[:22, 22:], quadrants[22:, :22], quadrants[22:, 22:] = 1, 2, 3
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.savez(tmp / 'labels.npz', labels=quadrants)  # synthetic labels: every class present
            with (tmp / 'chips.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['chip_id', 'imagery', 'labels', 'split'])
                for i, chip in enumerate(chips):
                    writer.writerow([chip, str(ROOT / f'samples/inputs/{chip}_2024.npz'), 'labels.npz',
                                     'validation' if i % 3 == 2 else 'train'])
            with (tmp / 'pairs.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['chip_id', 'before', 'after', 'conversion'])
                for chip in chips:
                    writer.writerow([chip, str(ROOT / f'samples/inputs/{chip}_2020.npz'),
                                     str(ROOT / f'samples/inputs/{chip}_2024.npz'),
                                     int('last_agriculture' in chip)])
            run = lambda *args: subprocess.run([sys.executable, *args], cwd=ROOT, check=True,  # noqa: E731
                                               capture_output=True)
            run('training/head/extract_tokens.py', '--chips', str(tmp / 'chips.csv'),
                '--output-dir', str(tmp / 'tokens'), '--device', 'cpu')
            run('training/head/train_head.py', '--tokens', str(tmp / 'tokens/tokens.csv'),
                '--output', str(tmp / 'head.pt'))
            run('training/head/select_thresholds.py', '--pairs', str(tmp / 'pairs.csv'),
                '--head', str(tmp / 'head.pt'), '--output', str(tmp / 'ops.json'), '--device', 'cpu')
            run('inference.py', '--before', str(ROOT / f'samples/inputs/{chips[0]}_2020.npz'),
                '--after', str(ROOT / f'samples/inputs/{chips[0]}_2024.npz'), '--head', str(tmp / 'head.pt'),
                '--operating-points', str(tmp / 'ops.json'), '--output', str(tmp / 'out.npz'), '--device', 'cpu')
            summary = json.loads((tmp / 'out.json').read_text())
            self.assertIn('temporal_score', summary)


if __name__ == '__main__':
    unittest.main()

"""Score the bundled samples and compare with the benchmark's saved scores.

    python scripts/verify_samples.py [--device cpu]
Requires the backbone checkpoint (see README.md). Writes outputs/sample_verification.json.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f2a.model import load_backbone, load_head, score_pair  # noqa: E402

TOLERANCE = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'samples/reference/scores.json').read_text())
    thresholds = json.loads((ROOT / 'heads/operating_points.json').read_text())
    backbone, head = load_backbone(device=args.device), load_head()
    rows, worst = [], 0.0
    for sample in reference['samples']:
        chips = [np.load(ROOT / 'samples/inputs' / f'{sample["chip_id"]}_{year}.npz')['imagery']
                 for year in reference['years']]
        result = score_pair(backbone, head, *chips, device=args.device)
        errors = {k: abs(result[k] - sample[k]) for k in ('temporal_score', 'after_only_score')}
        worst = max(worst, *errors.values())
        rows.append({'chip_id': sample['chip_id'], 'reference': sample['reference'],
                     **{k: result[k] for k in errors}, 'max_abs_error': max(errors.values())})
        flags = ['flagged' if result[f'{k}_score'] >= thresholds[k]['threshold'] else '-'
                 for k in ('temporal', 'after_only')]
        print(f'{sample["reference"]:34s} temporal {result["temporal_score"]:.3f} {flags[0]:8s}'
              f' after-only {result["after_only_score"]:.3f} {flags[1]:8s} error {max(errors.values()):.1e}')
    report = {'status': 'passed' if worst < TOLERANCE else 'failed', 'tolerance': TOLERANCE,
              'max_abs_error': worst, 'device': args.device, 'torch': torch.__version__,
              'samples': rows}
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/sample_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if worst >= TOLERANCE:
        raise SystemExit(f'FAIL: maximum score difference {worst:.2e} exceeds {TOLERANCE}')
    print(f'PASS: {len(rows)} samples reproduce the benchmark scores (max difference {worst:.1e}).')


if __name__ == '__main__':
    main()

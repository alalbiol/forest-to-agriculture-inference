"""Step 3 of head retraining: choose flag thresholds on labelled validation chip pairs.

    python training/head/select_thresholds.py --pairs validation_pairs.csv \
        --head heads/my_head.pt --output heads/my_operating_points.json [--max-fpr 0.1]

validation_pairs.csv columns: chip_id, before, after, conversion
  before/after  .npz chips as for inference.py; conversion 1 = forest converted to
  agriculture, 0 = any other outcome you want the flag to reject.
For each score (temporal, after-only) the threshold is the one with the highest recall
whose false-positive rate on these pairs is at most --max-fpr, as in the study (10%).
Use pairs that were not used to train the head. Pass the output to inference.py with
--operating-points.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f2a.model import DEFAULT_BACKBONE, load_backbone, load_head, score_pair  # noqa: E402


def choose(scores, positive, max_fpr):
    """Highest-recall threshold with FPR <= max_fpr (scores >= threshold are flagged)."""
    positive = np.asarray(positive, bool)
    if positive.all() or not positive.any():
        raise SystemExit('Validation pairs need both conversions and non-conversions.')
    fpr, tpr, thresholds = roc_curve(positive, np.asarray(scores, float))
    allowed = np.flatnonzero(fpr <= max_fpr)
    best = allowed[np.argmax(tpr[allowed])]
    return {'threshold': float(thresholds[best]), 'validation_recall': float(tpr[best]),
            'validation_fpr': float(fpr[best]), 'positive_n': int(positive.sum()),
            'negative_n': int((~positive).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-fpr', type=float, default=0.1)
    parser.add_argument('--backbone', type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with args.pairs.open() as f:
        rows = list(csv.DictReader(f))
    backbone, head = load_backbone(args.backbone, args.device), load_head(args.head)
    base = args.pairs.parent
    scores = {'temporal': [], 'after_only': []}
    for row in rows:
        chips = [np.load(base / row[k])['imagery'] for k in ('before', 'after')]
        result = score_pair(backbone, head, *chips, device=args.device)
        scores['temporal'].append(result['temporal_score'])
        scores['after_only'].append(result['after_only_score'])
    positive = [int(r['conversion']) == 1 for r in rows]
    output = {'note': f'Thresholds chosen on {args.pairs.name}: maximum recall with false-positive '
                      f'rate <= {args.max_fpr:.0%}. Scores are uncalibrated rankings.',
              'head': str(args.head),
              'temporal': {'method': 'postclassification', **choose(scores['temporal'], positive, args.max_fpr)},
              'after_only': {'method': 'endpoint_terrafm', **choose(scores['after_only'], positive, args.max_fpr)}}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f"temporal {output['temporal']['threshold']:.4f}, after-only {output['after_only']['threshold']:.4f} "
          f'-> {args.output}')


if __name__ == '__main__':
    main()

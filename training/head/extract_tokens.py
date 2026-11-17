"""Step 1 of head retraining: frozen-backbone tokens and patch labels for labelled chips.

    python training/head/extract_tokens.py --chips chips.csv --output-dir tokens/

chips.csv columns: chip_id, imagery, labels, split
  imagery  .npz with `imagery` (14, H, W) as for inference.py (one date per row)
  labels   .npz with `labels` (H, W) integers on the same grid: 0 primary forest,
           1 secondary forest, 2 agriculture, 3 mining; any other value is ignored
  split    train or validation
Each 14 x 14 patch takes the class covering at least 80% of it, otherwise -1 (ignored),
exactly as the released head's TerraClass labels were made. Writes <chip_id>.npz with
`tokens` (768, 14, 14, float16) and `token_labels` (14, 14), plus tokens.csv for step 2.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f2a.model import CLASSES, DEFAULT_BACKBONE, TOKEN_GRID, load_backbone, tokens  # noqa: E402

PURITY = 0.8


def token_labels(labels: np.ndarray) -> np.ndarray:
    indicators = np.stack([labels == k for k in range(len(CLASSES))]).astype(np.float32)
    fractions = F.interpolate(torch.from_numpy(indicators)[None], size=(TOKEN_GRID, TOKEN_GRID),
                              mode='area')[0].numpy()
    result = fractions.argmax(axis=0).astype(np.int64)
    result[fractions.max(axis=0) < PURITY] = -1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chips', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with args.chips.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or {'chip_id', 'imagery', 'labels', 'split'} - set(rows[0]):
        raise SystemExit('chips.csv needs columns chip_id, imagery, labels, split')
    backbone = load_backbone(args.backbone, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.chips.parent
    out_rows = []
    for row in rows:
        with np.load(base / row['imagery']) as saved:
            imagery = saved['imagery']
        with np.load(base / row['labels']) as saved:
            labels = saved['labels']
        if labels.shape != imagery.shape[1:]:
            raise ValueError(f'{row["chip_id"]}: labels {labels.shape} do not match imagery {imagery.shape}')
        target = args.output_dir / f'{row["chip_id"]}.npz'
        np.savez_compressed(target, tokens=tokens(backbone, [imagery], args.device)[0],
                            token_labels=token_labels(labels))
        out_rows.append({'tokens': target.resolve().as_posix(), 'split': row['split']})
    with (args.output_dir / 'tokens.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tokens', 'split'])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'{len(out_rows)} chips -> {args.output_dir / "tokens.csv"}')


if __name__ == '__main__':
    main()

"""Score a before/after Sentinel-1/2 chip pair for forest-to-agriculture conversion.

    python inference.py --before samples/inputs/<chip>_2020.npz \
                        --after  samples/inputs/<chip>_2024.npz \
                        --output outputs/<chip>.npz

Each input is an .npz with `imagery` (14, H, W): Sentinel-1 VV, VH (dB) followed by
Sentinel-2 B01-B12 surface-reflectance DN, as written by scripts/download_gee.py. An
optional `qa` (2, H, W) holds per-pixel S1/S2 observation counts; chips with more
than 10% unobserved pixels in either sensor are refused, as in the study, unless
--allow-gaps is given. The study used 440 m chips (44 x 44 pixels at 10 m).

Writes per-patch probability maps (14 x 14) to the .npz and a JSON summary beside it.
Scores are uncalibrated rankings, not probabilities of conversion or compliance verdicts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from f2a.model import (BAND_ORDER, CLASSES, DEFAULT_BACKBONE, DEFAULT_HEAD, ROOT, load_backbone,
                       load_head, score_pair, sha256)

MAX_GAP_FRACTION = 0.1


def read_chip(path: Path, allow_gaps: bool) -> np.ndarray:
    with np.load(path) as saved:
        imagery = saved['imagery'].astype(np.float32)
        qa = saved['qa'] if 'qa' in saved else None
    if imagery.ndim != 3 or imagery.shape[0] != len(BAND_ORDER):
        raise ValueError(f'{path}: expected imagery (14, H, W) in order {BAND_ORDER}, got {imagery.shape}')
    if qa is not None and not allow_gaps:
        for sensor, counts in zip(('Sentinel-1', 'Sentinel-2'), qa):
            gap = float((counts == 0).mean())
            if gap > MAX_GAP_FRACTION:
                raise ValueError(f'{path}: {sensor} has no observation for {gap:.0%} of pixels '
                                 f'(limit {MAX_GAP_FRACTION:.0%}); use --allow-gaps to score anyway')
    return imagery


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--before', type=Path, required=True, help='chip before the cutoff (e.g. 2020)')
    parser.add_argument('--after', type=Path, required=True, help='later chip (e.g. 2024)')
    parser.add_argument('--output', type=Path, required=True, help='output .npz; a .json is written beside it')
    parser.add_argument('--backbone', type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument('--head', type=Path, default=DEFAULT_HEAD)
    parser.add_argument('--operating-points', type=Path, default=ROOT / 'heads/operating_points.json',
                        help='thresholds JSON (e.g. from training/head/select_thresholds.py)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--allow-gaps', action='store_true')
    args = parser.parse_args()
    if not args.backbone.is_file():
        raise SystemExit(f'Backbone not found: {args.backbone}. Download it from EOTDL (see README.md).')
    before = read_chip(args.before, args.allow_gaps)
    after = read_chip(args.after, args.allow_gaps)
    if before.shape != after.shape:
        raise ValueError(f'before {before.shape} and after {after.shape} chips differ in shape')
    result = score_pair(load_backbone(args.backbone, args.device), load_head(args.head),
                        before, after, args.device)
    thresholds = json.loads(args.operating_points.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, p_before=result['p_before'], p_after=result['p_after'],
                        temporal_map=result['temporal_map'], classes=np.array(CLASSES))
    summary = {
        'before': str(args.before), 'after': str(args.after),
        'temporal_score': result['temporal_score'],
        'after_only_score': result['after_only_score'],
        'temporal_flag': result['temporal_score'] >= thresholds['temporal']['threshold'],
        'after_only_flag': result['after_only_score'] >= thresholds['after_only']['threshold'],
        'thresholds': {k: thresholds[k]['threshold'] for k in ('temporal', 'after_only')},
        'backbone_sha256': sha256(args.backbone), 'head_sha256': sha256(args.head),
        'device': args.device,
        'note': 'Uncalibrated scores; thresholds come from the Para/Mato Grosso 2020-2024 benchmark.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'temporal score {summary["temporal_score"]:.4f} '
          f'({"flagged" if summary["temporal_flag"] else "not flagged"}); '
          f'after-only score {summary["after_only_score"]:.4f} '
          f'({"flagged" if summary["after_only_flag"] else "not flagged"})')
    print(f'Wrote {args.output} and {args.output.with_suffix(".json")}')


if __name__ == '__main__':
    main()

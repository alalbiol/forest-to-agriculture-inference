"""Export the inference-only backbone from the original fine-tuning checkpoint.

    python scripts/export_checkpoint.py --source /path/to/TerraFM_EarlyFusion.pth
The original file (1.3 GB) also holds the optimiser state and the 19-class BigEarthNet
head, neither of which is used here. Only the `backbone.*` tensors are kept, unchanged,
and saved as a weights-only file. The source hash is checked first; the exported weights
are then compared tensor for tensor with the source and written to provenance/checkpoint.json.

For a backbone you fine-tuned yourself with training/backbone/train.py:
    python scripts/export_checkpoint.py --source <save_dir>/best_model.pth --any-source \
        --output checkpoints/my_backbone.pt
skips the hash check and writes the record beside the output instead of into provenance/.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f2a.model import DEFAULT_BACKBONE, load_backbone, sha256  # noqa: E402

SOURCE_SHA256 = '5a7c585e66752364360d54f21c6c4197b87cc85eec670c11650012d86c8d7bdc'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument('--any-source', action='store_true',
                        help='export a checkpoint other than the released one')
    args = parser.parse_args()
    source_sha256 = sha256(args.source)
    if not args.any_source and source_sha256 != SOURCE_SHA256:
        raise SystemExit('Source checkpoint hash differs from the one used in the study.')
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; remove it to re-export.')
    original = torch.load(args.source, map_location='cpu', weights_only=False)
    state = original['model_state_dict']
    backbone = {k[len('backbone.'):]: v.clone() for k, v in state.items() if k.startswith('backbone.')}
    dropped = sorted({k.split('.')[0] for k in state if not k.startswith('backbone.')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': backbone, 'architecture': 'terrafm_base',
                'input': '14 bands (VV, VH, B01-B12), 224 x 224, BigEarthNet normalisation',
                'source_sha256': source_sha256, 'source_epoch': int(original['epoch'])},
               args.output)
    model = load_backbone(args.output)
    reloaded = model.state_dict()
    assert reloaded.keys() == backbone.keys()
    assert all(torch.equal(reloaded[k], backbone[k]) for k in backbone)
    record = {'filename': args.output.name, 'bytes': args.output.stat().st_size,
              'sha256': sha256(args.output), 'tensors': len(backbone),
              'parameters': int(sum(v.numel() for v in backbone.values())),
              'source': {'filename': args.source.name, 'bytes': args.source.stat().st_size,
                         'sha256': source_sha256, 'epoch': int(original['epoch']),
                         'metrics': original.get('metrics'), 'dropped_modules': dropped,
                         'dropped_top_level_keys': [k for k in original if k != 'model_state_dict']}}
    target = args.output.with_suffix('.json') if args.any_source else ROOT / 'provenance/checkpoint.json'
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Exported {record["tensors"]} tensors ({record["bytes"] / 1e6:.0f} MB) to {args.output}')


if __name__ == '__main__':
    main()

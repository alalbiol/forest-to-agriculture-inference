"""Check the backbone checkpoint against provenance/checkpoint.json after downloading it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f2a.model import DEFAULT_BACKBONE, sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=DEFAULT_BACKBONE)
    args = parser.parse_args()
    expected = json.loads((ROOT / 'provenance/checkpoint.json').read_text())
    if not args.path.is_file():
        raise SystemExit(f'Missing {args.path}; download the EOTDL model snapshot (README.md).')
    if args.path.stat().st_size != expected['bytes'] or sha256(args.path) != expected['sha256']:
        raise SystemExit(f'Checkpoint hash mismatch: {args.path}')
    print(f'{args.path.name}: verified ({expected["bytes"] / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()

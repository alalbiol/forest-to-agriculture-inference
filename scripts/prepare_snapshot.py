"""Build the EOTDL model snapshot from a clean, tagged checkout plus the backbone checkpoint.

    python scripts/prepare_snapshot.py --tag v1.0.0
Writes dist/<tag>/forest-to-agriculture-inference/ with independent copies of every
Git-tracked file and the verified checkpoint, plus release-manifest.json. Never uploads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'forest-to-agriculture-inference'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'dist' / args.tag / NAME
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Commit source changes before building a release snapshot.')
    commit = git('rev-parse', 'HEAD')
    if commit != git('rev-parse', f'{args.tag}^{{commit}}'):
        raise SystemExit(f'Check out {args.tag} to build its snapshot.')
    readme = (ROOT / 'README.md').read_text()
    if not readme.startswith('---\n') or not re.search(rf'^name: {NAME}$', readme.split('---', 2)[1], re.M):
        raise SystemExit('README.md must start with EOTDL metadata naming this model.')
    sources = [ROOT / p for p in git('ls-files').splitlines() if p != '.gitignore']
    checkpoint = json.loads((ROOT / 'provenance/checkpoint.json').read_text())
    weights = ROOT / 'checkpoints' / checkpoint['filename']
    if digest(weights) != checkpoint['sha256']:
        raise SystemExit(f'Checkpoint hash mismatch: {weights}')
    sources.append(weights)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f'Expected a regular file: {source}')
        relative = source.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy rather than link, so the snapshot cannot change through the working tree.
        shutil.copyfile(source, target)
        entries.append({'path': relative.as_posix(), 'bytes': target.stat().st_size, 'sha256': digest(target)})
    manifest = {'name': NAME, 'release': args.tag, 'github_commit': commit,
                'github_release': f'https://github.com/alalbiol/{NAME}/releases/tag/{args.tag}',
                'eotdl_model': f'https://www.eotdl.com/models/{NAME}', 'eotdl_version': None,
                'files': entries,
                'note': 'Lists every payload file except itself. The EOTDL version is assigned on upload.'}
    (output / 'release-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'Snapshot ready: {output}; {len(entries)} files; '
          f'{sum(e["bytes"] for e in entries) / 1e6:.0f} MB. Nothing uploaded.')


if __name__ == '__main__':
    main()

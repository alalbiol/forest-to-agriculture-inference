"""Step 2 of head retraining: fit the 768 -> 4 land-use head with the study's recipe.

    python training/head/train_head.py --tokens tokens/tokens.csv --output heads/my_head.pt

tokens.csv columns: tokens (path to an .npz with `tokens` (768, 14, 14) and
`token_labels` (14, 14)), split (train or validation). Row order fixes token order.

Recipe of the released head, unchanged: tokens standardised with training mean and std
(std floored at 0.05); one linear layer; class weights sqrt(N / n_k), rescaled to mean 1;
AdamW (lr 1e-3, weight decay 0.01) on shuffled 4,096-token batches; seed 20261001;
up to 100 epochs, keeping the epoch with the lowest weighted validation loss and
stopping after 12 epochs without improvement (from epoch 20). Validation tokens are used
only for this choice. Given the benchmark's cached features in the benchmark's row order,
this reproduces the released head tensor for tensor (provenance/head_training_check.json).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f2a.model import CLASSES, EMBED_DIM  # noqa: E402

SEED = 20261001


def load(rows, split):
    xs, ys = [], []
    for row in rows:
        if row['split'] != split:
            continue
        with np.load(row['tokens']) as f:
            x = f['tokens'].astype(np.float32).reshape(-1, 196).T
            y = f['token_labels'].reshape(-1)
        valid = y >= 0
        xs.append(x[valid])
        ys.append(y[valid])
    if not xs:
        raise SystemExit(f'No {split} chips in the tokens table')
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)).long(), len(xs)


def fit(rows, checkpoint: Path, max_epochs=100):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)
    train_x, train_y, train_chips = load(rows, 'train')
    val_x, val_y, val_chips = load(rows, 'validation')
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(.05)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    count = torch.bincount(train_y, minlength=len(CLASSES))
    if (count == 0).any():
        raise SystemExit(f'Every class needs training tokens; counts {count.tolist()} for {CLASSES}')
    weights = count.float().sum().div(count.float()).sqrt()
    weights = weights / weights.mean()
    model = torch.nn.Linear(EMBED_DIM, len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.
        for batch in torch.randperm(len(train_y)).split(4096):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_x[batch]), train_y[batch], weight=weights)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = sum(float(F.cross_entropy(model(val_x[b]), val_y[b], weight=weights, reduction='sum'))
                           for b in torch.arange(len(val_y)).split(8192)) / float(weights[val_y].sum())
        history.append({'epoch': epoch, 'train_loss': total / len(train_y), 'validation_loss': val_loss})
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'classes': CLASSES,
                        'epoch': epoch, 'validation_loss': best, 'train_token_counts': count.tolist(),
                        'key': 'tokens', 'seed': SEED}, checkpoint)
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'epoch {epoch}: validation loss {val_loss:.5f}', flush=True)
        if epoch >= 20 and stale >= 12:
            break
    return {'train_chips': train_chips, 'validation_chips': val_chips, 'train_tokens': len(train_y),
            'validation_tokens': len(val_y), 'class_token_counts': count.tolist(),
            'class_weights': weights.tolist(), 'best_validation_loss': best, 'history': history}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; choose another output.')
    with args.tokens.open() as f:
        rows = list(csv.DictReader(f))
    base = args.tokens.parent
    for row in rows:
        row['tokens'] = str(base / row['tokens'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = fit(rows, args.output)
    report['tokens_table_sha256'] = hashlib.sha256(args.tokens.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output} (best validation loss {report["best_validation_loss"]:.5f})')


if __name__ == '__main__':
    main()

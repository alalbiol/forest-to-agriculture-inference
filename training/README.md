# Training

The released model has two trained parts, and this folder holds the code for both:

| Part | Code | Status |
|---|---|---|
| Land-use head (768 → 4) and flag thresholds | `head/` | Verified: reproduces the released head tensor for tensor and both released thresholds exactly |
| Backbone: TerraFM-B fine-tuned on BigEarthNet with early S1/S2 fusion | `backbone/` | Provided as used to produce the released backbone; not re-run for this release |

Most users who apply the model to a new region need only `head/`. You keep the frozen backbone, retrain the small head on local labels, and choose thresholds on local validation data. The model card advises this outside Pará and Mato Grosso.

## Retrain the land-use head on your labels

You need labelled chips (one date each, for training the head) and labelled chip pairs (for choosing thresholds). Chips use the same `.npz` format as `inference.py`; `scripts/download_gee.py` produces it.

**1. Tokens and patch labels.** `chips.csv` has the columns `chip_id, imagery, labels, split`. Each `labels` file holds `labels` (H, W) on the chip grid: 0 primary forest, 1 secondary forest, 2 agriculture, 3 mining; any other value is ignored. A patch takes a class only if that class covers at least 80% of it.

```bash
python training/head/extract_tokens.py --chips chips.csv --output-dir tokens/
```

**2. Fit the head.** This is the study's recipe, unchanged: standardised tokens, square-root inverse-frequency class weights, AdamW, seed 20261001, and early stopping on the weighted validation loss.

```bash
python training/head/train_head.py --tokens tokens/tokens.csv --output heads/my_head.pt
```

**3. Choose thresholds.** `validation_pairs.csv` has the columns `chip_id, before, after, conversion` (1 = forest → agriculture, 0 = anything else to reject). Use pairs that were not used in step 2.

```bash
python training/head/select_thresholds.py --pairs validation_pairs.csv \
    --head heads/my_head.pt --output heads/my_operating_points.json --max-fpr 0.1
python inference.py --before a_2020.npz --after a_2024.npz --output outputs/a.npz \
    --head heads/my_head.pt --operating-points heads/my_operating_points.json
```

All four classes need training patches. The study used 448 training chips (81,713 labelled patches) and 136 validation chips. Far fewer will give a noisier head.

**Verification** (`provenance/head_training_check.json`):
- Given the benchmark's cached features in the benchmark's order, `train_head.py` reproduces the released head exactly: same weights, standardisation, epoch 13 and validation loss.
- The patch-label rule reproduces the cached patch labels for all 732 benchmark chips.
- The threshold rule reproduces both released thresholds (0.5609 temporal, 0.7643 after-only) from the benchmark validation scores.

## Backbone fine-tuning on BigEarthNet

The same backbone, with its 19-class head, is released as `terrafm_early` in [sentinel-landcover-inference](https://github.com/alalbiol/sentinel-landcover-inference). That asset also holds the BigEarthNet test-set evaluation and the training code for all three fusion variants.

`backbone/train.py` is the AI4DIS script that produced the released backbone, trimmed to the TerraFM early-fusion configuration. The settings are in `backbone/finetune_terrafm_early_fusion.yaml`:

- **Start:** the public TerraFM-B weights, plus a new 14-channel early-fusion patch embedding (S1 VV, VH and S2 B01–B12 enter one convolution together).
- **Data:** BigEarthNet v2, 19-class multi-label, all 14 bands (ConfigILM band set `ALL`: VV, VH, B01–B12, the order `inference.py` uses), 224 × 224, BigEarthNet normalisation.
- **Schedule:**
  - epochs 1–5: backbone frozen; only the fusion embedding and the classification head train;
  - epoch 6 onwards: everything is unfrozen, and the optimiser and cosine schedule restart.
- **Optimiser and loss:** AdamW (lr 1e-4, weight decay 0.05), 5 warm-up epochs, gradient clip 1.0, mixed precision, horizontal/vertical flips, BCE loss, batch 32, seed 42, 10 epochs.
- **Selection:** the checkpoint with the best validation macro AP is kept. In the original run that was the 9th epoch (stored index 8): macro AP 0.776, micro AP 0.867.

Setup (GPU recommended):

```bash
python -m pip install -r requirements.txt -r training/backbone/requirements.txt
# 1. BigEarthNet v2 in ConfigILM's layout (LMDB + metadata parquet files):
#    https://bigearth.net and https://github.com/lhackel-tub/ConfigILM
# 2. TerraFM-B weights: https://github.com/mbzuai-oryx/TerraFM
# 3. Set the paths in the YAML, then:
python training/backbone/train.py --config training/backbone/finetune_terrafm_early_fusion.yaml
python scripts/export_checkpoint.py --source checkpoints_early_fusion/best_model.pth \
    --any-source --output checkpoints/my_backbone.pt
```

What was checked for this release:
- The trimmed classifier loads the original fine-tuning checkpoint strictly and gives identical outputs to the original AI4DIS class.
- Freezing leaves the same 8,278,291 parameters trainable as in the original run.
- A CPU training step runs end to end, and its checkpoint exports to the inference format (`tests/test_training.py`).

The full fine-tuning was **not** re-run. A re-run will not be bit-identical because of GPU nondeterminism, mixed precision and data order. A new backbone also changes every token, so retrain the head (above) after replacing it.

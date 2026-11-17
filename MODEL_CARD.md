# Forest-to-agriculture temporal scoring: model card, v1.0.0

## Task

Rank 440 m Sentinel-1/2 chips by evidence that primary forest present before a cutoff date was converted to agriculture by a later date. The output is two chip scores and their per-patch maps: a temporal score (primary forest before × agriculture after) and an after-only agriculture score. It is a screening ranking that points to where closer inspection is needed. It is not a land-use map, a deforestation detector for other end uses, or a compliance decision.

## Components

| Component | Description | Trained on |
|---|---|---|
| Backbone | TerraFM-B (ViT-B/16) with a 14-channel early S1/S2 fusion patch embedding, frozen | TerraFM-B public weights, fine-tuned for 10 epochs on BigEarthNet v2 19-class multi-label (best validation macro AP 0.776 at the 9th epoch); code in `training/backbone/` |
| Land-use head | Linear 768 → 4 on block-11 patch tokens, with stored input standardisation | 448 training chips (81,713 tokens) of the Brazil benchmark, labelled from TerraClass 2024; epoch chosen by validation loss on 136 chips; code in `training/head/` |
| Thresholds | Temporal 0.561, after-only 0.764 (`heads/operating_points.json`) | Maximum validation recall with ≤10% false positives among forest-loss controls |

Details and hashes: `provenance/origin.json`, `provenance/checkpoint.json`. Training code and how to retrain the head on local labels: [training/README.md](training/README.md). The released backbone file contains only the backbone tensors, copied unchanged from the original fine-tuning checkpoint. The optimiser state and the BigEarthNet classification head are dropped.

## Evaluation (from the benchmark)

Pará and Mato Grosso, Brazil; 2020 and 2024 annual composites; references from TerraClass maps and MapBiomas Alerta polygons, not field observations. Test set: 84 forest-loss chips that became agriculture, compared with 19 that became mining and 10 that became secondary forest, split by 10-km geographic blocks.

| | Temporal | After-only |
|---|---:|---:|
| AUROC, agriculture vs other post-loss uses | 0.963 | 0.964 |
| Recall at threshold (84 conversions) | 68 | 73 |
| False positives among 29 other post-loss chips | 1 | 2 |
| Flagged among 100 chips mapped as agriculture in both years | 3 | 97 |

A two-date spectral/radar random forest and a temporal spectral method performed comparably in ranking. The study does **not** show an advantage of TerraFM features over conventional temporal methods. Its result is that requiring before-date forest evidence prevents current agriculture from being read as recent conversion. The full analysis and data are in the benchmark asset.

## Verification of this release

`scripts/verify_samples.py` re-scores six bundled chip pairs from raw imagery and compares them with the benchmark's stored scores, which were computed from GPU-extracted features. On CPU the maximum difference was 4.1 × 10⁻⁶ (`provenance/validation.json`). Fresh Earth Engine downloads for two sample locations were bit-identical to the study imagery (`provenance/gee_smoke.json`). This demonstrates implementation agreement, not accuracy beyond the benchmark.

## Limitations

- Evaluated only in Pará and Mato Grosso with 2020/2024 composites. The sample is pasture-dominated (80 of 84 conversions) and most of it comes from Pará. Transfer to other biomes, crops, years or sensors is untested.
- The references are map-derived, so errors in TerraClass or MapBiomas propagate into both training and evaluation.
- Scores are uncalibrated, and the thresholds are tied to the benchmark's class mix. Choose operating points for a new region on local reference data.
- Annual medians blur the timing within a year, and clouds and missing observations change the composites. Chips with more than 10% unobserved pixels are refused by default.
- Chip size and resolution are fixed by training (440 m resized to 224 × 224). Large-area tiling is not implemented.
- Intended for research screening. Do not use it as the sole basis for legal or commercial decisions about a property or supplier.

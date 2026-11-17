---
name: forest-to-agriculture-inference
authors:
  - Alberto Albiol
  - Miquel Obrador-Reina
  - Juan B. Pedro
  - Albert Fernández
license: MIT AND CC-BY-SA-4.0
source: https://github.com/alalbiol/forest-to-agriculture-inference/tree/v1.0.0
---

# Forest-to-agriculture inference

Score a pair of Sentinel-1/2 image chips, one from before a cutoff date and one later, for conversion from primary forest to agriculture. The model is the temporal TerraFM rule evaluated in *Temporal Evidence in Forest-to-Agriculture Screening: A Benchmark in Pará and Mato Grosso, Brazil* (Albiol, Obrador-Reina, Pedro and Fernández). It uses a frozen TerraFM backbone fine-tuned on BigEarthNet and a linear land-use classifier applied to each image patch.

**Publication status:** the manuscript is under revision. It will be linked here, or its arXiv version, after acceptance.

**Release:** [GitHub v1.0.0](https://github.com/alalbiol/forest-to-agriculture-inference/releases/tag/v1.0.0). **EOTDL model:** [`forest-to-agriculture-inference`](https://www.eotdl.com/models/forest-to-agriculture-inference); the EOTDL version is recorded in `provenance/releases/` after upload.
GitHub holds the code, the land-use classifier, thresholds and six real example chips. The EOTDL snapshot additionally holds the 485 MB backbone checkpoint.

The benchmark data and the code that reproduces the paper's results are a separate asset: [brazil-forest-to-agriculture-benchmark](https://github.com/alalbiol/brazil-forest-to-agriculture-benchmark).

## What it computes

For each 440 m chip (44 × 44 pixels, 10 m):

1. Each year's 14-band composite is resized to 224 × 224, normalised, and passed through the frozen backbone. Block-11 tokens form a 14 × 14 grid of 768-d patch features.
2. The land-use classifier gives each patch probabilities of primary forest, secondary forest, agriculture and mining.
3. **Temporal score** = mean of the nine largest values of *p*<sub>before</sub>(primary forest) × *p*<sub>after</sub>(agriculture).
   **After-only score** = mean of the nine largest *p*<sub>after</sub>(agriculture), for comparison.

The after-only score detects current agriculture, but cannot tell when it arrived. In the benchmark it flagged 97 of 100 locations that were already agriculture in 2020; the temporal score flagged 3. Both scores ranked agricultural conversions against other post-loss land uses similarly (AUROC 0.963 temporal, 0.964 after-only).

Scores are **uncalibrated rankings**, not probabilities of conversion and not legal compliance verdicts. The supplied flags use thresholds chosen on the benchmark's validation split (Pará/Mato Grosso, 2020/2024). See [MODEL_CARD.md](MODEL_CARD.md) for scope and limitations.

## Quick start

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# CPU-only: python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Download the EOTDL snapshot (see below), or copy its `checkpoints/` folder into this repository, then:

```bash
python scripts/verify_checkpoint.py
python inference.py \
  --before samples/inputs/brazil_para__stable_agriculture__015_2020.npz \
  --after  samples/inputs/brazil_para__stable_agriculture__015_2024.npz \
  --output outputs/example.npz --device cpu
```

The `.npz` output holds the per-patch maps `p_before` and `p_after` (4 × 14 × 14, classes in `classes`) and `temporal_map` (14 × 14). The `.json` beside it holds both chip scores, the threshold flags and the checkpoint hashes. Once the model is loaded, scoring takes about 0.2 s per chip pair on a CPU.

Check all six samples against the scores saved in the benchmark:

```bash
python scripts/verify_samples.py --device cpu
```

| Sample | Reference (TerraClass) | Temporal | After-only |
|---|---|---:|---:|
| `brazil_para__last_agriculture__561423` | forest loss → agriculture | 0.902 flagged | 0.989 flagged |
| `brazil_para__last_agriculture__1038454` | forest loss → agriculture | 0.252 | 0.381 |
| `brazil_para__last_mining__682319` | forest loss → mining | 0.044 | 0.046 |
| `brazil_para__last_secondary_forest__715289` | forest loss → secondary forest | 0.720 flagged | 0.971 flagged |
| `brazil_matogrosso__stable_agriculture__041` | agriculture in 2020 and 2024 | 0.021 | 0.974 flagged |
| `brazil_para__stable_agriculture__015` | agriculture in 2020 and 2024 | 0.779 flagged | 0.998 flagged |

The samples include correct and incorrect outcomes on purpose. On CPU the scores agree with the stored GPU results to within 5 × 10⁻⁶ (`provenance/validation.json`).

## Input format

One `.npz` per chip and year:

- `imagery`, float32 (14, H, W): Sentinel-1 VV, VH in dB (as in Earth Engine `COPERNICUS/S1_GRD`), then Sentinel-2 B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 as surface-reflectance DN (`COPERNICUS/S2_SR_HARMONIZED`, approximately reflectance × 10000). **Do not rescale.**
- `qa`, optional (2, H, W): per-pixel Sentinel-1 and Sentinel-2 observation counts. Chips with more than 10% unobserved pixels for either sensor are refused, as in the study (override with `--allow-gaps`).

The study used annual median composites over 440 m boxes. Other chip sizes are resized to 224 × 224 but were not evaluated.

## Optional: download a chip from Earth Engine

```bash
python -m pip install -r requirements-gee.txt
earthengine authenticate
export GEE_PROJECT="your-earth-engine-enabled-project"
python scripts/download_gee.py --lon -50.8762 --lat -3.5342 --years 2020 2024 \
  --chip-id my_chip --output-dir downloads
python inference.py --before downloads/my_chip_2020.npz --after downloads/my_chip_2024.npz \
  --output outputs/my_chip.npz
```

The downloader reproduces the study's compositing: annual medians, the QA60 cloud mask, scenes less than 70% cloudy, Sentinel-1 IW VV/VH, and an explicit 44 × 44 grid. For two sample locations, fresh downloads were bit-identical to the study imagery (`provenance/gee_smoke.json`). You need your own Earth Engine account and project.

## Training and retraining

[training/README.md](training/README.md) covers both trained parts:

- **`training/head/`** retrains the land-use classifier on your own labelled chips and chooses thresholds on your own validation pairs. This is the recommended step before using the model outside Pará and Mato Grosso. The scripts reproduce the released classifier tensor for tensor, and both released thresholds exactly.
- **`training/backbone/`** is the BigEarthNet fine-tuning that produced the backbone. It is provided as used and was not re-run for this release.

## Repository guide

```text
inference.py      Command-line scoring of a before/after chip pair
f2a/              Preprocessing, backbone loading, land-use head and scores; TerraFM architecture
heads/            Land-use classifier (768 → 4) and validation-selected thresholds
checkpoints/      (EOTDL) TerraFM backbone, BigEarthNet early-fusion fine-tune
samples/          Six real 2020/2024 chip pairs and their benchmark scores
scripts/          Checkpoint and sample verification, Earth Engine download, export, snapshot
training/         Head retraining on new labels (verified) and backbone fine-tuning on BigEarthNet (as used)
provenance/       Checkpoint hashes and origin, training details, validation records
tests/            Input and training tests (python -m unittest discover -s tests)
```

## EOTDL download

```bash
eotdl auth login
eotdl models get forest-to-agriculture-inference --path ./downloads
```

The snapshot is self-contained: it holds the full tagged repository plus `checkpoints/`. See [EOTDL.md](EOTDL.md) for building and publishing it.

## Licence and attribution

- Code: [MIT](LICENSE). `f2a/terrafm.py` is the TerraFM architecture (MIT, © TerraFM Authors, MBZUAI). Parts of it are adapted from DINO and remain under Apache-2.0. See [THIRD_PARTY.md](THIRD_PARTY.md).
- Backbone checkpoint: [MIT](LICENSE). It is fine-tuned from the MIT-licensed TerraFM-B weights on BigEarthNet (CDLA-Permissive-1.0).
- Land-use classifier and thresholds (`heads/`): [CC BY-SA 4.0](LICENSE-HEAD). They were trained and selected on labels derived from TerraClass (CC BY-SA), and match the benchmark's data licence.
- Sample imagery: contains modified Copernicus Sentinel data (2020, 2024).

Prepared within AI4DIS, funded by the European Space Agency under Contract No. 4000147555/25/I-DT. The Universitat Politècnica de València participated as a subcontractor to EarthPulse S.L., the prime contractor. The views expressed are those of the authors and do not necessarily reflect the official position of the European Space Agency.

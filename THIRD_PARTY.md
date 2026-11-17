# Sources and third-party components

- **TerraFM** (`f2a/terrafm.py`): architecture code from https://github.com/mbzuai-oryx/TerraFM, MIT licence, © 2025 TerraFM Authors, Mohamed bin Zayed University of Artificial Intelligence. It is copied with one addition from the AI4DIS project: the 14-channel early-fusion patch embedding (`conv2d_s1_s2`, `s1_s2_embed`). The TerraFM licence also covers the released TerraFM weights from which the backbone was fine-tuned. Cite Danish et al., *TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation*, arXiv:2506.06281.
- **DINO**: parts of `f2a/terrafm.py` are adapted from https://github.com/facebookresearch/dino, © Facebook, Inc. and its affiliates, Apache License 2.0. Those portions remain under Apache-2.0.
- **BigEarthNet**: fine-tuning data for the backbone (Sumbul et al.), Community Data License Agreement – Permissive 1.0. It is not redistributed here.
- **ConfigILM** (MIT, © 2023 Leonard Wayne Hackel; https://github.com/lhackel-tub/ConfigILM): BigEarthNet v2 loader and band statistics used by `training/backbone/train.py`; an installed dependency, not bundled.
- **Weights & Biases**: optional logging in `training/backbone/train.py`; disabled by default.
- **TerraClass** (INPE/Embrapa, CC BY-SA 3.0) and **MapBiomas Alerta**: the sources of the benchmark references used to train the land-use head and select thresholds. This is why `heads/` is released under CC BY-SA 4.0. They are not redistributed here; see the benchmark asset.
- **Copernicus Sentinel-1/2**: the sample imagery contains modified Copernicus Sentinel data (2020, 2024), composited in Google Earth Engine (`COPERNICUS/S1_GRD`, `COPERNICUS/S2_SR_HARMONIZED`).
- **Earth Engine**: an optional external service that requires the user's own authentication and project.
- **PyTorch, NumPy, scikit-learn, earthengine-api, requests**: installed dependencies under their own licences.

No training dataset, manuscript, credentials or Google project configuration is included in this repository or the snapshot.

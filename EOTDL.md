# EOTDL model snapshot

This is an EOTDL **model entry** named `forest-to-agriculture-inference`. The root README
carries its EOTDL metadata and description. GitHub holds the code, the land-use
classifier, thresholds and samples. The snapshot is the complete tagged Git tree plus the
backbone checkpoint (`checkpoints/terrafm_b_bigearthnet_early_fusion.pt`, 485 MB),
which is not stored in Git.

## Published version

GitHub release `v1.0.0` (commit `4d6a7bc918345357690a980707f8cfd8cc66a8f9`) corresponds to EOTDL model version **1** (ID `6afc5814dbc0af5e134f7e00`). The EOTDL catalog checksum of every one of the 51 payload files, and the uploaded `release-manifest.json` itself, were checked against the snapshot built from that commit. See `provenance/releases/v1.0.0.json`. Changes on `main` after this tag are not part of the upload; a new upload creates a new EOTDL version.

## Build the snapshot

From a clean checkout of the tag, with the checkpoint in `checkpoints/`:

```bash
git checkout v1.0.0
python scripts/verify_checkpoint.py
python scripts/verify_samples.py
python scripts/prepare_snapshot.py --tag v1.0.0
```

The snapshot is written to `dist/v1.0.0/forest-to-agriculture-inference/` as
independent copies, with `release-manifest.json` (commit, tag, per-file SHA-256).
Existing folders are never overwritten. To rebuild the checkpoint from the original
1.3 GB fine-tuning file:

```bash
python scripts/export_checkpoint.py --source /path/to/TerraFM_EarlyFusion.pth
```

## Upload

Use a separate Python 3.12+ environment for the EOTDL client:

```bash
python3.12 -m venv /tmp/eotdl-uploader
source /tmp/eotdl-uploader/bin/activate
python -m pip install eotdl
eotdl auth login
eotdl models ingest -p ./dist/v1.0.0/forest-to-agriculture-inference
```

Push the GitHub repository and release first, so the `source` link resolves. After
upload, download the published manifest and check its hashes, add the release record,
run `python scripts/update_github_about.py` (after `gh auth login`), and add the model
to the AI4DIS project README's EOTDL resources table.

## Download

```bash
eotdl auth login
eotdl models get forest-to-agriculture-inference --version <N> --path ./downloads
```

Run `python scripts/verify_checkpoint.py` before inference.

[Official model ingestion instructions](https://www.eotdl.com/docs/models/ingest)

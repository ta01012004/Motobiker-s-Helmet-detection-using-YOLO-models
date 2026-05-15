# Checkpoints and Large Artifacts

Model weights are intentionally excluded from this Git repository.

## Why weights are not committed

GitHub blocks files larger than 100 MB and normal Git repositories become slow when datasets, videos, and repeated checkpoints are committed. This project therefore tracks code, documentation, configs, paper files, and small representative result images only.

## Recommended files to distribute separately

If you want to share a trained model, upload only the final selected checkpoint:

```text
runs/helmet_yolov8m_person-bike-helmet3/weights/best.pt
```

Recommended distribution options:

- GitHub Release asset
- Google Drive / OneDrive link
- DVC remote storage
- Git LFS, only if the repository owner accepts LFS quota usage

## Local expected paths

After downloading a checkpoint, pass it explicitly:

```bash
python src/infer_riders.py \
  --weights path/to/best.pt \
  --source path/to/images \
  --out-dir outputs/rider_inference
```

The original experiment used several local folders such as `models/`, `models_trained/`, and `runs/`. Those paths are ignored by Git and are not required as long as you pass explicit paths to the `src/` scripts.

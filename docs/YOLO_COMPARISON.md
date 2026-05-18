# YOLOv8n / YOLOv8s Comparison Runs

The baseline paper run uses YOLOv8m:

```text
runs/helmet_yolov8m_person-bike-helmet3/
```

To compare smaller YOLO variants with the same main setup, use:

```bash
sbatch scripts/slurm_train_compare.sh n
sbatch scripts/slurm_train_compare.sh s
```

The jobs train:

| Model | Initial weights | Output run |
| --- | --- | --- |
| YOLOv8n | `models/yolov8n.pt` | `runs_compare/yolov8n_120ep/` |
| YOLOv8s | `models/yolov8s.pt` | `runs_compare/yolov8s_120ep/` |

Each job:

1. trains for 120 epochs with the same data split and key hyperparameters as the selected YOLOv8m run;
2. saves training curves, confusion matrix, `results.csv`, and `best.pt`;
3. evaluates the resulting `best.pt` on the `test` split under `runs_compare_test/`.

After both jobs finish, generate the comparison table:

```bash
python scripts/summarize_comparison.py
```

Output:

```text
runs_compare/comparison_summary.csv
```

Submitted local job ids on this workspace:

```text
73325 YOLOv8n
73326 YOLOv8s
```

## Helmet-Only Heuristic Baseline

A simple non-spatial baseline is also provided:

```bash
python scripts/helmet_only_baseline.py \
  --weights runs_compare/yolov8s_120ep/weights/best.pt \
  --source path/to/images \
  --out-dir outputs/helmet_only_baseline \
  --device 0
```

This baseline only checks whether the detector finds any `helmet` boxes in an image. It does not match helmets to persons or motorcycles, so it should be treated as a weak baseline rather than a replacement for the rule-based rider-level pipeline.

Test-split outputs from the local 116-image `test.txt` split:

| Model checkpoint | Helmet detected images | No-helmet-detected images | Total helmet boxes |
| --- | ---: | ---: | ---: |
| YOLOv8m best.pt | 93 | 23 | 230 |
| YOLOv8n best.pt | 92 | 24 | 223 |
| YOLOv8s best.pt | 93 | 23 | 223 |

The corresponding CSV files are generated locally under:

```text
outputs/helmet_only_baseline/
```

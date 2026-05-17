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

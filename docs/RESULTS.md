# Results

The selected paper run is:

```text
runs/helmet_yolov8m_person-bike-helmet3/
```

This is a 120-epoch YOLOv8m run on the main three-class dataset. The paper reports the best mAP@0.5 epoch from this run:

| Metric | Value |
| --- | ---: |
| Precision | 0.8740 |
| Recall | 0.8719 |
| mAP@0.5 | 0.9190 |
| mAP@0.5:0.95 | 0.6905 |

The values correspond to epoch 57 in the local `results.csv`:

```text
precision=0.87399, recall=0.87193, mAP50=0.91895, mAP50-95=0.69046
```

## Main Figures

| Asset | Description |
| --- | --- |
| `assets/results/selected_run_curves.png` | Training and validation curves for the selected run |
| `assets/results/selected_run_confusion_matrix_normalized.png` | Normalized validation confusion matrix |
| `assets/results/selected_run_validation_predictions.jpg` | Representative validation predictions |
| `assets/results/rider_level_demo.jpg` | Qualitative rider-level post-processing output |

Use `assets/results/selected_run_curves.png` as the most appropriate single figure for Fig. 4 in the paper.

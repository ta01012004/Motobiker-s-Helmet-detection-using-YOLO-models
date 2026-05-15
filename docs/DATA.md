# Dataset

The main experiment uses 1,147 images from two sources:

| Source | Images | Original annotation format | Local folder |
| --- | ---: | --- | --- |
| Public helmet detection dataset | 764 | Pascal VOC XML | `helmetdata/` |
| Custom traffic data collected by the author | 383 | YOLO TXT | `dataloc/` |

The public dataset matches the 764-image Helmet Detection dataset commonly distributed on Kaggle:

https://www.kaggle.com/datasets/andrewmvd/helmet-detection/data

The custom data contains traffic images and extracted frames collected for this project. It is referenced in the paper as a custom YOLO-format traffic dataset.

## Main Split

The paper reports the split generated with seed 42:

| Split | Images | Ratio |
| --- | ---: | ---: |
| Train | 917 | about 80% |
| Validation | 114 | about 10% |
| Test | 116 | about 10% |
| Total | 1,147 | 100% |

The training labels use three primitive classes:

```text
0 person
1 motorcycle
2 helmet
```

## Notes

Large datasets, video files, zip archives, and model weights are intentionally excluded from Git. Recreate the local data folders from the public dataset and the custom collection, then update `configs/data.yaml` to point to the prepared YOLO directory.

The paper also mentions an optional combined dataset with 753 additional auto-labeled frames, giving 1,214 training images and 303 validation images. Those combined runs failed early and are not used as the main reported results.

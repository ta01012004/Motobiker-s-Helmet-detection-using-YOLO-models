# Algorithm

The project separates helmet-use recognition into two stages.

## 1. Primitive Object Detection

A pretrained YOLOv8m detector is fine-tuned on three primitive classes:

```text
person, motorcycle, helmet
```

The detector predicts bounding boxes, class ids, and confidence scores. The selected run uses:

| Item | Value |
| --- | --- |
| Model | YOLOv8m pretrained weights |
| Input size | 640 |
| Batch size | 16 |
| Epochs | 120 |
| Optimizer | SGD |
| Evaluation split | Validation |

## 2. Rule-Based Spatial Reasoning

The detector output is post-processed into rider-level labels:

```text
rider_helmet
rider_nohelmet
```

The reasoning module:

1. Filters low-confidence and very small detections per class.
2. Associates `person` boxes with nearby `motorcycle` boxes to form rider candidates.
3. Scores each `person` and `helmet` pair using interpretable spatial cues:
   - helmet center position relative to the person box
   - normalized person-helmet overlap
   - distance to the estimated head area
   - vertical alignment with the upper body/head region
   - helmet confidence
4. Greedily assigns each helmet to at most one rider.
5. Labels a rider as `rider_helmet` when the affinity score is at least 0.5; otherwise it is labeled `rider_nohelmet`.

The paper reports validated metrics only for the three-class detector. The final rider-level labels are qualitative outputs because rider-level ground truth has not yet been annotated.

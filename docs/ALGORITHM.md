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

## Baseline: Helmet-Only Heuristic

The helmet-only heuristic is a deliberately simple baseline. It uses only `helmet` detections from the trained detector and does not associate `person`, `motorcycle`, and `helmet` boxes.

For each image or frame:

1. Run the three-class detector.
2. Keep only detections whose class is `helmet`.
3. If at least one helmet is detected above the confidence threshold, label the scene as `helmet_detected`.
4. Otherwise, label the scene as `no_helmet_detected`.

This baseline is useful as a sanity check because it shows what can be achieved without spatial reasoning. It is not a true rider-level method: it cannot determine whether a detected helmet belongs to the correct motorcycle rider, and it can fail when a helmet belongs to a pedestrian, a passenger, a parked motorcycle, or a background object.

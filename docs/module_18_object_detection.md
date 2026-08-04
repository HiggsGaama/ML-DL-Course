# Module 18: Object Detection & Architecture Overview

## 1. The Physical Intuition: Finding Objects in Spatial Coordinates

Standard Image Classification is like answering a single multiple-choice question: *"What is the main subject of this photo?"* (`"Dog"`).

**Object Detection** is a dual-task challenge: finding WHERE every object is located AND WHAT category it belongs to:

```json
[
  {"class": "dog", "bbox": [ymin, xmin, ymax, xmax], "confidence": 0.96},
  {"class": "car", "bbox": [ymin, xmin, ymax, xmax], "confidence": 0.88}
]
```

It combines **Bounding Box Regression** (predicting continuous spatial coordinate tuples) and **Multiclass Classification** simultaneously across candidate regions.

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. Key Evaluation Concepts

```
INTERSECTION OVER UNION (IoU):
  IoU = Area of Overlap / Area of Union
  Measures overlap between predicted bounding box B_pred and ground truth box B_gt.
  IoU >= 0.5 is considered a successful hit!

NON-MAXIMUM SUPPRESSION (NMS):
  Post-processing filter step that deletes duplicate overlapping candidate boxes around the same object,
  keeping only the single box with highest confidence score.
```

---

### 2. Architectural Paradigms

1. **Two-Stage Detectors (R-CNN, Faster R-CNN)**:
   - *Stage 1*: Region Proposal Network (RPN) suggests candidate object regions (~2,000 proposals).
   - *Stage 2*: Classifier & Regressor evaluates each proposal.
   - *Trade-off*: Highly accurate, but slower inference (10-30 FPS).

2. **One-Stage Detectors (YOLO - You Only Look Once, SSD)**:
   - Divides image into an $S \times S$ grid. Single pass neural network predicts bounding boxes and class probabilities directly for all grid cells simultaneously!
   - *Trade-off*: Ultra-fast real-time inference (60-140+ FPS); perfect for real-time video streams.

---

## 3. Architecture & Visual Diagrams

### IoU Intersection over Union Geometry

```
┌─────────────────────────┐  (Predicted Box B_pred)
│                         │
│       ┌─────────────────┼───────┐  (Ground Truth Box B_gt)
│       │  INTERSECTION   │       │
└───────┼─────────────────┘       │
        │                         │
        └─────────────────────────┘

IoU = Area of Intersection / Area of Union
```

---

## 4. Practical Implementation: IoU and NMS in PyTorch

Let's write a pure PyTorch implementation of the **IoU (Intersection over Union)** metric calculation and **Non-Maximum Suppression (NMS)** logic:

```python
import torch
import torchvision

def compute_iou(box1, box2):
    """
    Calculates IoU between two bounding boxes formatted as [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

# Example Test
pred_box = [50, 50, 150, 150]
gt_box   = [70, 70, 160, 160]

print(f"Calculated IoU Overlap: {compute_iou(pred_box, gt_box):.4f}")

# PyTorch Torchvision Built-in NMS
boxes = torch.tensor([
    [50., 50., 150., 150.],
    [52., 51., 149., 152.],  # Duplicate overlapping box
    [200., 200., 300., 300.]
])
scores = torch.tensor([0.95, 0.88, 0.90])
iou_threshold = 0.5

keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
print(f"Boxes kept after NMS filtering: {keep_indices.tolist()}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the IoU formula.

### Equation 1: Intersection over Union (IoU)
$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

- $|A \cap B|$: Area of intersection overlap between predicted box $A$ and ground-truth box $B$.
- $|A \cup B|$: Combined union area of both boxes.
- Subtracting $|A \cap B|$ from $|A| + |B|$ prevents double-counting the overlap area in the denominator!

---

## 6. Real-World Production Gotchas & Failure Modes

### Mean Average Precision (mAP)
Primary benchmark metric for object detection. Calculates Area Under Precision-Recall Curve averaged across class categories at specified IoU thresholds (e.g., $\text{mAP@50}$ or $\text{mAP@[.50:.95]}$).

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why do one-stage detectors (YOLO) run dramatically faster than two-stage detectors (Faster R-CNN)?
   - *Answer/Explanation*: One-stage detectors frame bounding box regression and classification as a single unified tensor forward pass over grid cells without generating candidate region proposals.

2. **Exercise**: Calculate IoU manually for two identical overlapping $100 \times 100$ boxes shifted by 10 pixels.

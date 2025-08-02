import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
CONF_THRESH = 0.001

# === Download dataset==
with open("/home/artem/Downloads/VisDrone2YOLO-main/convert_json/VisDrone2019-DET_test_coco_start.json") as f:
    gt = json.load(f)
with open("/home/artem/Downloads/y12+SAHI.json") as f:
    preds = json.load(f)

# === Create dict ===
gt_by_image_cat = defaultdict(list)
for ann in gt["annotations"]:
    gt_by_image_cat[(ann["image_id"], ann["category_id"])].append(ann["bbox"])

pred_by_image_cat = defaultdict(list)
for ann in preds:
    pred_by_image_cat[(ann["image_id"], ann["category_id"])].append((ann["bbox"], ann["score"]))

# === Create list of classes ===
category_id_to_name = {cat["id"]: cat["name"] for cat in gt["categories"]}
all_class_ids = list(category_id_to_name.keys())

# === IoU-function ===
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0

# === AP for one class ===
def compute_ap_class(class_id, iou_thresh=0.5):
    y_true = []
    y_scores = []

    keys = [key for key in gt_by_image_cat if key[1] == class_id]

    for key in keys:
        gt_boxes = gt_by_image_cat[key]
        pred_boxes = pred_by_image_cat.get(key, [])

        matched = np.zeros(len(gt_boxes), dtype=bool)

        for pred_box, score in sorted(pred_boxes, key=lambda x: -x[1]):
            found_match = False
            for i, gt_box in enumerate(gt_boxes):
                if not matched[i] and compute_iou(pred_box, gt_box) >= iou_thresh:
                    matched[i] = True
                    found_match = True
                    break
            y_true.append(1 if found_match else 0)
            y_scores.append(score)

    if len(set(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_scores)

# === calculate mAP ===
map50_all_classes = []
for class_id in all_class_ids:
    ap50 = compute_ap_class(class_id, iou_thresh=0.5)
    class_name = category_id_to_name[class_id]
    map50_all_classes.append(ap50)
    print(f"Class {class_name:15s} (id={class_id:2d}): mAP@50 = {ap50:.3f}")

map50 = np.mean(map50_all_classes)
map5095 = np.mean([compute_ap_class(class_id, iou_thresh=iou)
                   for iou in IOU_THRESHOLDS for class_id in all_class_ids])

print("\n====== Overall Metrics ======")
print(f"mAP@50 (mean across classes)     = {map50:.3f}")
print(f"mAP@50-95 (mean across classes) = {map5095:.3f}")



# === calculate mAP@75 ===
map75_all_classes = []

for class_id in all_class_ids:
    ap75 = compute_ap_class(class_id, iou_thresh=0.75)
    class_name = category_id_to_name[class_id]
    map75_all_classes.append(ap75)
    print(f"Class {class_name:15s} (id={class_id:2d}): mAP@75 = {ap75:.3f}")

map75 = np.mean(map75_all_classes)

print("\n====== mAP@75 Summary ======")
print(f"mAP@75 (mean across classes) = {map75:.3f}")

# === calculate mAP@90 ===
map90_all_classes = []

for class_id in all_class_ids:
    ap90 = compute_ap_class(class_id, iou_thresh=0.90)
    class_name = category_id_to_name[class_id]
    map90_all_classes.append(ap90)
    print(f"Class {class_name:15s} (id={class_id:2d}): mAP@90 = {ap90:.3f}")

map90 = np.mean(map90_all_classes)

print("\n====== mAP@90 Summary ======")
print(f"mAP@90 (mean across classes) = {map90:.3f}")


# === calculate mAP@50-95 ДЛЯ КАЖДОГО КЛАССА ===
map5095_per_class = {}

for class_id in all_class_ids:
    aps = []
    for iou_thresh in IOU_THRESHOLDS:
        ap = compute_ap_class(class_id, iou_thresh=iou_thresh)
        aps.append(ap)
    mean_ap = np.mean(aps)
    class_name = category_id_to_name[class_id]
    map5095_per_class[class_id] = mean_ap
    print(f"Class {class_name:15s} (id={class_id:2d}): mAP@50-95 = {mean_ap:.3f}")

# Overall
overall_map5095 = np.mean(list(map5095_per_class.values()))
print(f"\n====== Overall mAP@50-95 ======")
print(f"mAP@50-95 (mean across classes) = {overall_map5095:.3f}")


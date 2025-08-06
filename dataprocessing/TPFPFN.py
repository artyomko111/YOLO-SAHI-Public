import os
 
classes = {0: "class 1",
           1: "class 2",
           2: "class 3",
           3: "class 4",
           4: "class 5",
           5: "class 6",
           6: "class 7",
           7: "class 8",
           8: "class 9",
           9: "class 10"}
 
def iou(box1, box2):
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
 
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
 
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
 
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
 
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
 
    union_area = box1_area + box2_area - inter_area
 
    iou = inter_area / union_area if union_area > 0 else 0
 
    return iou
 
 
def read_boxes(file_path):
    boxes = []
    confidences = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                category = int(parts[0])
                box = [float(part) for part in parts[1:5]]
                confidence = float(parts[5])
                boxes.append((category, box))
                confidences.append((category, confidence))
            elif len(parts) == 5:
                category = int(parts[0])
                box = [float(part) for part in parts[1:5]]
                boxes.append((category, box))
    return boxes, confidences
 
 
def evaluate_folder(annotation_dir, result_dir):
    stats = {}
    for annot_file in os.listdir(annotation_dir):
        annot_path = os.path.join(annotation_dir, annot_file)
        result_path = os.path.join(result_dir, annot_file)
 
        annot_boxes, _ = read_boxes(annot_path)  # 标注不需要置信度
        result_boxes, result_confidences = read_boxes(result_path) if os.path.exists(result_path) else ([], [])
 
        for category, _ in annot_boxes:
            if category not in stats:
                stats[category] = {'annotated': 0, 'predicted': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'confidences': []}
            stats[category]['annotated'] += 1
 
        for category, result_box in result_boxes:
            if category not in stats:
                stats[category] = {'annotated': 0, 'predicted': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'confidences': []}
            stats[category]['predicted'] += 1
 
            ious = [iou(result_box, box) for cat, box in annot_boxes if cat == category]
            if max(ious, default=0) >= 0.5:
                stats[category]['tp'] += 1
            else:
                stats[category]['fp'] += 1
 
        for category, confidence in result_confidences:
            stats[category]['confidences'].append(confidence)
 
        for category, annot_box in annot_boxes:
            ious = [iou(annot_box, box) for cat, box in result_boxes if cat == category]
            if max(ious, default=0) < 0.5:
                stats[category]['fn'] += 1
 
    for category, data in stats.items():
        confidences = data['confidences']
        data['min_conf'] = min(confidences, default=0)
        data['max_conf'] = max(confidences, default=0)
        data['ave_conf'] = sum(confidences) / len(confidences) if confidences else 0
        data['precision'] = data['tp'] / (data['tp'] + data['fp']) if data['tp'] + data['fp'] > 0 else 0
        data['recall'] = data['tp'] / (data['tp'] + data['fn']) if data['tp'] + data['fn'] > 0 else 0
        del data['confidences']  # 为了清晰，删除置信度列表
 
    return stats
 
# path
annotation_dir = '/home/artem/Downloads/VisDrone2YOLO-main/VisDrone2019-DET-test-dev/labels'
result_dir = '/home/artem/Downloads/labels-20250720T090426Z-1-001/labels'
 
# calculate TP, FP, FN
stats = evaluate_folder(annotation_dir, result_dir)
print("index | class | Instance | Detection | TP | FP | FN | Precision | Recall | Min Conf | Max Conf | Ave Conf")
for category, data in sorted(stats.items()):
    print(f"{category} | {classes[category]} | {data['annotated']} | {data['predicted']} | {data['tp']} | {data['fp']} | {data['fn']} | {data['precision']:.2f} | {data['recall']:.2f} | {data['min_conf']:.2f} | {data['max_conf']:.2f} | {data['ave_conf']:.2f}")

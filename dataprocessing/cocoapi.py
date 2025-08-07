from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Path to ground truth annotations (COCO-style JSON)
cocoGt = COCO("/home/artem/Downloads/annotations_yolo.json")

# Path to model predictions (COCO-format prediction JSON)
cocoDt = cocoGt.loadRes("/home/artem/Downloads/yolov10_json_new/predictions.json")

# Create evaluation object
cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')  # Use 'segm' or 'keypoints' if needed

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

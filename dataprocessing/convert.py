import os
import json
from PIL import Image

# === Настройки ===
images_dir = "/home/artem/Downloads/VisDrone2YOLO-main/VisDrone2019-DET-test-dev/images"       # 📁 Папка с изображениями (.jpg/.png)
labels_dir = "/home/artem/Downloads/VisDrone2YOLO-main/VisDrone2019-DET-test-dev/labels"       # 📁 Папка с YOLO .txt файлами
output_json = "/home/artem/Downloads/annotations_y11_2.json"    # 📄 Куда сохранить итоговый JSON

# Список классов — ВАЖНО: порядок как в YOLO (0 -> class_names[0], 1 -> class_names[1], и т.д.)
class_names = [
    "pedestrian",       # ID 1
    "people",           # ID 2
    "bicycle",          # ID 3
    "car",              # ID 4
    "van",              # ID 5
    "truck",            # ID 6
    "tricycle",         # ID 7
    "awning-tricycle",  # ID 8
    "bus",              # ID 9
    "motor",            # ID 10
]

# COCO categories (ID начинается с 1, чтобы избежать нуля как ID)
categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]

images = []
annotations = []
annotation_id = 1
image_id = 1

# 🔁 Обработка изображений и аннотаций
for filename in os.listdir(images_dir):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(images_dir, filename)
    label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    # Получаем размеры изображения
    img = Image.open(image_path)
    width, height = img.size

    image_dict = {
        "id": os.path.splitext(filename)[0],  # используем строковый image_id как у тебя
        "file_name": filename,
        "width": width,
        "height": height
    }
    images.append(image_dict)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_center, y_center, w, h = map(float, parts)
            cls_id = int(cls_id)

            # YOLO -> COCO bbox
            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w *= width
            h *= height

            annotations.append({
                "id": annotation_id,
                "image_id": os.path.splitext(filename)[0],  # строка!
                "category_id": cls_id + 1,  # COCO требует ID >= 1
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

# 💾 Финальная сборка COCO JSON
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"COCO annotations saved to {output_json}")


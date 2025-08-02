import json

# Load your COCO annotations JSON file
with open('/home/artem/Downloads/VisDrone2YOLO-main/convert_json/VisDrone2019-DET_test_coco_start.json') as f:
    data = json.load(f)

# Load the prediction results from result.json
with open('/home/artem/Downloads/YOLOv12SAHI.json') as f:
    predictions = json.load(f)

# Initialize dictionaries to store counts
class_counts = {}
total_images = set()
total_instances = 0
true_positives = {class_name: 0 for class_name in [item['name'] for item in data['categories']]}
false_positives = {class_name: 0 for class_name in [item['name'] for item in data['categories']]}
false_negatives = {class_name: 0 for class_name in [item['name'] for item in data['categories']]}

# Loop through annotations to count instances and images for each class
for annotation in data['annotations']:
    class_id = annotation['category_id']
    class_name = next(item for item in data['categories'] if item['id'] == class_id)['name']
    
    # Count the number of instances
    if class_name not in class_counts:
        class_counts[class_name] = {'images': set(), 'instances': 0}
    
    # Add image to the set (to avoid double counting images)
    class_counts[class_name]['images'].add(annotation['image_id'])
    # Count the instances (objects) in the annotations
    class_counts[class_name]['instances'] += 1

    # Update the total counts
    total_images.add(annotation['image_id'])
    total_instances += 1

# Now calculate True Positives (TP) and False Positives (FP)
# For each prediction in the result.json, compare with ground truth annotations
for prediction in predictions:
    predicted_class = next(item for item in data['categories'] if item['id'] == prediction['category_id'])['name']
    matching_annotation = any(
        annotation['category_id'] == prediction['category_id'] and annotation['image_id'] == prediction['image_id']
        for annotation in data['annotations']
    )
    
    # If a matching annotation is found, count it as a True Positive (TP)
    if matching_annotation:
        true_positives[predicted_class] += 1
    else:
        # If no matching annotation is found, count it as a False Positive (FP)
        false_positives[predicted_class] += 1

# Calculate False Negatives (FN) for each class
for annotation in data['annotations']:
    class_name = next(item for item in data['categories'] if item['id'] == annotation['category_id'])['name']
    predicted = any(
        prediction['category_id'] == annotation['category_id'] and prediction['image_id'] == annotation['image_id']
        for prediction in predictions
    )
    if not predicted:
        false_negatives[class_name] += 1

# Prepare the table with the first line as "All"
total_true_positives = sum(true_positives.values())
total_false_positives = sum(false_positives.values())
total_false_negatives = sum(false_negatives.values())
all_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
all_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

table = [['All', len(total_images), total_instances, total_true_positives, total_false_positives, total_false_negatives, all_precision, all_recall]]

# Fill the table for each class and calculate precision and recall
for class_name, counts in class_counts.items():
    tp = true_positives[class_name]
    fp = false_positives[class_name]
    fn = false_negatives[class_name]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # Avoid division by zero
    table.append([class_name, len(counts['images']), counts['instances'], tp, fp, fn, precision, recall])

# Print the table with precision and recall
print(f"{'Class':<20} {'Images':<20} {'Instances':<20} {'True Positives':<20} {'False Positives':<20} {'False Negatives':<20} {'Precision':<20} {'Recall':<20}")
for row in table:
    print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<20} {row[5]:<20} {row[6]:<20.3f} {row[7]:<20.3f}")


import os
import cv2
import numpy as np
from preprocess import preprocess_image
from pyramid import create_gaussian_pyramid,create_laplacian_pyramid
from prepare_templates import prepare_templates

# --- Matching Functions ---

def normalized_cross_correlation(patch, template):
    patch_mean = np.mean(patch)
    template_mean = np.mean(template)
    numerator = np.sum((patch - patch_mean) * (template - template_mean))
    denominator = np.sqrt(np.sum((patch - patch_mean)**2) * np.sum((template - template_mean)**2))
    return 0 if denominator == 0 else numerator / denominator

def match_template_at_scale(image, template, class_label, threshold=0.85):
    h, w = template.shape
    img_h, img_w = image.shape
    detections = []
    for y in range(0, img_h - h + 1, 4):
        for x in range(0, img_w - w + 1, 4):
            patch = image[y:y+h, x:x+w]
            score = normalized_cross_correlation(patch, template)
            if score > threshold:
                detections.append({'bbox': [x, y, x+w, y+h], 'score': score, 'class': class_label})
    return detections

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-5)

def non_max_suppression(detections, iou_threshold=0.85):
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [d for d in detections if compute_iou(current['bbox'], d['bbox']) < iou_threshold]
    return keep

def match_all_templates(test_pyramid, templates_by_class, threshold=0.85, iou_threshold=0.85):
    all_detections = []

    for level_index, test_scaled in enumerate(test_pyramid):
        for class_label, template_pyramid in templates_by_class.items():
            for template in template_pyramid:
                detections = match_template_at_scale(test_scaled, template, class_label, threshold)

                # Rescale bounding boxes to original image coordinates
                scale = 2 ** level_index
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
                    all_detections.append(det)

    return non_max_suppression(all_detections, iou_threshold)


# --- MAIN ---

if __name__ == "__main__":
    template_dir = "IconDataset/png"
    test_dir = "Task2Dataset/images"
    output_dir = "Task2Dataset/results"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading and preprocessing templates...")
    templates_by_class = prepare_templates(template_dir, num_levels=5)

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])

    for filename in test_files:
        print(f"\nProcessing {filename}...")

        test_image_path = os.path.join(test_dir, filename)
        test_image = preprocess_image(test_image_path, grayscale=True)
        original_image = cv2.imread(test_image_path)

        test_pyramid = create_gaussian_pyramid(test_image, num_levels=4)
        detections = match_all_templates(test_pyramid, templates_by_class, threshold=0.8, iou_threshold=0.85)


        print(f" → {len(detections)} objects detected.")
        for det in detections:
            print(f"   - {det['class']} at {det['bbox']} (score: {det['score']:.2f})")

    print("\n✅ Matching complete. Results saved in:", output_dir)

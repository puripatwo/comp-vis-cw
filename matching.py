import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_image
from pyramid import create_gaussian_pyramid, create_laplacian_pyramid
from prepare_templates import prepare_templates


def normalized_cross_correlation(patch, template):
    assert patch.shape == template.shape, "Patch and template must have the same shape"
    assert patch.ndim == 3 and patch.shape[2] == 3, "Inputs must be RGB images"

    ncc_channels = []
    for c in range(3):
        patch_c = patch[:, :, c].flatten()
        template_c = template[:, :, c].flatten()

        patch_mean = np.mean(patch_c)
        template_mean = np.mean(template_c)

        numerator = np.sum((patch_c - patch_mean) * (template_c - template_mean))
        denominator = np.sqrt(np.sum((patch_c - patch_mean) ** 2) * np.sum((template_c - template_mean) ** 2))

        ncc = numerator / denominator if denominator > 1e-8 else 0
        ncc_channels.append(ncc)
    
    return np.mean(ncc_channels)


def local_search(image, template, bbox, score, init_step=8):
    h, w = template.shape[:2]
    img_h, img_w = image.shape[:2]

    best_bbox = bbox
    best_score = score

    step = init_step
    iterations = 0
    while step >= 1 and iterations < 50:
        improved = False
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0:
                    continue

                ny, nx = best_bbox[0] + dy, best_bbox[1] + dx
                if 0 <= ny < img_h - h and 0 <= nx < img_w - w:
                    patch = image[ny:ny+h, nx:nx+w]
                    current_score = normalized_cross_correlation(patch, template)

                    if current_score > best_score:
                        best_score = current_score
                        best_bbox = [ny, nx, ny+h, nx+w]
                        improved = True

        if not improved:
            step //= 2
        iterations += 1

    return best_bbox, best_score


def find_matches_with_local_search(image, template_pyramid, class_label, max_samples=1000, early_stop_threshold=0.95):
    img_h, img_w = image.shape[:2]

    saved_positions = set()
    best_score = -1
    best_bbox = [0, 0, 0, 0]

    for level_index, template_scaled in enumerate(template_pyramid):
        print(f"Level: {level_index}, {template_scaled.shape}")

        h, w = template_scaled.shape[:2]
        if h > 500:
            continue

        for i in range(max_samples):
            while True:
                y = np.random.randint(0, img_h - h)
                x = np.random.randint(0, img_w - w)

                if (y, x) not in saved_positions:
                    saved_positions.add((y, x))
                    break

            patch = image[y:y+h, x:x+w]
            score = normalized_cross_correlation(patch, template_scaled)

            if score < 0:
                continue

            if score > best_score:
                best_score = score
                best_bbox = [y, x, y+h, x+w]

            if score >= early_stop_threshold:
                break
    
        if best_score > 0:
            best_bbox, best_score = local_search(image, template_scaled, best_bbox, best_score)
    
    detection = {
        'bbox': best_bbox,
        'score': best_score,
        'class': class_label,
        }
    return detection


def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    
    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    interArea = max(0, xB - xA) * max(0, yB - yA)

    union = areaA + areaB - interArea
    return interArea / (union + 1e-5) if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining = []
        for det in detections:
            if det['class'] != best['class']:
                remaining.append(det)
                continue

            iou = compute_iou(best['bbox'], det['bbox'])
            if iou <= iou_threshold:
                remaining.append(det)

        detections = remaining
    return keep


def match_all_templates(test_image, templates_by_class, score_threshold, nms_threshold):
    all_detections = []

    i = 0
    for class_label, template_pyramid in templates_by_class.items():
        i += 1
        print(f"Class: {class_label} ({i})")

        detection = find_matches_with_local_search(
                        test_image,
                        template_pyramid,
                        class_label,
                        max_samples=1000,
                        early_stop_threshold=0.95,
                    )

        y1, x1, y2, x2 = detection['bbox']
        detection['bbox'] = [int(y1), int(x1), int(y2), int(x2)]

        if detection['score'] > score_threshold:
            all_detections.append(detection)

    return non_max_suppression(all_detections, nms_threshold)


def evaluate_detections_with_class(gt_objects, det_objects, iou_threshold):
    matched = set()
    ious = []

    for det_idx, det in enumerate(det_objects):
        # print("det", det['class'])
        for gt_idx, gt in enumerate(gt_objects):
            # print("gt", gt[4])
            if gt_idx in matched:
                continue

            gt_box, gt_class = gt['bbox'], gt['class']
            det_box, det_class = det['bbox'], det['class']
            
            iou = compute_iou(det_box, gt_box)
            if iou >= iou_threshold and det_class == gt_class:
                matched.add(gt_idx)
                ious.append(iou)
                break

    TP = len(matched)
    FP = len(det_objects) - TP
    FN = len(gt_objects) - TP
    accuracy = TP / (TP + FP + FN + 1e-6)
    avg_iou = np.mean(ious) if ious else 0.0

    return {
        'True Positives': TP,
        'False Positives': FP,
        'False Negatives': FN,
        'Accuracy': accuracy,
        'Average IoU': avg_iou
    }

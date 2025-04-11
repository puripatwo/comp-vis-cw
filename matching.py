import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_image
from pyramid import create_gaussian_pyramid, create_laplacian_pyramid
from prepare_templates import prepare_templates


def normalized_cross_correlation(patch, template):
    patch_mean = np.mean(patch)
    template_mean = np.mean(template)
    numerator = np.sum((patch - patch_mean) * (template - template_mean))
    denominator = np.sqrt(np.sum((patch - patch_mean)**2) * np.sum((template - template_mean)**2))
    return 0 if denominator == 0 else numerator / denominator


def match_template_at_scale(image, template, class_label, threshold, stride):
    h, w = template.shape
    img_h, img_w = image.shape

    detections = []
    for y in range(0, img_h - h + 1, stride):
        for x in range(0, img_w - w + 1, stride):
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


def non_max_suppression(detections, iou_threshold):
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [d for d in detections if compute_iou(current['bbox'], d['bbox']) < iou_threshold]
    return keep


def match_all_templates(test_image, templates_by_class, threshold, iou_threshold, stride):
    all_detections = []

    i = 0
    for class_label, template_pyramid in templates_by_class.items():
        i += 1
        # print(f"Class: {class_label} ({i})")
        for level_index, template_scaled in enumerate(template_pyramid[1:], start=1):
            # print(f"Level: {level_index}, {template_scaled.shape}")

            detections = match_template_at_scale(test_image, template_scaled, class_label, threshold, stride)

            scale = 1 / (2 ** level_index)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                w = x2 - x1
                h = y2 - y1
                if w < 8 or h < 8:
                    continue

                det['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                det['scale'] = scale
                all_detections.append(det)

    return non_max_suppression(all_detections, iou_threshold)


def evaluate_detections_with_class(gt_objects, det_objects, iou_threshold):
    """
    gt_objects: List of [x1, y1, x2, y2, class_label]
    det_objects: List of {'bbox': [x1, y1, x2, y2], 'class': class_label}
    """
    matched = set()
    ious = []

    for det_idx, det in enumerate(det_objects):
        # print("det", det['class'])
        for gt_idx, gt in enumerate(gt_objects):
            # print("gt", gt[4])
            if gt_idx in matched:
                continue

            gt_box, gt_class = gt[:4], gt[4]
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

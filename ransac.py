import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sift_matching import extract_sift_features, match_features, draw_feature_matches

def get_keypoint_coordinates(matches, kp1, kp2):
    """ Convert matches to two lists of (x, y) coordinates """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    return src_pts, dst_pts

def compute_homography(p1, p2):
    """ Compute homography from 4+ points using SVD (DLT algorithm) """
    A = []
    for (x, y), (xp, yp) in zip(p1, p2):
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iters=2000):
    best_H = None
    max_inliers = 0
    best_inliers = []

    for _ in range(max_iters):
        indices = np.random.choice(len(src_pts), 4, replace=False)
        p1, p2 = src_pts[indices], dst_pts[indices]

        try:
            H = compute_homography(p1, p2)
        except np.linalg.LinAlgError:
            continue

        projected = cv2.perspectiveTransform(src_pts.reshape(-1,1,2), H).reshape(-1, 2)
        errors = np.linalg.norm(dst_pts - projected, axis=1)

        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers

    return best_H, best_inliers


def polygon_iou(p1, p2):
    """Compute IoU between two quadrilaterals using mask intersection"""
    img_shape = (500, 500)  # adjust if needed

    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)

    cv2.fillPoly(mask1, [np.int32(p1)], 1)
    cv2.fillPoly(mask2, [np.int32(p2)], 1)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def apply_nms(detections, iou_threshold=0.4, center_dist_thresh=40):
    def get_center(box):
        return np.mean(box.reshape(4, 2), axis=0)

    suppressed = [False] * len(detections)
    kept = []

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    for i, det in enumerate(detections):
        if suppressed[i]:
            continue
        kept.append(det)
        center_i = get_center(det["box"])

        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            center_j = get_center(detections[j]["box"])
            iou = polygon_iou(det["box"], detections[j]["box"])
            dist = np.linalg.norm(center_i - center_j)

            if iou > iou_threshold or dist < center_dist_thresh:
                suppressed[j] = True  # suppress lower-scoring overlapping detection

    return kept

def is_box_reasonable(box, image_shape, min_area_ratio=0.0005, max_area_ratio=0.6, max_aspect_ratio=10):
    pts = box.reshape(4, 2)
    edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    width = (edges[0] + edges[2]) / 2
    height = (edges[1] + edges[3]) / 2
    area = cv2.contourArea(np.int32(box))

    if width < 1 or height < 1:
        return False

    image_area = image_shape[0] * image_shape[1]
    area_ratio = area / (image_area + 1e-5)

    aspect_ratio = max(width / (height + 1e-5), height / (width + 1e-5))

    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False
    if aspect_ratio > max_aspect_ratio:
        return False

    return True


def detect_objects_from_all_templates(test_image_path, icon_dir):
    test_image = cv2.imread(test_image_path)
    kp2, desc2 = extract_sift_features(test_image)

    detections = []

    for icon_path in sorted(glob.glob(os.path.join(icon_dir, "*.png"))):
        label = os.path.splitext(os.path.basename(icon_path))[0]
        icon = cv2.imread(icon_path)
        kp1, desc1 = extract_sift_features(icon)

        matches = match_features(desc1, desc2)
        if len(matches) < 4:
            continue

        src_pts, dst_pts = get_keypoint_coordinates(matches, kp1, kp2)
        H, inliers = ransac_homography(src_pts, dst_pts)
        # num_inliers = np.sum(inliers)

        # if len(matches) >= 9:
        #     inlier_ratio = num_inliers / len(matches)
        #     if inlier_ratio >= 0.2:
        #         print(f"[✓] Detected: {label} ({num_inliers} inliers, {inlier_ratio:.2f} ratio)")

        #         h, w = icon.shape[:2]
        #         corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        #         projected_corners = cv2.perspectiveTransform(corners, H)

        #         if is_box_reasonable(projected_corners, test_image.shape):
        #             detections.append({
        #                 "label": label,
        #                 "box": projected_corners,
        #                 "score": inlier_ratio
        #             })
        
        if len(matches) >= 10:  # use match count only
            print(f"[✓] Detected: {label} ({len(matches)} raw matches)")

            h , w = icon.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            projected_corners = cv2.perspectiveTransform(corners, H)

            if is_box_reasonable(projected_corners, test_image.shape):
                detections.append({
                    "label": label,
                    "box": projected_corners,
                    "score": len(matches)
                })


    return test_image, detections



def draw_detections(image, detections):
    vis = image.copy()
    for det in detections:
        box = det["box"]
        label = det["label"]
        cv2.polylines(vis, [np.int32(box)], isClosed=True, color=(0, 255, 0), thickness=2)
        pt = tuple(np.int32(box[0][0]))
        cv2.putText(vis, label, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


if __name__ == "__main__":
    test_image_path = "Task3Dataset/images/test_image_2.png"
    icon_dir = "IconDataset/png"

    test_img, detections = detect_objects_from_all_templates(test_image_path, icon_dir)

    detections = apply_nms(detections)

    print(f"\nTotal detections after NMS: {len(detections)}")

    output = draw_detections(test_img, detections)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Detection Results")
    plt.axis("off")
    plt.show()
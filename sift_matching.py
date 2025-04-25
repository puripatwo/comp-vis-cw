import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import random
import csv
import time

def load_annotations_csv(filepath):
    """
    Load ground truth from a .csv file with a header:
    classname,top,left,bottom,right
    Returns: list of {'label': str, 'bbox': [left, top, right, bottom]}
    """
    gt = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) != 5:
                continue
            label = row[0].strip()
            try:
                top, left, bottom, right = map(int, row[1:])
            except ValueError:
                continue
            bbox = [top, left, bottom, right]
            gt.append({'label': label, 'bbox': bbox})
    return gt


def normalize_label(label):
    """Removes numeric prefix and lowercases the class name."""
    # Replace any dash with standard '-' just in case
    label = label.replace('–', '-')
    # Split on the last '-' and keep only the name
    return label.split('-')[-1].strip().lower()


def _iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB-xA), max(0, yB-yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

def match_features(desc1, desc2, ratio_thres=0.75):
    """
    Mutual Lowe's ratio test with forward-backward consistency.
    """
    if desc1 is None or desc2 is None:
        return []
    forward = []
    for i, d1 in enumerate(desc1):
        dists = np.linalg.norm(desc2 - d1, axis=1)
        if len(dists) < 2:
            continue
        idx = np.argsort(dists)[:2]
        if dists[idx[0]] < ratio_thres * dists[idx[1]]:
            forward.append((i, idx[0]))
    matches = []
    for q, t in forward:
        d_back = np.linalg.norm(desc1 - desc2[t], axis=1)
        idx_back = np.argsort(d_back)[:2]
        if idx_back[0] == q and d_back[idx_back[0]] < ratio_thres * d_back[idx_back[1]]:
            matches.append(cv2.DMatch(_queryIdx=q, _trainIdx=t, _distance=d_back[idx_back[0]]))
    return matches


def normalize_points(pts, eps=1e-8):
    """
    Normalize 2D points for DLT (zero mean, average distance sqrt(2)).
    Handles degenerate cases where std ≈ 0.
    """
    mean = np.mean(pts, axis=0)
    std = np.std(pts - mean)

    if std < eps:
        std = eps  # avoid division by zero

    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T

    return pts_norm[:, :2], T


def compute_homography(src_pts, dst_pts):
    """
    Compute homography using normalized DLT.
    """
    src_norm, T1 = normalize_points(src_pts)
    dst_norm, T2 = normalize_points(dst_pts)

    A = []
    for i in range(len(src_pts)):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H_norm = V[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T2) @ H_norm @ T1
    return H / H[2, 2]


def symmetric_transfer_error(H, src_pts, dst_pts, eps=1e-8):
    """
    Compute symmetric transfer error with safe division.
    Returns large error for invalid projections.
    """
    N = src_pts.shape[0]
    src_h = np.hstack([src_pts, np.ones((N, 1))])
    dst_h = np.hstack([dst_pts, np.ones((N, 1))])

    proj_dst = (H @ src_h.T).T
    proj_dst_z = proj_dst[:, 2:3]
    proj_dst_z[proj_dst_z < eps] = eps  # Avoid division by zero
    proj_dst /= proj_dst_z

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(N, np.inf)

    proj_src = (H_inv @ dst_h.T).T
    proj_src_z = proj_src[:, 2:3]
    proj_src_z[proj_src_z < eps] = eps
    proj_src /= proj_src_z

    # If any NaNs still got through, treat them as high-error
    if np.isnan(proj_dst).any() or np.isnan(proj_src).any():
        return np.full(N, np.inf)

    err1 = np.linalg.norm(proj_dst[:, :2] - dst_pts, axis=1)
    err2 = np.linalg.norm(proj_src[:, :2] - src_pts, axis=1)
    return err1**2 + err2**2


def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iters=1000):
    """
    Mimics OpenCV's cv2.findHomography with cv2.RANSAC.
    Inputs:
      - src_pts, dst_pts: Nx1x2 float32 arrays
    Returns:
      - Best homography H and inlier mask (Nx1 uint8)
    """
    assert src_pts.shape == dst_pts.shape
    N = src_pts.shape[0]
    if N < 4:
        return None, None

    src_pts_flat = src_pts.reshape(N, 2)
    dst_pts_flat = dst_pts.reshape(N, 2)

    best_H = None
    best_inliers = np.zeros(N, dtype=bool)
    best_score = 0

    for _ in range(max_iters):
        idx = random.sample(range(N), 4)
        src_sample = src_pts_flat[idx]
        dst_sample = dst_pts_flat[idx]

        try:
            H = compute_homography(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue

        errors = symmetric_transfer_error(H, src_pts_flat, dst_pts_flat)
        inliers = errors < threshold**2
        score = np.sum(inliers)

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_H = H

    if best_H is None or best_inliers.sum() < 4:
        return None, None

    # Re-estimate H using all inliers
    inlier_src = src_pts_flat[best_inliers]
    inlier_dst = dst_pts_flat[best_inliers]
    try:
        best_H = compute_homography(inlier_src, inlier_dst)
    except np.linalg.LinAlgError:
        return None, None

    mask = best_inliers.astype(np.uint8).reshape(-1, 1)
    return best_H, mask


def extract_sift_features(image, grayscale=True):
    """Extract SIFT keypoints and descriptors."""
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nOctaveLayers=6)
    return sift.detectAndCompute(image, None)


def detect_at_scale(icon_label, icon_shape, kp1, desc1, kp2, desc2,
                    scale_factor, reproj_thresh, min_inliers, ratio):
    """
    Run matching and RANSAC on a scaled test image for one icon.
    Returns list of detection dicts.
    """
    if desc1 is None or desc2 is None:
        return []
    matches = match_features(desc1, desc2, ratio)
    if len(matches) < min_inliers:
        return []
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    H, mask = ransac_homography(src_pts, dst_pts, threshold=reproj_thresh)
    if H is None or mask is None:
        return []
    inliers = [matches[i] for i in range(len(matches)) if mask[i]]
    if len(inliers) < min_inliers:
        return []
    A = H[:2,:2]
    sx, sy = np.linalg.norm(A[:,0]), np.linalg.norm(A[:,1])
    scale_est = (sx+sy)/2
    angle = np.degrees(np.arctan2(A[1,0], A[0,0]))
    h_t, w_t = icon_shape
    corners = np.array([[0,0],[w_t,0],[w_t,h_t],[0,h_t]], np.float32).reshape(-1,1,2)
    proj = cv2.perspectiveTransform(corners, H).astype(int)
    proj = (proj / scale_factor).astype(int)
    pts = proj.reshape(-1,2)
    x0,y0 = pts.min(axis=0)
    x1,y1 = pts.max(axis=0)
    bbox = [int(x0),int(y0),int(x1),int(y1)]
    return [{
        'label': icon_label,
        'score': len(inliers),
        'scale': scale_est,
        'angle': angle,
        'proj': proj,
        'bbox': bbox
    }]


def batch_multiscale_detect(icons_dir, test_path,
                            scales=[0.5,0.75,1.0,1.25,1.5],
                            ratio=0.75, reproj_thresh=5.0,
                            min_inliers=8, iou_thresh=0.3,
                            visualize=False):
    """
    Run detection over multiple scales and apply class-aware NMS.
    Returns the final detection list (after NMS).
    """
    img_orig = cv2.imread(test_path)
    kp2_full, desc2_full = extract_sift_features(img_orig)
    all_dets = []
    for scale in scales:
        img = cv2.resize(img_orig, None, fx=scale, fy=scale)
        kp2, desc2 = extract_sift_features(img)
        for icon_path in glob.glob(os.path.join(icons_dir, '*.png')):
            icon_label = os.path.splitext(os.path.basename(icon_path))[0]
            icon = cv2.imread(icon_path)
            kp1, desc1 = extract_sift_features(icon)
            icon_shape = icon.shape[:2]
            dets = detect_at_scale(icon_label, icon_shape,
                                   kp1, desc1, kp2, desc2,
                                   scale, reproj_thresh,
                                   min_inliers, ratio)
            all_dets.extend(dets)

    print(f"Raw detections: {len(all_dets)}")
    dets = sorted(all_dets, key=lambda d: d['score'], reverse=True)
    final = []
    while dets:
        curr = dets.pop(0)
        final.append(curr)
        dets = [d for d in dets if d['label'] != curr['label'] or _iou(curr['bbox'], d['bbox']) < iou_thresh]
    print(f"After NMS: {len(final)}")

    if visualize:
        vis = img_orig.copy()
        for d in final:
            cv2.polylines(vis, [d['proj']], True, (0, 255, 0), 2)
            x, y = d['bbox'][0], d['bbox'][1]
            cv2.putText(vis, f"{d['label']} s={d['scale']:.2f}",
                        (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return final

def evaluate_predictions(detections, gt, iou_thresh=0.85):
    TP, FP, FN = 0, 0, 0
    matched = set()

    for d in detections:
        best_iou = 0
        best_match = -1
        for i, g in enumerate(gt):
            if i in matched:
                continue
            if normalize_label(g['label']) == normalize_label(d['label']):
                iou = _iou(d['bbox'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

        if best_iou >= iou_thresh:
            TP += 1
            matched.add(best_match)
        else:
            FP += 1

    FN = len(gt) - len(matched)

    acc = TP / (TP + FP + FN) if (TP + FP + FN) else 0
    tpr = TP / (TP + FN) if (TP + FN) else 0
    fpr = FP / (TP + FP) if (TP + FP) else 0

    print(f"\nEvaluation")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Accuracy: {acc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")

    return acc, tpr, fpr, TP, FP, FN


def batch_evaluate_all(icons_dir, images_dir, annos_dir,
                       image_ext='.png', anno_ext='.csv',
                       iou_eval_thresh=0.85):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(image_ext)])
    total_TP, total_FP, total_FN = 0, 0, 0
    total_runtime = 0
    num_images = 0

    for img_name in image_files:
        start_time = time.time()

        image_path = os.path.join(images_dir, img_name)
        base = os.path.splitext(img_name)[0]
        anno_path = os.path.join(annos_dir, base + anno_ext)

        if not os.path.exists(anno_path):
            print(f"[Warning] Annotation file not found for: {img_name}")
            continue

        print(f"\n--- Evaluating {img_name} ---")
        detections = batch_multiscale_detect(icons_dir, image_path, visualize=False)
        gt = load_annotations_csv(anno_path)
        visualize_detections_vs_gt(image_path, detections, gt, iou_thresh=0.85)
        acc, tpr, fpr, TP, FP, FN = evaluate_predictions(detections, gt, iou_thresh=iou_eval_thresh)

        total_TP += TP
        total_FP += FP
        total_FN += FN
        num_images += 1
        total_runtime += time.time() - start_time

    if num_images == 0:
        print("No valid image-annotation pairs found.")
        return

    avg_acc = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0

    avg_tpr = total_TP / (total_TP + total_FN) if total_TP + total_FN else 0
    avg_fpr = total_FP / (total_TP + total_FN) if total_TP + total_FN else 0
    avg_fnr = total_FN / (total_TP + total_FN) if total_TP + total_FN else 0

    avg_runtime = total_runtime / num_images

    print(f"\nOverall Evaluation Across {num_images} Images")
    print(f"Total TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average TPR: {avg_tpr:.4f}")
    print(f"Average FPR: {avg_fpr:.4f}")
    print(f"Average FNR: {avg_fnr:.4f}")
    print(f"Average Runtime:  {avg_runtime:.4f} seconds/image")
    
    return avg_acc, avg_tpr, avg_fpr, avg_fnr


def visualize_detections_vs_gt(image_path, detections, gt, iou_thresh=0.85):
    """
    Overlay both detections and ground truth on image.
    Green = detections, Blue = ground truth.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis = image.copy()

    # Draw ground truth boxes (blue)
    for g in gt:
        x0, y0, x1, y1 = g['bbox']
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, f"GT: {g['label']}", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw predicted boxes (green)
    for d in detections:
        x0, y0, x1, y1 = d['bbox']
        matching_ious = [
            _iou(d['bbox'], g['bbox']) 
            for g in gt 
            if normalize_label(g['label']) == normalize_label(d['label'])
        ]
        iou = max(matching_ious) if matching_ious else 0
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"DET: {normalize_label(d['label'])} ({iou:.2f})", (x0, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(vis)
    plt.axis('off')
    plt.title("Green = Detection, Blue = Ground Truth")
    plt.show()


if __name__ == '__main__':
    batch_evaluate_all(
        icons_dir='IconDataset/png',
        images_dir='Task3Dataset/images',
        annos_dir='Task3Dataset/annotations',
        image_ext='.png',
        anno_ext='.csv',
        iou_eval_thresh=0.85
    )
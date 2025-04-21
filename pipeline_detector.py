import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time


def extract_sift_features(image, grayscale=True):
    """Extract SIFT keypoints and descriptors."""
    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)


def _iou(boxA, boxB):
    """Compute Intersection over Union between two bboxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0


def load_icons(icons_dir):
    """Precompute keypoints and descriptors for all icons."""
    icon_data = []
    for icon_path in glob.glob(os.path.join(icons_dir, '*.png')):
        label = os.path.splitext(os.path.basename(icon_path))[0]
        img = cv2.imread(icon_path)
        kp, desc = extract_sift_features(img)
        icon_data.append({
            'label': label,
            'shape': img.shape[:2],  # (h, w)
            'kp': kp,
            'desc': desc,
            'img': img
        })
    return icon_data


def match_features(desc1, desc2, ratio_thres=0.75):
    """Mutual Lowe's ratio test with forward-backward consistency."""
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


def detect_at_scale(icon_label, icon_img, kp1, desc1, kp2, desc2,
                    scale_factor, reproj_thresh, min_inliers, ratio):
    """Run matching and RANSAC on a scaled test image for one icon."""
    matches = match_features(desc1, desc2, ratio)
    if len(matches) < min_inliers:
        return []

    # Prepare point arrays
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # RANSAC homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None or mask is None:
        return []
    inlier_matches = [m for m, msk in zip(matches, mask.ravel()) if msk]
    if len(inlier_matches) < min_inliers:
        return []

    # Estimate scale and rotation
    A = H[:2, :2]
    sx, sy = np.linalg.norm(A[:,0]), np.linalg.norm(A[:,1])
    scale_est = (sx + sy) / 2
    angle = np.degrees(np.arctan2(A[1,0], A[0,0]))

    # Project icon corners back to original image
    h_t, w_t = icon_img.shape[:2]
    corners = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]]).reshape(-1,1,2)
    proj = cv2.perspectiveTransform(corners, H)
    proj_orig = (proj / scale_factor).reshape(-1,2).astype(int)

    # Compute axis-aligned bbox
    x0, y0 = proj_orig.min(axis=0)
    x1, y1 = proj_orig.max(axis=0)
    bbox = [int(x0), int(y0), int(x1), int(y1)]

    return [{
        'label': icon_label,
        'score': len(inlier_matches),
        'scale': scale_est,
        'angle': angle,
        'proj': proj_orig.reshape(-1,2),
        'bbox': bbox
    }]


def batch_multiscale_detect(icons_dir, test_path,
                            scales=[0.5,0.75,1.0,1.25,1.5],
                            ratio=0.75, reproj_thresh=5.0,
                            min_inliers=6, nms_iou=0.3):
    """Run detection, NMS, timing breakdown, and visualize results."""
    # Load icons once
    icon_data = load_icons(icons_dir)

    # Load test image
    test_img = cv2.imread(test_path)
    h_orig, w_orig = test_img.shape[:2]

    all_dets = []
    timings = []

    # Multi-scale detection
    for scale in scales:
        test_img_scaled = cv2.resize(test_img, None, fx=scale, fy=scale)
        kp2, desc2 = extract_sift_features(test_img_scaled)

        for item in icon_data:
            label, img_icon = item['label'], item['img']
            kp1, desc1 = item['kp'], item['desc']

            start = time.time()
            dets = detect_at_scale(label, img_icon, kp1, desc1,
                                   kp2, desc2,
                                   scale, reproj_thresh,
                                   min_inliers, ratio)
            timings.append(time.time() - start)
            all_dets.extend(dets)

    print(f"Raw detections: {len(all_dets)}")

    # Class-aware NMS
    final = []
    dets_sorted = sorted(all_dets, key=lambda d: d['score'], reverse=True)
    while dets_sorted:
        top = dets_sorted.pop(0)
        final.append(top)
        dets_sorted = [d for d in dets_sorted
                       if d['label'] != top['label']
                       or _iou(top['bbox'], d['bbox']) < nms_iou]
    print(f"After NMS: {len(final)}")

    # Visualize
    vis = test_img.copy()
    for d in final:
        pts = d['proj'].reshape(-1,1,2)
        cv2.polylines(vis, [pts], True, (0,255,0), 2)
        x, y = d['bbox'][:2]
        cv2.putText(vis, f"{d['label']} s={d['scale']:.2f}",
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print(f"Average detection time per icon-scale: {np.mean(timings):.3f}s")


if __name__ == '__main__':
    icons_dir = 'IconDataset/png'
    test_path = 'Task3Dataset/images/test_image_17.png'
    batch_multiscale_detect(icons_dir, test_path)





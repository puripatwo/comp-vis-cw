import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import random

# === Utility Functions ===

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts - mean)
    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def compute_homography(src_pts, dst_pts):
    src_norm, T1 = normalize_points(src_pts)
    dst_norm, T2 = normalize_points(dst_pts)

    A = []
    for i in range(len(src_norm)):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    _, _, V = np.linalg.svd(A)
    H_norm = V[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ H_norm @ T1
    return H / H[2, 2]

def symmetric_transfer_error(H, src_pts, dst_pts, eps=1e-8):
    N = src_pts.shape[0]
    src_h = np.hstack([src_pts, np.ones((N, 1))])
    dst_h = np.hstack([dst_pts, np.ones((N, 1))])

    proj_dst = (H @ src_h.T).T
    proj_dst_z = proj_dst[:, 2:3]
    proj_dst_z[proj_dst_z < eps] = eps
    proj_dst /= proj_dst_z

    try:
        H_inv = np.linalg.inv(H)
        proj_src = (H_inv @ dst_h.T).T
        proj_src_z = proj_src[:, 2:3]
        proj_src_z[proj_src_z < eps] = eps
        proj_src /= proj_src_z
    except np.linalg.LinAlgError:
        return np.full(N, np.inf)

    if np.isnan(proj_dst).any() or np.isnan(proj_src).any():
        return np.full(N, np.inf)

    err1 = np.linalg.norm(proj_dst[:, :2] - dst_pts, axis=1)
    err2 = np.linalg.norm(proj_src[:, :2] - src_pts, axis=1)
    return err1**2 + err2**2

# === Model Wrapper ===

class HomographyModel:
    def fit(self, data):
        src_pts = data[:, 0:2]
        dst_pts = data[:, 2:4]
        return compute_homography(src_pts, dst_pts)

    def get_error(self, data, H):
        src_pts = data[:, 0:2]
        dst_pts = data[:, 2:4]
        return symmetric_transfer_error(H, src_pts, dst_pts)

# === Generic RANSAC ===

def random_partition(n, data_size):
    all_idxs = np.arange(data_size)
    np.random.shuffle(all_idxs)
    return all_idxs[:n], all_idxs[n:]

def ransac(data, model, n, k, t, d, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs]
        test_points = data[test_idxs]

        try:
            maybemodel = model.fit(maybeinliers)
        except np.linalg.LinAlgError:
            iterations += 1
            continue

        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs]

        if len(alsoinliers) > d:
            betterdata = np.vstack((maybeinliers, alsoinliers))
            try:
                bettermodel = model.fit(betterdata)
            except np.linalg.LinAlgError:
                iterations += 1
                continue
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations += 1

    if bestfit is None:
        return None, None

    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit
    
def extract_sift_features(image, grayscale=True):
    """Extract SIFT keypoints and descriptors."""
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)


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

    # Convert matches to flat (N, 4) array: [x1, y1, x2, y2]
    pts = np.hstack([
        np.float32([kp1[m.queryIdx].pt for m in matches]),
        np.float32([kp2[m.trainIdx].pt for m in matches])
    ])

    model = HomographyModel()
    H, info = ransac(pts, model, n=4, k=1000, t=reproj_thresh**2, d=min_inliers, return_all=True)

    if H is None or info is None:
        return []

    # Build inlier list using returned indices
    inlier_mask = np.zeros(len(matches), dtype=np.uint8)
    inlier_mask[info['inliers']] = 1
    inliers = [matches[i] for i in range(len(matches)) if inlier_mask[i]]

    if len(inliers) < min_inliers:
        return []

    # Extract affine part for estimating scale and rotation
    A = H[:2, :2]
    sx, sy = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
    scale_est = (sx + sy) / 2
    angle = np.degrees(np.arctan2(A[1, 0], A[0, 0]))

    # Project icon corners
    h_t, w_t = icon_shape
    corners = np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]], dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).astype(int)
    proj = (proj / scale_factor).astype(int)

    pts = proj.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bbox = [int(x0), int(y0), int(x1), int(y1)]

    return [{
        'label': icon_label,
        'score': len(inliers),
        'scale': scale_est,
        'angle': angle,
        'proj': proj,
        'bbox': bbox
    }]



def _iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB-xA), max(0, yB-yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0


def batch_multiscale_detect(icons_dir, test_path,
                            scales=[0.5,0.75,1.0,1.25,1.5],
                            ratio=0.75, reproj_thresh=5.0,
                            min_inliers=6, iou_thresh=0.3):
    """
    Run detection over multiple scales and apply class-aware NMS.
    """
    img_orig = cv2.imread(test_path)
    h_orig, w_orig = img_orig.shape[:2]
    kp2_full, desc2_full = extract_sift_features(img_orig)
    all_dets = []
    for scale in scales:
        img = cv2.resize(img_orig, None, fx=scale, fy=scale)
        kp2, desc2 = extract_sift_features(img)
        for icon_path in glob.glob(os.path.join(icons_dir,'*.png')):
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
    final = []
    dets = sorted(all_dets, key=lambda d: d['score'], reverse=True)
    while dets:
        curr = dets.pop(0)
        final.append(curr)
        dets = [d for d in dets if d['label']!=curr['label'] or _iou(curr['bbox'], d['bbox'])<iou_thresh]
    print(f"After NMS: {len(final)}")
    vis = img_orig.copy()
    for d in final:
        cv2.polylines(vis, [d['proj']], True, (0,255,0),2)
        x,y = d['bbox'][0], d['bbox'][1]
        cv2.putText(vis, f"{d['label']} s={d['scale']:.2f}",
                    (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    batch_multiscale_detect('IconDataset/png','Task3Dataset/images/test_image_6.png')


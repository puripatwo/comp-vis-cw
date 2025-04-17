import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_sift_features(image, grayscale=True):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# ratio_thres: how strict the matching is
def match_features(desc1, desc2, ratio_thres=0.85):
    good_matches = []

    for i, d1 in enumerate(desc1):
        # Compute L2 distance to all descriptors in desc2
        distances = np.linalg.norm(desc2 - d1, axis=1)
        
        if len(distances) < 2:
            continue  # Not enough descriptors

        nearest = np.argsort(distances)[:2]
        m_dist, n_dist = distances[nearest[0]], distances[nearest[1]]

        if m_dist < ratio_thres * n_dist:
            match = cv2.DMatch(_queryIdx=i, _trainIdx=nearest[0], _distance=m_dist)
            good_matches.append(match)

    return good_matches

# Draw matches, change num_matches to show more matches
def draw_feature_matches(img1, kp1, img2, kp2, matches, title="Matches", num_matches=20):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# TESTING
if __name__ == "__main__":

    icon_path = "IconDataset/png/001-lighthouse.png"
    test_image_path = "Task3Dataset/images/test_image_2.png"

    icon = cv2.imread(icon_path)
    test_image = cv2.imread(test_image_path)

    # Extract features
    kp1, desc1 = extract_sift_features(icon)
    kp2, desc2 = extract_sift_features(test_image)

    matches = match_features(desc1, desc2)

    # Visualize
    print(f"Found {len(matches)} matches")
    draw_feature_matches(icon, kp1, test_image, kp2, matches, title="SIFT Raw Matches")

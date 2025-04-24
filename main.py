import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from canny import canny_detector
from hough_transform import detect_hough_lines

from preprocess import preprocess_image
from prepare_templates import prepare_templates
from matching import match_all_templates, evaluate_detections_with_class

from sift_matching import batch_evaluate_all


def detect_edges(image):
    """Apply Sobel filtering to detect edges"""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    edges = np.hypot(grad_x, grad_y)
    return (edges / edges.max() * 255).astype(np.uint8)


def testTask1(folderName):
    # assume that this folder name has a file list.txt that contains the annotation
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    totalError = []
    task1Data = pd.read_csv(folderName+"/list.txt")
    
    for index, row in task1Data.iterrows():
        image_path = row["FileName"]
        target_angle = row["AngleInDegrees"]

        # 1. Load an image in grayscale.
        image = cv2.imread(f"Task1Dataset/{image_path}", cv2.IMREAD_GRAYSCALE)

        # 2. Check if the image is loaded correctly.
        if image is None:
            print(f"Error: Image {image_path} not found or unable to load.")
            continue

        # 3. Perform Canny edge detection.
        edges_custom = canny_detector(image, 50, 150)
        edges = cv2.Canny(image, 50, 150)

        # 4. Detect hough lines.
        lines_custom = detect_hough_lines(edges_custom, threshold_ratio=0.7)
        image_with_lines_custom = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if lines_custom is not None and len(lines_custom) == 2:
            # Extract rho and theta for both lines
            rho1, theta1 = lines_custom[0, 0]
            rho2, theta2 = lines_custom[1, 0]

            # Compute line equations in the form Ax + By = C
            a1, b1 = np.cos(theta1), np.sin(theta1)
            a2, b2 = np.cos(theta2), np.sin(theta2)

            A1, B1, C1 = a1, b1, rho1
            A2, B2, C2 = a2, b2, rho2

            # Solve for intersection (Ax + By = C system)
            det = A1 * B2 - A2 * B1
            if abs(det) > 1e-6:  # Ensure lines are not parallel
                x_int = (C1 * B2 - C2 * B1) / det
                y_int = (A1 * C2 - A2 * C1) / det
                intersection = (int(x_int), int(y_int))

                # Compute endpoints for both lines
                x01, y01 = int(a1 * rho1), int(b1 * rho1)
                x11, y11 = int(x01 + 1000 * (-b1)), int(y01 + 1000 * (a1))
                x21, y21 = int(x01 - 1000 * (-b1)), int(y01 - 1000 * (a1))

                x02, y02 = int(a2 * rho2), int(b2 * rho2)
                x12, y12 = int(x02 + 1000 * (-b2)), int(y02 + 1000 * (a2))
                x22, y22 = int(x02 - 1000 * (-b2)), int(y02 - 1000 * (a2))

                def split_line(x1, y1, x2, y2, x_int, y_int, y_tolerance=2):
                    """Splits a line at intersection favoring 'top' (lower y) and then 'right' (higher x) direction."""
                    if abs(y1 - y2) <= y_tolerance:  # If y-values are very close, decide based on x
                        before = (x1, y1) if x1 > x2 else (x2, y2)  # Favor right (higher x)
                    else:
                        before = (x1, y1) if y1 < y2 else (x2, y2)  # Favor higher (lower y)
                    after = (x1, y1) if before == (x2, y2) else (x2, y2)
                    return before, (int(x_int), int(y_int)), after

                # Split the first and second line
                before1, inter1, after1 = split_line(x11, y11, x21, y21, x_int, y_int)
                before2, inter2, after2 = split_line(x12, y12, x22, y22, x_int, y_int)

                # Draw split lines
                cv2.line(image_with_lines_custom, before1, inter1, (255, 0, 0), 2)
                cv2.line(image_with_lines_custom, inter1, after1, (0, 255, 0), 2)
                cv2.line(image_with_lines_custom, before2, inter2, (255, 0, 0), 2)
                cv2.line(image_with_lines_custom, inter2, after2, (0, 255, 0), 2)
                cv2.circle(image_with_lines_custom, intersection, 5, (0, 0, 255), -1)

                # 5. Calculate angle using "before intersection" segments
                v1 = np.array([before1[0] - inter1[0], before1[1] - inter1[1]])
                v2 = np.array([before2[0] - inter2[0], before2[1] - inter2[1]])

                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)

                dot_product = np.dot(v1, v2)
                angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_degrees = np.degrees(angle_radians)

                error = 0
                error = float(abs(angle_degrees - target_angle))
                totalError.append(error)
                # print(index + 1, angle_degrees, target_angle)

        # # Display the results
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 3, 1)
        # plt.imshow(image, cmap="gray")
        # plt.title("Original Image")
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(edges_custom, cmap="gray")
        # plt.title("Detected Edges")
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.imshow(image_with_lines_custom)
        # plt.title("Hough Lines")
        # plt.axis("off")

        # plt.show()

    print(totalError)
    return (totalError)


def testTask2(iconDir, testDir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    template_dir = iconDir
    test_dir = os.path.join(testDir, "images")

    output_dir = os.path.join(testDir, "results")
    os.makedirs(output_dir, exist_ok=True)

    target_dir = os.path.join(testDir, "annotations")

    # 1. Load and preprocess the templates.
    print("Loading and preprocessing templates...")
    scales = []
    templates_by_class = prepare_templates(template_dir, num_levels=3, preprocess=False, scales=scales, laplacian=True)

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
    all_tp, all_fp, all_fn, all_ious = 0, 0, 0, []
    total_runtime = 0
    n_images = len(test_files)

    for filename in test_files:
        start_time = time.time()

        name = filename.split(".")[0]
        print(f"\nProcessing {name}...")

        target_path = os.path.join(target_dir, f"{name}.csv")
        target_df = pd.read_csv(target_path)

        gt_objects = []
        for index, row in target_df.iterrows(): 
            gt_objects.append({
                "bbox": [row["left"], row["top"], row["right"], row["bottom"]],
                "class": f"0{row["classname"]}"
                })

        # 2. Load and preprocess the test images.
        test_image_path = os.path.join(test_dir, filename)
        # test_image = preprocess_image(test_image_path, grayscale=True)
        test_image = cv2.imread(test_image_path)
        original_image = cv2.imread(test_image_path)

        # 3. Perform template matching.
        detections = match_all_templates(test_image, templates_by_class, score_threshold=0.75, nms_threshold=0.60)

        # 4. Print out the detected objects.
        print(f"{len(detections)} objects detected.")
        for det in detections:
            cls = det['class']
            bbox = det['bbox']
            score = det['score']
            print(f"{cls} at {bbox} (score: {score:.2f})")

            # 5. Visualize the detected objects.
            y1, x1, y2, x2 = bbox
            cv2.rectangle(original_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(original_image, f"{cls}: {score:.2f}", (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 0), thickness=1)
            
        # 6. Evaluate the results.
        metrics = evaluate_detections_with_class(gt_objects, detections, iou_threshold=0.85)

        all_tp += metrics['True Positives']
        all_fp += metrics['False Positives']
        all_fn += metrics['False Negatives']
        all_ious.append(metrics['Average IoU'])

        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        total_runtime += time.time() - start_time
        print(f"Runtime: {total_runtime}")
            
        output_path = os.path.join(output_dir, f"laplacian_{filename}")
        cv2.imwrite(output_path, original_image)

    print("\nMatching complete. Results saved in:", output_dir)

    # 7. Final summary.
    tpr = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0
    fpr = all_fp / (all_tp + all_fn) if all_tp + all_fn else 0
    fnr = all_fn / (all_tp + all_fn) if all_tp + all_fn else 0

    accuracy = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0
    avg_iou = np.mean(all_ious)
    avg_runtime = total_runtime / n_images

    print(f"\nEvaluation Results:")
    print(f"True Positive Rate:   {tpr}")
    print(f"False Positive Rate:  {fpr}")
    print(f"False Negative Rate:  {fnr}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Average IoU:      {avg_iou:.4f}")
    print(f"Average Runtime:  {avg_runtime:.4f} seconds/image")

    return (accuracy, tpr, fpr, fnr)


def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    icons_dir = iconFolderName
    images_dir = os.path.join(testFolderName, "images")
    annos_dir = os.path.join(testFolderName, "annotations")

    avg_acc, avg_tpr, avg_fpr, avg_fnr = batch_evaluate_all(icons_dir, images_dir, annos_dir)
    
    return (avg_acc, avg_tpr, avg_fpr, avg_fnr)


if __name__ == "__main__":
    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task 2 and Task 3.", type=str, required=False)
    parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
    args = parser.parse_args()
    if(args.Task1Dataset!=None):
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        testTask1(args.Task1Dataset)
    if(args.IconDataset!=None and args.Task2Dataset!=None):
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask2(args.IconDataset,args.Task2Dataset)
    if(args.IconDataset!=None and args.Task3Dataset!=None):
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask3(args.IconDataset,args.Task3Dataset)

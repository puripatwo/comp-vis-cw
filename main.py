import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from canny import canny_detector
from hough_transform import detect_hough_lines


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
    task1Data = pd.read_csv(folderName+"/list.txt")
    
    for index, row in task1Data.iterrows():
        image_path = row["FileName"]

        # 1. Load an image in grayscale.
        image = cv2.imread(f"Task1Dataset/{image_path}", cv2.IMREAD_GRAYSCALE)

        # 2. Check if the image is loaded correctly.
        if image is None:
            print(f"Error: Image {image_path} not found or unable to load.")
            continue

        # 3. Perform Canny edge detection.
        edges_custom = canny_detector(image, 50, 150)

        # 4. Detect hough lines.
        lines_custom = detect_hough_lines(edges_custom, threshold_ratio=0.7)
        image_with_lines_custom = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Visualize hough lines
        if lines_custom is not None:
            for line in lines_custom:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = int(a * rho)
                y0 = int(b * rho)
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image_with_lines_custom, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5. Display the results.
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(edges_custom, cmap="gray")
        plt.title("Detected Edges")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image_with_lines_custom)
        plt.title("Hough Lines")
        plt.axis("off")

        plt.show()

        

    return 0


def testTask2(iconDir, testDir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


if __name__ == "__main__":

    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
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



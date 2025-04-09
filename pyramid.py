import cv2
import numpy as np


# Create Gaussian pyramid with image and number of pyramid levels (num_levels)
# returns list: pyramid of images
def create_gaussian_pyramid(image, num_levels=4):

    gaussian_pyramid = [image]
    current_image = image.copy()
    
    for _ in range(1, num_levels):
        downsampled = cv2.pyrDown(current_image)
        gaussian_pyramid.append(downsampled)
        current_image = downsampled
    
    return gaussian_pyramid


# Create Laplacian pyramid with gaussian pyramid
# returns list: Laplacian pyramid of images
def create_laplacian_pyramid(image, num_levels=4):

    # Create Gaussian pyramid
    gaussian_pyramid = [image]
    current_image = image.copy()
   
    for _ in range(1, num_levels):
        downsampled = cv2.pyrDown(current_image)
        gaussian_pyramid.append(downsampled)
        current_image = downsampled
   
    # Create Laplacian pyramid
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        expanded = cv2.pyrUp(gaussian_pyramid[i+1], 
                              dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    
    # Add the smallest gaussian level as the last level
    laplacian_pyramid.append(gaussian_pyramid[-1])
   
    return laplacian_pyramid


def visualize_gaussian_pyramid(pyramid):
    normalized_pyramid = [
        cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for level in pyramid
    ]
    
    max_height = max(level.shape[0] for level in normalized_pyramid)
    total_width = sum(level.shape[1] for level in normalized_pyramid)
    
    canvas = np.zeros((max_height, total_width), dtype=np.uint8)

    current_x = 0
    for level in normalized_pyramid:

        if level.shape[0] < max_height:
            pad_height = max_height - level.shape[0]
            level = np.pad(level, ((0, pad_height), (0, 0)), mode='constant')
        
        canvas[:level.shape[0], current_x:current_x+level.shape[1]] = level
        current_x += level.shape[1]
    
    return canvas


def visualize_laplacian_pyramid(laplacian_pyramid):
    normalized_pyramid = [
        cv2.normalize(np.abs(level), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for level in laplacian_pyramid
    ]
   
    max_height = max(level.shape[0] for level in normalized_pyramid)
    total_width = sum(level.shape[1] for level in normalized_pyramid)
   
    canvas = np.zeros((max_height, total_width), dtype=np.uint8)

    current_x = 0
    for level in normalized_pyramid:
        if level.shape[0] < max_height:
            pad_height = max_height - level.shape[0]
            level = np.pad(level, ((0, pad_height), (0, 0)), mode='constant')
       
        canvas[:level.shape[0], current_x:current_x+level.shape[1]] = level
        current_x += level.shape[1]
   
    return canvas


import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataset_dir = "Task2Dataset/images"
    image_files = [f for f in os.listdir(dataset_dir)]
    
    for image_filename in image_files:

        image_path = os.path.join(dataset_dir, image_filename)
        
        test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if test_image is None:
            print(f"Failed to read image: {image_filename}")
            continue
        
        # Create Gaussian pyramid
        gaussian_pyramid = create_gaussian_pyramid(test_image)
        plt.figure(figsize=(15, 5))
        plt.imshow(visualize_gaussian_pyramid(gaussian_pyramid), cmap='gray')
        plt.title(f'Gaussian Pyramid - {image_filename}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Create Laplacian pyramid
        laplacian_pyramid = create_laplacian_pyramid(test_image)
        plt.figure(figsize=(15, 5))
        plt.imshow(visualize_laplacian_pyramid(laplacian_pyramid), cmap='gray')
        plt.title(f'Laplacian Pyramid - {image_filename}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

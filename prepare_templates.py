import os
from preprocess import preprocess_image
from pyramid import create_gaussian_pyramid, create_custom_gaussian_pyramid, create_laplacian_pyramid, visualize_gaussian_pyramid, visualize_laplacian_pyramid
import cv2


def prepare_templates(template_dir, num_levels=4, preprocess=False, scales=[], laplacian=False):
    """
    Loads all training templates, preprocesses them (grayscale + background masked), and builds Gaussian pyramids.

    Returns:
        dict: {class_label: [pyramid_level_0, ..., pyramid_level_n]}
    """
    templates_by_class = {}

    if laplacian:
        print("Using Laplacian Pyramid...")
    else:
        print("Using Gaussian Pyramid...")

    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            class_label = os.path.splitext(filename)[0]  # "class01" from "class01.png"
            path = os.path.join(template_dir, filename)

            if preprocess:
                processed = preprocess_image(path, grayscale=True)
            else:
                processed = cv2.imread(path)
            
            if processed is None:
                print(f"Skipping {filename} due to load error.")
                continue

            pyramid = create_custom_gaussian_pyramid(processed, scales)
            if len(scales) == 0:
                if laplacian:
                    pyramid = create_laplacian_pyramid(processed, num_levels)
                else:
                    pyramid = create_gaussian_pyramid(processed, num_levels)
            templates_by_class[class_label] = pyramid

    print(f"Loaded {len(templates_by_class)} templates.")
    return templates_by_class


import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataset_dir = "IconDataset/png"
    gaussian_pyramid = prepare_templates(dataset_dir, num_levels=3, preprocess=False, laplacian=True)
    
    for key, values in gaussian_pyramid.items():
        # Create Gaussian pyramid
        plt.figure(figsize=(15, 5))
        plt.imshow(visualize_gaussian_pyramid(values), cmap='gray')
        plt.title(f'Gaussian Pyramid - {key}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

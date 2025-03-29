import os
from preprocess import preprocess_image
from pyramid import create_laplacian_pyramid

def prepare_templates(template_dir, num_levels=4):
    """
    Loads all training templates, preprocesses them (grayscale + background masked),
    and builds Gaussian pyramids.

    Returns:
        dict: {class_label: [pyramid_level_0, ..., pyramid_level_n]}
    """
    templates_by_class = {}

    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            class_label = os.path.splitext(filename)[0]  # "class01" from "class01.png"
            path = os.path.join(template_dir, filename)

            processed = preprocess_image(path, grayscale=True)
            if processed is None:
                print(f"Skipping {filename} due to load error.")
                continue

            pyramid = create_laplacian_pyramid(processed, num_levels)
            templates_by_class[class_label] = pyramid

    print(f"Loaded {len(templates_by_class)} templates.")
    return templates_by_class

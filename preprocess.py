import cv2
import os
import matplotlib.pyplot as plt

# Load an image, convert it to grayscale then set the background to black
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    # Apply the mask to the image
    processed_image = cv2.bitwise_and(image, image, mask=mask)

    return processed_image

######################################
###### Visualise processed image below

def visualize_image(processed):

    plt.figure(figsize=(6, 6))

    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    plt.imshow(processed, cmap='gray')
    plt.title("Processed image (Background Black)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    test_image_path = "Task2Dataset/images/test_image_1.png"

    if os.path.exists(test_image_path):
        processed = preprocess_image(test_image_path)
        if processed is not None:
            visualize_image(processed)
    else:
        print(f"Error: File '{test_image_path}' not found.")

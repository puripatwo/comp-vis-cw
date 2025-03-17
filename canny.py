import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


def canny_detector(img, weak_th=None, strong_th=None):
    """Perform Canny edge detection"""
    
    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Calculating the gradients 
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    
    # Converting Cartesian coordinates to polar  
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Setting thresholds for double thresholding
    mag_max = np.max(mag)
    if not weak_th:
        weak_th = mag_max * 0.1
    if not strong_th:
        strong_th = mag_max * 0.5
    
    # Getting dimensions
    height, width = img.shape
    
    # Non-maximum suppression
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)

            # Determine neighbors based on gradient direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x+1, i_y
            elif grad_ang > 22.5 and grad_ang <= 67.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x+1, i_y+1
            elif grad_ang > 67.5 and grad_ang <= 112.5:
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y+1
            elif grad_ang > 112.5 and grad_ang <= 157.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y+1
                neighb_2_x, neighb_2_y = i_x+1, i_y-1
            elif grad_ang > 157.5 and grad_ang <= 180:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x+1, i_y
            
            # Non-maximum suppression comparisons
            if 0 <= neighb_1_x < width and 0 <= neighb_1_y < height:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue
            if 0 <= neighb_2_x < width and 0 <= neighb_2_y < height:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0
    
    # Double thresholding
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                mag[i_y, i_x] = weak_th
            else:
                mag[i_y, i_x] = strong_th
    
    # Return the final edge-detected image
    return mag

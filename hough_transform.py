import numpy as np


# Returns a array of shape (N, 1, 2)
# N is the number of detected lines
# Each line is represented in polar coords as [rho, theta]
def detect_hough_lines(edges, threshold_ratio, rho_res=1, theta_res=np.pi/180, min_rho_diff=10, min_theta_diff=np.pi/90):
    """Perform hough transform"""

    # Get the height and width of edge image
    image_height, image_width = edges.shape
    
    # Compute rho
    diag_len = int(np.sqrt(image_height**2 + image_width**2))
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    
    # Initialize the accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int64) 
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # For each edge point compute the corresponding rho for every theta
    edge_points = np.argwhere(edges > 0)
    for (y, x) in edge_points:
        for theta_index, (cos_val, sin_val) in enumerate(zip(cos_t, sin_t)):
            rho_val = x * cos_val + y * sin_val
            rho_val = int(round(rho_val))
            rho_index = np.argmin(np.abs(rhos - rho_val))
            accumulator[rho_index, theta_index] += 1

    # Determine threshold to filter out lines that are too close together
    dynamic_threshold = threshold_ratio * np.max(accumulator)

    # Store detected lines
    detected_lines = []
    rho_indices, theta_indices = np.where(accumulator > dynamic_threshold)
    for rho_idx, theta_idx in zip(rho_indices, theta_indices):
        rho_value = rhos[rho_idx]
        theta_value = thetas[theta_idx]
        detected_lines.append([rho_value, theta_value])
    
    # Sort lines by accumulator votes
    sorted_lines = sorted(
        detected_lines,
        key=lambda line: -accumulator[
            np.argmin(np.abs(rhos - line[0])),
            np.argmin(np.abs(thetas - line[1]))]
    )
    
    # Filter out lines that are too close
    filtered_lines = []
    for rho, theta in sorted_lines:
        is_unique = True

        for existing_rho, existing_theta in filtered_lines:
            if (abs(rho - existing_rho) <= min_rho_diff and
                abs(theta - existing_theta) <= min_theta_diff):
                is_unique = False
                break
            
        if is_unique:
            filtered_lines.append([rho, theta])
    
    # Reshape to match cv2.HoughLines format
    result = np.array(filtered_lines, dtype=np.float32).reshape(-1, 1, 2)
    return result

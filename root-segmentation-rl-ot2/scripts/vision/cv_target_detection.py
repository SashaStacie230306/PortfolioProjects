# cv_target_detection

import cv2
import numpy as np

def detect_dark_target_coordinates(image_path):
    """
    Detects the darkest blob (assumed target) and maps it to simulation coordinates.
    Returns (x, y, z) tuple for pipette.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold to find dark regions (adjust threshold value if needed)
    _, mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No dark marker found in image.")

    # Pick the largest dark region (assumed target)
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        raise ValueError("Bad contour detected.")

    # Calculate centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    print(f"Detected pixel coordinates: x={cx}, y={cy}")

    # Image dimensions
    height, width = image.shape

    # Map to simulation coordinates
    sim_x = (cx / width) * 0.5 - 0.25   # [-0.25, 0.25]
    sim_y = (cy / height) * 0.5 - 0.25
    sim_z = 0.32  # Fixed Z value for pipette

    print(f" Simulation coordinates: x={sim_x:.4f}, y={sim_y:.4f}, z={sim_z}")
    return np.array([sim_x, sim_y, sim_z], dtype=np.float32)

if __name__ == "__main__":
    image_path = "Robotics_tasks/textures/03.png"
    coords = detect_dark_target_coordinates(image_path)

import cv2
import imutils
from imutils import contours
from scipy.spatial import distance as dist
import numpy as np

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def measure_objects(image_path, reference_width):
    """
    Measure objects in an image given a reference object width.

    Parameters:
        image_path (str): Path to the input image.
        reference_width (float): Width of the reference object in the same units you want output (e.g., inches or cm).

    Returns:
        None. Displays the image with measurements.
    """

    # Load image and preprocess
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right
    (cnts, _) = contours.sort_contours(cnts)

    pixels_per_metric = None

    for c in cnts:
        # Ignore small contours that could be noise
        if cv2.contourArea(c) < 100:
            continue

        # Compute rotated bounding box of contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order points: top-left, top-right, bottom-right, bottom-left
        box = imutils.perspective.order_points(box)

        # Compute midpoints between corners
        (tl, tr, br, bl) = box
        (topMidX, topMidY) = midpoint(tl, tr)
        (bottomMidX, bottomMidY) = midpoint(bl, br)
        (leftMidX, leftMidY) = midpoint(tl, bl)
        (rightMidX, rightMidY) = midpoint(tr, br)

        # Compute Euclidean distances between midpoints
        width_pixels = dist.euclidean((topMidX, topMidY), (bottomMidX, bottomMidY))
        height_pixels = dist.euclidean((leftMidX, leftMidY), (rightMidX, rightMidY))

        # If pixels_per_metric not set, assume this contour is the reference object
        if pixels_per_metric is None:
            pixels_per_metric = width_pixels / reference_width
            # Optionally, you can check if width_pixels > height_pixels and swap if needed
            # print(f"Reference object pixels per unit: {pixels_per_metric:.2f}")
            continue

        # Compute size of the object
        width = width_pixels / pixels_per_metric
        height = height_pixels / pixels_per_metric

        # Draw the bounding box
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

        # Draw midpoints for width
        cv2.circle(image, (int(topMidX), int(topMidY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(bottomMidX), int(bottomMidY)), 5, (255, 0, 0), -1)

        # Draw midpoints for height
        cv2.circle(image, (int(leftMidX), int(leftMidY)), 5, (0, 0, 255), -1)
        cv2.circle(image, (int(rightMidX), int(rightMidY)), 5, (0, 0, 255), -1)

        # Put text of measurements
        cv2.putText(image, "{:.2f} units".format(width),
                    (int(topMidX - 15), int(topMidY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(image, "{:.2f} units".format(height),
                    (int(rightMidX + 10), int(rightMidY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the output image
    cv2.imshow("Measured Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # Replace 'test_image.jpg' with your image path
    # Replace 3.5 with your reference object width in inches or cm
    measure_objects("test_image.jpg", reference_width=3.5)

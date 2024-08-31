import cv2
import matplotlib.pyplot as plt
import numpy as np

def darken_dark_areas(img, darkening_factor, brightness_threshold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 2] < brightness_threshold
    hsv[:, :, 2] = np.where(mask, np.clip(hsv[:, :, 2] * darkening_factor, 0, 255), hsv[:, :, 2])
    darkened_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return darkened_img

def canny(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    # triangle = np.array([[(0, 200), (width, 185), (230, 115)]]) test_img4
    # triangle = np.array([[(0, 400), (width, 500), (265, 335)]]) test_img6
    triangle = np.array([[(200, 0), (1200, 0), (805,785)]])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, (255, 255, 255))  # Fills with white color (255, 255, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_img

# Correct path to the image file
image_path = 'test_img7.jpg'

# Load image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found. Please check the file path and try again.")

lane_img = image.copy()

plt.imshow(cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB))
plt.show()

darkening_factor = 0.80
brightness_threshold = 200

lane_img = darken_dark_areas(lane_img, darkening_factor, brightness_threshold)


canny_img = canny(lane_img)


masked_image = region_of_interest(canny_img)


lines = cv2.HoughLinesP(masked_image, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)


line_img = display_lines(lane_img, lines)


combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)


plt.imshow(cv2.cvtColor(combo_img, cv2.COLOR_BGR2RGB))
plt.show()

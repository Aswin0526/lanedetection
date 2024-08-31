import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Global variables for averaging lines
avgLeft = (0, 0, 0, 0)
avgRight = (0, 0, 0, 0)

# Function to increase contrast using CLAHE
def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return enhanced_img


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    if denom == 0:
        return np.array([np.nan, np.nan])
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def movingAverage(avg, new_sample, N=20):
    if avg == 0:
        return new_sample
    avg -= avg / N
    avg += new_sample / N
    return avg

# Function to convert image to grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# Function to apply Canny edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Function to apply Gaussian blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Function to mask region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to draw lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    global avgLeft, avgRight
    largestLeftLineSize = 0
    largestRightLineSize = 0
    largestLeftLine = (0, 0, 0, 0)
    largestRightLine = (0, 0, 0, 0)

    if lines is None:
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        return
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if slope > 0.5:  # right
                if size > largestRightLineSize:
                    largestRightLineSize = size
                    largestRightLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif slope < -0.5:  # left
                if size > largestLeftLineSize:
                    largestLeftLineSize = size
                    largestLeftLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    imgHeight, imgWidth = img.shape[:2]
    upLinePoint1 = np.array([0, int(imgHeight - (imgHeight / 3))])
    upLinePoint2 = np.array([int(imgWidth), int(imgHeight - (imgHeight / 3))])
    downLinePoint1 = np.array([0, imgHeight])
    downLinePoint2 = np.array([imgWidth, imgHeight])
    
    p3 = np.array([largestLeftLine[0], largestLeftLine[1]])
    p4 = np.array([largestLeftLine[2], largestLeftLine[3]])
    upLeftPoint = seg_intersect(upLinePoint1, upLinePoint2, p3, p4)
    downLeftPoint = seg_intersect(downLinePoint1, downLinePoint2, p3, p4)
    if np.isnan(upLeftPoint[0]) or np.isnan(downLeftPoint[0]):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        return
    cv2.line(img, (int(upLeftPoint[0]), int(upLeftPoint[1])), (int(downLeftPoint[0]), int(downLeftPoint[1])), [0, 0, 255], 8)

    avgx1, avgy1, avgx2, avgy2 = avgLeft
    avgLeft = (movingAverage(avgx1, upLeftPoint[0]), movingAverage(avgy1, upLeftPoint[1]), movingAverage(avgx2, downLeftPoint[0]), movingAverage(avgy2, downLeftPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)

    p5 = np.array([largestRightLine[0], largestRightLine[1]])
    p6 = np.array([largestRightLine[2], largestRightLine[3]])
    upRightPoint = seg_intersect(upLinePoint1, upLinePoint2, p5, p6)
    downRightPoint = seg_intersect(downLinePoint1, downLinePoint2, p5, p6)
    if np.isnan(upRightPoint[0]) or np.isnan(downRightPoint[0]):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        return
    cv2.line(img, (int(upRightPoint[0]), int(upRightPoint[1])), (int(downRightPoint[0]), int(downRightPoint[1])), [0, 0, 255], 8)

    avgx1, avgy1, avgx2, avgy2 = avgRight
    avgRight = (movingAverage(avgx1, upRightPoint[0]), movingAverage(avgy1, upRightPoint[1]), movingAverage(avgx2, downRightPoint[0]), movingAverage(avgy2, downRightPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgRight
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)

# Function to perform Hough transform and detect lines
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Function to overlay the lines on the original image
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Reading in an image
image = mpimg.imread('carimg1.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
plt.show()


# Increase contrast of the image
enhanced_image = increase_contrast(image)

# Convert the image to grayscale
gray = grayscale(enhanced_image)

# Apply Gaussian blur
blur_gray = gaussian_blur(gray, 5)

# Apply Canny edge detection
edges = canny(blur_gray, 50, 150)

# Define a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(50, imshape[0]), (450, 320), (490, 320), (imshape[1]-50, imshape[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)

# Define the Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_len = 40
max_line_gap = 20
line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

# Overlay the lines on the original image
result = weighted_img(line_img, enhanced_image)

# Display the final image
plt.imshow(result)
plt.show()

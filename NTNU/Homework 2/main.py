import cv2 as cv
import numpy as np

img = cv.imread('Lenna.png')

#convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#apply gaussian image blur
blurred_img = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)

def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row -1)/2)
    pad_width = int((kernel_col -1)/2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    #place original image in the center
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    return output

def sobel_filter(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = convolution(image, sobel_x)
    gradient_y = convolution(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x**2 + np.square(gradient_y))
    angles = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    angles[angles < 0] += 180
    gradient_magnitude = gradient_magnitude.astype('uint8')
    return gradient_magnitude, angles


def non_maximum_suppression(image, angles):
    image_row, image_col = image.shape
    suppressed = np.zeros(image.shape)
    for i in range(1, image_row - 1):
        for j in range(1, image_col - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detection(image):
    image, angle = sobel_filter(blurred_img)
    image = non_maximum_suppression(image, angle)
    high_threshold = 100
    low_threshold = 50
    _, img_low = cv.threshold(image, low_threshold, 255, cv.THRESH_BINARY)
    _, img_high = cv.threshold(image, high_threshold, 255, cv.THRESH_BINARY)
    img_final = np.logical_and(img_low, img_high).astype(np.uint8) * 255
    test = hysteresis(img_final, img_low)
    return test

result_img = canny_edge_detection(blurred_img)

cv.imshow('Gaussian blur', result_img)
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()


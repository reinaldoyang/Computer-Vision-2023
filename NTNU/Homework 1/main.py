import cv2 as cv
import numpy as np

img = cv.imread('Lenna.png')

# print(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
m, n = gray.shape

def average_filtering(gray, kernel_size):
    m, n = gray.shape
    mask = np.ones((1, kernel_size), dtype = int) / kernel_size
    img_new = np.zeros([m, n], dtype = np.uint8)
    kernel_radius = kernel_size // 2

    # Convolve the image with the horizontal kernel
    for i in range(m):
        for j in range(kernel_radius, n - kernel_radius):
            img_new[i, j] = np.sum(gray[i, j - kernel_radius: j + kernel_radius + 1] * mask)
    img_new = img_new.astype(np.uint8)
    # Convolve the result of the horizontal convolution with the vertical kernel
    img_new_2 = np.zeros([m, n], dtype=np.uint8)
    for i in range(kernel_radius, m - kernel_radius):
        for j in range(n):
            img_new_2[i, j] = np.sum(img_new[i - kernel_radius: i + kernel_radius + 1, j] * mask)
    return img_new_2

img_filtered = average_filtering(gray, 11)
print(img_filtered)
cv.imshow("Filtered Image", img_filtered)
cv.imwrite('./filter_11.jpg',img_filtered)
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()
import cv2
import numpy as np

ima = cv2.imread("Image.png")
sample = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)

#flipping the kernel
def conv_transform(image):
    image_copy = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy

# Convolving the Sobel Kernel over the image
def conv(image, kernel):
    kernel = conv_transform(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h//2
    w = kernel_w//2

    image_conv = np.zeros(image.shape)

    for i in range(h, image_h-h):
        for j in range(w, image_w-w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = (sum + kernel[m][n] * image[i-h+m][j-w+n])
            
            image_conv[i][j] = sum

    return image_conv

# Storing the values for Sobel Kernel to find gradient of Y
kernel = np.zeros(shape=(3,3))
kernel[0, 0] = 1
kernel[0, 1] = 2
kernel[0, 2] = 1
kernel[1, 0] = 0
kernel[1, 1] = 0
kernel[1, 2] = 0
kernel[2, 0] = -1
kernel[2, 1] = -2
kernel[2, 2] = -1

gy = conv(sample, kernel)
cv2.imshow("gradient y", gy)
cv2.imwrite("gradientY.png", gy)

# Storing the values for Sobel Kernel to find gradient of X
kernel[0, 0] = 1
kernel[0, 1] = 0
kernel[0, 2] = -1
kernel[1, 0] = 2
kernel[1, 1] = 0
kernel[1, 2] = -2
kernel[2, 0] = 1
kernel[2, 1] = 0
kernel[2, 2] = -1

gx = conv(sample, kernel)
cv2.imshow("gradient x", gx)
cv2.imwrite("gradientX.png", gx)

cv2.waitKey(0)
cv2.destroyAllWindows()


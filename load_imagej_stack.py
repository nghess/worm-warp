import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt


def normalize_img(image):
    image = image.astype(float)
    image -= np.amin(image.astype(float))
    image /= np.amax(image)
    return image * 255

img = io.imread('stacks/MAX_DOI_231016_001.nd2 - DOI_231016_001.nd2 (series 1) - C=0.tif')
img = cv2.resize(img, (0, 0), fx=.5, fy=.5)


# Apply median blur
blurred = cv2.medianBlur(img[:,:,0], ksize=5)  # ksize is the kernel size, must be an odd number
mask = normalize_img(blurred)
img = normalize_img(img[:,:,0])

print(np.amax(mask))
print(np.amin(mask))

blank = np.zeros_like(mask)

binary = np.where(mask < .95, blank, 1)
mask = img * binary


print(binary[:100,:100])

cv2.imshow('preview', np.array(mask, dtype=np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows
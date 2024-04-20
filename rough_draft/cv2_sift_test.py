import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the two images
img1 = cv2.imread('test_sequence/image1.png', cv2.IMREAD_GRAYSCALE)  # 0 for grayscale
img2 = cv2.imread('test_sequence/image2.png', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Match descriptors using FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Ratio test to find good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Compute homography
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Check if anything's wrong with M
# Compute homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Check for issues
if M is None:
    print("Homography matrix is None. Check your keypoint matches.")
else:
    print("Shape of M:", M.shape)
    print("Type of M:", M.dtype)
    
    # Ensure it's in the correct type
    M = M.astype('float32')
    
    # Warp the second image
    height, width = img1.shape
    try:
        aligned_img2 = cv2.warpPerspective(img2, M, (width, height))
    except Exception as e:
        print(f"An error occurred: {e}")

# Warp the second image to align with the first
aligned_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

cv2.imwrite("test_sequence/img1_align.png", img1)
cv2.imwrite("test_sequence/img2_align.png", aligned_img2)

# Plotting the aligned images
plt.subplot(131), plt.imshow(img1, cmap='gray'), plt.title('Image 1')
plt.subplot(132), plt.imshow(img2, cmap='gray'), plt.title('Image 2')
plt.subplot(133), plt.imshow(aligned_img2, cmap='gray'), plt.title('Aligned Image 2')
plt.show()

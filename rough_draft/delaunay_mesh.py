import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def sift_images(src_img, tgt_img):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    keypoints1, descriptors1 = sift.detectAndCompute(src_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(tgt_img, None)

    # Match descriptors using FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    tgt_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    return src_pts, tgt_pts


src = cv2.imread('test_sequence/image2.png', cv2.IMREAD_GRAYSCALE)  # 0 for grayscale
tgt = cv2.imread('test_sequence/image1.png', cv2.IMREAD_GRAYSCALE)

src_pts, tgt_pts = sift_images(src, tgt)
#hull = ConvexHull(src_pts)

src_delaunay = Delaunay(src_pts)

plt.figure()
plt.triplot(src_pts[:, 0], src_pts[:, 1], src_delaunay.simplices)
plt.plot(src_pts[:, 0], src_pts[:, 1], 'o')
plt.show()

print("Done!")
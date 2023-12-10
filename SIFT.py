import cv2
import numpy as np 
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

foreground_path = "./linear_interpolation/img/2-1.jpg"
middleground_path = "./linear_interpolation/img/2-2.jpg"
background_path = "./linear_interpolation/img/2-3.jpg"

foreground = cv2.imread(foreground_path, cv2.IMREAD_GRAYSCALE)
# middleground = cv2.imread(middleground_path)
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

# cv2.imshow("origin", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(foreground, None)
kp2, des2 = sift.detectAndCompute(background, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = foreground.shape
    pts = np.float32([[0,0], [0, h-1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    background = cv2.polylines(background, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor=None, 
                matchesMask=matchesMask,  # draw only inliers
                flags = 2)


result = cv2.drawMatches(foreground, kp1, background, kp2, good, None, **draw_params)

plt.imshow(result, 'gray'), plt.show()
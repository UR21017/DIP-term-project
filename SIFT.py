import cv2
import numpy as np 
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

foreground_path = "./linear_interpolation/img/2-1.jpg"
# middleground_path = "./linear_interpolation/img/2-2.jpg"
background_path = "./linear_interpolation/img/2-3.jpg"

foreground = cv2.imread(foreground_path, cv2.IMREAD_GRAYSCALE)
# middleground = cv2.imread(middleground_path)
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

# cv2.imshow("origin", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def SIFT(foreground, background):

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
       
       # 將background 沿著polylines切割
        mask = np.zeros_like(background)
        cv2.fillPoly(mask, [np.int32(dst)], 255)
        background_crop = cv2.bitwise_and(background, mask)

        pts_rect = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        M_inv = cv2.getPerspectiveTransform(dst, pts_rect)
        background = cv2.warpPerspective(background_crop, M_inv, (w, h))
        foreground = foreground
        
        cv2.imshow("background", background)
        cv2.imshow("foreground", foreground)

        

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor=None, 
                    matchesMask=matchesMask,  # draw only inliers
                    flags = 2)


    result = cv2.drawMatches(foreground, kp1, background, kp2, good, None, **draw_params)

    plt.imshow(result, 'gray'), plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

result = SIFT(foreground, background)

# # Stereo Vision Depth Estimation
# stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(foreground, background)

# # Normalize and apply color map to the disparity map
# depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
# depth_map_colored = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_JET)

# # Depth-based Blending
# blurred_background = cv2.GaussianBlur(background, (15, 15), 0)
# result = cv2.addWeighted(blurred_background, 0.7, depth_map_colored, 0.3, 0)

# # Display the result
# plt.imshow(result), plt.show()
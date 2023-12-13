import matplotlib.pyplot as plt
import numpy as np
import cv2

from matplotlib.backend_bases import MouseButton

import glob

# 图片路径列表
folder_path = "./linear_interpolation/img2/"
image_paths = glob.glob(f"{folder_path}*.jpg")

# image_paths = ["./linear_interpolation/img2/1.jpg", "./linear_interpolation/img2/2.jpg", "./linear_interpolation/img2/3.jpg", "./linear_interpolation/img2/4.jpg", "./linear_interpolation/img2/5.jpg"]
current_image_index = 0  # 当前显示的图片索引




MIN_MATCH_COUNT = 10

foreground_path = "./linear_interpolation/img2/1.jpg"
# middleground_path = "./linear_interpolation/img/2-2.jpg"
background_path = "./linear_interpolation/img2/2.jpg"

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
       
       # 將background 沿著polylines切割
        mask = np.zeros_like(background)
        cv2.fillPoly(mask, [np.int32(dst)], 255)
        background_crop = cv2.bitwise_and(background, mask)

        pts_rect = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        M_inv = cv2.getPerspectiveTransform(dst, pts_rect)
        background = cv2.warpPerspective(background_crop, M_inv, (w, h))
        foreground = foreground 
        
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor=None, 
                    matchesMask=matchesMask,  # draw only inliers
                    flags = 2)


    result = cv2.drawMatches(foreground, kp1, background, kp2, good, None, **draw_params)

    return background




def load_image(index):
    # 加载图像并进行处理（例如，转换为灰度）
    image_path = image_paths[index]
    background = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    foreground = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    
    # processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = SIFT(foreground,  background)

    return processed_image

def on_click(event):
    if event.button is MouseButton.LEFT:
        global current_image_index

        # 切换到下一张图片
        current_image_index = (current_image_index + 1) % len(image_paths)

        # 更新显示的图像
        ax.clear()
        ax.imshow(load_image(current_image_index), cmap='gray')  # 指定cmap为'gray'
        ax.set_title(f"Image {current_image_index + 1}")

        # 刷新图形
        plt.draw()

# 创建初始图形
fig, ax = plt.subplots()
ax.imshow(load_image(current_image_index), cmap='gray')  # 指定cmap为'gray'
ax.set_title(f"Image {current_image_index + 1}")

# 将鼠标点击事件连接到更新图像的函数
plt.connect('button_press_event', on_click)

# 显示图形
plt.show()
import cv2
import numpy as np

def linear_interpolation(images, weights):
    # images: 一系列的圖像
    # weights: 對應於每個圖像的權重，用於線性插值
    
    # 確保權重的總和為1
    weights = np.array(weights)
    weights /= weights.sum()

    # 初始化插值後的圖像
    interpolated_image = np.zeros_like(images[0], dtype=np.float32)

    # 進行線性插值
    for i in range(len(images)):
        interpolated_image += images[i] * weights[i]

    # 四捨五入到 8 位圖像
    interpolated_image = np.round(interpolated_image).astype(np.uint8)

    return interpolated_image

# 例子
image1 = cv2.imread('img/2-1.jpg')
image2 = cv2.imread('img/2-2.jpg')
image3 = cv2.imread('img/2-3.jpg')

# 假設有三個不同焦點的圖像，可以設定不同的權重
weights = [0.2, 0.6, 0.2]

# 進行線性插值
interpolated_image = linear_interpolation([image1, image2, image3], weights)

# 顯示原始圖像和插值後的圖像
cv2.imshow('Original Image 1', image1)
cv2.imshow('Original Image 2', image2)
cv2.imshow('Original Image 3', image3)
cv2.imshow('Interpolated Image', interpolated_image)
cv2.imwrite("./results/1.jpg", interpolated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
